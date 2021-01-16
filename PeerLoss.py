#################################
#
# File: PeerLoss.py
# This code modifies some code from cleanlab.
# 
#################################
import numpy as np
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from scipy.optimize import minimize
from scipy import sparse

from sklearn.linear_model import LogisticRegression as logreg
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator
import inspect
from cleanlab.util import (
    assert_inputs_are_valid,
    value_counts,
)
from cleanlab.latent_estimation import (
    estimate_py_noise_matrices_and_cv_pred_proba,
    estimate_py_and_noise_matrices_from_probabilities,
    estimate_cv_predicted_probabilities,
)
from cleanlab.latent_algebra import (
    compute_py_inv_noise_matrix,
    compute_noise_matrix_from_inverse,
)
from cleanlab.pruning import get_noise_indices
from cleanlab.classification import LearningWithNoisyLabels

def sigmoid(z):
    return 1./(1. + np.exp(-z))

def safe_sparse_dot(a, b, dense_output=False):
    if a.ndim > 2 or b.ndim > 2:
        if sparse.issparse(a):
            # sparse is always 2D. Implies b is 3D+
            # [i, j] @ [k, ..., l, m, n] -> [i, k, ..., l, n]
            b_ = np.rollaxis(b, -2)
            b_2d = b_.reshape((b.shape[-2], -1))
            ret = a @ b_2d
            ret = ret.reshape(a.shape[0], *b_.shape[1:])
        elif sparse.issparse(b):
            # sparse is always 2D. Implies a is 3D+
            # [k, ..., l, m] @ [i, j] -> [k, ..., l, j]
            a_2d = a.reshape(-1, a.shape[-1])
            ret = a_2d @ b
            ret = ret.reshape(*a.shape[:-1], b.shape[1])
        else:
            ret = np.dot(a, b)
    else:
        ret = a @ b

    if (sparse.issparse(a) and sparse.issparse(b)
            and dense_output and hasattr(ret, "toarray")):
        return ret.toarray()
    return ret

def logistic(w, x):
    c = 0.
    if w.size == x.shape[1] + 1:
        c = w[-1]
        w = w[:-1]
    
    z = safe_sparse_dot(x, w) + c 
    return sigmoid(z)

def log_logistic(w, x, y, epsilon=1e-12):
    z = logistic(w, x)
    out = y*np.log(z + epsilon) + (1-y)*np.log(1-z+epsilon)
    return -out

class PeerLoss:
    """ Plug Peer Loss into Logistic Regression
    """
    def __init__(self, A, delta=[1., 1.], alpha=0.5):
        self.weights = None
        self.A = np.ones(A.shape)
        self.alpha = alpha

        for d in delta:
            if d == 0:
                raise ValueError("Delta cannot be zero.")

        for i in range(len(A)):
            self.A[i] = 1./delta[A[i]]

    def predict_prob(self, X):
        if self.weights is None:
            raise ValueError("Estimator not fitted.")
        
        return logistic(self.weights, X)

    def predict(self, X):
        prob = self.predict_prob(X)
        return (prob > 0.5).astype(np.int)

    def fit(self, X, y, sample_weight=None):
        if self.weights is None:
            self.weights = np.random.normal(size=(X.shape[1]+1))

        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])

        def peer_loss(w):
            l = log_logistic(w, X, y) - self.alpha * log_logistic(w, xt, yt, epsilon=1e-5)
            out = np.mean(self.A * (sample_weight * l))
            # print(f"Peer Loss: {out}.")
            return out

        xt = np.random.permutation(X)
        yt = np.random.permutation(y)
        res = minimize(peer_loss, self.weights, method='BFGS')
        self.weights = res.x

class SurrogateLoss(LearningWithNoisyLabels):
    def __init__(
        self,
        clf = None,
        seed = None,
        # Hyper-parameters (used by .fit() function)
        cv_n_folds = 5,
        prune_method = 'prune_by_noise_rate',
        converge_latent_estimates = False,
        pulearning = None,
        noise_matrix = None
    ):
        super().__init__(clf, seed, cv_n_folds, prune_method, converge_latent_estimates, pulearning)
        self.noise_matrix = noise_matrix

    def fit(
        self, 
        X,
        s,
        sample_weight=None,
        psx = None,
        thresholds = None,
        noise_matrix = None,
        inverse_noise_matrix = None, 
    ):
        '''This method implements the confident learning. It counts examples that are likely
        labeled correctly and incorrectly and uses their ratio to create a predicted
        confusion matrix.
        This function fits the classifier (self.clf) to (X, s) accounting for the noise in
        both the positive and negative sets.
        Parameters
        ----------
        X : np.array
          Input feature matrix (N, D), 2D numpy array
        s : np.array
          A binary vector of labels, s, which may contain mislabeling.
        psx : np.array (shape (N, K))
          P(s=k|x) is a matrix with K (noisy) probabilities for each of the N examples x.
          This is the probability distribution over all K classes, for each
          example, regarding whether the example has label s==k P(s=k|x). psx should
          have been computed using 3 (or higher) fold cross-validation.
          If you are not sure, leave psx = None (default) and
          it will be computed for you using cross-validation.
        thresholds : iterable (list or np.array) of shape (K, 1)  or (K,)
          P(s^=k|s=k). If an example has a predicted probability "greater" than
          this threshold, it is counted as having hidden label y = k. This is
          not used for pruning, only for estimating the noise rates using
          confident counts. This value should be between 0 and 1. Default is None.
        noise_matrix : np.array of shape (K, K), K = number of classes
          A conditional probablity matrix of the form P(s=k_s|y=k_y) containing
          the fraction of examples in every class, labeled as every other class.
          Assumes columns of noise_matrix sum to 1. 
        inverse_noise_matrix : np.array of shape (K, K), K = number of classes
          A conditional probablity matrix of the form P(y=k_y|s=k_s) representing
          the estimated fraction observed examples in each class k_s, that are
          mislabeled examples from every other class k_y. If None, the
          inverse_noise_matrix will be computed from psx and s.
          Assumes columns of inverse_noise_matrix sum to 1.
        Output
        ------
          Returns (noise_mask, sample_weight)'''
        # print("[DEBUG][STAT] Pruning.")
        # Check inputs
        X = X.values
        s = s.values
        assert_inputs_are_valid(X, s, psx)
        if noise_matrix is not None and np.trace(noise_matrix) <= 1:
            t = np.round(np.trace(noise_matrix), 2)
            raise ValueError("Trace(noise_matrix) is {}, but must exceed 1.".format(t))
        if inverse_noise_matrix is not None and np.trace(inverse_noise_matrix) <= 1:
            t = np.round(np.trace(inverse_noise_matrix), 2)
            raise ValueError("Trace(inverse_noise_matrix) is {}, but must exceed 1.".format(t))

        # Number of classes
        self.K = len(np.unique(s))

        # 'ps' is p(s=k)
        self.ps = value_counts(s) / float(len(s))

        self.confident_joint = None
        # If needed, compute noise rates (fraction of mislabeling) for all classes. 
        # Also, if needed, compute P(s=k|x), denoted psx.
        
        # Set / re-set noise matrices / psx; estimate if not provided.
        if self.noise_matrix is None:
            if noise_matrix is not None:
                self.noise_matrix = noise_matrix
                if inverse_noise_matrix is None:
                    self.py, self.inverse_noise_matrix = compute_py_inv_noise_matrix(self.ps, self.noise_matrix)
            if inverse_noise_matrix is not None:
                self.inverse_noise_matrix = inverse_noise_matrix
                if noise_matrix is None:
                    self.noise_matrix = compute_noise_matrix_from_inverse(self.ps, self.inverse_noise_matrix)
            if noise_matrix is None and inverse_noise_matrix is None:
                if psx is None:
                    self.py, self.noise_matrix, self.inverse_noise_matrix, self.confident_joint, psx = \
                    estimate_py_noise_matrices_and_cv_pred_proba(
                        X = X,
                        s = s,
                        clf = self.clf,
                        cv_n_folds = self.cv_n_folds,
                        thresholds = thresholds,
                        converge_latent_estimates = self.converge_latent_estimates,
                        seed = self.seed,
                    )
                else: # psx is provided by user (assumed holdout probabilities)
                    self.py, self.noise_matrix, self.inverse_noise_matrix, self.confident_joint = \
                    estimate_py_and_noise_matrices_from_probabilities(
                        s = s, 
                        psx = psx,
                        thresholds = thresholds,
                        converge_latent_estimates = self.converge_latent_estimates,
                    )
        else:
            self.py, self.inverse_noise_matrix = compute_py_inv_noise_matrix(self.ps, self.noise_matrix)

        if psx is None: 
            psx = estimate_cv_predicted_probabilities(
                X = X,
                labels = s,
                clf = self.clf,
                cv_n_folds = self.cv_n_folds,
                seed = self.seed,
            )

        # if pulearning == the integer specifying the class without noise.
        if self.K == 2 and self.pulearning is not None: # pragma: no cover
            # pulearning = 1 (no error in 1 class) implies p(s=1|y=0) = 0
            self.noise_matrix[self.pulearning][1 - self.pulearning] = 0
            self.noise_matrix[1 - self.pulearning][1 - self.pulearning] = 1
            # pulearning = 1 (no error in 1 class) implies p(y=0|s=1) = 0
            self.inverse_noise_matrix[1 - self.pulearning][self.pulearning] = 0
            self.inverse_noise_matrix[self.pulearning][self.pulearning] = 1
            # pulearning = 1 (no error in 1 class) implies p(s=1,y=0) = 0
            self.confident_joint[self.pulearning][1 - self.pulearning] = 0
            self.confident_joint[1 - self.pulearning][1 - self.pulearning] = 1

        # This is the actual work of this function.

        # Get the indices of the examples we wish to prune
        self.noise_mask = get_noise_indices(
            s,
            psx,
            inverse_noise_matrix = self.inverse_noise_matrix,
            confident_joint = self.confident_joint,
            prune_method = self.prune_method,
        ) 

        X_mask = ~self.noise_mask
        X_pruned = X[X_mask]
        s_pruned = s[X_mask]

        # print("[DEBUG][STAT] Fitting Surrogate Loss.")
        # Check if sample_weight in clf.fit(). Compatible with Python 2/3.
        if hasattr(inspect, 'getfullargspec') and \
                'sample_weight' in inspect.getfullargspec(self.clf.fit).args or \
                hasattr(inspect, 'getargspec') and \
                'sample_weight' in inspect.getargspec(self.clf.fit).args:
            # Re-weight examples in the loss function for the final fitting
            # s.t. the "apparent" original number of examples in each class
            # is preserved, even though the pruned sets may differ.
            self.sample_weight = np.ones(np.shape(s_pruned))
            for k in range(self.K): 
                self.sample_weight[s_pruned == k] = 1.0 / self.noise_matrix[k][k]
            # print("[DEBUG][IF] True branch.")
            self.clf.fit(X_pruned, s_pruned, sample_weight=self.sample_weight)
        else:
            # This is less accurate, but its all we can do if sample_weight isn't available.
            print("[INFO][IF] False branch.")
            self.clf.fit(X_pruned, s_pruned)
        # print("[DEBUG][STAT] Fitted.")
        return self.clf