import numpy as np 
import pandas as pd
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
from fairlearn.reductions import ConditionalSelectionRate
from cleanlab.classification import LearningWithNoisyLabels

_GROUP_ID = "group_id"
_EVENT = "event"
_LABEL = "label"
_LOSS = "loss"
_PREDICTION = "pred"
_ALL = "all"
_SIGN = "sign"
_DIFF = "diff"

class ProxyEqualizedOdds(ConditionalSelectionRate):
    def __init__(self, error_rate=[[0.3, 0.3], [0.0, 0.0]]):
        super().__init__()
        self.error_rate = error_rate
    
    def load_data(self, X, y, **kwargs):
        """Load the specified data into the object."""
        super().load_data(X, y,
                          event=pd.Series(y).apply(lambda y: _LABEL + "=" + str(int(y))),
                          **kwargs)

    def gamma(self, predictor):
        """Calculate the degree to which constraints are currently violated by the predictor."""
        pred = predictor(self.X)
        self.tags[_PREDICTION] = pred
        expect_event = self.tags.groupby(_EVENT).mean()
        expect_group_event = self.tags.groupby(
            [_EVENT, _GROUP_ID]).mean()

        num_grp = len(self.error_rate)
        tprs = [0 for _ in range(num_grp)]
        # print(expect_group_event)
        for i in range(num_grp):
            tprs[i] = expect_group_event.loc[(expect_group_event.index.get_level_values(_GROUP_ID) == i)].groupby([_EVENT]).mean()
            expect_group_event.loc[('label=1', i), 'pred'] = (1 - self.error_rate[i][0]) * tprs[i].loc['label=1', 'pred'] + self.error_rate[i][0] * tprs[i].loc['label=0', 'pred']
            expect_group_event.loc[('label=0', i), 'pred'] = (1 - self.error_rate[i][1]) * tprs[i].loc['label=0', 'pred'] + self.error_rate[i][1] * tprs[i].loc['label=1', 'pred']

        # neg = expect_group_event.loc[(expect_group_event.index.get_level_values(_GROUP_ID) == 0.0)].groupby([_EVENT]).mean()
        # pos = expect_group_event.loc[(expect_group_event.index.get_level_values(_GROUP_ID) == 1.0)].groupby([_EVENT]).mean()

        # expect_group_event.loc[('label=1.0', 1), 'pred'] = (1 - self.error_rate[1][0]) * pos.loc['label=1.0', 'pred'] + self.error_rate[1][1] * pos.loc['label=0.0', 'pred']
        # expect_group_event.loc[('label=0.0', 1), 'pred'] = (1 - self.error_rate[1][1]) * pos.loc['label=0.0', 'pred'] + self.error_rate[1][0] * pos.loc['label=1.0', 'pred']

        # expect_group_event.loc[('label=1.0', 0), 'pred'] = (1 - self.error_rate[0][0]) * neg.loc['label=1.0', 'pred'] + self.error_rate[0][1] * neg.loc['label=0.0', 'pred']
        # expect_group_event.loc[('label=0.0', 0), 'pred'] = (1 - self.error_rate[0][1]) * neg.loc['label=0.0', 'pred'] + self.error_rate[0][0] * neg.loc['label=1.0', 'pred']

        expect_event = expect_group_event.groupby(_EVENT).mean()
        expect_group_event[_DIFF] = expect_group_event[_PREDICTION] - expect_event[_PREDICTION]

        # expect_group_event[_DIFF] = expect_group_event[_PREDICTION] - expect_event[_PREDICTION]
        g_unsigned = expect_group_event[_DIFF]
        g_signed = pd.concat([g_unsigned, -g_unsigned],
                             keys=["+", "-"],
                             names=[_SIGN, _EVENT, _GROUP_ID])
        self._gamma_descr = str(expect_group_event[[_PREDICTION, _DIFF]])
        return g_signed


class ProxyEqualizedOdds2(ConditionalSelectionRate):
    def __init__(self, delta=[0.3, 0.0]):
        super().__init__()
        self.delta = delta
    
    def load_data(self, X, y, **kwargs):
        """Load the specified data into the object."""
        super().load_data(X, y,
                          event=pd.Series(y).apply(lambda y: _LABEL + "=" + str(y)),
                          **kwargs)

    def gamma(self, predictor):
        """Calculate the degree to which constraints are currently violated by the predictor."""
        pred = predictor(self.X)
        pred_mean = pred.mean()

        self.tags[_PREDICTION] = pred
        expect_event = self.tags.groupby(_EVENT).mean()
        expect_group_event = self.tags.groupby(
            [_EVENT, _GROUP_ID]).mean()

        neg = expect_group_event.loc[(expect_group_event.index.get_level_values(_GROUP_ID) == 0.0)].groupby([_EVENT]).mean()
        pos = expect_group_event.loc[(expect_group_event.index.get_level_values(_GROUP_ID) == 1.0)].groupby([_EVENT]).mean()
        # print(pos)
        expect_group_event.loc[('label=1.0', 1), 'pred'] = (pred_mean + self.delta[1] * (pos.loc['label=1.0', 'pred'] - pos.loc['label=0.0', 'pred'])) / 2
        expect_group_event.loc[('label=0.0', 1), 'pred'] = (pred_mean + self.delta[1] * (pos.loc['label=0.0', 'pred'] - pos.loc['label=1.0', 'pred'])) / 2

        expect_group_event.loc[('label=1.0', 0), 'pred'] = (pred_mean + self.delta[0] * (neg.loc['label=1.0', 'pred'] - neg.loc['label=0.0', 'pred'])) / 2
        expect_group_event.loc[('label=0.0', 0), 'pred'] = (pred_mean + self.delta[0] * (neg.loc['label=0.0', 'pred'] - neg.loc['label=1.0', 'pred'])) / 2

        expect_event = expect_group_event.groupby(_EVENT).mean()
        expect_group_event[_DIFF] = expect_group_event[_PREDICTION] - expect_event[_PREDICTION]

        # expect_group_event[_DIFF] = expect_group_event[_PREDICTION] - expect_event[_PREDICTION]
        g_unsigned = expect_group_event[_DIFF]
        g_signed = pd.concat([g_unsigned, -g_unsigned],
                             keys=["+", "-"],
                             names=[_SIGN, _EVENT, _GROUP_ID])
        self._gamma_descr = str(expect_group_event[[_PREDICTION, _DIFF]])
        return g_signed

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