import numpy as np
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from scipy.optimize import minimize
from scipy import sparse

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

def log_logistic(w, x, y):
    z = logistic(w, x)
    out = y*np.log(z) + (1-y)*np.log(1-z)
    return out

class PeerLoss:
    """ Plug Peer Loss into Logistic Regression
    """
    def __init__(self, A, delta=[1., 1.]):
        self.weights = None
        self.A = np.ones(A.shape)

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
        xt = np.random.permutation(X)
        yt = np.random.permutation(y)

        def peer_loss(w):
            l = log_logistic(w, X, y) - log_logistic(w, xt, yt)
            out = np.mean((sample_weight * l))
            # print(f"Peer Loss: {out}.")
            return out
        
        res = minimize(peer_loss, self.weights, method='BFGS')
        self.weights = res.x
