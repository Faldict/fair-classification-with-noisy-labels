#################################
#
# File: utils.py
# Author: NIPS Anoymous Authors
# 
#################################
import random
import numpy as np
from cleanlab.latent_estimation import estimate_py_noise_matrices_and_cv_pred_proba, estimate_py_and_noise_matrices_from_probabilities

def flip(data, A, error_rate=[[0.3, 0.3], [0.0, 0.0]]):
    # print(np.mean(np.array(data)))
    corrupted_data = np.zeros(data.shape)
    for i in range(len(data)):
        p = random.uniform(0,1)
        if p < error_rate[A[i]][data[i]]:
            corrupted_data[i] = 1 - data[i]
        else:
            corrupted_data[i] = data[i]
    return corrupted_data

def generate_noise_matrix(s, Y):
    psx = np.zeros((Y.shape[0], 2))
    for i in range(Y.shape[0]):
        psx[i, int(Y[i])] = 1.
    py, noise_matrix, inverse_noise_matrix, _ = estimate_py_and_noise_matrices_from_probabilities(s, psx)
    print(noise_matrix)
    return noise_matrix

def estimation(X, Y, A, ngroups=2):
    est_error_rates = []
    # print(X.shape, Y.shape, A.shape)
    for z in range(ngroups):
        print(f"[DEBUG][EST] Estimating Group {z}")
        X_t = X[A == z]
        Y_t = Y[A == z]
        # print(X_t.shape, Y_t.shape)
        est_py, est_nm, est_inv, confident_joint, psx = estimate_py_noise_matrices_and_cv_pred_proba(
            X=X_t,
            s=Y_t,
        )
        print(f"[DEBUG] Estimated Noise Matrix {est_nm}.")
        est_error_rates.append([1-est_nm[0][0], 1-est_nm[1][1]])
    return est_error_rates

def accuracy(predictions, labels):
    t, num = 0., 0.
    for y_p, y in zip(predictions, labels):
        num += 1.
        t += (y_p == y)
    return float("{0:.2f}".format(t / num * 100))

def tpr(predictions, labels, attribute, group=None):
    t, num = 0., 0.
    for y_prime, y, z in zip(predictions, labels, attribute):
        if group is None:
            if y == 1:
                num += 1.
                if y_prime == 1:
                    t += 1
        else:
            if z == group and y == 1:
                num += 1.
                if y_prime == 1.:
                    t += 1
    return t/num

def fpr(predictions, labels, attribute, group=None):
    t, num = 0., 0.
    for y_prime, y, z in zip(predictions, labels, attribute):
        if group is None:
            if y == 0:
                num += 1.
                if y_prime == 1:
                    t += 1
        else:
            if z == group and y == 0:
                num += 1.
                if y_prime == 1.:
                    t += 1
    return t/num

def violation(predictions, labels, attribute, ngroups=2, grp=None):
    if grp is not None:
        return max(abs(tpr(predictions, labels, attribute, group=grp) - tpr(predictions, labels, attribute)), abs(fpr(predictions, labels, attribute, group=grp) - fpr(predictions, labels, attribute)))
    else:
        mv = 0.
        for g in range(ngroups):
            v = violation(predictions, labels, attribute, ngroups=ngroups, grp=g) 
            mv = max(mv, v)
        return float("{0:.2f}".format(mv * 100))