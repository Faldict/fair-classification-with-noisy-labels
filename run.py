import shap
import copy
import time
import json
import argparse

import numpy as np
import pandas as pd

from fairlearn.reductions import ExponentiatedGradient, GridSearch
from fairlearn.reductions import DemographicParity, ErrorRate, EqualizedOdds
from cleanlab.classification import LearningWithNoisyLabels
from cleanlab.latent_estimation import estimate_latent, estimate_confident_joint_and_cv_pred_proba, estimate_py_and_noise_matrices_from_probabilities
from sklearn import svm, neighbors, tree
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from utils import flip, accuracy, violation, generate_noise_matrix, estimation
from ProxyConstraint import ProxyEqualizedOdds, ProxyEqualizedOdds2
from PeerLoss import PeerLoss, SurrogateLoss

from datasets import *

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='adult', help='Dataset to run.')
parser.add_argument('--ngroups', type=int, default=1, help="Number of sensitive groups.")
parser.add_argument('--constraint', type=float, default=0.05, help='Fairness constraint.')
args = parser.parse_args()

error_rate = [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.15, 0.0], [0.0, 0.15], [0.20, 0.25]]
error_rate = error_rate[:2**args.ngroups]

if args.dataset.lower() == 'compas':
    X_scaled, Y, A = compas()
elif args.dataset.lower() == 'law':
    X_scaled, Y, A = law()
elif args.dataset.lower() == 'apnea':
    X_scaled, Y, A = apnea()
elif args.dataset.lower() == 'arrest':
    X_scaled, Y, A = compas_arrest_race(ngroups=args.ngroups)
elif args.dataset.lower() == 'violent':
    X_scaled, Y, A = compas_violent_race(ngroups=args.ngroups)
elif args.dataset.lower() == 'german':
    X_scaled, Y, A = german()
elif args.dataset.lower() == 'credit':
    X_scaled, Y, A = credit()
elif args.dataset.lower() == 'adult' and args.ngroups > 1:
    X_scaled, Y, A = balanced_adult(ngroups=args.ngroups)
else:
    X_scaled, Y, A = adult()

X_train, X_test, Y_train, Y_test, A_train, A_test = train_test_split(X_scaled, 
                                                    Y, 
                                                    A,
                                                    test_size = 0.2,
                                                    random_state=0,
                                                    stratify=Y)

# Work around indexing bug
X_train = X_train.reset_index(drop=True)
A_train = A_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
A_test = A_test.reset_index(drop=True)
# A_test = A_test.map({ 0:"female", 1:"male"})

# flip across different groups
Y_noised = flip(Y_train, A_train, error_rate=error_rate)
noise_matrix = generate_noise_matrix(Y_noised, Y_train)
est_error_rate = estimation(X_train.values, Y_noised, A_train.values, ngroups=2**args.ngroups)
print(f"True error rate is {error_rate}.\nEstimated error rate is {est_error_rate}.")

# Learning with Noisy Labels
lnl = LearningWithNoisyLabels(clf=LogisticRegression())
lnl.fit(X=X_train.values, s=Y_noised, noise_matrix=noise_matrix)
Y_lnlt = lnl.predict(X_train.values).astype(int)
lnl.fit(X=X_train.values, s=Y_noised)
Y_lnle = lnl.predict(X_train.values).astype(int)


def run_corrupt(fairness_constraints):
    all_results = {}
    all_results['eps'] = fairness_constraints
    all_results['accuracy'] = {
        'train': [],
        'test': []
    }

    all_results['violation'] = {
        'train': [],
        'test': []
    }

    all_results['violation_male'] = {
        'train': [],
        'test': []            
    }

    all_results['violation_female'] = {
        'train': [],
        'test': []
    }

    for eps in fairness_constraints:
        begin = time.time()

        print(f"[INFO][RUN] Corrupt")
        sweep = ExponentiatedGradient(LogisticRegression(solver='liblinear', fit_intercept=True),
                            constraints=EqualizedOdds(),
                            eps=eps)        

        try:
            sweep.fit(X_train, Y_noised, sensitive_features=A_train)

            prediction_train = sweep.predict(X_train)
            prediction_test = sweep.predict(X_test)
        except:
            print(f"Fairlearn can't fit at fairness constraint {eps}")
            pass

        all_results['accuracy']['train'].append(accuracy(prediction_train, Y_train))
        all_results['accuracy']['test'].append(accuracy(prediction_test, Y_test))

        all_results['violation']['train'].append(violation(prediction_train, Y_train, A_train))
        all_results['violation']['test'].append(violation(prediction_test, Y_test, A_test))

        all_results['violation_male']['train'].append(violation(prediction_train, Y_train, A_train, grp=1))
        all_results['violation_male']['test'].append(violation(prediction_test, Y_test, A_test, grp=1))         

        all_results['violation_female']['train'].append(violation(prediction_train, Y_train, A_train, grp=0))
        all_results['violation_female']['test'].append(violation(prediction_test, Y_test, A_test, grp=0))
        print(f"Running fairness constraint: {eps}, Training Accuracy: {all_results['accuracy']['train'][-1]}, Test Accuracy: {all_results['accuracy']['test'][-1]}, Training Violation: {all_results['violation']['train'][-1]}, Test Violation: {all_results['violation']['test'][-1]}, Time cost: {time.time() - begin}")

    acc = np.array(all_results['accuracy']['test'])
    v = np.array(all_results['violation']['test'])
    all_results['accuracy']['mean'] = acc.mean()
    all_results['accuracy']['std'] = acc.std()
    all_results['violation']['mean'] = v.mean()
    all_results['violation']['std'] = v.std()
    return all_results

def run_clean(fairness_constraints):
    print(f"[INFO][RUN] Clean")
    all_results = {}
    all_results['eps'] = fairness_constraints
    all_results['accuracy'] = {
        'train': [],
        'test': []
    }

    all_results['violation'] = {
        'train': [],
        'test': []
    }

    all_results['violation_male'] = {
        'train': [],
        'test': []            
    }

    all_results['violation_female'] = {
        'train': [],
        'test': []
    }

    for eps in fairness_constraints:
        begin = time.time()  
        sweep = LogisticRegression(solver='liblinear', fit_intercept=True)

        try:
            sweep.fit(X_train, Y_train)

            prediction_train = sweep.predict(X_train)
            prediction_test = sweep.predict(X_test)
        except:
            print(f"Fairlearn can't fit at fairness constraint {eps}")
            pass

        all_results['accuracy']['train'].append(accuracy(prediction_train, Y_train))
        all_results['accuracy']['test'].append(accuracy(prediction_test, Y_test))

        all_results['violation']['train'].append(violation(prediction_train, Y_train, A_train))
        all_results['violation']['test'].append(violation(prediction_test, Y_test, A_test))

        all_results['violation_male']['train'].append(violation(prediction_train, Y_train, A_train, grp=1))
        all_results['violation_male']['test'].append(violation(prediction_test, Y_test, A_test, grp=1))         

        all_results['violation_female']['train'].append(violation(prediction_train, Y_train, A_train, grp=0))
        all_results['violation_female']['test'].append(violation(prediction_test, Y_test, A_test, grp=0))
        print(f"Running fairness constraint: {eps}, Training Accuracy: {all_results['accuracy']['train'][-1]}, Test Accuracy: {all_results['accuracy']['test'][-1]}, Training Violation: {all_results['violation']['train'][-1]}, Test Violation: {all_results['violation']['test'][-1]}, Time cost: {time.time() - begin}")
    acc = np.array(all_results['accuracy']['test'])
    v = np.array(all_results['violation']['test'])
    all_results['accuracy']['mean'] = acc.mean()
    all_results['accuracy']['std'] = acc.std()
    all_results['violation']['mean'] = v.mean()
    all_results['violation']['std'] = v.std()
    return all_results

def run_peerloss(fairness_constraints, alpha=0.5, est=False):
    print(f"[INFO][RUN] Peer Loss with alpha = {alpha}")
    all_results = {}
    all_results['eps'] = fairness_constraints
    all_results['accuracy'] = {
        'train': [],
        'test': []
    }

    all_results['violation'] = {
        'train': [],
        'test': []
    }

    all_results['violation_male'] = {
        'train': [],
        'test': []            
    }

    all_results['violation_female'] = {
        'train': [],
        'test': []
    }
    
    if est:
        delta = [1 - est_error_rate[i][0] - est_error_rate[i][1] for i in range(len(est_error_rate))]
    else:
        delta = [1 - error_rate[i][0] - error_rate[i][1] for i in range(len(error_rate))]

    for eps in fairness_constraints:
        begin = time.time()

        sweep = ExponentiatedGradient(PeerLoss(A_train, delta=delta, alpha=alpha),
                    constraints=EqualizedOdds(),
                    eps=eps)   
 
        sweep.fit(X_train, Y_noised, sensitive_features=A_train)

        prediction_train = sweep.predict(X_train)
        prediction_test = sweep.predict(X_test)

        all_results['accuracy']['train'].append(accuracy(prediction_train, Y_train))
        all_results['accuracy']['test'].append(accuracy(prediction_test, Y_test))

        all_results['violation']['train'].append(violation(prediction_train, Y_train, A_train))
        all_results['violation']['test'].append(violation(prediction_test, Y_test, A_test))

        all_results['violation_male']['train'].append(accuracy(prediction_train, Y_train))
        all_results['violation_male']['test'].append(accuracy(prediction_test, Y_test))         

        all_results['violation_female']['train'].append(accuracy(prediction_train, Y_train))
        all_results['violation_female']['test'].append(accuracy(prediction_test, Y_test))

        print(f"Running fairness constraint: {eps}, Training Accuracy: {all_results['accuracy']['train'][-1]}, Test Accuracy: {all_results['accuracy']['test'][-1]}, Training Violation: {all_results['violation']['train'][-1]}, Test Violation: {all_results['violation']['test'][-1]}, Time cost: {time.time() - begin}")

    acc = np.array(all_results['accuracy']['test'])
    v = np.array(all_results['violation']['test'])
    all_results['accuracy']['mean'] = acc.mean()
    all_results['accuracy']['std'] = acc.std()
    all_results['violation']['mean'] = v.mean()
    all_results['violation']['std'] = v.std()
    return all_results

def run_surrogate(fairness_constraints, est=False):
    print(f"[INFO][RUN] Surrogate Loss.")
    all_results = {}
    all_results['eps'] = fairness_constraints
    all_results['accuracy'] = {
        'train': [],
        'test': []
    }

    all_results['violation'] = {
        'train': [],
        'test': []
    }

    all_results['violation_male'] = {
        'train': [],
        'test': []            
    }

    all_results['violation_female'] = {
        'train': [],
        'test': []
    }
    
    for eps in fairness_constraints:
        begin = time.time()

        if not est:
            surrogate_clf = SurrogateLoss(clf=LogisticRegression(solver='liblinear', fit_intercept=True), noise_matrix=noise_matrix)
        else:
            surrogate_clf = SurrogateLoss(clf=LogisticRegression(solver='liblinear', fit_intercept=True))

        sweep = ExponentiatedGradient(surrogate_clf,
                    constraints=ProxyEqualizedOdds(error_rate=error_rate),
                    eps=eps)   

        sweep.fit(X_train, Y_noised, sensitive_features=A_train)

        prediction_train = sweep.predict(X_train)
        prediction_test = sweep.predict(X_test)

        all_results['accuracy']['train'].append(accuracy(prediction_train, Y_train))
        all_results['accuracy']['test'].append(accuracy(prediction_test, Y_test))

        all_results['violation']['train'].append(violation(prediction_train, Y_train, A_train))
        all_results['violation']['test'].append(violation(prediction_test, Y_test, A_test))

        all_results['violation_male']['train'].append(accuracy(prediction_train, Y_train))
        all_results['violation_male']['test'].append(accuracy(prediction_test, Y_test))         

        all_results['violation_female']['train'].append(accuracy(prediction_train, Y_train))
        all_results['violation_female']['test'].append(accuracy(prediction_test, Y_test))

        print(f"Running fairness constraint: {eps}, Training Accuracy: {all_results['accuracy']['train'][-1]}, Test Accuracy: {all_results['accuracy']['test'][-1]}, Training Violation: {all_results['violation']['train'][-1]}, Test Violation: {all_results['violation']['test'][-1]}, Time cost: {time.time() - begin}")
    
    acc = np.array(all_results['accuracy']['test'])
    v = np.array(all_results['violation']['test'])
    all_results['accuracy']['mean'] = acc.mean()
    all_results['accuracy']['std'] = acc.std()
    all_results['violation']['mean'] = v.mean()
    all_results['violation']['std'] = v.std()
    return all_results

tolerance = [args.constraint for _ in range(5)]
clean_result = run_clean(tolerance)
sl_result = run_surrogate(tolerance, est=True)
pl_result = run_peerloss(tolerance, est=True, alpha=0.1)
print(f"clean\t{clean_result['accuracy']['mean']} \pm {clean_result['accuracy']['std']}\t{clean_result['violation']['mean']} \pm {clean_result['violation']['std']}")
print(f"clean\t{sl_result['accuracy']['mean']} \pm {sl_result['accuracy']['std']}\t{sl_result['violation']['mean']} \pm {sl_result['violation']['std']}")
print(f"clean\t{pl_result['accuracy']['mean']} \pm {pl_result['accuracy']['std']}\t{pl_result['violation']['mean']} \pm {pl_result['violation']['std']}")