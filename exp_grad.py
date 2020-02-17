import shap
import copy
import time
import json
import numpy as np
import pandas as pd

from fairlearn.reductions import ExponentiatedGradient, GridSearch
from fairlearn.reductions import DemographicParity, ErrorRate, EqualizedOdds
from cleanlab.classification import LearningWithNoisyLabels
from sklearn import svm, neighbors, tree
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from utils import flip, accuracy, violation
from ProxyConstraint import ProxyEqualizedOdds, ProxyEqualizedOdds2

error_rate = [[0.3, 0.4], [0.0, 0.0]]
shap.initjs()

X_raw, Y = shap.datasets.adult()

A = X_raw["Sex"]
X = X_raw.drop(labels=['Sex'],axis = 1)
X = pd.get_dummies(X)

sc = StandardScaler()
X_scaled = sc.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

le = LabelEncoder()
Y = le.fit_transform(Y)

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

def run_clean(fairness_constraints):
    print(f"Start running experiment with clean data.")
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

        sweep = ExponentiatedGradient(LogisticRegression(solver='liblinear', fit_intercept=True),
                        constraints=EqualizedOdds(),
                        eps=eps)     

        try:
            sweep.fit(X_train, Y_train, sensitive_features=A_train)

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

        print(f"Running fairness constraint: {eps}, Training Accuracy: {all_results['accuracy']['train']}, Test Accuracy: {all_results['accuracy']['test']}, Training Violation: {all_results['violation']['train']}, Test Violation: {all_results['violation']['test']}, Time cost: {time.time() - begin}")
    
    return all_results

def run(fairness_constraints, use_proxy=False):
    print(f"Start running experiment with Proxy: {use_proxy}.")
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

        if use_proxy:
            sweep = ExponentiatedGradient(LogisticRegression(solver='liblinear', fit_intercept=True),
                        constraints=ProxyEqualizedOdds(error_rate=error_rate),
                        eps=eps)
        else:
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

        print(f"Running fairness constraint: {eps}, Training Accuracy: {all_results['accuracy']['train']}, Test Accuracy: {all_results['accuracy']['test']}, Training Violation: {all_results['violation']['train']}, Test Violation: {all_results['violation']['test']}, Time cost: {time.time() - begin}")
    
    return all_results

def run_estimation(fairness_constraints, isEstimate=True):
    def NearestNeighbor(X, A, i):
        # print(X_train.shape)
        distance = max(np.linalg.norm(X[i] - X[0]), np.linalg.norm(X[i] - X[1]))
        nn = 0
        for j in range(len(X)):
            if i == j:
                continue
            if A[i]== A[j] and np.linalg.norm(X[i] - X[j]) < distance:
                distance = np.linalg.norm(X[i] - X[j])
                nn = j
        return nn

    def estimate_delta(X, A, Y):
        c1 = np.array([0., 0.])
        t = np.array([0., 0.])
        num = np.array([0., 0.])
        for i in range(len(X)):
            num[int(A[i])] += 1.
            if Y[i] == 1:
                j = NearestNeighbor(X, A, i)
                # print(i, j)
                t[int(A[i])] += Y[i] == Y[j]
                c1[int(A[i])] += 1
        c1 = 2 * c1 / num
        c2 = 2 * t / num
        print(f"c1: {c1}, c2: {c2}")
        return np.sqrt(2 * c2 - c1 * c1)

    if isEstimate:
        print(f"Start running proxy fairness constraint with estimated delta.")
        delta = estimate_delta(X_train.values, A_train.values, Y_noised)
        print(f"Estimated delta is {delta}.")
    else:
        print("Start running proxy fairness constraint with known delta.")
        delta = np.array([1-error_rate[0][0]-error_rate[0][1], 1-error_rate[1][0]-error_rate[1][1]])
        print(f"The known delta is {delta}.")

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

        sweep = ExponentiatedGradient(LogisticRegression(solver='liblinear', fit_intercept=True),
                        constraints=ProxyEqualizedOdds2(delta=delta),
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

        print(f"Running fairness constraint: {eps}, Training Accuracy: {all_results['accuracy']['train']}, Test Accuracy: {all_results['accuracy']['test']}, Training Violation: {all_results['violation']['train']}, Test Violation: {all_results['violation']['test']}, Time cost: {time.time() - begin}")

    return all_results

fairness_constraints = [0.004 * i for i in range(1, 11)]
results = {}
results['proxy_fairness_constraint_with_estimated_delta'] = run_estimation(fairness_constraints, isEstimate=True)
results['proxy_fairness_constraint_with_know_delta'] = run_estimation(fairness_constraints, isEstimate=False)
results['clean_data'] = run_clean(fairness_constraints)
results['surrogate_fairness_constraint'] = run(fairness_constraints, use_proxy=True)
results['corrupted_data'] = run(fairness_constraints, use_proxy=False)

filename = f"result{int(error_rate[0][0] * 100)}{int(error_rate[0][1] * 100)}{int(error_rate[1][0] * 100)}{int(error_rate[1][1] * 100)}"

with open(f"logs/{filename}.json", 'w') as f:
    json.dump(results, f)