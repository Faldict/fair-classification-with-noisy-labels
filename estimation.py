import shap
import copy
import time
import numpy as np
import pandas as pd
from utils import flip, accuracy, violation
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from fairlearn.reductions import ExponentiatedGradient, GridSearch
from fairlearn.reductions import DemographicParity, ErrorRate, EqualizedOdds
from ProxyConstraint import ProxyEqualizedOdds2

error_rate = [[0.35, 0.45], [0.10, 0.06]]

X_raw, Y = shap.datasets.adult()
# X_raw['label'] = Y
# df_majority = X_raw[X_raw.label == False]
# df_minority = X_raw[X_raw.label == True]
# df_minority_upsampled = resample(df_majority, replace=True, n_samples=7841, random_state=123)
# df = pd.concat([df_minority, df_minority_upsampled])
# print(df.label.value_counts())
# Y = df['label'].values
# X_raw = df.drop('label', axis=1)
# print(X_raw.head())

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

X_train = X_train.values
A_train = A_train.values
X_test = X_test.reset_index(drop=True)
A_test = A_test.reset_index(drop=True)
# print(X_train)

Y_noised = flip(Y_train, A_train, error_rate=error_rate)

print(np.mean(Y_train), np.mean(Y_noised))

def NearestNeighbor(i):
    # print(X_train.shape)
    distance = max(np.linalg.norm(X_train[i,] - X_train[0,]), np.linalg.norm(X_train[i,] - X_train[1,]))
    nn = 0
    for j in range(len(X_train)):
        if i == j:
            continue
        if A_train[i]== A_train[j] and np.linalg.norm(X_train[i] - X_train[j]) < distance:
            distance = np.linalg.norm(X_train[i] - X_train[j])
            nn = j
            # print(nn, distance)
    return nn

def estimate_delta(X, A, Y):
    c1 = np.array([0., 0.])
    t = np.array([0., 0.])
    num = np.array([0., 0.])
    for i in range(len(X)):
        num[int(A[i])] += 1.
        if Y[i] == 1:
            j = NearestNeighbor(i)
            # print(i, j)
            t[int(A[i])] += Y[i] == Y[j]
            c1[int(A[i])] += 1
    c1 = 2 * c1 / num
    c2 = 2 * t / num
    print(f"c1: {c1}, c2: {c2}")
    return np.sqrt(2 * c2 - c1 * c1)

delta = estimate_delta(X_train, A_train, Y_noised)
print(f"True Error Rate: {error_rate}, Estimated Delta: {delta}.")

def run_clean(fairness_constraints):
    print(f"Start running experiment with clean data.")

    unmitigated_predictor = LogisticRegression(solver='liblinear', fit_intercept=True)

    # unmitigated_predictor.fit(X_train, Y_train)
    unmitigated_predictor.fit(X_train, Y_train)
    sweep = GridSearch(LogisticRegression(solver='liblinear', fit_intercept=True),
                        constraints=EqualizedOdds(),
                        grid_size=71)        

    sweep.fit(X_train, Y_train, sensitive_features=A_train)
    predictors = [ unmitigated_predictor ] + [ z.predictor for z in sweep.all_results]

    all_results_train, all_results_test = [], []
    for predictor in predictors:
        prediction_train = predictor.predict(X_train)
        prediction_test = predictor.predict(X_test)

        all_results_train.append({'accuracy': accuracy(prediction_train, Y_train), 'violation': violation(prediction_train, Y_train, A_train)})
        all_results_test.append({'accuracy': accuracy(prediction_test, Y_test), 'violation': violation(prediction_test, Y_test, A_test)})
    # print(all_results_train)
    # print(all_results_test)

    best_train, best_test = [], []
    for constraint in fairness_constraints:
        best = 0.0
        for result in all_results_train:
            if result['violation'] <= constraint and result['accuracy'] > best:
                best = result['accuracy']
        best_train.append(best)

        best = 0.0
        for result in all_results_test:
            if result['violation'] <= constraint and result['accuracy'] > best:
                best = result['accuracy']
        best_test.append(best)
    
    return best_train, best_test

def run_estimation(fairness_constraints, proxy=False, lnl=False):
    print(f"Start running experiment with Proxy: {proxy}, Learning with Noisy Labels: {lnl}.")
    all_results_train, all_results_test = [], []

    for eps in fairness_constraints:
        begin = time.time()

        if proxy and lnl:
            clf = ExponentiatedGradient(LogisticRegression(solver='liblinear', fit_intercept=True),
                        constraints=ProxyEqualizedOdds2(delta=delta),
                        eps=eps)
            sweep = LearningWithNoisyLabels(clf=clf)
        elif proxy:
            sweep = ExponentiatedGradient(LogisticRegression(solver='liblinear', fit_intercept=True),
                        constraints=ProxyEqualizedOdds2(delta=delta),
                        eps=eps)

        elif lnl:
            clf = ExponentiatedGradient(LogisticRegression(solver='liblinear', fit_intercept=True),
                        constraints=EqualizedOdds(),
                        eps=eps)
            sweep = LearningWithNoisyLabels(clf=clf)
        else:
            sweep = ExponentiatedGradient(LogisticRegression(solver='liblinear', fit_intercept=True),
                            constraints=EqualizedOdds(),
                            eps=eps)        

        sweep.fit(X_train, Y_noised, sensitive_features=A_train)
        
        prediction_train = sweep.predict(X_train)
        prediction_test = sweep.predict(X_test)

        accuracy_train = accuracy(prediction_train, Y_train)
        accuracy_test = accuracy(prediction_test, Y_test)
        all_results_train.append(accuracy_train)
        all_results_test.append(accuracy_test)

        print(f"Running fairness constraint: {eps}, Training Accuracy: {accuracy_train}, Test Accuracy: {accuracy_test}, Training Violation: {violation(prediction_train, Y_train, A_train)}, Test Violation: {violation(prediction_test, Y_test, A_test)}, Time cost: {time.time() - begin}")
    
    return all_results_train, all_results_test

fairness_constraints = [0.008 * i for i in range(1, 11)]

train_result1, test_result1 = run(fairness_constraints, proxy=True, lnl=False)
# train_result2, test_result2 = run(fairness_constraints, proxy=False, lnl=True)
# train_result3, test_result3 = run(fairness_constraints, proxy=False, lnl=False)
train_result2, test_result2 = run_clean(fairness_constraints)
# train_result4, test_result4 = run(fairness_constraints, proxy=True, lnl=True)

with open('logs/est_result2.txt', 'w') as f:
    for i in range(len(fairness_constraints)):
        # f.write(f"{fairness_constraints[i]}\t{train_result1[i]}\t{test_result1[i]}\t{train_result2[i]}\t{test_result2[i]}\t{train_result3[i]}\t{test_result3[i]}\t{train_result4[i]}\t{test_result4[i]}\n")
        f.write(f"{fairness_constraints[i]}\t{train_result2[i]}\t{test_result2[i]}\t{train_result1[i]}\t{test_result1[i]}\n")