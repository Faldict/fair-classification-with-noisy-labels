import shap
import copy
import time
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
from ProxyConstraint import ProxyEqualizedOdds
from PeerLoss import PeerLoss

error_rate = [[0.35, 0.45], [0.10, 0.20]]
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

X_train = X_train.reset_index(drop=True)
A_train = A_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
A_test = A_test.reset_index(drop=True)
# A_test = A_test.map({ 0:"female", 1:"male"})

# flip across different groups
Y_noised = flip(Y_train, A_train, error_rate=error_rate)

print(f"Start running experiment with Peer Loss")

delta = [1.-error_rate[0][0]-error_rate[0][1], 1.-error_rate[1][0]-error_rate[1][1]]

fairness_constraints = [0.008 * i for i in range(1, 11)]
all_results_train, all_results_test = [], []

for eps in fairness_constraints:  
    sweep = ExponentiatedGradient(PeerLoss(A_train, delta),
                        # constraints=ProxyEqualizedOdds(error_rate=error_rate),
                        constraints=EqualizedOdds(),
                        eps=eps)        
    sweep.fit(X_train, Y_noised, sensitive_features=A_train)

    prediction_train = sweep.predict(X_train)
    prediction_test = sweep.predict(X_test)

    accuracy_train = accuracy(prediction_train, Y_train)
    accuracy_test = accuracy(prediction_test, Y_test)
    all_results_train.append(accuracy_train)
    all_results_test.append(accuracy_test)

    print(f"Running fairness constraint {eps}, Train Accuracy {accuracy_train}, Test Accuracy {accuracy_test}")

with open('logs/PeerLossResult2.txt', 'w') as f:
    for eps, result_train, result_test in zip(fairness_constraints, all_results_train, all_results_test):
        f.write(f"{eps}\t{result_train}\t{result_test}\n")