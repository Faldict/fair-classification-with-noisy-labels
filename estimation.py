import shap
import copy
import time
import numpy as np
import pandas as pd
from utils import flip
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

error_rate = [[0.21, 0.39], [0.28, 0.04]]

X_raw, Y = shap.datasets.adult()
X_raw['label'] = Y
df_majority = X_raw[X_raw.label == False]
df_minority = X_raw[X_raw.label == True]
df_minority_upsampled = resample(df_minority, replace=True, n_samples=24720, random_state=123)
df = pd.concat([df_majority, df_minority_upsampled])
print(df.label.value_counts())
Y = df['label'].values
X_raw = df.drop('label', axis=1)
print(X_raw.head())

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

def estimation():
    c1 = np.array([0., 0.])
    t = np.array([0., 0.])
    num = np.array([0., 0.])
    for i in range(len(X_train)):
        num[int(A_train[i])] += 1.
        if Y_noised[i] == 1:
            j = NearestNeighbor(i)
            # print(i, j)
            t[int(A_train[i])] += Y_noised[i] == Y_noised[j]
            c1[int(A_train[i])] += 1
    c1 = 2 * c1 / num
    c2 = 2 * t / num
    print(f"c1: {c1}, c2: {c2}")
    return np.sqrt(2 * c2 - c1 * c1)

delta = estimation()
print(f"True Error Rate: {error_rate}, Estimated Delta: {delta}.")