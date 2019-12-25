import random
import numpy as np

def flip(data, A, error_rate=[[0.3, 0.3], [0.0, 0.0]]):
    corrupted_data = np.zeros(data.shape)
    for i in range(len(data)):
        p = random.uniform(0,1)
        if p < error_rate[A[i]][data[i]]:
            corrupted_data[i] = 1 - data[i]
        else:
            corrupted_data[i] = data[i]
    return corrupted_data

def accuracy(predictions, labels):
    t, num = 0., 0.
    for y_p, y in zip(predictions, labels):
        num += 1.
        t += (y_p == y)
    return t / num

def tpr(predictions, labels, attribute, group=0):
    t, num = 0., 0.
    for y_prime, y, z in zip(predictions, labels, attribute):
        # print(y_prime, y, z)
        if z == group and y == 1:
            num += 1.
            if y_prime == 1.:
                t += 1
    return t/num

def violation(predictions, labels, attribute):
    return abs(tpr(predictions, labels, attribute, group=0) - tpr(predictions, labels, attribute, group=1))