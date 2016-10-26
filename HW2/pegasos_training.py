import numpy as np
import copy

def train_linearSVM(X, Y, L, max_epochs=200, step_size = 1e-1, threshold=1e-2):
    t = 0  
    X = np.concatenate((np.ones([400,1]), X), axis=1)
    w = np.zeros(len(X[0]))
    prev = copy.copy(w)
    for epoch in range(max_epochs):
        for i in range(len(X)):
            t += 1
            step_size = 1/(t*L)
            multiplier = np.array([1.]+[1.-step_size*L for j in range(len(w)-1)])
            w *= multiplier
            if Y[i]*(w.dot(X[i])) < 2:
                w += step_size*Y[i]*X[i]
        if np.linalg.norm(w-prev) < threshold:
            print 'Ran for', epoch, 'epochs'
            break
        prev = copy.copy(w)
    return w


def gauss_kernel(x1, x2, gamma):
    v = x1-x2
    return np.exp(-gamma*v.dot(v))

def train_gaussianSVM(X, Y, gamma, max_epochs=500, step_size=1e-1):
    L = 0.02
    t = 0
    n = len(X)
    K = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            K[i][j] = gauss_kernel(X[i], X[j], gamma)
            K[j][i] = K[i][j]
    alpha = np.zeros(len(X))
    for epoch in range(max_epochs):
        for i in range(len(X)):
            t += 1
            step_size = 1/(t*L)
            alpha[i] *= 1-step_size*L
            if Y[i]*(alpha.dot(K[i])) < 1:
                alpha[i] += step_size*Y[i]
    return alpha

