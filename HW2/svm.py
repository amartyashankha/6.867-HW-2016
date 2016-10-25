from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import pylab as pl
from cvxopt import matrix, solvers
from plotBoundary import plotDecisionBoundary

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['figure.autolayout'] = 'true'

THRESH = 1e-5

def innerProd(x1, x2):
    return np.asscalar(x1*(x2.T))

def gaussianKer(gamma):
    def gaussianRBF(x1, x2):
        diff = x1 - x2
        return np.exp(-gamma * np.asscalar(diff*(diff.T)))
    return gaussianRBF

def computeFromSupports(x, X, Y, ker, alpha, b, n):
    return sum([Y[i]*ker(x,X[i])*alpha[i] for i in range(n)]) + b

def getScoreFn(X, Y, ker, alpha, b, n):
    def f(x):
        return computeFromSupports(x, X, Y, ker, alpha, b, n)
    return f



def svm(X, Y, ker = innerProd, C = None):
    n = X.shape[0]
    K = np.zeros((n,n))

    for i in range(n):
        for j in range(n):
            K[i,j] = Y[i] * ker(X[i], X[j]) * Y[j]

    P = matrix(K)
    q = matrix(-np.ones(n))
    A = matrix(np.matrix([Y]).astype(float))
    b = matrix([0.0])

    if C:
        G = matrix(np.vstack((-np.identity(n), np.identity(n))))
        h = matrix(np.hstack((np.zeros(n), C * np.ones(n))))
    else:
        G = matrix(-np.identity(n))
        h = matrix(np.zeros(n))
    
    solvers.options['show_progress'] = False
    solution = solvers.qp(P, q, G, h, A, b)
    alpha = (np.array(solution['x']).T)[0]

    print "alpha: ", alpha

    b = 0

    for i in range(n):
        if alpha[i] < THRESH:
            continue
        if C and alpha[i] > (C-THRESH):
            continue
        b = Y[i] - computeFromSupports(X[i], X, Y, ker, alpha, 0, n)
        break

    svs = 0
    for i in range(n):
        if (alpha[i] > THRESH) and (not C or alpha[i] > (C-THRESH)):
            svs +=1
    print "Support vectors: ",svs
        
    w = sum([Y[i]*X[i]*alpha[i] for i in range(n)]) 
    gm = 1/np.sqrt(w*w.T)
    print "Geometric margin: ",gm

    return getScoreFn(X, Y, ker, alpha, b, n)

def test():
    X = np.matrix([[2,2],[2,3],[0,-1],[-3,-2]])
    Y = np.array([1,1,-1,-1])
    cls = svm(X,Y)
    plotDecisionBoundary(X,Y,cls, [-1,0,1], title = 'SVM', fname = 'svm_small')

#test()
