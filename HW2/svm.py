from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import pylab as pl
from cvxopt import matrix, solvers

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

def getClassifier(X, Y, ker, alpha, b, n):
    def f(x0, X1):
        x = np.hstack([x0,X1])
        return computeFromSupports(x, X, Y, ker, alpha, b, n)
    return f

def getScoreFn(X, Y, ker, alpha, b, n):
    def f(x):
        return computeFromSupports(x, X, Y, ker, alpha, b, n)
    return f



def svm(X, Y, ker = innerProd, C = None, fname = None):
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
    
    solution = solvers.qp(P, q, G, h, A, b)
    alpha = (np.array(solution['x']).T)[0]

    b = 0

    for i in range(n):
        if alpha[i] < THRESH:
            continue
        if C and alpha[i] > (C-THRESH):
            continue
        b = Y[i] - computeFromSupports(X[i], X, Y, ker, alpha, 0, n)
        break

    w = sum([Y[i]*X[i]*alpha[i] for i in range(n)]) + b
    
    cls = getClassifier(X, Y, ker, alpha, b, n)
    
    if fname:
        x1 = []
        x2 = []
        y1 = []
        y2 = []

        for i in range(n):
            if Y[i] > 0:
                y2.append(X[i,1])
                x2.append(X[i,0])
            else:
                y1.append(X[i,1])
                x1.append(X[i,0])

        plt.plot(x1,y1,'o')
        plt.plot(x2,y2,'o')

        xl = []
        yl = []

        for i in np.arange(-4, 3, 0.1):
            try:
                j =  scipy.optimize.brentq(partial(cls, i), -1000, 1000)
            except ValueError:
                pass
            else:
                xl.append(i)
                yl.append(j)

        plt.plot(xl,yl)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(fname)

    return getScoreFn(X, Y, ker, alpha, b, n)

def P21():
    X = np.matrix([[2,2],[2,3],[0,-1],[-3,-2]])
    Y = np.array([1,1,-1,-1])
    svm(X,Y, plot = True)

def P21p():
    X = np.matrix([[2,2],[2,3],[0,-1],[-3,-2], [2,1.9]])
    Y = np.array([1,1,-1,-1, -1])
    svm(X,Y, C = 0.01, plot = True)
