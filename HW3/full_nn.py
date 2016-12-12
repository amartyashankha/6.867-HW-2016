import numpy as np
from functools import partial
from math import sqrt

EPS = 1e-9

def vec_grad(x, f):
    ret = []
    old_f = f(x)
    for i in range(len(x)):
        x[i] += EPS
        new_f = f(x)
        x[i] -= EPS
        ret.append((new_f-old_f)/EPS)
    return np.hstack(ret)

def grad(x, f):
    return (f(x+EPS)-f(x-EPS))/(2*EPS)


class NN:
    def query(self, x):
        self.a[0] = x

        for i in range(1, self.L):
            self.z[i] = (self.W[i].T) * (self.a[i-1]) + self.b[i]
            self.a[i] = self.f(self.z[i])

        self.a[self.L-1] = self.output_f(self.z[self.L-1])
        return self.a[self.L-1]

    def train(self, x, y):
        self.query(x)

        self.delta[self.L-1] = (vec_grad(self.a[self.L-1], partial(self.l, y)) * vec_grad(self.z[self.L-1],self.output_f)).T

        for i in range(self.L-2, 0, -1):
            self.delta[i] = np.diag(grad(self.z[i], self.f)) * self.W[i+1] * self.delta[i+1]

        for i in range(1, self.L):
            self.W[i] -= self.eta * self.a[i-1] * self.delta[i].T
            self.b[i] -= self.eta * self.delta[i]

    def print_weights(self):
        for i in range(self.L):
            print "================"
            print i
            print self.W[i]
            print self.b[i]

    def __init__(self, L, dim_x, dim_y, m, f, output_f, l, eta = 1e-3, WEIGHT_BIAS = 0.0):
        self.W = [None, np.ones((dim_x, m)) * WEIGHT_BIAS + np.random.normal(0, 1/sqrt(m), (dim_x, m))]
        self.b = [None, np.zeros((m,1))]

        for i in range(2, L-1):
            self.W.append(np.ones((m, m)) * WEIGHT_BIAS + np.random.normal(0, 1/sqrt(m), (m,m)))
            self.b.append(np.zeros((m,1)))

        self.W.append(np.ones((m, dim_y)) * WEIGHT_BIAS + np.random.normal(0, 1/sqrt(m), (m,dim_y)))
        self.b.append(np.zeros((dim_y,1)))

        self.f = f
        self.output_f = output_f
        self.l = l
        self.L = L
        self.m = m
        self.a = [None] * L
        self.z = [None] * L
        self.delta = [None] * L
        self.eta = eta
