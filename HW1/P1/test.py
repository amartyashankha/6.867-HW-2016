import numpy as np
import matplotlib.pyplot as plt
import loadParametersP1
import loadFittingDataP1
from scipy.stats import multivariate_normal
import pdb
import math
import copy
import timeit

class SGD(object):
    
    def __init__(self, X, y,
                 step_size=1e-5):
        self.X = X
        self.y = y
        self.step_size = step_size
    
    def compute_objective(self, theta, idx = None):
        if idx == None:
            idx = range(len(self.y))
        return sum([(theta.dot(X[i])-y[i])**2 for i in idx])
    
    def compute_numerical_gradient(self, theta, eps=1e-7, idx=None):
        grad = np.zeros(len(theta))
        for i in range(len(theta)):
            if idx != None and i != idx:
                continue
            f = self.compute_objective(theta)
            theta[i] = theta[i] + eps
            f_eps = self.compute_objective(theta)
            theta[i] = theta[i] - eps
            grad[i] = (f_eps-f)/eps
        return grad
        
    def compute_gradient(self, theta, idx=None):
        if idx == None:
            idx = range(len(self.y))
        grad = np.zeros(self.X.shape[1])
        for i in idx:
            grad += 2 * (self.X[i].dot(theta) - self.y[i]) * self.X[i]
        return grad*len(self.y)*1.0/len(idx)
    
    def line_search(self, theta, search_dir, step_size=1e-6, idx=None):
        best_val = self.compute_objective(theta-step_size*search_dir)
        while True:
            new_val = self.compute_objective(theta-2.0*step_size*search_dir, idx)
            if new_val < best_val:
                step_size *= 2.0
                best_val = new_val
            else:
                break
        while step_size > 1e-9:
            new_val = self.compute_objective(theta-0.5*step_size*search_dir, idx)
            if new_val < best_val:
                step_size *= 0.5
                best_val = new_val
            else:
                break
        return step_size
    
    def step(self, theta, stochastic=False, minibatch_size=1, ftol=1e1, gtol=1e3):
        log = []
        idx = None
        prev_objective = self.compute_objective(theta)
        beta = 0.8
        t = 0.0
        t0 = 1e6
        k = 1.0
        step_size = 1e-6
        while True:
            if stochastic:
                idx = np.random.choice(self.X.shape[0], minibatch_size, replace=False)
            
            grad = self.compute_gradient(theta, idx)
            
            #step_size = self.line_search(theta, grad, step_size)

            step_size = (t0+10000*t)**-k
            #step_size = 1e-6
            
            theta = theta - step_size * grad
            
            tmp = self.compute_objective(theta)
            log.append((theta, tmp, grad))
            
            if abs(prev_objective-tmp) < ftol*step_size:
                print "function convergence"
                print minibatch_size, len(log), log[-1][1]
                break
            if t%10 == 0 and np.linalg.norm(self.compute_gradient(theta)) < gtol:
                print "gradient convergence"
                print minibatch_size, len(log), log[-1][1]
                break
            
            prev_objective = tmp
            t += 1.0
            if t%10000 == 0:
                print log[-1][1], np.linalg.norm(self.compute_gradient(theta))
                print "step size", step_size
            #if t > 10000:
            #    break
        return log

X,y = loadFittingDataP1.getData()

optimizer = SGD(X,y)

X_dagger = np.linalg.inv(X.T.dot(X)).dot(X.T)
theta_opt = X_dagger.dot(y)
optimizer.compute_objective(X_dagger.dot(y))

theta_0 = (np.random.random(X.shape[1])-0.5)*0.3+0.5 * theta_opt

batch_sizes = [1,2,3,5,10,20,30,40,50,60,70,80,90,100]
batch_sizes.reverse()
logs = []
line_search_times =[]
for bs in batch_sizes[:]:
    start = timeit.timeit()
    log = optimizer.step(copy.copy(theta_0), stochastic=True, minibatch_size=bs);
    end = timeit.timeit()
    line_search_times.append(end-start)
    print "time", end-start
    logs.append(log)

print line_search_times
