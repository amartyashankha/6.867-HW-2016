import numpy as np
import matplotlib.pyplot as plt
import P1.loadParametersP1
import P1.loadFittingDataP1
from scipy.stats import multivariate_normal
import pdb
import math
import copy

class GD(object):
    
    def __init__(self, x0, objective,
                 gradient=None,
                 step_size=0.1):
        self.x0 = x0
        self.objective = objective
        self.gradient = gradient
        self.step_size = step_size
    
    def compute_gradient(self, x, idx=None, eps=1e-6):
        if self.gradient != None:
            return self.gradient(x)
        grad = np.array([0.0 for i in range(len(x))])
        if idx != None:
            f_x = self.objective(x)
            x[idx] += eps
            f_eps = self.objective(x)
            x[idx] -= eps
            grad[idx] = (f_eps-f_x)/eps
            return grad
        X = copy.copy(x)
        for i in range(len(x)):
            f_x = self.objective(X)
            #print X, X[i]+eps, self.objective(X), eps
            X[i] = X[i] + eps
            #print X, X[i], self.objective(X)
            f_eps = self.objective(X)
            X[i] = X[i] - eps
            grad[i] = (f_eps-f_x)/eps
        return grad
        
    
    def step(self, stochastic=False, gtol=1e-8):
        log = []
        while True:
            grad = self.compute_gradient(self.x0)
            log.append((self.x0, self.objective(self.x0)))
            if np.linalg.norm(grad) < gtol:
                break
            self.x0 = self.x0 - self.step_size * grad
        return log

class SGD(object):
    
    def __init__(self, X, y,
                 step_size=1e-5):
        self.X = X
        self.y = y
        self.step_size = step_size
    
    def compute_objective(self, theta):
        return sum([(theta.dot(self.X[i])-self.y[i])**2 for i in range(len(self.y))])
    
    def compute_numerical_gradient(self, theta, eps=1e-7, idx=None):
        grad = np.zeros(len(theta))
        for i in range(len(theta)):
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
        return grad
    
    def line_search(self, theta, search_dir, step_size=1e-6):
        best_val = self.compute_objective(theta-step_size*search_dir)
        while True:
            new_val = self.compute_objective(theta-2.0*step_size*search_dir)
            if new_val < best_val:
                step_size *= 2.0
            else:
                break
        while step_size > 1e-9:
            new_val = self.compute_objective(theta-0.5*step_size*search_dir)
            if new_val < best_val:
                step_size *= 0.5
            else:
                break
        return step_size
    
    def step(self, theta, stochastic=False, minibatch_size=1, ftol=1e-7, gtol=1e-1, stepSize = None):
        log = []
        idx = None
        prev_objective = self.compute_objective(theta)
        beta = 0.8
        t = 0.0
        while True:
            if stochastic:
                idx = np.random.randint(self.X.shape[0], size=minibatch_size)
            
            grad = self.compute_gradient(theta, idx)	    
            #print(theta)
            
            if stepSize:
                step_size = stepSize
            else:
                step_size = self.line_search(theta, grad)
                
            theta = theta - step_size * grad
            
            tmp = self.compute_objective(theta)
            log.append((theta, tmp, grad))
            
            if abs(prev_objective-tmp) < ftol:
                print("Converged f")
                print(abs(prev_objective-tmp))
                break

            if np.linalg.norm(self.compute_gradient(theta)) < gtol:
                print("Converged g")
                break
            
            prev_objective = tmp
            t += 1.0
            #if t % 10000 == 0:
                #print(theta)
            if t > 100000:
                print("t too high")
                break
        return log
