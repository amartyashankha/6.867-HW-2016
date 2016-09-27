import pylab as pl
from loadParametersP1 import getData
import numpy as np
import math


def row_to_col(v):
	return np.asmatrix(v).T

def grad_desc(f,g, x, eta = 0.01, thr = 0.00000000000001):
	"""
	Assumes function is R^n -> R
	"""
	gradient, diff = g(x), 1
	while diff > thr:
		print(x)
		xn = x - (gradient * eta)
		diff = abs(f(xn) - f(x))
		x, gradient = xn, g(x)
	return x

def gauss(x,u,sig):
	arg = (-1/2)*((x-u).T)*(sig.I)*(x-u)
	n = sig.shape[0]
	return np.asscalar(-1*((2*math.pi)**(-n/2))*(np.linalg.det(sig)**(-1/2))*np.exp(arg))

def gauss_deriv(x,u,sig):
	return -gauss(x,u,sig)*(sig.I)*(x-u)

def quadbowl(x,A,B):
	return (0.5*(x.T)*A*x)-((x.T)*B)

def quadbowl_deriv(x,A,B):
	return (A*x - B)

def apply(f,a,b):
	return (lambda x: f(x,a,b))

(gaussMean,gaussCov,quadBowlA,quadBowlB) = getData()
gaussMean, gaussCov  = row_to_col(gaussMean), np.asmatrix(gaussCov)
quadBowlB, quadBowlA = row_to_col(quadBowlB), np.asmatrix(quadBowlA)

gauss_deriv_f = apply(gauss_deriv, gaussMean, gaussCov)
gauss_f = apply(gauss, gaussMean, gaussCov)
quadbowl_deriv_f = apply(quadbowl_deriv, quadBowlA, quadBowlB)
quadbowl_f = apply(quadbowl, quadBowlA, quadBowlB)

#xm = grad_desc(gauss_f, gauss_deriv_f,np.asmatrix([0.00,0.00]).T)
xm = grad_desc(quadbowl_f, quadbowl_deriv_f,np.asmatrix([0.00,0.00]).T)
