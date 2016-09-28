import matplotlib.pyplot as plt
import math
import pylab as pl
import P2.loadFittingDataP2 as load2
import P3.regressData as load3
import numpy as np
import grad_desc as gd

def getWML(phis, X, t):
	M,N = len(phis), len(X)
	Phi = np.asmatrix([[phis[j](X[i]) for j in range(M)] for i in range(N)])
	PSI = ((Phi.T*Phi).I)*(Phi.T)
	t = vmat(t)
	return PSI*t

def getWRidge(phis, X, t, lbd):
	M,N = len(phis), len(X)
	Phi = np.asmatrix([[phis[j](X[i]) for j in range(M)] for i in range(N)])
	PSI = ((lbd*np.identity(M)+Phi.T*Phi).I)*(Phi.T)
	t = np.asmatrix(t).T
	return PSI*t

def apply(f, n):
	return lambda x: f(x,n)

def linearSum(w, phis):
	return (lambda x: sum([float(w[i])*phis[i](x) for i in range(len(w))]))

def getBasis(M, f):
	return [apply(f,n) for n in range(M)]

def poly(x,n):
	return x**n

def cosn(x,n):
	return np.cos(n*math.pi*x)

def vmat(v):
	return np.asmatrix(v).T

def SSE(f, X, Y):
	ret = 0 
	for (x,y) in zip(X,Y):
		ret += (f(x) - y)**2
	return ret

def SSE_vec_fn(phi, X, Y):
	def sse(w):
		n,m = len(X), len(phi)
		dif = [(sum([w[i]*phi[i](X[j]) for i in range(m)]) - Y[j]) for j in range(n)]
		return np.sum(np.square(dif))
	return sse

def SSE_grad_fn(phi, X, Y):
	def grad(w):
		n,m = len(X), len(phi)
		dif = [sum([2*(sum([w[i]*phi[i](X[j]) for i in range(m)]) - Y[j])*phi[k](X[j]) for j in range(n)]) for k in range(m)]
		return dif
	return grad

def P2(M, F):
	X,Y = load2.getData(False)
	wml = getWML(getBasis(M,F), X, Y)
	est = linearSum(wml,getBasis(M,F))
	xml   = np.arange(0,1,0.01)
	yml   = np.array(est(xml))
	yac   = np.cos(math.pi*xml) + np.cos(2*math.pi*xml)

	plt.plot(X,Y,'o')
	plt.plot(xml,yml)
	plt.plot(xml,yac)
	plt.xlabel('x')
	plt.ylabel('y')
	plt.savefig('P2_'+str(M)+'_'+str(F)+'.png')
	plt.close()

	return wml

def P3(M, lbd, dataFunc, plot = True, part1 = False):
	F = poly
	X,Y = dataFunc()
	wml = getWRidge(getBasis(M,F), X,Y, lbd)
	est = linearSum(wml,getBasis(M,F))
	xml   = np.arange(min(X),max(X),0.01)
	yml   = np.array(est(xml))

	fname = 'P3_'+str(M)+'_'+str(lbd)+'_'+str(dataFunc.__name__)+'.png'

	print(fname, SSE(est,X,Y))

	if plot:
		plt.plot(X,Y,'o')
		plt.plot(xml,yml)
		if part1:
			yac   = np.cos(math.pi*xml) + np.cos(2*math.pi*xml)
			plt.plot(xml,yac)
		plt.xlabel('x')
		plt.ylabel('y')

		plt.savefig(fname)
		plt.close()

	return est

def validate(f, dataFunc, plot = True):
	X,Y = dataFunc()
	xml   = np.arange(min(X),max(X),0.01)
	yml   = np.array(f(xml)).reshape(-1)

	fname = 'Testing_'+str(dataFunc.__name__)+'.png'
	err = SSE(f,X,Y)
	print(fname, err)

	if plot:
		plt.plot(X,Y,'o')
		plt.plot(xml,yml)
		plt.xlabel('x')
		plt.ylabel('y')

		plt.savefig(fname)
		plt.close()

	return err

def P2_2(M, F, point = None):
	X,Y = load2.getData(False)

	phis = getBasis(M,poly)
	wml = getWML(phis, X, Y)
	sse = SSE_vec_fn(phis,X,Y)
	grad = SSE_grad_fn(phis,X,Y)

	wml = np.asarray(wml.T)[0]
	if point == None:
		point = wml
	print("wml: ",wml)
	print("SSE: ",sse(point))
	print("Analytic Grad: ",grad(point))
	ng = gd.GD([0]*M, sse)
	print("Numeric Grad:",ng.compute_gradient(point, eps =1e-8))

	phiX = np.asarray([[phis[i](x) for i in range(M)] for x in X])
	sgd = gd.SGD(phiX, Y)
	#sgd_log = sgd.step(np.ones(M), stochastic = True, minibatch_size = 1, ftol = 1e-9, gtol = 1e-5, stepSize = 0.1)
	gd_log = sgd.step(np.zeros(M), stochastic = False, ftol = 1e-9, gtol = 1e-5, stepSize = 0.01)[-1]
	#print("Stochastic: ", sgd_log[-1])
	print("Batch GD: ", gd_log)

#stuff for problem 3
#f = P3(10, 0.5, load3.regressAData)
#validate(f, load3.validateData)

#Problem 2.2
P2_2(5,poly)
