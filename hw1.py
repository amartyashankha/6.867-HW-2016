import matplotlib.pyplot as plt
import math
import pylab as pl
import P2.loadFittingDataP2 as load2
import P3.regressData as load3
import numpy as np

def getWML(phis, X, t):
	M,N = len(phis), len(X)
	Phi = np.asmatrix([[phis[j](X[i]) for j in range(M)] for i in range(N)])
	print(Phi)
	PSI = ((Phi.T*Phi).I)*(Phi.T)
	print(PSI)
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

def P2(M, F):
	X,Y = load2.getData(False)
	wml = getWML(getBasis(M,F), X, Y)
	print(wml)
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

f = P3(1, 0.0001, load3.regressAData)

validate(f, load3.validateData)
