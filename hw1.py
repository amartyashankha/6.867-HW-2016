import matplotlib.pyplot as plt
import math
import pylab as pl
import P2.loadFittingDataP2 as load2
import P3.regressData as load3
import numpy as np

def getWML(phis, X, t):
	M,N = len(phis), len(X)
	Phi = np.asmatrix([[phis[j](X[i]) for j in range(M)] for i in range(N)])
	PSI = ((Phi.T*Phi).I)*(Phi.T)
	t = np.asmatrix(t).T
	return PSI*t

def getWRidge(phis, X, t, lbd):
	M,N = len(phis), len(X)
	Phi = np.asmatrix([[phis[j](X[i]) for j in range(M)] for i in range(N)])
	PSI = ((Phi.T*Phi).I + lbd*np.identity(M))*(Phi.T)
	t = np.asmatrix(t).T
	return PSI*t


def apply(f, n):
	return lambda x: f(x,n)

def linearSum(w, phis):
	return (lambda x: sum([w[i]*phis[i](x) for i in range(len(w))]))

def getBasis(M, f):
	return [apply(f,n) for n in range(M)]

def poly(x,n):
	return x**n

def cosn(x,n):
	return np.cos(n*math.pi*x)

def P2(M, F):
	X,Y = load2.getData(False)
	wml = getWML(getBasis(M,F), X,Y)
	est = linearSum(wml,getBasis(M,F))
	xml   = np.arange(0,1,0.01)
	yml   = np.array(est(xml)).reshape(-1)
	yac   = np.cos(math.pi*xml) + np.cos(2*math.pi*xml)

	plt.plot(X,Y,'o')
	plt.plot(xml,yml)
	plt.plot(xml,yac)
	plt.xlabel('x')
	plt.ylabel('y')
	plt.show()

def P3_1(M, lbd):
	F = poly
	X,Y = load2.getData(False)
	wml = getWRidge(getBasis(M,F), X,Y, lbd)
	est = linearSum(wml,getBasis(M,F))
	xml   = np.arange(0,1,0.01)
	yml   = np.array(est(xml)).reshape(-1)
	yac   = np.cos(math.pi*xml) + np.cos(2*math.pi*xml)

	plt.plot(X,Y,'o')
	plt.plot(xml,yml)
	plt.plot(xml,yac)
	plt.xlabel('x')
	plt.ylabel('y')
	plt.savefig('P3_1_'+str(M)+'_'+str(lbd)+'.png')
	plt.close()
	
P2(8,poly)
P3_1(8,0)
P3_1(8,0.005)
P3_1(8,0.05)
P3_1(8,1)

