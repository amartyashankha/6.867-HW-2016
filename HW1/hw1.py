import matplotlib.pyplot as plt
import math
import pylab as pl
import P2.loadFittingDataP2 as load2
import P3.regressData as load3
import numpy as np
import grad_desc as gd

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['figure.autolayout'] = 'true'

def getWML(phis, X, t):
    M,N = len(phis), len(X)
    Phi = np.asmatrix([[phis[j](X[i]) for j in range(M)] for i in range(N)])
#   print("DETERMINTANT OF MATRIX:", np.linalg.det(Phi.T*Phi))
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
        if f == cosn:
            return [apply(f,n) for n in range(1,M+1)]
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

def P2(M, F, F2 = None):
    X,Y = load2.getData(False)
    wml = getWML(getBasis(M,F), X, Y)
    print(wml)
    est = linearSum(wml,getBasis(M,F))
    print(SSE(est,X,Y))
    xml   = np.arange(0,1,0.01)
    yml   = np.array(est(xml))
    yac   = np.cos(math.pi*xml) + np.cos(2*math.pi*xml)

    plt.plot(xml,yml, label = 'Polynomial basis regression')
    plt.plot(xml,yac, label = 'Actual function')
    if F2:
        wml2 = getWML(getBasis(M,F2), X, Y)
        est = linearSum(wml2,getBasis(M,F2))
        print(SSE(est,X,Y))
        yml   = np.array(est(xml))
        plt.plot(xml,yml, label = 'Cosine basis regression')
    plt.plot(X,Y,'o', markersize = 12, markeredgewidth=2 ,markeredgecolor= '#1e8c19', markerfacecolor = 'none', label = 'Training data')
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.title('Linear Regression (M = '+str(M-1)+')')
    plt.title('Polynomial and Cosine basis functions (M=8)')
    axes = plt.gca()
    axes.set_ylim([-2,3])
    axes.grid(False)
    plt.legend()
    plt.savefig('P2_'+str(M)+'_'+str(F.__name__)+'.png')
    plt.close()

    return wml

def P3_1(M, lbds, dataFunc):
    F = poly
    X,Y = getData()

    xml   = np.arange(min(X),max(X),0.01)
    for lbd in lbds:
        wml = getWRidge(getBasis(M,F), X,Y, lbd)
        est = linearSum(wml,getBasis(M,F))
        yml   = np.array(est(xml))
        print(SSE(est,X,Y))
        plt.plot(xml,yml, label = 'lambda = '+str(lbd))

    yac   = np.cos(math.pi*xml) + np.cos(2*math.pi*xml)
    plt.plot(xml,yac, label = 'Actual function', ls ='--')
    plt.plot(X,Y,'o', markersize = 12, markeredgewidth=2 ,markerfacecolor = 'none', label = 'Training data')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Ridge regression for M = '+str(M-1))
    axes = plt.gca()
    axes.grid(False)

    fname = 'P312.png'
    plt.savefig(fname)
    plt.close()

    return est

def P3_2(Ms, lbds, train, test, val, plotv = False, plot = False):
    F = poly
    X,Y = train()
    X2,Y2 = test()
    X3,Y3 = val()
    print(len(X),len(Y))
    print(len(X2),len(Y2))
    print(len(X3),len(Y3))

    xml   = np.arange(-3,3,0.01)
    Mbest, Lbest, SSEbest = None, None, 1e10
    for M in Ms:
        for lbd in lbds:
            wml = getWRidge(getBasis(M,F), X,Y, lbd)
            est = linearSum(wml,getBasis(M,F))
            sse = SSE(est,X3,Y3)

            if sse<SSEbest:
                SSEbest = sse
                Mbest = M
                Lbest = lbd
                #print("Try: ",sse, M, lbd)

            if plotv:
                yml   = np.array(est(xml))
                plt.plot(xml,yml, label = 'lambda = '+str(lbd)+', M = '+str(M-1))

    wml = getWRidge(getBasis(Mbest,F), X,Y, Lbest)
    est = linearSum(wml,getBasis(Mbest,F))

    sse_test = SSE(est, X2, Y2)

    print("Best: ",SSEbest, Mbest, Lbest, sse_test)
    if plot:
        yml   = np.array(est(xml))
        plt.plot(xml,yml, label = 'lambda = '+str(Lbest)+', M = '+str(Mbest-1))
        plt.plot(X2,Y2,'o', markersize = 12, markeredgewidth=2 , label = 'Testing data')
        plt.plot(X3,Y3,'o', markersize = 12, markeredgewidth=2 ,markerfacecolor = 'none', label = 'Validation data')
        plt.plot(X,Y,'o', markersize = 12, markeredgewidth=2 ,label = 'Training data')

    if plotv:
        plt.plot(X3,Y3,'o', markersize = 12, markeredgewidth=2 ,markerfacecolor = 'none', label = 'Validation data')

    if plot or plotv:
        plt.legend(loc = 4)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Ridge regression trained on '+str(train.__name__[7]))
        axes = plt.gca()
        axes.grid(False)
        axes.set_ylim([-3,3])

        fname = 'P33'+str(train.__name__[7])+'.png'
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
    print("***************************************")
    print(M)
    print("wml: ",wml)
    #print("SSE: ",sse(point))
    #print("Analytic Grad: ",grad(point))
    #print("Analytic Grad Norm: ",np.linalg.norm(grad(point)))
    ng = gd.GD([0]*M, sse)
    #print("Numeric Grad:",ng.compute_gradient(point, eps =1e-8))
    #print("Numeric Grad Norm:",np.linalg.norm(ng.compute_gradient(point, eps =1e-8)))

    phiX = np.asarray([[phis[i](x) for i in range(M)] for x in X])
    sgd = gd.SGD(phiX, Y)
    sgd_log = sgd.step(np.random.rand(M), stochastic = True, minibatch_size = 1, ftol = 1e-9, gtol = 1e-3, stepSize = 0.1)
    gd_log = sgd.step(np.random.rand(M), stochastic = False, ftol = 1e-9, gtol = 1e-5, stepSize = 0.02)
    print("Stochastic: ", sgd_log[-1])
    #print("Stochastic second last: ", sgd_log[-2])
    print("Stochastic len: ", len(sgd_log))
    #print("Closed form grad at Stoch: ", grad(sgd_log[-1][0]))

    #print("Numeric grad at Stoch: ", ng.compute_gradient(sgd_log[-1][0], eps =1e-8))
    #print("In code grad at Stoch: ", sgd.compute_gradient(sgd_log[-1][0]))
    #print("In code numeric grad at Stoch: ", sgd.compute_numerical_gradient(sgd_log[-1][0]))

    print("Batch GD: ", gd_log[-1])
    print("Batch GD len: ", len(gd_log))

#stuff for problem 3
#f = P3(10, 0.5, load3.regressAData)
#validate(f, load3.validateData)

#Problem 2.2
#P2_2(5,poly)

def gen_p2():
    P2(1,poly)
    P2(2,poly)
    P2(4,poly)
    P2(6,poly)
    P2(8,poly)
    P2(11,poly)

def gen_p22():
    P2_2(5,poly)

def gen_p24():
    wml = P2(8,cosn, poly)
    lefts = range(1,9)

#    plt.xlabel('i')
    #plt.ylabel('$w_i$')
    plt.title('Maximum likelihood weights for M=8')
    plt.bar(lefts, wml, align = 'center', tick_label = lefts, alpha = 0.4)
    axes = plt.gca()
    axes.set_ylim([0,1])
    axes.grid(False)
    plt.savefig('bar1.png')
    plt.close()

    true = np.zeros(8)
    true[0] = 1
    true[1] = 1
#    plt.xlabel('i')
#    plt.ylabel('$w_i$')
    plt.title('Actual weights')
    plt.bar(lefts, true, align = 'center', tick_label = lefts, alpha = 0.4)
    axes = plt.gca()
    axes.grid(False)
    axes.set_ylim([0,1])
    plt.savefig('bar2.png')
    plt.close()

def gen_p31():
    lbds = [0, 0.001, 0.05,1]
    P3_1(3, lbds, load2.getData)
    P3_1(11, lbds, load2.getData)

def gen_p32():
    Ms = range(1,15)
    lbds = np.arange(0,1,0.001)
    P3_2(Ms, lbds, load3.regressAData, load3.regressBData, load3.validateData, plot = True)
    P3_2(Ms, lbds, load3.regressBData, load3.regressAData, load3.validateData, plot = True)

def gen_p33():
    Ms = [10,3]
    lbds = [0,0.5]
    P3_2(Ms, lbds, load3.regressAData, load3.regressBData, load3.validateData, plotv = True)
    P3_2(Ms, lbds, load3.regressBData, load3.regressAData, load3.validateData, plotv = True)

gen_p24()

