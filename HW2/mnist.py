import numpy as np
from svm import svm, innerProd, gaussianKer
from load_mnist import load
from PIL import Image
from sklearn.linear_model import LogisticRegression
import time
import matplotlib.pyplot as plt
from pegasos_training import train_linearSVM

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['figure.autolayout'] = 'true'

# parameters
setsA = [['1'], ['3'], ['4'], ['0','2','4','6','8']]
setsB = [['7'], ['5'], ['9'], ['1','3','5','7','9']]
setA = ['1']
setB = ['7']

def countErrors(scoreFn, X, Y):
    return np.asscalar(sum([(scoreFn(x)*y) < 0 for x,y in zip(X,Y)]))

def errorRate(scoreFn, X, Y):
    return (1.0*countErrors(scoreFn, X, Y))/len(X)

def getPegasosFn(w):
    def f(x):
        return w[0] + w[1:].dot(x)
    return f

def normalize(x):
    return 2*x.astype(float)*(1/255.0) - np.ones(x.shape)

def showErrors(scoreFn, X, Y, i):
    ind = i
    for x,y in zip(X,Y):
        if (scoreFn(x)*y) < 0:
            if i==0:
                img = Image.fromarray(x.reshape((28,28))).convert('RGB')
                img.save('mis'+str(ind)+'.png')
                break
            i-=1

def LRFn(X,Y):
    clf = LogisticRegression()
    clf.fit(X,Y)
    def f(x):
        #print x
        #print x.reshape(1,-1)
        return clf.predict(x.reshape(1,-1))
    return f

#(trainX, trainY, testX, testY, valX, valY) = (normalize(arr) for arr in load(setA, setB))

def part1():
    C = 1
    ker = innerProd
    for (setA,setB) in zip(setsA,setsB):
        (trainX, trainY, testX, testY, valX, valY) = load(setA, setB)
        #(trainX, testX, valX) = (normalize(arr) for arr in (trainX,testX,valX))
        print "================"
        print setA, setB, "C: ", C, "ker: linear"
        svm_fn = svm(np.matrix(trainX),np.asarray(trainY.T), C = C, ker = ker)
        print "SVM train: ", errorRate(svm_fn, trainX, trainY)
        print "SVM test: ", errorRate(svm_fn, testX, testY)

        #showErrors(svm_fn, testX, testY, 0)
        #showErrors(svm_fn, valX, valY, 1)
        #showErrors(svm_fn, valX, valY, 2)

        lr_fn = LRFn(trainX, trainY)
        print "LR train: ", errorRate(lr_fn, trainX, trainY)
        print "LR test: ", errorRate(lr_fn, testX, testY)

def part2():
    Cs = [0.1, 1, 10]
    gammas = [0.01, 0.1, 1, 10]

    for (setA,setB) in zip(setsA,setsB):
        best = (1e9, -1, -1)
        print "======================"
        (trainX, trainY, testX, testY, valX, valY) = load(setA, setB)
        #(trainX, testX, valX) = (normalize(arr) for arr in (trainX,testX,valX))
        print setA, setB
        for C in Cs:
            for gamma in gammas:
                svm_fn = svm(np.matrix(trainX),np.asarray(trainY.T), C = C, ker = gaussianKer(gamma))
                errs = errorRate(svm_fn, valX, valY)
                print "SVM validating with C="+str(C)+" gamma="+str(gamma)+": ", errs
                best = min(best, (errs,C,gamma))

        svm_fn = svm(np.matrix(trainX),np.asarray(trainY.T), C = best[1], ker = gaussianKer(best[2]))
        print best
        errs = errorRate(svm_fn, testX, testY)
        print "SVM testing with C="+str(C)+" gamma="+str(gamma)+": ", errs

    #showErrors(svm_fn, testX, testY, 0)

def part3_1():
    C = 1
    ker = innerProd
    sizes = np.arange(50, 1000, 50)
    lambdas = [None, 0.2, 0.02, 2]
    epochs = [None, 50, 50, 50]
    (trainX, trainY, testX, testY, valX, valY) = load(setA, setB, tr_size=1000,tst_size=0,val_size =0)

    for (lbd, epoch) in zip(lambdas,epochs):
        times = []
        for size in sizes:
            X = trainX[:size]
            Y = trainY[:size]

            t1 = int(round(time.time() * 1000))
            if lbd:
                train_linearSVM(X, Y, lbd, max_epochs = epoch)
            else:
                svm(np.matrix(X),np.asarray(Y.T), C = 1, ker = innerProd)
            t2 = int(round(time.time() * 1000))
            print size, t2-t1
            times.append(t2-t1)
        if lbd:
            plt.plot(sizes, times, label="Pegasos lambda = "+str(lbd))
        else:
            plt.plot(sizes, times, label="SVM QP")

    plt.xlabel('number of training samples')
    plt.ylabel('time (ms)')
    plt.title('SVM vs Pegasos running time')
    plt.legend()
    plt.savefig('comparison.png')

def part3_2():
    C = 1
    ker = innerProd
    sizes = np.arange(50, 1000, 50)
    lambdas = [None, 0.2, 0.02, 2]
    epochs = [None, 25, 25, 25]

    for setA in setsA:
        for setB in setsB:
            print "========================"
            print setA, setB
            (trainX, trainY, testX, testY, valX, valY) = load(setA, setB)
            for (lbd, epoch) in zip(lambdas,epochs):
                cls = None
                if lbd:
                    cls = getPegasosFn(train_linearSVM(trainX, trainY, lbd, max_epochs = epoch))
                else:
                    cls = svm(np.matrix(trainX),np.asarray(trainY.T), C = 1, ker = innerProd)
                print lbd, errorRate(cls,valX, valY)

#part1()
part3_2()


