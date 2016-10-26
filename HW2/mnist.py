import numpy as np
from svm import svm, innerProd, gaussianKer
from load_mnist import load
from PIL import Image
from sklearn.linear_model import LogisticRegression
# import your SVM training code

# parameters
setA = ['1']
setB = ['7']
#setA = ['0','2','4','6','8']
#setB = ['1','3','5','7','9']

def countErrors(scoreFn, X, Y):
    return sum([(scoreFn(x)*y) < 0 for x,y in zip(X,Y)])

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
    print "HERE"
    def f(x):
        #print x
        #print x.reshape(1,-1)
        return clf.predict(x.reshape(1,-1))
    return f

(trainX, trainY, testX, testY, valX, valY) = load(setA, setB)
#(trainX, trainY, testX, testY, valX, valY) = (normalize(arr) for arr in load(setA, setB))

def part1():
    C = 1
    ker = innerProd
    #svm_fn = svm(np.matrix(trainX),np.asarray(trainY.T), C = C, ker = ker)
    #print "SVM train: ", countErrors(svm_fn, trainX, trainY)
    #print "SVM test: ", countErrors(svm_fn, testX, testY)

    #showErrors(svm_fn, testX, testY, 0)
    #showErrors(svm_fn, valX, valY, 1)
    #showErrors(svm_fn, valX, valY, 2)

    lr_fn = LRFn(trainX, trainY)
    print "LR train: ", countErrors(lr_fn, trainX, trainY)
    print "LR test: ", countErrors(lr_fn, testX, testY)

def part2():
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = np.arange(0, 2, 0.2)
    best = (1e9, -1, -1)

    for C in Cs:
        for gamma in gammas:
            svm_fn = svm(np.matrix(trainX),np.asarray(trainY.T), C = C, ker = gaussianKer(gamma))
            errs = countErrors(svm_fn, valX, valY)
            print "SVM validating with C="+str(C)+" gamma="+str(gamma)+": ", errs
            best = min(best, (errs,C,gamma))

    svm_fn = svm(np.matrix(trainX),np.asarray(trainY.T), C = best[1], ker = gaussianKer(best[2]))
    print best
    errs = countErrors(svm_fn, testX, testY)
    print "SVM testing with C="+str(C)+" gamma="+str(gamma)+": ", errs

    #showErrors(svm_fn, testX, testY, 0)

part1()


