import numpy as np
from svm import svm, innerProd, gaussianKer
from load_mnist import load
from PIL import Image
# import your SVM training code

# parameters
setA = ['1']
setB = ['7']
#setA = ['0','2','4','6','8']
#setB = ['1','3','5','7','9']
C = 1
ker = innerProd

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

(trainX, trainY, testX, testY, valX, valY) = load(setA, setB)
#(trainX, trainY, testX, testY, valX, valY) = (normalize(arr) for arr in load(setA, setB))

def part1():
    svm_fn = svm(np.matrix(trainX),np.asarray(trainY.T), C = C, ker = ker)
    print "SVM train: ", countErrors(svm_fn, trainX, trainY)
    print "SVM validate: ", countErrors(svm_fn, valX, valY)
    print "SVM test: ", countErrors(svm_fn, testX, testY)

    svm_fn = svm(np.matrix(trainX),np.asarray(trainY.T), C = C, ker = ker)
    print "SVM train: ", countErrors(svm_fn, trainX, trainY)
    print "SVM validate: ", countErrors(svm_fn, valX, valY)
    print "SVM test: ", countErrors(svm_fn, testX, testY)

#showErrors(svm_fn, testX, testY, 0)
#showErrors(svm_fn, valX, valY, 1)
#showErrors(svm_fn, valX, valY, 2)

