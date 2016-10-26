from numpy import *
from plotBoundary import *
import pylab as pl
from svm import svm, innerProd, gaussianKer
# import your SVM training code

# parameters
names = ['4']
Cs = [0.01, 0.1, 1, 10, 100]
gamma = 1
ker = innerProd
if gamma:
    ker = gaussianKer(gamma)

def countErrors(scoreFn, X, Y):
    return sum([(scoreFn(x)*y) < 0 for x,y in zip(X,Y)])
    
for C in Cs:
    for name in names:
        # load data from csv files
        print '============'
        print "C: ",C," data: ",name, "gamma: ", gamma
        train = loadtxt('data/data'+name+'_train.csv')
        # use deep copy here to make cvxopt happy
        X = train[:, 0:2].copy()
        Y = train[:, 2:3].copy()

        # Carry out training, primal and/or dual

        predictSVM = svm(matrix(X),asarray(Y.T)[0], C = C, ker = ker)

        # Define the predictSVM(x) function, which uses trained parameters

        # plot training results
        plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], fname =  name+'_'+str(C)+'_'+str(gamma)+'_train', title = 'SVM Train')

        errs = countErrors(predictSVM, X, Y)
        print "Training errors "+name+": ", errs, (1.0*errs)/len(X)


        # load data from csv files
        validate = loadtxt('data/data'+name+'_validate.csv')
        X = validate[:, 0:2]
        Y = validate[:, 2:3]
        # plot validation results
        plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Validate', fname =  name+'_'+str(C)+'_'+str(gamma)+'_val')

        errs = countErrors(predictSVM, X, Y)
        print "Validation errors "+name+": ", errs, (1.0*errs)/len(X)
