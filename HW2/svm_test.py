from numpy import *
from plotBoundary import *
import pylab as pl
from svm import svm, innerProd, gaussianKer
# import your SVM training code

# parameters
name = '3'
C = 0.01
ker = gaussianKer(1)
print '======Training======'
# load data from csv files
train = loadtxt('data/data'+name+'_train.csv')
# use deep copy here to make cvxopt happy
X = train[:, 0:2].copy()
Y = train[:, 2:3].copy()

# Carry out training, primal and/or dual

predictSVM = svm(matrix(X),asarray(Y.T)[0], C = C, ker = ker)

# Define the predictSVM(x) function, which uses trained parameters
### TODO ###

# plot training results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title =  name+'_'+str(C)+'_train.png')


print '======Validation======'
# load data from csv files
validate = loadtxt('data/data'+name+'_validate.csv')
X = validate[:, 0:2]
Y = validate[:, 2:3]
# plot validation results
plotDecisionBoundary(X, Y, predictSVM, [-1, 0, 1], title = 'SVM Validate')
pl.show()
