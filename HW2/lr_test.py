from numpy import *
from plotBoundary import *
import pylab as pl
from sklearn.linear_model import LogisticRegression
import sys

# import your LR training code

# parameters
name = sys.argv[1]
print '======Training======'
# load data from csv files
train = loadtxt('data/data'+name+'_train.csv')
X = train[:,0:2]
Y = train[:,2:3]

# Carry out training.
L = float(sys.argv[2])
C = 1.0/L 
clf = LogisticRegression(C=1e5)
if len(sys.argv) > 3:
    clf = LogisticRegression(C=C, penalty='l1')
else:
    clf = LogisticRegression(C=C)

clf.fit(X, Y)

# Define the predictLR(x) function, which uses trained parameters
def predictLR(x):
    return clf.predict(x.reshape(1, -1))

# plot training results
plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Train')

print '======Validation======'
# load data from csv files
validate = loadtxt('data/data'+name+'_validate.csv')
X = validate[:,0:2]
Y = validate[:,2:3]

# plot validation results
plotDecisionBoundary(X, Y, predictLR, [0.5], title = 'LR Validate')
pl.show()
