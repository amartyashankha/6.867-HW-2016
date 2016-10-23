from numpy import *
from plotBoundary import *
import pylab as pl
# import your LR training code

# parameters
name = 'ls'
print '======Training======'
# load data from csv files
train = loadtxt('data/data'+name+'_train.csv')
X = train[:,0:2]
Y = train[:,2:3]

# Carry out training.
### TODO ###

# Define the predictLR(x) function, which uses trained parameters
### TODO ###

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
