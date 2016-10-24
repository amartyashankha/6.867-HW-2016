from numpy import *
from plotBoundary import *
import pylab as pl
# import your LR training code
from pegasos_training import train_linearSVM

# load data from csv files
train = loadtxt('data/data3_train.csv')
X = train[:,0:2]
Y = train[:,2:3]
L = 2**-5

# Carry out training.
w = train_linearSVM(X, Y, L)


# Define the predict_linearSVM(x) function, which uses global trained parameters, w
def predict_linearSVM(x):
    return w[0] + w[1:].dot(x)


# plot training results
plotDecisionBoundary(X, Y, predict_linearSVM, [-1,0,1], title = 'Linear SVM')
pl.show()

