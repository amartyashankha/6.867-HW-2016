from numpy import *
from plotBoundary import *
import pylab as pl
# import your LR training code
from pegasos_training import train_gaussianSVM, gauss_kernel

# load data from csv files
train = loadtxt('data/data3_train.csv')
X = train[:,0:2]
Y = train[:,2:3]

# Carry out training.
epochs = 1000;
lmbda = .02;
gamma = 2e-2;

### TODO: Compute the kernel matrix ###


### TODO: Implement train_gaussianSVM ###
alpha = train_gaussianSVM(X, Y, gamma)


# Define the predict_gaussianSVM(x) function, which uses trained parameters, alpha
def predict_gaussianSVM(x):
    return sum([alpha[i]*gauss_kernel(x,X[i],gamma) for i in range(len(X))])

# plot training results
plotDecisionBoundary(X, Y, predict_gaussianSVM, [-1,0,1], title = 'Gaussian Kernel SVM')
pl.show()
