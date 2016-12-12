from full_nn import *
import numpy as np
from copy import deepcopy
from random import randint
from plotBoundary import *
from PIL import Image

def relu(x):
    return np.maximum(x,0)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def cross_entropy(y,x):
    x = np.log(x)
    return - y.T*x

def one_hot(i, k):
    ret = np.zeros(k)
    ret[int(i)] = 1
    return ret

def predictor(network):
    def f(x):
        return np.argmax(network.query(x))
    return f

def cont_predictor(network):
    def f(x):
        return network.query(x)[0,0]
    return f

def easy_test():
    NUM_EPOCHS = 20

    data = np.loadtxt('data/data_3class.csv')
    n, m = data.shape

    network = NN(3, 2, 3, 5, relu, softmax, cross_entropy)

    for i in range(NUM_EPOCHS * n):
        ind = randint(0, n-1)
        network.train(np.matrix(data[ind][:-1]).T, one_hot(data[ind][2],3))

    ers = 0
    for i in range(n):
        ers+= (int(data[i][2]) != np.argmax(network.query(np.matrix(data[i][:-1]).T)))

    print(ers, n)

def error_rate(network, X, Y):
    n = X.shape[0]
    ers = sum([(int(Y[i]) != np.argmax(network.query(np.matrix(X[i]).T))) for i in range(n)])
    return ers/(1.0 * n)

def misclassifications(network, X, Y, save = False):
    n = X.shape[0]
    ret = []
    c   = 0
    for i in range(n):
        val = np.argmax(network.query(np.matrix(X[i]).T))
        if int(Y[i]) != val:
            ret.append((val, int(Y[i])))
            if save:
                unnorm = (X[i] + np.ones(X[i].shape))*(255.0/2)
                unnorm = np.uint8(unnorm)
                img = Image.fromarray(unnorm.reshape((28,28))).convert('RGB')
                img.save('digs/thought_'+str(val)+'_actually_'+str(int(Y[i]))+'_'+str(c)+'.png')
            c += 1

    return ret

def train_multiclass(data, hidden_layers, hidden_nodes, classes, dim_x, error_iters = 10, conv_thresh = 0.01, max_iters = 100, eta = 1e-3):
    (trainX, trainY, valX, valY, testX, testY) = data
    n_train, n_val, n_test = (trainX.shape[0], valX.shape[0], testX.shape[0])

    network = NN(hidden_layers+2, dim_x, classes, hidden_nodes, relu, softmax, cross_entropy)
    mn  = 1
    best_net = None
    errors = []

    for c in range(max_iters):
        for i in range(n_train):
            ind = randint(0, n_train-1)
            network.train(np.matrix(trainX[ind]).T, one_hot(trainY[ind],classes))

            
        ers = error_rate(network, valX, valY)

        errors.append(ers)
        last_few = errors[-error_iters:]
        #print ers
        if ers < mn:
            mn = ers
            best_net = deepcopy(network)
        if len(last_few) == error_iters and ((np.max(last_few) - np.min(last_few)) <= conv_thresh):
            break

    #print "Final validation error: ", error_rate(best_net, valX, valY)
    #print "Final training error: ", error_rate(best_net, trainX, trainY)
    #print "Final testing error: ", error_rate(best_net, testX, testY)
    #misclassifications(network, testX, testY, save = True)

    return error_rate(best_net, testX, testY)
