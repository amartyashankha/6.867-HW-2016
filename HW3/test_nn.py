from full_nn import *
import numpy as np
from random import randint

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

    print ers, n

def P1_4(i, layers, hidden_nodes):
    train = np.loadtxt('../HW2/data/data'+str(i)+'_train.csv')

    X = train[:,0:2].copy()
    Y = train[:,2:3].copy()

    network = NN(layers, 2, 2, hidden_nodes, relu, softmax, cross_entropy)

    for i in range(NUM_EPOCHS * n):
        ind = randint(0, n-1)
        network.train(np.matrix(data[ind][:-1]).T, one_hot(data[ind][2],3))

    ers = 0
    for i in range(n):
        ers+= (int(data[i][2]) != np.argmax(network.query(np.matrix(data[i][:-1]).T)))

    print ers, n

P1_4(1)
