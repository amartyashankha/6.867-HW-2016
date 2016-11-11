import numpy as np
from PIL import Image

def normalize(x):
    return 2*x.astype(float)*(1/255.0) - np.ones(x.shape)
        
def load_MNIST_data(tr_size = 200, tst_size = 150, val_size = 150):
    trainX = []
    trainY = []

    testX = []
    testY = []

    valX = []
    valY = []

    for num in range(10):
        data = np.loadtxt('../HW2/data/mnist_digit_'+str(num)+'.csv')
        X = data[:, 0:784].copy()
        trainX.append(X[:tr_size])
        trainY.append(np.ones(tr_size) * num)

        testX.append(X[tr_size:tst_size+tr_size])
        testY.append(np.ones(tst_size) * num)

        valX.append(X[tst_size+tr_size:val_size+tst_size+tr_size])
        valY.append(np.ones(val_size) * num)

    trainX = np.vstack(trainX)
    trainY = np.hstack(trainY)
    testX = np.vstack(testX)
    testY = np.hstack(testY)
    valX = np.vstack(valX)
    valY = np.hstack(valY)

    (trainX, testX, valX) = [normalize(data) for data in (trainX, testX, valX)]
    return (trainX, trainY, testX, testY, valX, valY)
