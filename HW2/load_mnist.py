import numpy as np

def load(setA, setB, tr_size = 200, tst_size = 150, val_size = 150):
    trainX = []
    trainY = []

    testX = []
    testY = []

    valX = []
    valY = []

    for num in setA:
        data = np.loadtxt('data/mnist_digit_'+num+'.csv')
        X = data[:, 0:784].copy()
        trainX.append(X[:tr_size])
        trainY.append(np.ones(tr_size))

        testX.append(X[tr_size:tst_size+tr_size])
        testY.append(np.ones(tst_size))

        valX.append(X[tst_size+tr_size:val_size+tst_size+tr_size])
        valY.append(np.ones(val_size))

    for num in setB:
        data = np.loadtxt('data/mnist_digit_'+num+'.csv')
        X = data[:, 0:784].copy()
        trainX.append(X[:tr_size])
        trainY.append(-np.ones(tr_size))

        testX.append(X[tr_size:tst_size+tr_size])
        testY.append(-np.ones(tst_size))

        valX.append(X[tst_size+tr_size:val_size+tst_size+tr_size])
        valY.append(-np.ones(val_size))

    trainX = np.vstack(trainX)
    trainY = np.hstack(trainY)
    testX = np.vstack(testX)
    testY = np.hstack(testY)
    valX = np.vstack(valX)
    valY = np.hstack(valY)

    return (trainX, trainY, testX, testY, valX, valY)
