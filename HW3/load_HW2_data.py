import numpy as np

def load_HW2_data(i):
    train = np.loadtxt('../HW2/data/data'+str(i)+'_train.csv')
    val   = np.loadtxt('../HW2/data/data'+str(i)+'_validate.csv')
    test  = np.loadtxt('../HW2/data/data'+str(i)+'_test.csv')

    trainX = train[:,0:2].copy()
    trainY = (train[:,2:3].copy()+1)/2
    valX   = val[:,0:2].copy()
    valY   = (val[:,2:3].copy()+1)/2
    testX  = test[:,0:2].copy()
    testY  = (test[:,2:3].copy()+1)/2
    
    return (trainX, trainY, valX, valY, testX, testY)
