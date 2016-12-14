from load_lyrics import *
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn import metrics

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['figure.autolayout'] = 'true'

def test_classifiers(test_ratio = 0.05):
    Xs = []
    Ys = []

    norms = [False, False, True]
    one_hot = [True, False, False]
    labels = ['one hot','freq', 'normalized freq']

    feats = [1, 5, 10, 25, 50, 100, 200, 500, 800, 1500, 2000, 2500, 3000]
    for (norm,one_hot,label) in zip(norms,one_hot,labels):
        (X, Y, i, words) = load_class(norm = norm, one_hot = one_hot, divs = 1)
        Xs.append(X)
        Ys.append(Y)

        n_samp = len(Y)
        n_test = int(test_ratio * n_samp)
        trainX = X[:n_samp - n_test]
        trainY = Y[:n_samp - n_test]
        testX = X[n_samp - n_test:]
        testY = Y[n_samp - n_test:]

        res = []
        for feat in feats:
            ch2 = SelectKBest(chi2, k = feat)        
            trX = ch2.fit_transform(trainX, trainY)      
            tstX = ch2.transform(testX)       
            lr = LinearRegression(n_jobs = -1)

            lr.fit(trX, trainY)
            print "Finished fitting " + str(feat)

            pred = lr.predict(tstX)
            mse = np.sqrt(metrics.mean_squared_error(testY, pred))
            print "RMSE of prediction: ", mse
            res.append(mse)

        plt.plot(feats, res, label = label)

    plt.xlabel('Number of features selected')
    plt.ylabel('RMSE')
    ax = plt.gca()
    ax.ticklabel_format(useOffset=False)
    plt.title('Accuracy of various feature models')
    plt.legend()
    plt.savefig('feats.png')
    plt.show()

if __name__ == "__main__":
    test_classifiers(test_ratio = 0.1)



