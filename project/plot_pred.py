from load_lyrics import *
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn import metrics

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['figure.autolayout'] = 'true'

def test_classifiers(test_ratio = 0.05, norm = False, one_hot = True, feat_sel = 5000):
    (X, Y, i, words) = load_class(norm = norm, one_hot = one_hot, divs = 1)
            #min_year = 1990, max_year = 2010)
    n_samp = len(Y)
    n_test = int(test_ratio * n_samp)
    trainX = X[:n_samp - n_test]
    trainY = Y[:n_samp - n_test]
    testX = X[n_samp - n_test:]
    testY = Y[n_samp - n_test:]


    lr = LinearRegression(n_jobs = -1)

    lr.fit(trainX, trainY)
    print "Finished fitting " 

    pred = lr.predict(testX)
    print "R2: ", metrics.r2_score(testY, pred)
    coefs = zip(np.abs(lr.coef_), words)
    coefs.sort(reverse = True)
    coefs = [coef[1] for coef in coefs]
    print coefs[:50]
    

    print "MSE of prediction: ", metrics.mean_squared_error(testY, pred)
    #mode = np.bincount(testY).argmax()
    mode = np.mean(trainY)
    print "MSE of baseline: ", metrics.mean_squared_error(testY, [mode]*len(testY))
    print "R2 baseline: ", metrics.r2_score(testY, [mode] * len(testY))
    
    testY += 1960
    pred  += 1960
    yrs = np.arange(1960,2015,5)
    plt.scatter(pred, testY)
    plt.xticks(yrs, yrs, rotation = 'vertical')
    plt.xlabel('Actual release year')
    plt.ylabel('Predicted release year')
    plt.title('Results of Linear Regression')
    plt.show()

if __name__ == "__main__":
    test_classifiers(test_ratio = 0.1)



