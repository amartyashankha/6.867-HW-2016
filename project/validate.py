from load_lyrics import *
import numpy as np


from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn import metrics

def error(a, b):
    dif = a-b
    return 
def test_classifiers(test_ratio = 0.1, norm = False, one_hot = True, 
        val_ratio = 0.2)

    (X, Y, i, words) = load_class(norm = norm, one_hot = one_hot, divs = 1)
            #min_year = 1990, max_year = 2010)
    n_samp = len(Y)
    n_test = int(test_ratio * n_samp)
    n_val  = int(val_ratio * n_samp)

    trainX = X[:n_samp - n_test - n_val]
    trainY = Y[:n_samp - n_test - n_val]
    testX  = X[n_samp - n_test - n_val: n_samp - n_val]
    testY  = Y[n_samp - n_test - n_val: n_samp - n_val]
    valX   = X[n_samp - n_val: ]
    valY   = Y[n_samp - n_val: ]

    alphas = np.hstack((np.arange(0, 1, 0.1), 2, 5))
    res    = []

    for alpha in alphas:
        classifier = Ridge(alpha = alpha)
        classifier.fit(trainX, trainY)

        print "Finished fitting ", alpha

        pred = classifier.predict(valX)
        print "Validation RMSE: ", metrics.mean_squared_error(valY, pred)

if __name__ == "__main__":
    test_classifiers(test_ratio = 0.1)



