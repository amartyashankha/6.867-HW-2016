from load_lyrics import *
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn import metrics


def test_classifiers(test_ratio = 0.05, norm = False, one_hot = True):
    (X, Y, i) = load_class(norm = norm, one_hot = one_hot, divs = 5)
    n_samp = len(Y)
    n_test = int(test_ratio * n_samp)
    trainX = X[:n_samp - n_test]
    trainY = Y[:n_samp - n_test]
    testX = X[n_samp - n_test:]
    testY = Y[n_samp - n_test:]

    classifiers = [AdaBoostClassifier(),
                   #RidgeClassifier(tol=1e-2, solver="lsqr"),
                   #Perceptron(n_iter=50),
                   #PassiveAggressiveClassifier(n_iter=50),
                   #KNeighborsClassifier(n_neighbors=10),
                   RandomForestClassifier(n_estimators=200),
                   RandomForestClassifier(n_estimators=200, criterion = 'entropy'),
                   #LinearSVC(loss='l2', penalty='l2', dual=False, tol=1e-3),
                   #SGDClassifier(alpha=.0001, n_iter=200, penalty='l2'),
                   #SVC()
                   ]


    for classifier in classifiers:
        classifier.fit(trainX, trainY)
        print "="*80
        print "Finished fitting " + str(classifier)[:50]

        pred = classifier.predict(testX)
        print metrics.classification_report(testY, pred)
        print classifier.score(testX, testY)

if __name__ == "__main__":
    test_classifiers(test_ratio = 0.25)

