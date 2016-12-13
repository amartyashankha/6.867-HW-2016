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
def test_classifiers(test_ratio = 0.05, norm = False, one_hot = True, feat_sel = 5000):
    (X, Y, i, words) = load_class(norm = norm, one_hot = one_hot, divs = 1)
            #min_year = 1990, max_year = 2010)
    n_samp = len(Y)
    n_test = int(test_ratio * n_samp)
    trainX = X[:n_samp - n_test]
    trainY = Y[:n_samp - n_test]
    testX = X[n_samp - n_test:]
    testY = Y[n_samp - n_test:]


    regressors = [
                   #LinearRegression(),
                   RidgeCV(alphas = (0, 0.1, 0.5, 1, 5), store_cv_values = True),
                   ]
    classifiers = [#AdaBoostClassifier(),
                   #RidgeClassifier(tol=1e-2, solver="lsqr"),
                   #Perceptron(n_iter=50),
                   #PassiveAggressiveClassifier(n_iter=50),
                   #KNeighborsClassifier(n_neighbors=10),
                   #RandomForestClassifier(),
                   #RandomForestClassifier(n_estimators=200, criterion = 'entropy'),
                   #LinearSVC(loss='l2', penalty='l2', dual=False, tol=1e-3),
                   #MultinomialNB(alpha=0.1),
                   #BernoulliNB(alpha=0.1),
                   #SGDClassifier(alpha=.0001, n_iter=200, penalty='l2'),
                   #SVC()
                   ]


    for classifier in np.hstack((regressors,classifiers)):
        classifier.fit(trainX, trainY)
        print "="*80
        print "Finished fitting " + str(classifier)[:50]

        pred = classifier.predict(testX)
        if classifier in classifiers:
            #print metrics.classification_report(testY, pred)
            print "classification score: ",classifier.score(testX, testY)
        else:
            print "R2: ", metrics.r2_score(testY, pred)
            coefs = zip(np.abs(classifier.coef_), words)
            coefs.sort(reverse = True)
            coefs = [coef[1] for coef in coefs]
            print coefs[:50]
        

        print "MSE of prediction: ", metrics.mean_squared_error(testY, pred)
        mode = np.bincount(testY).argmax()
        print "MSE of predicting mode: ", metrics.mean_squared_error(testY, [mode]*len(testY))

if __name__ == "__main__":
    test_classifiers(test_ratio = 0.1)



