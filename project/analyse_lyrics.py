from load_lyrics import *
import numpy as np
import sklearn.ensemble
import sklearn.svm


def get_unique_5y(thresh):
    (words, songs) = load(norm = False)

    years = np.arange(1960, 2011, 5)
    sums = np.zeros((len(years)-1, 5000))
    nums = np.zeros(len(years)-1)

    for song in songs:
        ind = 0
        if years[ind] >= song.year:
            continue
        while years[ind] < song.year:
            ind+=1
        ind -= 1
        for i in range(5000):
            #sums[ind][i] += song.freq_dic.get(i,0)
            sums[ind][i] += (i in song.freq_dic)
        nums[ind] += 1

    vecs = [sums[i]/nums[i] for i in range(len(nums))]
    tots = np.zeros(5000)
    for i in range(len(nums)):
        tots += vecs[i]

    res = []
    for y in range(len(nums)):
        meaningful = [(words[i], vecs[y][i]) for i in range(5000) if ((tots[i] > 0 ) and (vecs[y][i]/(1.0*tots[i])) > (thresh/len(nums)))]
        res.append(sorted(meaningful, key = (lambda x: x[1]), reverse = True)[:15])
        print years[y], [item[0] for item in res[y]]

def build_regressor(test_ratio = 0.05, norm = True):
    (X, Y, i) = load_regress(norm)
    Y = np.array(Y)
    Y -= 1970
    n_samp = len(Y)
    n_test = int(test_ratio * n_samp)
    trainX = X[:n_samp - n_test]
    trainY = Y[:n_samp - n_test]
    testX = X[n_samp - n_test:]
    testY = Y[n_samp - n_test:]

    adb = sklearn.ensemble.AdaBoostRegressor(n_estimators = 50)
    adb.fit(trainX,trainY)
    print "Finished fitting model"

    predY = adb.predict(testX)

    for (x,y) in zip(predY, testY):
        print x,y

    print adb.score(testX, testY)

def build_classifier(test_ratio = 0.05, norm = True):
    (X, Y, i) = load_class(norm, divs = 5)
    n_samp = len(Y)
    n_test = int(test_ratio * n_samp)
    trainX = X[:n_samp - n_test]
    trainY = Y[:n_samp - n_test]
    testX = X[n_samp - n_test:]
    testY = Y[n_samp - n_test:]

    #adb = sklearn.ensemble.AdaBoostClassifier(base_estimator = sklearn.svm.SVC(probability = True), n_estimators = 3)
    adb = sklearn.ensemble.AdaBoostClassifier(n_estimators = 200)
    #adb = sklearn.svm.SVC(C = 0.01)
    adb.fit(trainX,trainY)
    print "Finished fitting model"

    predY = adb.predict(testX)

    for (x,y) in zip(predY, testY):
        print x,y

    print adb.score(testX, testY)

build_classifier(test_ratio = 0.06)
#build_regressor(test_ratio = 0.02)
#build_learner(norm = False)
#get_unique_5y(3.5)
