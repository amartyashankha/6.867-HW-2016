import numpy as np
from scipy.sparse import dok_matrix
import matplotlib.pyplot as plt

years = {}

ydata = open('data/tracks_per_year.txt').readlines()
for line in ydata:
    proc = line.split('<SEP>')
    years[proc[1]] = int(proc[0])

class LyricVector():
    def __init__(self, line, norm = True):
        vals = line.split(',')
        self.track_id = vals[0]
        self.mxm_id   = vals[1]
        self.freq_dic = {}
        for item in vals[2:]:
            proc = item.split(':')
            self.freq_dic[int(proc[0])] = int(proc[1])
        if norm:
            N = sum(self.freq_dic.values())
            for i in self.freq_dic:
                self.freq_dic[i] /= float(N)
        self.year = years.get(self.track_id, None)

def load(norm):
    data = open('mxm_dataset_train.txt').readlines()[:15000]
    words = [line.split(',') for line in data if line[0] == '%'][0]
    songs = [LyricVector(line, norm) for line in data if line[0] == 'T']
    songs = [song for song in songs if song.year != None]
    print "Finished loading data"
    return (words, songs)

def plot_years(word):
    (words, songs) = load(norm = False)
    word = words.index(word)
    yrs  = np.arange(1960, 2015, 5)
    cnt  = np.zeros(len(yrs))
    num  = np.zeros(len(yrs))
    tot  = np.zeros(len(yrs))
    for song in songs:
        ind = (max(song.year, 1960) - 1960)/5
        cnt[ind] += song.freq_dic.get(word, 0)
        num[ind] += (word in song.freq_dic)
        tot[ind] += 1.0

    cnt/= tot
    num/= tot
    print yrs
    print cnt
    print num
    plt.plot(yrs, cnt)
    plt.plot(yrs, num)
    plt.show()

def load_regress(norm, one_hot = True, dense = False):
    data = open('mxm_dataset_train.txt').readlines()[:15000]
    (words, songs) = load(norm and (not one_hot))
    X = dok_matrix((len(songs), 5000), dtype = np.float32)
    Y = []
    ids = []
    for i in range(len(songs)): 
        song = songs[i]
        for j in song.freq_dic:
            if one_hot:
                X[i,j] = 1
            else:
                X[i, j] = song.freq_dic[j]
        Y.append(song.year)
        ids.append(song.track_id)
    if dense:
        X = X.toarray() # remove this later to keep sparse matrix
    print "Finished making regression matrix"
    return (X,Y,ids)

def load_class(norm, min_year = 1970, max_year = 2016, divs = 10, one_hot = True):
    (X,Y,ids) = load_regress(norm, one_hot)
    Y = np.array(Y)
    Y = np.clip(Y, min_year, max_year)
    Y = ((Y - min_year)/divs).astype(int)
    return (X,Y,ids)

#plot_years('quiet')
