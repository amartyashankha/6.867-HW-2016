import numpy as np
from scipy.sparse import dok_matrix

import matplotlib.pyplot as plt
import sys
import os
#from joblib impoort Parallel, delayed
sys.path.append(sys.path[0]+'/feature_extractor')
from get_features import get_features

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['figure.autolayout'] = 'true'

def looad_txt_data():
	years = {}

	ydata = open('data/tracks_per_year.txt').readlines()
	for line in ydata:
		proc = line.split('<SEP>')
		years[proc[1]] = int(proc[0])

def get_year(song):
    fpath  = '/mnt/snap/data/'
    fpath += song[2]+'/'
    fpath += song[3]+'/'
    fpath += song[4]+'/'
    fpath += song +'.h5'

    if os.path.isfile(fpath):
        year = get_features(fpath, 'year')['year']
        if year != 0:
            return song+ "," + str(year)+'\n'

def save_year_data():
    data = open('mxm_dataset_train.txt').readlines()
    songs = [line.split(',')[0] for line in data if line[0] == 'T']
    lines = Parallel(n_jobs = -1, verbose = 50)(delayed(get_year)(song) for song in songs)
    f  = open('year_save.txt', 'w')
    for line in lines:
        if line:
            f.write(line)
    f.close()

year_dic = None
def load_year_data():
    global year_dic
    lines = open('year_save.txt').readlines()
    year_dic = {}
    for line in lines:
        (i, y) = line.split(',')
        year_dic[i] = int(y)

class LyricVector():
    def __init__(self, line, norm = False):
        global year_dic
        vals = line.split(',')
        self.track_id = vals[0]
        self.mxm_id   = vals[1]
        self.freq_dic = {}
        for item in vals[2:]:
            proc = item.split(':')
            self.freq_dic[int(proc[0])-1] = int(proc[1])
        if norm:
            N = sum(self.freq_dic.values())
            for i in self.freq_dic:
                self.freq_dic[i] /= float(N)

        if year_dic == None:
            load_year_data()
        self.year = year_dic.get(self.track_id, None)

def load(norm):
    data = open('mxm_dataset_train.txt').readlines()
    words = [line[1:].split(',') for line in data if line[0] == '%'][0]
    songs = [LyricVector(line, norm) for line in data if line[0] == 'T']
    songs = [song for song in songs if song.year != None]
    print "Finished loading data"
    return (words, songs)

def plot_years(word_str, div = 5):
    (words, songs) = load(norm = False)
    word = words.index(word_str)
    yrs  = np.arange(1960, 2015, div)
    cnt  = np.zeros(len(yrs))
    num  = np.zeros(len(yrs))
    tot  = np.zeros(len(yrs))
    for song in songs:
        ind = (max(song.year, 1960) - 1960)/div
        cnt[ind] += song.freq_dic.get(word, 0)
        num[ind] += (word in song.freq_dic)
        tot[ind] += 1.0

    print tot
    print num
    num/= tot
    num*=100
    #plt.plot(yrs, cnt)
    plt.xticks(yrs, yrs, rotation = 'vertical')
    plt.xlabel('Year')
    plt.ylabel('% songs that mentioned ' + word_str)
    plt.title(word_str +' in song lyrics')
    plt.plot(yrs, num)
    plt.show()

def load_class(norm, min_year = 1960, max_year = 2016, divs = 10, one_hot = True, dense = False):
    (words, songs) = load(norm and (not one_hot))
    print len(songs)
    X = dok_matrix((len(songs), 5001), dtype = np.float32)
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

    Y = np.array(Y)
    Y = np.clip(Y, min_year, max_year)
    Y = ((Y - min_year)/divs).astype(int)

    return (X,Y,ids, words)

#plot_years('yonder')

#if __name__ == '__main__':
    #save_year_data()
