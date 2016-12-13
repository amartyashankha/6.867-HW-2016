import numpy as np
from scipy.sparse import dok_matrix

#import matplotlib.pyplot as plt
import sys
import os
from joblib import Parallel, delayed
sys.path.append(sys.path[0]+'/feature_extractor')
from get_features import get_features
import pickle


def get_vals(song, feats):
    fpath  = '/mnt/snap/data/'
    fpath += song[2]+'/'
    fpath += song[3]+'/'
    fpath += song[4]+'/'
    fpath += song +'.h5'

    if os.path.isfile(fpath):
        vals = get_features_simple(fpath, feats)
        return song+ "," + vals.join(',')

def save_data(feature_list):
    data = open('mxm_dataset_train.txt').readlines()
    songs = [line.split(',')[0] for line in data if line[0] == 'T']
    feats = ['key','loudness','tempo']
    lines = Parallel(n_jobs = -1, verbose = 50)(delayed(get_year)(song,feats) for song in songs)
    pickle.
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

if __name__ == '__main__':
    save_year_data()
