from generate_paths import get_all_files
from get_features import get_features, get_features_simple, get_feature_list
import cPickle as pickle
import numpy as np
import tempfile
import shutil
import os

from progressbar import ProgressBar
from joblib import Parallel, delayed
import pdb

files = get_all_files()


feat_list = filter(lambda x: x[:6] != 'artist', get_feature_list())
feat_list = ['song_idx']


folder = tempfile.mkdtemp()
lengths_name = os.path.join(folder, 'lengths')
lengths = np.memmap(lengths_name, dtype='uint32',
                         shape=len(files), mode='w+')
feats_name = os.path.join(folder, 'feats')
feats = np.memmap(feats_name, dtype='uint8',
                         shape=(len(files),200000), mode='w+')

import timeit

delay_h5 = 0.0

def fun(f, i, delay=[]):
    feat = get_features_simple(f, feat_list) 
    p = pickle.dumps(feat)
    lengths[i] = len(p)
    feats[i][:len(p)] = map(ord, p)

def extract(file_rng):
    features = pickle.load(open('features'+'.pkl', 'rb'))
    for f in file_rng:
        if f not in features:
            features[f] = {}
    Parallel(n_jobs=-1)(delayed(fun)(f, i)
        for i,f in ProgressBar()(enumerate(file_rng)))
            
    final = [None for _ in files]
    final = [pickle.loads(''.join(map(chr, feats[i][:lengths[i]])))
                        for i in range(len(file_rng))]
    final = [{feat_list[j]: f[j]
                            for j in range(len(f))}
                                     for f in final]
    for i,f in enumerate(file_rng):
        features[f].update(final[i])


    f_name = 'features'
    pickle.dump(features, open(f_name+'.pkl', 'wb'))


total = 100000
batch = 2000
iters = total/batch
for i in range(iters):
    extract(files[i*batch:(i+1)*batch])
