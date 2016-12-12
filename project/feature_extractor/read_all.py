from generate_paths import get_all_files
from get_features import get_features_simple, get_feature_list
import cPickle as pickle
import numpy as np
import tempfile
import shutil
import os

from progressbar import ProgressBar
from joblib import Parallel, delayed

files = get_all_files()[:1000]


feat_list = filter(lambda x: x[:6] != 'artist', get_feature_list())
feat_list = ['year', 'song_id']


folder = tempfile.mkdtemp()
lengths_name = os.path.join(folder, 'lengths')
lengths = np.memmap(lengths_name, dtype='uint32',
                         shape=len(files), mode='w+')
feats_name = os.path.join(folder, 'feats')
feats = np.memmap(feats_name, dtype='uint8',
                         shape=(len(files),500000), mode='w+')

print feats.shape
def fun(f, i, features):
    if f not in features:
        features[f] = {}
    actual_feat_list = set(feat_list) - set(features[f].keys())
    feat = get_features_simple(f, list(actual_feat_list))
    p = pickle.dumps(feat)
    lengths[i] = len(p)
    for j,char in enumerate(p):
        feats[i][j] = ord(char)

def extract(file_rng):
    features = pickle.load(open('features'+'.pkl', 'rb'))
    Parallel(n_jobs=-1)(delayed(fun)(f, i, features)
        for i,f in ProgressBar()(enumerate(file_rng)))

    final = [None for _ in files]
    final = [pickle.loads(''.join(map(chr, feats[i][:lengths[i]])))
                        for i in range(len(file_rng))]
    def rem_features(f):
        return sorted(list(set(feat_list)-set(features[f].keys())))
    final = [{rem_features(f)[j]: f[j]
                            for j in range(len(f))}
                                     for f in final]
    final = {file_rng[i]: final[i] for i in range(len(final))}
    for i in enumerate(file_rng):
        features[f] = features[f] | final[i]


    f_name = 'features'
    pickle.dump(final, open(f_name+'.pkl', 'wb'))


extract(files)
