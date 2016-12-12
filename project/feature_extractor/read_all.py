from generate_paths import get_all_files
from get_features import get_features_simple, get_feature_list
import cPickle as pickle
import numpy as np
import tempfile
import shutil
import os

from progressbar import ProgressBar
from joblib import Parallel, delayed

files = get_all_files()[:10]

feat_list = filter(lambda x: x[:6] != 'artist', get_feature_list())
feat_list = map(lambda x: 'get_'+x, feat_list)

folder = tempfile.mkdtemp()
lengths_name = os.path.join(folder, 'lengths')
lengths = np.memmap(lengths_name, dtype='uint32',
                         shape=len(files), mode='w+')
features_name = os.path.join(folder, 'features')
features = np.memmap(features_name, dtype='uint8',
                         shape=(len(files),1000000), mode='w+')


print features.shape
def fun(f, i):
    if i%1000 == 0:
        print i
    feat = get_features_simple(f, feat_list)
    p = pickle.dumps(feat)
    lengths[i] = len(p)
    for j,char in enumerate(p):
        features[i][j] = ord(char)


Parallel(n_jobs=8)(delayed(fun)(f, i)
    for i,f in ProgressBar()(enumerate(files)))

pickle.dump(lengths, open('feature'+'.pkl', 'wb'))

final = [None for _ in files]
final = np.array([pickle.loads(''.join(map(chr, features[i][:lengths[i]])))
                    for i in range(len(features))])
final = [{feat_list[j][4:]: f[j] for j in range(len(feat_list))} for f in final]

pickle.dump(final, open('feature'+'.pkl', 'wb'))
