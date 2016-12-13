from generate_paths import get_all_files
from get_features import get_features_simple, get_feature_list
import six.moves.cPickle as pickle
import numpy as np
import tempfile
import shutil
import os

from progressbar import ProgressBar
from joblib import Parallel, delayed
import pdb

files = get_all_files()


feat_list = filter(lambda x: x[:6] != 'artist', get_feature_list())
feat_list = ['key','mode','loudness','danceability','duration','energy','mode', 'year', 'song_id', 'end_of_fade_in', 'start_of_fade_out']
#feat_list.extend(['bas_start', 'beats_start', 'sections_start', 'segments_loudness_max', 'segments_timbre', 'segments_pitches', 'tatums_start'])
feat_list = ['segments_pitches', 'year', 'song_id']
feat_list.sort()

f_name = 'features_inc'
f_name = feat_list[0]

folder = tempfile.mkdtemp()
lengths_name = os.path.join(folder, 'lengths')
lengths = np.memmap(lengths_name, dtype='uint32',
                         shape=len(files), mode='w+')
feats_name = os.path.join(folder, 'feats')
feats = np.memmap(feats_name, dtype='uint8',
                         shape=(len(files),1000000), mode='w+')
feat_dump = np.memmap(feats_name, dtype='int32',
                         shape=(len(files),500,12), mode='w+')

import timeit
import gc 

delay_h5 = 0.0

def fun(f, i):
    feat = get_features_simple(f, feat_list) 
    return feat
    #p = pickle.dumps(feat)
    #lengths[i] = len(p)
    #feats[i][:len(p)] = map(ord, p)

def retrieve(i, final):
    final[i] = pickle.loads(''.join(map(chr, feats[i][:lengths[i]])))

def extract(file_rng):
    final = Parallel(n_jobs=-1)(delayed(fun)(f, i)
        for i,f in enumerate(file_rng))
            
    #pdb.set_trace()
    #print('Pickling Results ...'

    #final = [None for _ in range(batch)]
    #print("Retrieving ... "
    #Parallel(n_jobs=-1, max_nbytes=1e0)(delayed(retrieve)(i, final)
    #        for i in ProgressBar()(range(len(file_rng))))
    #for i in ProgressBar()(range(len(file_rng))):
    #    retrieve(i)
    
    features = [{feat_list[j]: f[j] for j in range(len(f))}
                                     for f in final]
    for i,f in enumerate(file_rng):
        pickle.dump((f, features[i]), pickle_file, pickle.HIGHEST_PROTOCOL)

    del final
    gc.collect()



total = 1000000
batch = 1000
iters = int(total/batch)
pickle_file = open(f_name+'.pkl', 'wb')

for i in range(0, iters):
    start = timeit.timeit()
    print('Starting Batch', i)
    extract(files[i*batch:(i+1)*batch])
    print('Finished Batch', i, 'in', timeit.timeit()-start, 'seconds')

pickle_file.close()
