import numpy as np
import cPickle as pickle

features = []
with open('feature_extractor/features_inc.pkl', 'rb') as pickle_file:
    for _ in range(5):
        entry = pickle.load(pickle_file, encoding='bytes', fix_imports=True)
        features.append(entry)
    print features
