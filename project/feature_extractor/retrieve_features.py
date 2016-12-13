import six.moves.cPickle as pickle
import numpy as np

def get_features(feat_name):
    features = {}
    with open(feat_name+'.pkl', 'rb') as pickle_file:
        try:
            for _ in range(5000):
                entry = pickle.load(pickle_file, encoding='bytes', fix_imports=True)
                if 'year' in entry[1] and len(entry[1][feat_name]) > 0:
                    features[entry[0]] = {}
                    if feat_name[:4] in ['bars', 'beat', 'sect']:
                        if feat_name[:8] == 'sections':
                            features[entry[0]][feat_name] = len(entry[1][feat_name])
                        else:
                            features[entry[0]][feat_name] = entry[1][feat_name][0]
                        entry[1][feat_name] = np.diff(entry[1][feat_name])
                    mean_vals = np.mean(entry[1][feat_name], axis=0)
                    var_vals = np.var(entry[1][feat_name], axis=0)
                    if feat_name[-7:] in ['_timbre', 'pitches']:
                         for i in range(12):
                             features[entry[0]][feat_name+'-'+str(i)+'_mean'] = mean_vals[i]
                             features[entry[0]][feat_name+'-'+str(i)+'_var'] = var_vals[i]
                    else:
                         features[entry[0]][feat_name+'_mean'] = mean_vals
                         features[entry[0]][feat_name+'_var'] = var_vals 
        except EOFError:
            pass
    return features
