import numpy as np

def get_weights(y):
    counts = np.bincount(y)
    n_buckets = sum(counts > 0)
    n_samps = len(y)
    w = (1.0 * n_samps)/(n_buckets * counts[y])
    return w

