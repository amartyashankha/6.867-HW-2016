import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def scatter_dense(x, y):
    xmin = min(x)
    xmax = max(x)
    ymin = min(y)
    ymax = max(y)

    plt.hexbin(x, y, bins=10, cmap=plt.cm.YlOrRd_r)
    plt.axis([xmin, xmax, ymin, ymax])
    plt.title("With a log color scale")
    cb = plt.colorbar()
    cb.set_label('log10(N)')
