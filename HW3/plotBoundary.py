import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

def plot_heat(X, y, predictor, res = 200, fname = None):
    y = np.asarray(y.T, dtype=int)[0]
    eps = .1
    xmin = np.min(X[:,0]) - eps; xmax = np.max(X[:,0]) + eps
    ymin = np.min(X[:,1]) - eps; ymax = np.max(X[:,1]) + eps
    ax = tidyPlot(xmin, xmax, ymin, ymax, xlabel = 'x', ylabel = 'y')
    ima = np.array([[predictor(np.matrix([xi, yi]).T) for xi in np.linspace(xmin, xmax, res)] \
        for yi in np.linspace(ymin, ymax, res)])
    im = ax.imshow(np.flipud(ima), interpolation = 'none',
        extent = [xmin, xmax, ymin, ymax],
        cmap = 'viridis')  
    plt.colorbar(im)
    colors = [['r', 'g', 'b'][l] for l in y]
    ax.scatter(X[:,0], X[:,1], c = colors, marker = 'o', s=80,
        edgecolors = 'none')
    if fname:
        plt.savefig(fname)
    else:
        plt.show()

def tidyPlot(xmin, xmax, ymin, ymax, center = False, title = None,
    xlabel = None, ylabel = None):
    plt.figure(facecolor="white")
    ax = plt.subplot()
    if center:
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_smart_bounds(True)
        ax.spines['bottom'].set_smart_bounds(True)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    else:
        ax.spines["top"].set_visible(False)    
        ax.spines["right"].set_visible(False)    
        ax.get_xaxis().tick_bottom()  
        ax.get_yaxis().tick_left()
        eps = .05
        plt.xlim(xmin-eps, xmax+eps)
        plt.ylim(ymin-eps, ymax+eps)
        if title: ax.set_title(title)
        if xlabel: ax.set_xlabel(xlabel)
        if ylabel: ax.set_ylabel(ylabel)
        return ax
