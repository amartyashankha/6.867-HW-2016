from conv_param import ArtistConvNet

import numpy as np
import matplotlib.pyplot as plt
import prettyplotlib as ppl
from sklearn.linear_model import LogisticRegression
import pickle 
import math
import sys 

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['figure.autolayout'] = 'true'

data = pickle.load( open( "param_data.pkl", "rb" ) )

dropout = 1.0 
penalty = 0.0
augment = False

test = 'dropout'

fil1=7
str1=3
depth1=16

fil2=5
str2=3
depth2=16

fil_vals = range(3,8)
str_vals = range(1,5)
depth_vals = [8, 16, 32, 64]

layer = 'first'
if len(sys.argv) > 1:
    if sys.argv[1] == 'second':
        layer = 'second'
    if sys.argv[1] == 'depth':
        layer = 'depth'

if layer != 'depth':
    res = np.empty((len(fil_vals), len(str_vals)))

    for i in fil_vals:
        if layer == 'first':
            fil1 = i
        else:
            fil2 = i
        for j in str_vals:
            if layer == 'first':
                str1 = j
            else:
                str2 = j
            print i, j
            if (fil1,str1,depth1,fil2,str2,depth2) not in data:
                conv_net = ArtistConvNet(invariance=False,
                                         fil1=fil1,str1=str1,depth1=depth1,
                                         fil2=fil2,str2=str2,depth2=depth2)
                                 
                log= conv_net.train_model()
                data = pickle.load( open( "param_data.pkl", "rb" ) )
                data[(fil1,str1,depth1,fil2,str2,depth2)] = log 
                pickle.dump(data, open( "param_data.pkl", "wb" ) )
            else:
                log= data[(fil1,str1,depth1,fil2,str2,depth2)]
            res[i-3][j-1] = max([log[x]["val_acc"] for x in log])

else:
    res = np.empty((len(depth_vals), len(depth_vals)))

    for i, depth1 in enumerate(depth_vals):
        for j, depth2 in enumerate(depth_vals):
            print i, j
            if (fil1,str1,depth1,fil2,str2,depth2) not in data:
                conv_net = ArtistConvNet(invariance=False,
                                         fil1=fil1,str1=str1,depth1=depth1,
                                         fil2=fil2,str2=str2,depth2=depth2)
                                 
                log= conv_net.train_model()
                data = pickle.load( open( "param_data.pkl", "rb" ) )
                data[(fil1,str1,depth1,fil2,str2,depth2)] = log 
                pickle.dump(data, open( "param_data.pkl", "wb" ) )
            else:
                log= data[(fil1,str1,depth1,fil2,str2,depth2)]
            res[i][j] = max([log[x]["val_acc"] for x in log])
    fil_vals = depth_vals
    str_vals = depth_vals


fig, ax = ppl.subplots(1)

print res

ppl.pcolormesh(fig, ax, res,
                yticklabels=fil_vals,
                xticklabels=str_vals)

# ax.set_ylabel('Size of ' + layer + ' convolutional filter')
# ax.set_xlabel('Stride of ' + layer + ' convolutional filter')
ax.set_ylabel('Depth of First Convolutional Layer')
ax.set_xlabel('Depth of Second Convolutional Layer')
title = 'Validation' + " Accuracy Heatmap" 

ax.set_title(title)

plt.show()
