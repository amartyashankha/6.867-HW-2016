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

fil1=5
str1=2
depth1=16

fil2=5
str2=2
depth2=16

fil_vals = range(3,8)
str_vals = range(1,4)

layer = 'first'
if len(sys.argv) > 2 and sys.argv[2] == '2':
    layer = 'second'
           
res = np.empty((5, 3))
acc = 'val'

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
        if (fil1,str1,depth1,fil2,str2,depth2) not in data:
            conv_net = ArtistConvNet(invariance=False,
                                     fil1=fil1,str1=str1,depth1=depth1,
                                     fil2=fil2,str2=str2,depth2=depth2)
                             
            log= conv_net.train_model()
            data[(fil1,str1,depth1,fil2,str2,depth2)] = log 
            pickle.dump(data, open( "param_data.pkl", "wb" ) )
        else:
            log= data[(fil1,str1,depth1,fil2,str2,depth2)]
        res[i-3][j-1] = max([log[x]["val_acc"] for x in log])

fig, ax = ppl.subplots(1)

print res

ppl.pcolormesh(fig, ax, res,
                xticklabels=fil_vals,
                yticklabels=str_vals)

ax.set_ylabel('Size of ' + layer + ' pooling filter')
ax.set_xlabel('Stride of ' + layer + ' pooling filter')
title = ('Validation' if acc == 'val' else 'Training') + " Accuracy Heatmap" 

ax.set_title(title)

plt.show()
