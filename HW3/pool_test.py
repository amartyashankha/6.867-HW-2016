from conv_pooling import ArtistConvNet

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

acc = 'val'
if len(sys.argv) > 1 and  sys.argv[1] == 'train':
    acc = 'train'

win1 = 3
win2 = 2
str1 = 1
str2 = 2

data = pickle.load( open( "pooling_data.pkl", "rb" ) )

end = 8
if len(sys.argv) > 2:
    end = int(sys.argv[2])

layer = 'first'
if len(sys.argv) > 3 and sys.argv[3] == '2':
    layer = 'second'
           
win1range = range(2,end)
res = np.empty((len(win1range), (win1range[-1])))

for i in win1range:
    if layer == 'first':
        win1 = i
    else:
        win2 = i
    for j in range(1,win1range[-1]+1):
        if layer == 'first':
            str1 = j
        else:
            str2 = j
        if (win1, str1, win2, str2) not in data:
            conv_net = ArtistConvNet(invariance=False,
                         win1=win1, str1=str1,
                         win2=win2, str2=str2)
            result = conv_net.train_model()
            data[(win1, str1, win2, str2)] = result
            pickle.dump(data, open( "pooling_data.pkl", "wb" ) )
        else:
            result = data[(win1, str1, win2, str2)]
        if layer == 'first':
            res[win1-2][str1-1] = result[acc+"_acc"]
        else:
            res[win2-2][str2-1] = result[acc+"_acc"]

fig, ax = ppl.subplots(1)

print res

ppl.pcolormesh(fig, ax, res,
                xticklabels=range(1,win1range[-1]+1),
                yticklabels=win1range)

ax.set_ylabel('Size of ' + layer + ' pooling filter')
ax.set_xlabel('Stride of ' + layer + ' pooling filter')
title = ('Validation' if acc == 'val' else 'Training') + " Accuracy Heatmap" 

ax.set_title(title)

ppl.plt.show()
