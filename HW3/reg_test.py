from conv_reg import ArtistConvNet

import numpy as np
import matplotlib.pyplot as plt
import prettyplotlib as ppl
from sklearn.linear_model import LogisticRegression
import pickle 
import math

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['figure.autolayout'] = 'true'

data = pickle.load( open( "reg_data.pkl", "rb" ) )

dropout = 1.0 
penalty = 0.0
augment = False

test = 'dropout'

weight_vals = np.linspace(0.0, 0.2, 20)
weight_val = []
weight_train = []
dropout_vals = np.linspace(0.5, 1.0, 20)
dropout_val = []
dropout_train = []
for dropout in dropout_vals:
    if (dropout, penalty, augment) not in data:
        conv_net = ArtistConvNet(invariance=False,
                                dropout=dropout,
                                weight_penalty=penalty,
                                augment=augment)
        log = conv_net.train_model()
        data[(dropout, penalty, augment)] = log
        pickle.dump(data, open( "reg_data.pkl", "wb" ) )
    else:
        log = data[(dropout, penalty, augment)]
    if test == 'weight':
        weight_val.append(max([log[x]["val_acc"] for x in log]))
        weight_train.append(max([log[x]["train_acc"] for x in log]))
    else:
        dropout_val.append(max([log[x]["val_acc"] for x in log]))
        dropout_train.append(max([log[x]["train_acc"] for x in log]))

if test == 'weight':
    X = weight_vals
    plt.plot(X, weight_val, label='Validation Accuracy')
    plt.plot(X, weight_train, label='Training Accuracy')
if test == 'dropout':
    X = 1.0-np.array(dropout_vals)
    plt.plot(X, dropout_val, label='Validation Accuracy')
    plt.plot(X, dropout_train, label='Training Accuracy')

title = 'Weight Penalty' if test == 'weight' else 'Dropout Probability'

plt.xlabel(title)
plt.title(title + ' Versus Accuracy') 
plt.legend(loc='Upper right')

plt.show()
'''

if (dropout, penalty, augment) not in data:
    conv_net = ArtistConvNet(invariance=False,
                            dropout=dropout,
                            weight_penalty=penalty,
                            augment=augment)
    log = conv_net.train_model()
    data[(dropout, penalty, augment)] = log
    pickle.dump(data, open( "reg_data.pkl", "wb" ) )
else:
    log = data[(dropout, penalty, augment)]

X = sorted(log.keys())
Yval = [log[x]["val_acc"] for x in X]
Ytrain = [log[x]["train_acc"] for x in X]

plt.plot(X, Yval, label='Validation')
plt.plot(X, Ytrain, label='Training')

augment = True
if (dropout, penalty, augment) not in data:
    conv_net = ArtistConvNet(invariance=False,
                            dropout=dropout,
                            weight_penalty=penalty,
                            augment=augment)
    log = conv_net.train_model()
    data[(dropout, penalty, augment)] = log
    pickle.dump(data, open( "reg_data.pkl", "wb" ) )
else:
    log = data[(dropout, penalty, augment)]

X_aug = sorted(log.keys())
Yval_aug = [log[x]["val_acc"] for x in X_aug]
Ytrain_aug = [log[x]["train_acc"] for x in X_aug]

plt.plot(X_aug, Yval_aug, label='Augmented Validation')
plt.plot(X_aug, Ytrain_aug, label='Augmented Training')

plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
'''
