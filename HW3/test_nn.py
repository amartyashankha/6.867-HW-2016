from load_HW2_data import *
from train_nn import *
from load_MNIST_data import *
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['figure.autolayout'] = 'true'


def test_HW2_data(i, L, m):
    data = load_HW2_data(i)
    network = train_multiclass(data, L, m, 2, 2)

    # plot test data
    plot_heat(data[4], data[5], predictor(network), fname = 'figs/'+str(i)+'_'+str(L)+'_'+str(m)+'.png')

def test_MNIST():
    data = load_MNIST_data()
    dim_x = data[0].shape[1]
    network = train_multiclass(data, 2, 100, 10, 784, max_iters = 10, eta = 1e-2)

def test_all_MNIST():
    Ls = [1,2]
    ms = np.hstack([range(10,50,10), (range(50, 550, 50))])
    print ms
    data = load_MNIST_data()
    dim_x = data[0].shape[1]
    for L in Ls:
        res = []
        for m in ms:
            tries = []
            for k in range(3):
                err = train_multiclass(data, L, m, 10, dim_x, max_iters = 10, eta = 1e-2)
                tries.append(err)
            print L, m, np.min(tries)
            res.append(np.min(tries))
        plt.plot(ms, res, label="L="+str(L))
    plt.xlabel('Number of nodes per layer')
    plt.ylabel('Testing error')
    plt.title('testing error vs number of nodes')
    plt.legend()
    plt.savefig('comparison.png')

def test_all_data():
    ds = [1,2,3]

    Ls = [1,2,1,2]
    ms = [5,5,500,500]
    etas = [1e-2,1e-2,1e-3,1e-3]
    for i in ds:
        print "==================="
        for j in range(len(Ls)):
            L = Ls[j]
            m = ms[j]
            eta = etas[j]
            print "L: ", L, " M: ", m," i: ",i
            test_HW2_data(i, L, m)

#test_all_data()
#test_HW2_data(4, 1, 5)
test_all_MNIST()
