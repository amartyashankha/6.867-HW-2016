from load_HW2_data import *
from train_nn import *
from load_MNIST_data import *

def test_HW2_data(i):
    data = load_HW2_data(i)
    network = train_multiclass(data, 2, 400, 2, 2)

    # plot test data
    plot_heat(data[4], data[5], predictor(network))

def test_MNIST():
    data = load_MNIST_data()
    dim_x = data[0].shape[1]
    network = train_multiclass(data, 2, 300, 10, 784, max_iters = 50, eta = 5e-4)

#test_HW2_data(3)
test_MNIST()
