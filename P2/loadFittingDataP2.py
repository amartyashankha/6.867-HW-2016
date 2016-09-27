import matplotlib.pyplot as plt
import pylab as pl

def getData(ifPlotData=False):
    # load the fitting data and (optionally) plot out for examination
    # return the X and Y as a tuple

    data = pl.loadtxt('P2/curvefittingp2.txt')

    X = data[0,:].T
    Y = data[1,:].T

    if ifPlotData:
        plt.plot(X,Y,'o')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    return (X,Y)
