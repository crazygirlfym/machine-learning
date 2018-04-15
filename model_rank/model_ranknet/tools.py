# -*-- coding:utf-8 -*--
import math
import random
import time
import matplotlib.pyplot as plt
#The transfer function of neurons, g(x)
def logFunc(x):
    return (1.0/(1.0+math.exp(-x)))

#The derivative of the transfer function, g'(x)
def logFuncDerivative(x):
    return math.exp(-x)/(pow(math.exp(-x)+1,2))

def random_float(low,high):
    return random.random()*(high-low) + low

#Initializes a matrix of all zeros
def makeMatrix(I, J):
    m = []
    for i in range(I):
        m.append([0]*J)
    return m


def plotErrorRate(errorRate):
    '''
    Plot error rate using matplotlib
    '''

    plt.plot(errorRate)
    plt.ylabel('Error Rate')
    plt.show()