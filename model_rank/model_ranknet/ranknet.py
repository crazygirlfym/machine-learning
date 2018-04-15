# -*-- coding:utf-8 -*--



import math
import random
import copy
import time
import matplotlib.pyplot as plt
from model_ranknet.tools import *

'''
Ranknet

Learning to rank using gradient descent. A simplified implementation of the
algorithm described in http://research.microsoft.com/en-us/um/people/cburges/papers/icml_ranking.pdf
'''
class ranknet:
    def __init__(self, network, iterations=25):
        self.network= network
        self.iterations = iterations

    def countMisorderedPairs(self, X_train, pairs):
        '''
        Let the network classify all pairs of patterns. The highest output determines the winner.
        Count how many times the network makes the wrong judgement
        errorRate = numWrong/(Total)
        '''

        misorderedPairs = 0

        for pair in pairs:
            self.network.propagate(X_train[pair[0]])
            self.network.propagate(X_train[pair[1]])
            if self.network.prevOutputActivation <= self.network.activation_output:
                misorderedPairs += 1

        return misorderedPairs / float(len(pairs))

    def train(self, X_train, pairs):
        '''
        Train the network on all patterns for a number of iterations.
        Training:
            Propagate A (Highest ranked document)
            Propagate B (Lower ranked document)
            Backpropagate
        Track the number of misordered pairs for each iteration.
        '''

        errorRate = []
        start = time.time()

        print('Training the neural network...')
        for epoch in range(self.iterations):
            print('*** Epoch %d ***' % (epoch + 1))
            for pair in pairs:
                self.network.propagate(X_train[pair[0]])
                self.network.propagate(X_train[pair[1]])
                self.network.backpropagate()
            errorRate.append(self.countMisorderedPairs(X_train, pairs))
            # Debug:
            print('Error rate: %.2f' % errorRate[epoch])
            # self.weights()
        m, s = divmod(time.time() - start, 60)
        print('Training took %dm %.1fs' % (m, s))
        plotErrorRate(errorRate)






