from __future__ import print_function
from model_ranknet.data_utils import readDataset, extractPairsOfRatedSites
from model_ranknet.nn_net import *
from model_ranknet.ranknet import *


if __name__ == '__main__':
    X_train, y_train, Query = readDataset('Data/train.txt')
    pairs = extractPairsOfRatedSites(y_train, Query)
    Tree_Net = NN(46, 20, 0.001)
    rank_net = ranknet(Tree_Net, 20)
    rank_net.train(X_train, pairs)

    X_train, y_train, Query = readDataset('Data/test.txt')
    # Extract document pairs
    pairs = extractPairsOfRatedSites(y_train, Query)
    print('Testset errorRate: ' + str(rank_net.countMisorderedPairs(X_train, pairs)))
