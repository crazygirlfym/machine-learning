#--*- coding:utf-8 -*--

import numpy as np

def extractQueryData(split):
    '''
    Extract the query features from a dataset line
    Format:
    <query-id><document-id><inc><prob>
    '''
    queryFeatures = split[1].split(':')[1]
    # queryFeatures.append(split[50])
    # queryFeatures.append(split[53])
    # queryFeatures.append(split[56])

    return queryFeatures


def extractFeatures(split):
    '''
    Extract the query to document features used
    as input to the neural network
    '''
    features = []
    for i in range(2, 48):
        features.append(float(split[i].split(':')[1]))
    return features

def readDataset(path):
    '''
    Dataset - LETOR 4.0
    Dataset format: svmlight / libsvm format
    <label> <feature-id>:<feature-value>... #docid = <feature-value> inc = <feature-value> prob = <feature-value>
    We have a total of 46 features
    '''

    X_train = [] #<feature-value>[46]
    y_train = [] #<label>
    Query = []   #<query-id><document-id><inc><prob>

    print('Reading training data from file...')

    with open(path, 'r') as file:
        for line in file:
            split = line.split()
            y_train.append(int(split[0]))
            X_train.append(extractFeatures(split))
            Query.append(extractQueryData(split))
    y_train /= np.max(y_train)
    print('Read %d lines from file...' %(len(X_train)))
    # X_train = np.asarray(X_train)
    # y_train = np.asarray(y_train)
    # Query = np.asarray(Query)
    return (X_train, y_train, Query)


def extractPairsOfRatedSites(y_train, Query):
    '''
    For each queryid, extract all pairs of documents
    with different relevance judgement and save them in
    a list with the most relevant in position 0
    '''
    pairs = []
    for i in range(0, len(Query)):
        for j in range(i+1, len(Query)):
            #Only look at queries with the same id
            if(Query[i][0] != Query[j][0]):
                break
            #Document pairs found with different rating
            if(Query[i][0] == Query[j][0] and y_train[i] != y_train[j]):
                #Sort by saving the largest index in position 0
                if(y_train[i] > y_train[j]):
                    pairs.append([i, j])
                else:
                    pairs.append([j, i])
    print('Found %d document pairs' %(len(pairs)))
    return pairs

if __name__ == '__main__':
    X_train, y_train, Query = readDataset('../model_ranknet/Data/train.txt')
    pairs = extractPairsOfRatedSites(y_train, Query)