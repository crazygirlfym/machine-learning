## -*-- coding:utf-8 -*--

import numpy as np

def softmax_loss(W, X, y ,reg):
    """
            Inputs:
            - W: A numpy array of shape (D, C) containing weights.
            - X: A numpy array of shape (N, D) containing a minibatch of data.
            - y: A numpy array of shape (N,) containing training labels; y[i] = c means
                 that X[i] has label c, where 0 <= c < C.
            - reg: (float) regularization strength

            Returns a tuple of:
            - loss as single float
            - gradient with respect to weights W; an array of same shape as W
    """


    loss = 0.0
    dW = np.zeros(W.shape)
    num_train, dim = X.shape

    f = X.dot(W)  # N by C
    # Considering the Numeric Stability  稳定性  解决方法就是 减去一个f_max

    ## refer to http://cs231n.github.io/linear-classify/#svm
    f_max = np.reshape(np.max(f, axis=1), (num_train, 1))  # N by 1
    prob = np.exp(f - f_max) / np.sum(np.exp(f - f_max), axis=1, keepdims=True)
    y_trueClass = np.zeros_like(prob)

    ## 转换为one-hot
    y_trueClass[range(num_train), y] = 1.0  # N by C

    ## 交叉熵
    loss += -np.sum(y_trueClass * np.log(prob)) / num_train + 0.5 * reg * np.sum(W * W)


    ## 梯度计算


    dW += -np.dot(X.T, y_trueClass - prob) / num_train + reg * W
    return loss, dW






