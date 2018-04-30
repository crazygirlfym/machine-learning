# -*-- coding:utf-8 -*--

import numpy as np

def svm_loss(W, X, y, reg):
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

    scores = X.dot(W)  # N by C
    num_train = X.shape[0]
    scores_correct = scores[np.arange(num_train), y]  # 1 by N
    scores_correct = np.reshape(scores_correct, (num_train, 1))  # N by 1


    ## 回归问题， 采用hinge loss   L_i = \sum _ {j \neq y_i} max( 0, s_j - s_{y_i} + 1)
    margins = scores - scores_correct + 1.0  # N by C
    margins[np.arange(num_train), y] = 0.0
    margins[margins <= 0] = 0.0
    loss += np.sum(margins) / num_train
    loss += 0.5 * reg * np.sum(W * W)

    ## 对于求梯度


    ### 可以参考loss的表达是， 其实这里是在公式的基础上去 delta = 1
    # if margin > 0:
    #     loss += margin
    #     dW[:, y[i]] += -X[i, :]  # compute the correct_class gradients
    #     dW[:, j] += X[i, :]  # compute the wrong_class gradients




    # compute the gradient
    margins[margins > 0] = 1.0
    row_sum = np.sum(margins, axis=1)  # 1 by N
    margins[np.arange(num_train), y] = -row_sum
    dW += np.dot(X.T, margins) / num_train + reg * W  # D by C

    return loss, dW

