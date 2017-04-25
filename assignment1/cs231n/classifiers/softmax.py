import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  C = W.shape[1]
  _scores = np.zeros((N, C))

  for i in range(N):
    for j in range(C):
      _scores[i, j] = np.dot(X[i, :], W[:, j])

  norm_vec = _scores.max(axis = 1)
  sum_vec = np.zeros(N)
  L_i = np.zeros((N, C))
  for r in range(N):
    for col in range(C):
      _scores[r][col] -= norm_vec[r]
      _scores[r][col] = np.exp(_scores[r][col])
      sum_vec[r] += _scores[r][col]

    for col in range(C):
      L_i[r][col] = -np.log(_scores[r][col]/sum_vec[r])

  for i in range(N):
    loss += L_i[i, y[i]]

  loss /= N
  loss += 0.5 * reg * np.sum(W*W)

  d_exp = _scores / np.sum(_scores, axis=1, keepdims=True)
  for i in range(N):
    d_exp[i, y[i]] -= 1

  d_exp /= N
  dW = np.dot(X.T, d_exp) + reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  N = X.shape[0]
  C = W.shape[1]

  _scores = np.dot(X, W)
  _scores -= np.matrix(_scores.max(axis = 1)).T
  _scores = np.exp(_scores)

  L_i = -np.log(_scores / np.sum(_scores, axis=1, keepdims=True))
  loss = np.sum(L_i[range(N),y])/N + 0.5 * reg * np.sum(W*W)

  d_exp = _scores / np.sum(_scores, axis=1, keepdims=True)
  d_exp[range(N),y] -= 1
  d_exp /= N
  dW = np.dot(X.T, d_exp) + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
