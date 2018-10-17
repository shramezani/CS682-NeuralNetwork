import numpy as np
from random import shuffle



def softmax(x):
    x -= np.max(x)
    result = np.exp(x)/np.sum(np.exp(x))
    return result

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
  num_of_inputs = X.shape[0]
  num_of_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(num_of_inputs):
    #compute loss_i
    scores_i = X[i].dot(W)  #f_i
    normalized_scores_i = softmax(scores_i) #softmax(f_i)
    score_of_correct_class = normalized_scores_i[y[i]]  #
    l_i = -np.log(score_of_correct_class)
    loss +=l_i  
    normalized_scores_i[y[i]] -= 1
    dW += X[i].reshape((len(X[i]),1)).dot(normalized_scores_i.reshape((1,len(normalized_scores_i))))
    
    
    
  loss /=num_of_inputs
  dW /= num_of_inputs
  loss += reg * np.sum(W * W)
  dW += reg*2*W
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
  num_of_inputs = X.shape[0]
  num_of_classes = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  exp_shifted_scores = np.exp(scores-np.max(scores, axis=1, keepdims = True))
  normalized_scores = exp_shifted_scores/np.sum(exp_shifted_scores,axis=1, keepdims =True)
  log_normalized_scores = -np.log(normalized_scores)
  loss = np.sum(log_normalized_scores[np.arange(num_of_inputs),y])
  loss /=num_of_inputs
  loss += reg * np.sum(W * W)

  normalized_scores[np.arange(num_of_inputs),y] -=1
  dW = X.T.dot(normalized_scores)
  dW /= num_of_inputs
  dW += reg*2*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

