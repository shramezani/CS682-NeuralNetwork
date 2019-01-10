from builtins import range
from builtins import object
import numpy as np

from cs682.layers import *
from cs682.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        W1 = np.random.randn(input_dim, hidden_dim) * weight_scale+ 0.0
        b1 = np.zeros(hidden_dim)
        W2 = np.random.randn(hidden_dim, num_classes) * weight_scale+ 0.0
        b2 = np.zeros(num_classes)
        
        self.params['W1']=W1
        self.params['b1']=b1
        self.params['W2']=W2
        self.params['b2']=b2
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
    
    
    def soft(self, x):
        
        shifted_x = x- np.max(x, axis=1, keepdims=True)
        exp_sum = np.sum(np.exp(x), axis=1, keepdims=True)
        log_probs = shifted_x - np.log(exp_sum)
        probs = np.exp(log_probs)
        scores = probs
        return scores
        
        
    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        
        W1=self.params['W1']
        b1=self.params['b1']
        W2=self.params['W2']
        b2=self.params['b2']
        
        
        #forward pass
        layer_1,cache_1 = affine_relu_forward(X, W1, b1)
        layer_2, cache_2 = affine_forward(layer_1, W2, b2)
        scores = layer_2
       
        
        
         
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        
        #softmax
        #shifted_layer_2 = layer_2- np.max(layer_2, axis=1, keepdims=True)
        #exp_sum = np.sum(np.exp(layer_2), axis=1, keepdims=True)
        #log_probs = shifted_layer_2 - np.log(exp_sum)
        #probs = np.exp(log_probs)
        #scores = probs
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        
        loss, dsoft_dlayer2 = softmax_loss(layer_2, y)
        loss += 0.5*self.reg*(np.sum(W1*W1) + np.sum(W2*W2))
        
        dlayer2_dlayer1,dw2,db2= affine_backward(dsoft_dlayer2,cache_2)
        dx,dw1,db1 = affine_relu_backward(dlayer2_dlayer1,cache_1)
        
        grads['W1']=dw1+self.reg*W1
        grads['W2']=dw2+self.reg*W2
        grads['b1']=db1
        grads['b2']=db2
    
    
    
    
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        self.params['W1']=np.random.randn(input_dim, hidden_dims[0]) * weight_scale+ 0.0
        self.params['b1']=np.zeros(hidden_dims[0])
        
        
        for i in range(2,self.num_layers):
            self.params['W'+str(i)]=np.random.randn(hidden_dims[i-2],hidden_dims[i-1]) * weight_scale+ 0.0
            self.params['b'+str(i)]=np.zeros(hidden_dims[i-1])
        self.params['W'+str(self.num_layers)]=np.random.randn(hidden_dims[-1], num_classes) * weight_scale+ 0.0
        self.params['b'+str(self.num_layers)]=np.zeros(num_classes)
        
        if(self.normalization!=None):
            for i in range(1,self.num_layers):
                self.params['gamma'+str(i)]=np.ones(hidden_dims[i-1])
                self.params['beta'+str(i)]=np.zeros(hidden_dims[i-1])
            
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        #forward pas
        cache ={}
        sigma_W_square = 0
        if self.normalization!=None:
            out_layer,cache[1] = affine_layer_or_batch_relu_forward(self.normalization,X, self.params['W1'], self.params['b1'],self.params['gamma1'], self.params['beta1'], self.bn_params[0],self.use_dropout,self.dropout_param )
        else:
            out_layer,cache[1] = affine_relu_forward_dout(X, self.params['W1'], self.params['b1'],self.use_dropout,self.dropout_param )
        sigma_W_square +=np.sum(self.params['W1']*self.params['W1'])
        #dout, c = dropout_forward(out_layer, dropout_param)
        #cache[1].append(c)
        
        
        
        if self.normalization!=None:
            for i in range(2,self.num_layers):
                out_layer, cache[i] = affine_layer_or_batch_relu_forward(self.normalization,out_layer, self.params['W'+str(i)], self.params['b'+str(i)],self.params['gamma'+str(i)], self.params['beta'+str(i)], self.bn_params[i-1],self.use_dropout,self.dropout_param )
                sigma_W_square +=np.sum(self.params['W'+str(i)]*self.params['W'+str(i)])
        else:
            
            for i in range(2,self.num_layers):
                out_layer, cache[i] = affine_relu_forward_dout(out_layer, self.params['W'+str(i)], self.params['b'+str(i)],self.use_dropout,self.dropout_param )
                sigma_W_square +=np.sum(self.params['W'+str(i)]*self.params['W'+str(i)])

    
        out_layer, cache[self.num_layers] = affine_forward(out_layer,
                                                           self.params['W'+str(self.num_layers)],
                                                           self.params['b'+str(self.num_layers)])
        sigma_W_square +=np.sum(self.params['W'+str(self.num_layers)]*self.params['W'+str(self.num_layers)])
        scores = out_layer
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dsoft = softmax_loss(out_layer, y)
        loss += 0.5*self.reg*(sigma_W_square)
        
        dlayer,grads['W'+str(self.num_layers)],grads['b'+str(self.num_layers)]= affine_backward(dsoft,cache[self.num_layers])
        grads['W'+str(self.num_layers)]+=self.reg*self.params['W'+str(self.num_layers)]
        
        
        if self.normalization!=None:
            for i in range(self.num_layers-1, 0, -1):
                dlayer,grads['W'+str(i)],grads['b'+str(i)],grads['gamma'+str(i)],grads['beta'+str(i)]= affine_layer_or_batch_relu_backward(self.normalization,dlayer,cache[i],self.use_dropout)
                grads['W'+str(i)]+=self.reg*self.params['W'+str(i)] 
        
        else:
            
            for i in range(self.num_layers-1, 0, -1):
                dlayer,grads['W'+str(i)],grads['b'+str(i)] = affine_relu_backward_dout(dlayer,cache[i],self.use_dropout)
                grads['W'+str(i)]+=self.reg*self.params['W'+str(i)]
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
    
    
    
def affine_layer_or_batch_relu_forward(norm,x, w, b,gamma, beta, bn_param, droup_out, do_param):
        """
        Convenience layer that perorms an affine transform followed by batch_ReLU
        Inputs:
        - x: Input to the affine layer
        - w, b: Weights for the affine layer
       
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        fc_cache=None
        batch_cache = None
        relu_cache =None
        do_cache=None
        out, fc_cache = affine_forward(x, w, b)
        if norm =='batchnorm':
            out, batch_cache=batchnorm_forward(out, gamma, beta, bn_param)
        elif norm =='layernorm':
            out, batch_cache=layernorm_forward(out, gamma, beta, bn_param)
                
        out, relu_cache = relu_forward(out)
        if droup_out:
            out, do_cache = dropout_forward(out, do_param)
        cache = (fc_cache, batch_cache, relu_cache,do_cache)
        return out, cache
    
def affine_layer_or_batch_relu_backward(norm,dout, cache,drop_out):
        """
        Backward pass for the affine--batch_relu convenience layer
        """
        fc_cache,batch_cache, relu_cache,do_cache = cache
        if drop_out:
            dout = dropout_backward(dout,do_cache)
        dout = relu_backward(dout, relu_cache)
        if norm =='batchnorm':
            dout, dgamma, dbeta = batchnorm_backward_alt(dout, batch_cache)
        elif norm =='layernorm':
            dout, dgamma, dbeta = layernorm_backward(dout, batch_cache)
        
        dx, dw, db = affine_backward(dout, fc_cache)
        return dx, dw, db,dgamma, dbeta
def affine_relu_forward_dout(x, w, b, droup_out, do_param):
    do_cache=None
    a, fc_cache = affine_forward(x, w, b)
    out, relu_cache = relu_forward(a)
    if droup_out:
            out, do_cache = dropout_forward(out, do_param)
    cache = (fc_cache, relu_cache,do_cache)
    return out, cache


def affine_relu_backward_dout(dout, cache,drop_out):

    fc_cache, relu_cache,do_cache = cache
    if drop_out:
            dout = dropout_backward(dout,do_cache)
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    
    return dx, dw, db

