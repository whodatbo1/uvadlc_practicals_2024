################################################################################
# MIT License
#
# Copyright (c) 2024 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course (UvA) | Fall 2024
# Date Created: 2024-10-28
################################################################################
"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np


class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """

    def __init__(self, in_features, out_features, input_layer=False):
        """
        Initializes the parameters of the module.

        Args:
          in_features: size of each input sample
          out_features: size of each output sample
          input_layer: boolean, True if this is the first layer after the input, else False.

        TODO:
        Initialize weight parameters using Kaiming initialization.
        Initialize biases with zeros.
        Hint: the input_layer argument might be needed for the initialization

        Also, initialize gradients with zeros.
        """

        # Note: For the sake of this assignment, please store the parameters
        # and gradients in this format, otherwise some unit tests might fail.
        self.params = {'weight': None, 'bias': None} # Model parameters
        self.grads = {'weight': None, 'bias': None} # Gradients

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        self.in_features = in_features
        self.out_features = out_features

        self.__initialize_parameters(input_layer)
        self.__initialize_gradients()

        assert self.params['weight'].shape == (self.out_features, self.in_features)
        assert self.params['bias'].shape == (1, self.out_features)

        #######################
        # END OF YOUR CODE    #
        #######################

    # Initialize the weights and biases
    # Assume we are using the fan_in method for the initialization
    # Inspiration has been taken from the Pytorch implementation at https://github.com/pytorch/pytorch/blob/0adb5843766092fba584791af76383125fd0d01c/torch/nn/init.py#L389
    # And the paper: https://arxiv.org/pdf/1502.01852
    def __initialize_parameters(self, input_layer=False):
        if input_layer:
            weight_std = np.sqrt(1 / self.in_features)
        else:
            weight_std = np.sqrt(2 / self.in_features)

        weights = np.random.randn(self.out_features, self.in_features) * weight_std
        assert weights.shape == (self.out_features, self.in_features)

        biases = np.zeros((1, self.out_features))
        assert biases.shape == (1, self.out_features)

        self.params['weight'] = weights
        self.params['bias'] = biases

    # Initialize the gradients to 0
    def __initialize_gradients(self):
        self.grads['weight'] = np.zeros((self.out_features, self.in_features))
        self.grads['bias'] = np.zeros((1, self.out_features))

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        out = x @ self.params['weight'].T + self.params['bias']

        self.x_cache = x

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.

        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        self.grads['weight'] = dout.T @ self.x_cache
        self.grads['bias'] = np.sum(dout, axis=0)

        dx = dout @ self.params['weight']

        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        pass
        #######################
        # END OF YOUR CODE    #
        #######################

    def update_parameters(self, lr):
        self.params['weight'] -= lr * self.grads['weight']
        self.params['bias'] -= lr * self.grads['bias']

class ELUModule(object):
    """
    ELU activation module.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # Apply the ELU activation function element-wise
        out = np.where(x >= 0, x, self.alpha * (np.exp(x) - 1))

        # Cache the input for backward pass
        self.x_cache = x

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # If the input was >= 0, the derivative is 1, otherwise it's alpha * exp(x)
        dx = np.where(self.x_cache >= 0, dout, dout * self.alpha * np.exp(self.x_cache))

        #######################
        # END OF YOUR CODE    #
        #######################
        return dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################
        self.x_cache = None
        #######################
        # END OF YOUR CODE    #
        #######################

    def update_parameters(self, lr):
        pass


class SoftMaxModule(object):
    """
    Softmax activation module.
    """

    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        # Use the Max Trick to stabilize the computation
        x_max = np.max(x, axis=1, keepdims=True)
        x_exp = np.exp(x - x_max)
        sum_c = np.sum(x_exp, axis=1, keepdims=True)
        out = x_exp / sum_c

        assert out.shape == x.shape

        # Cache the derivative of the softmax function for backward pass
        self.x_cache = x
        self.out_cache = out

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################
        
        assert dout.shape == self.out_cache.shape

        # Naive implementation
        # We compute the Jacobian for each input in the batch
        batch_size = dout.shape[0]
        data_dim = self.x_cache.shape[1]
        dx = np.zeros((batch_size, data_dim, data_dim)) # NxDxD
        for i in range(batch_size):
            # Non-diagonal entries
            outer = - np.outer(self.out_cache[i], self.out_cache[i])
            dx[i] = outer
            # The derivative for diagonal entries has an extra term SM(x)_ii
            diag = np.diag(self.out_cache[i])
            dx[i] += diag

        dL_dx = np.einsum('nij,nj->ni', dx, dout)

        #######################
        # END OF YOUR CODE    #
        #######################

        return dL_dx

    def clear_cache(self):
        """
        Remove any saved tensors for the backward pass.
        Used to clean-up model from any remaining input data when we want to save it.

        TODO:
        Set any caches you have to None.
        """
        #######################
        # PUT YOUR CODE HERE  #
        #######################

        self.dsm_dx_cache = None

        #######################
        # END OF YOUR CODE    #
        #######################

    def update_parameters(self, lr):
        pass

class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """

    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss

        TODO:
        Implement forward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        labels = self.__get_labels(x, y)

        out = -np.sum(labels * np.log(x)) / x.shape[0]

        self.labels = labels
        self.out = out

        #######################
        # END OF YOUR CODE    #
        #######################

        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.

        TODO:
        Implement backward pass of the module.
        """

        #######################
        # PUT YOUR CODE HERE  #
        #######################

        labels = self.__get_labels(x, y)
        
        s = labels.shape[0]
        dx = - (labels / x) / s

        #######################
        # END OF YOUR CODE    #
        #######################

        return dx
    
    # Create a one-hot matrix of label
    def __get_labels(self, x, y):
        labels = np.zeros_like(x)
        labels[np.arange(y.shape[0]), y] = 1
        return labels
    
    def update_parameters(self, lr):
        pass