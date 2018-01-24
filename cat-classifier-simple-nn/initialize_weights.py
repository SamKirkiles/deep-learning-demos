# -*- coding: utf-8 -*-

import numpy as np

def init_weights(layer_dims):
    
    """ Initializes a dictionary of random weights depending on the size of
    the networks hidden layer and number of nodes in each layer """
 
    
    parameters = {}

    for i in range(1,len(layer_dims)):
        np.random.seed(1)
        parameters["W" + str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1]) * np.sqrt(2/layer_dims[i-1])
        parameters["b" + str(i)] = np.zeros((layer_dims[i],1))
    
    return parameters