# -*- coding: utf-8 -*-

import numpy as np
import activation as a


def cost(Y, A):
    m = Y.shape[1]
    return -(1/m) *  np.sum((Y * np.log(A) + (1-Y) * np.log(1-A)), dtype=np.float64)

def forward_propagate(parameters,X, L,dropout=False):
    
    """ computes the forward propagation of the nerual network """
    
    caches = {}
    
    caches["Z1"] = parameters["W1"].dot(X) + parameters["b1"]
    caches["a1"] = a.relu(caches["Z1"])

    if dropout == True:    
        caches["D1"] = np.random.rand(caches["a1"].shape[0],caches["a1"].shape[1]) < 0.8
        caches["a1"] *= caches["D1"]
        caches["a1"] /= 0.5

 
    caches["Z2"] = parameters["W2"].dot(caches["a1"]) + parameters["b2"]
    caches["a2"] = a.relu(caches["Z2"])
    

    if dropout == True:    
        caches["D2"] = np.random.rand(caches["a2"].shape[0],caches["a2"].shape[1]) < 0.8
        caches["a2"] *= caches["D2"]
        caches["a2"] /= 0.5    

    # on the last layer we would like to compute the sigmoid for each examples
    caches["Z3"] = parameters["W3"].dot(caches["a2"]) + parameters["b3"]
    caches["a3"] = a.sigmoid(caches["Z3"])
    
    return caches["a3"], caches