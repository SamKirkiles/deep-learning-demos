# -*- coding: utf-8 -*-

import numpy as np
import activation as a


def cost(Y, A):
    m = Y.shape[1]
    return -(1/m) *  np.sum((Y * np.log(A) + (1-Y) * np.log(1-A)), dtype=np.float64)

def forward_propagate(parameters,X, L,dropout=False):
    
    """ computes the forward propagation of the nerual network """
    
    caches = {}
    
    A_prev = X
    Z_temp = 0;
    
    for i in range (1,L-1):
        

        
        Z_temp = parameters["W" + str(i)].dot(A_prev) + parameters["b" + str(i)]
        A_prev = a.relu(Z_temp)
        
        if (dropout == True):
            drop = np.random.rand(A_prev.shape[0],A_prev.shape[1]) < 0.8
            A_prev *= drop
            A_prev /= 0.8

        
        caches["Z" + str(i)] = Z_temp
        caches["a" + str(i)] = A_prev
        
    # on the last layer we would like to compute the sigmoid for each examples
    Z = parameters["W" + str(L-1)].dot(A_prev) + parameters["b" + str(L-1)]
    AL = a.sigmoid(Z)
    
    caches["Z" + str(L-1)] = Z
    caches["a" + str(L-1)] = AL

    
    return AL, caches
