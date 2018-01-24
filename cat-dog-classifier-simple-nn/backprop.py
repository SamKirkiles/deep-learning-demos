# -*- coding: utf-8 -*-

import activation as a
import numpy as np
import forwardpropagate as fp
import copy

def back_propagation(X, y, al, L,  parameters, caches,dropout=False):
    
    
    m = X.shape[1]
    
    grads = {}
    
    da3 = (-np.divide(y, caches["a3"]) + np.divide(1 - y, 1 - caches["a3"]))
    dz3 = da3 * a.sigmoid_back(caches["a3"])
    
    grads["W3"] = (1/m) *  (dz3.dot(caches["a2"].T))
    grads["b3"] = (1/m) * np.sum(dz3, axis=1, keepdims=True)
    
    #potential spot for dropout
    da2 = parameters["W3"].T.dot(dz3)
    
    if dropout == True:
        da2 *= caches["D2"]
        da2 /= 0.5
    
    dz2 = da2 * a.relu_back(caches["a2"])
    
    grads["W2"] = (1/m) * dz2.dot(caches['a1'].T)
    grads["b2"] = (1/m) * np.sum(dz2, axis=1, keepdims=True)
    
    da1 = parameters["W2"].T.dot(dz2)
    
    if dropout == True:
        da1 *= caches["D1"]
        da1 /= 0.5
        
    dz1 = da1 * a.relu_back(caches["a1"])
    
    grads["W1"] = ((1/m) * dz1.dot(X.T))
    grads["b1"] = (1/m) * np.sum(dz1, axis=1, keepdims=True)
    
    return grads



def check_gradients(X,Y,parameters,L):
    
    gradapprox = {}
    
    for i in range(1,L):
        
        params = ["W","b"]
        
        for p in params:
            epsilon = 0.0001
            
            parameters1 = copy.deepcopy(parameters)
            parameters2 = copy.deepcopy(parameters)
            
            parameters1[p + str(i)][0,0] += epsilon
            parameters2[p + str(i)][0,0] -= epsilon
                
            fp1, fp1cache = fp.forward_propagate(parameters1, X, L)
            fp2, fp2cache = fp.forward_propagate(parameters2, X, L)
            
            cost1 = fp.cost(Y, fp1)
            cost2 = fp.cost(Y, fp2)
            
            gradapprox[p + str(i)] = (cost1 - cost2) / (2. *epsilon)
        
    return gradapprox