# -*- coding: utf-8 -*-

import activation as a
import numpy as np
import forwardpropagate as fp
import copy

def back_propagation(X, y, al, L,  parameters, caches):
    
    
    m = X.shape[1]
    
    grads = {}
        
    """ Another way of writing this line is:
       grads["W" + str(L-1)] = (1/m) * (caches['a3']-y).dot(caches['a2'].T)
    """
    
    dzl = (-np.divide(y, al) + np.divide(1 - y, 1 - al))
    dzl *=  a.sigmoid_back(caches["a3"])
    grads["W" + str(L-1)] = (1/m) *  (dzl.dot(caches["a2"].T))
    
    grads["b" + str(L-1)] = (1/m) * np.sum(dzl, axis=1, keepdims=True)
    
    
    # (498, 4)
    
    dzl2 = (dzl.T.dot(parameters["W3"]) * a.relu_back(caches["Z2"]).T)
    
    grads["W" + str(L-2)] = (1/m) * caches['a1'].dot(dzl2)
    grads["b" + str(L-2)] = (1/m) * np.sum(dzl2, axis=0, keepdims=True).T
    
    
    dzl3 = (dzl2.dot(parameters["W2"]) * a.relu_back(caches["Z1"]).T)
    
    grads["W" + str(L-3)] = ((1/m) * X.dot(dzl3)).T
    grads["b" + str(L-3)] = (1/m) * np.sum(dzl3, axis=0, keepdims=True).T

    

    return grads



def check_gradients(X,Y,parameters,L):
    
    gradapprox = {}
    
    for i in range(1,L):
        
        params = ["W","b"]
        
        for p in params:
            epsilon = 0.000001
            
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