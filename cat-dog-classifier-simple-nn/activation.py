# -*- coding: utf-8 -*-

import numpy as np

def sigmoid(Z):
    #print(-Z)
    return 1 / (1 + np.exp(-Z))

def sigmoid_back(Z):
    return Z * (1-Z)

def relu(Z):
    return Z * (Z > 0)

def relu_back(Z):
    return 1 * (Z>0)