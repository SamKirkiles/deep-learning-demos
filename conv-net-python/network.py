# -*- coding: utf-8 -*-

import numpy as np
import copy


from layers_fast import conv_fast, conv_fast_back
from layers import (relu, relu_back,conv_forward_naive, conv_back_naive, max_pooling, max_pooling_back,
fully_connected, softmax, softmax_back, softmax_cost)

from karpathy_tester import karpathy_conv_forward_naive, karpathy_max_pool_forward_naive

def forward_propagate(x,y,_weights,debug=True):
    
    activation_caches = {}
    
    m = x.shape[0]
    

    activation_caches["conv1"] = conv_fast(x,_weights["W1"],_weights["B1"],2,1)
    activation_caches["A1"] = relu(activation_caches["conv1"])
    activation_caches["pool1"] = max_pooling(activation_caches["A1"],2,2)
    
    # Sanity check to make sure that our convolution vectorization is correct

    if debug:
        # Conv

        kconv,kcache = karpathy_conv_forward_naive(x,_weights["W1"],_weights["B1"],{'stride':1,'pad':2})
        assert np.mean(np.isclose(activation_caches["conv1"],kconv)) == 1.0

        conv1_verify = conv_forward_naive(x,_weights["W1"],_weights["B1"],2,1)
        assert np.mean(np.isclose(activation_caches["conv1"],conv1_verify)) == 1.0

        kpool1,kcache1 = karpathy_max_pool_forward_naive(activation_caches["A1"],{'pool_height':2,'pool_width':2,'stride':2})
        assert np.mean(np.isclose(activation_caches["pool1"],kpool1)) == 1.0

    activation_caches["conv2"] = conv_fast(activation_caches["pool1"],_weights["W2"],_weights["B2"],2,1)   
    activation_caches["A2"] = relu(activation_caches["conv2"])
    activation_caches["pool2"]=max_pooling(activation_caches["A2"],2,2)       
    activation_caches["Ar2"] = activation_caches["pool2"].reshape((m, activation_caches["pool2"].shape[1] * 
                     activation_caches["pool2"].shape[2] * activation_caches["pool2"].shape[3]))
    
    
    if debug:
        conv2_verify = conv_forward_naive(activation_caches["pool1"],_weights["W2"],_weights["B2"],2,1)
        assert np.mean(np.isclose(activation_caches["conv2"],conv2_verify)) == 1.0

    activation_caches["Z3"] = fully_connected(activation_caches["Ar2"], _weights["W3"],_weights["B3"])
    activation_caches["A3"] = relu(activation_caches["Z3"])
    
    activation_caches["Z4"] = fully_connected(activation_caches["A3"],_weights["W4"],_weights["B4"])
    activation_caches["A4"] = softmax(activation_caches["Z4"])
    
    cost = np.mean(softmax_cost(y, activation_caches["A4"],m))
    
    return activation_caches,cost

def backward_propagate(x,y,weights,caches,debug=True):
    
    if debug:
        print("Testing Backprop")
        
    m = x.shape[0]

    
    softmax_grad = softmax_back(caches["A4"], y, m)
    db4 = np.sum(softmax_grad)
    dw4 = caches["A3"].T.dot(softmax_grad)
        

    da3 = (weights["W4"].dot(softmax_grad.T)).T
    dz3 = relu_back(caches["Z3"],da3)
    dw3 = caches["Ar2"].T.dot(dz3)
    db3 = np.sum(dz3)
        
    dpool2 = weights["W3"].dot(dz3.T).T
    dpool2_reshape = dpool2.reshape(caches["pool2"].shape)
    
    da2 = max_pooling_back(caches["A2"], caches["pool2"],dpool2_reshape)
    dz2 = relu_back(caches["conv2"],da2)
    
    
    df2,dpool1,db2 = conv_fast_back(caches["pool1"],weights["W2"],dz2,2,1)      
    
    if debug:
        df2_naive,dpool1_naive,db2_naive= conv_back_naive(caches["pool1"],weights["W2"],2,1,dz2)   
        assert np.mean(np.isclose(df2,df2_naive)) == 1.0
        assert np.mean(np.isclose(dpool1,dpool1_naive)) == 1.0

    da1 = max_pooling_back(caches["A1"], caches["pool1"],dpool1)
    dz1 = relu_back(caches["conv1"],da1)
    df1,dinput,db1 = conv_fast_back(x,weights["W1"],dz1,2,1) 
    
    if debug:
        df1_naive,dinput_naive,df1_naive= conv_back_naive(x,weights["W1"],2,1,dz1)   
        assert np.mean(np.isclose(df1,df1_naive)) == 1.0
        assert np.mean(np.isclose(dinput,dinput_naive)) == 1.0

    return df1,df2,dw3,dw4,db1,db2,db3,db4


def backward_propagate_check(df1,df2,dw3,dw4,db1,db2,db3,db4, x,y,weights):
    grad_W4 = check_gradients(x,y,weights,"W4")
    print("Gradient Approx W4:     " + str(grad_W4))
    print("Gradient Calculated W4: " + str(dw4[0,0]) + "\n")
    
    grad_B4 = check_gradients(x,y,weights,"B4",False,False,True)
    print("Gradient Approx B4:     " + str(grad_B4))
    print("Gradient Calculated B4: " + str(db4) + "\n")

    grad_W3 = check_gradients(x,y,weights,"W3")
    print("Gradient Approx W3:     " + str(grad_W3))
    print("Gradient Calculated W3: " + str(dw3[0,0]) + "\n")
    
    grad_B3 = check_gradients(x,y,weights,"B3",False,False,True)
    print("Gradient Approx B3:     " + str(grad_B3))
    print("Gradient Calculated B3: " + str(db3) + "\n")

    grad_W2 = check_gradients(x,y,weights,"W2",True,False)
    print("Gradient Approx W2:     " + str(grad_W2))
    print("Gradient Calculated W2: " + str(df2[0,0,0,0]) + "\n")
    
    grad_B2 = check_gradients(x,y,weights,"B2",False,False,True)
    print("Gradient Approx B2:     " + str(grad_B2))
    print("Gradient Calculated B2: " + str(db2[0]) + "\n")

    grad_W1 = check_gradients(x,y,weights,"W1",True,False)
    print("Gradient Approx W1:     " + str(grad_W1))
    print("Gradient Calculated W1: " + str(df1[0,0,0,0]) + "\n")

    grad_B1 = check_gradients(x,y,weights,"B1",False,False,True)
    print("Gradient Approx B1:     " + str(grad_B1))
    print("Gradient Calculated B1: " + str(db1[0]) + "\n")

def check_gradients(x,y,weights,key,four_dimensional=False,inline_checker=False,bias=False):
    
    epsilon = 0.0001
    
    
    parameters1 = copy.deepcopy(weights)
    parameters2 = copy.deepcopy(weights)
    
    if inline_checker:
        activation_caches_1,cost1 = forward_propagate(x,y,parameters1,1,debug=False)
        activation_caches_2,cost2 = forward_propagate(x,y,parameters2,2,debug=False)
    elif bias:
        parameters1[key][0] += epsilon
        parameters2[key][0] -= epsilon
        
        activation_caches_1,cost1 = forward_propagate(x,y,parameters1,debug=False)
        activation_caches_2,cost2 = forward_propagate(x,y,parameters2,debug=False)

    else:
        if four_dimensional == True:
            parameters1[key][0,0,0,0] += epsilon
            parameters2[key][0,0,0,0] -= epsilon
        else:
            parameters1[key][0,0] += epsilon
            parameters2[key][0,0] -= epsilon
    
        activation_caches_1,cost1 = forward_propagate(x,y,parameters1,debug=False)
        activation_caches_2,cost2 = forward_propagate(x,y,parameters2,debug=False)

    return (cost1 - cost2) / (2. *epsilon)

def get_weights():
    np.random.seed(455)
    weights = {}
    weights["W1"] = np.random.rand(5,5,3,16) * 1e-3
    print(np.std(weights["W1"]))

    weights["B1"] = np.zeros(16)
    
    weights["W2"] = np.random.rand(5,5,16,16) * 1e-3
    weights["B2"] = np.zeros(16)
    
    weights["W3"] = np.random.rand(1024,20) *1e-3
    weights["B3"] = np.zeros((1,1))
    
    weights["W4"] = np.random.rand(20,10) * 1e-3
    weights["B4"] = np.zeros((1,1)) 
    
    return weights

def get_weights_from_file(filename):
    
    weights = {}
    weights_load = np.load(filename).item()
    weights["W1"] = weights_load.get('W1')
    weights["W2"] = weights_load.get('W2')
    weights["W3"] = weights_load.get('W3')
    weights["W4"] = weights_load.get('W4')
    
    weights["B1"] = weights_load.get('B1')
    weights["B2"] = weights_load.get('B2')
    weights["B3"] = weights_load.get('B3')
    weights["B4"] = weights_load.get('B4')


    return weights
