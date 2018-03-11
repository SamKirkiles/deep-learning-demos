# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cifar10
import copy

from layers import (relu, relu_back,conv_forward_naive, conv_back_naive, max_pooling, max_pooling_back,
fully_connected, softmax, softmax_back, softmax_cost)

from layers_fast import conv_fast, conv_fast_back

import time

# Load in MNIST data and convert to one hot encoding. Only load in data if needed.

try:
    train_x_raw
    train_y
    train_y_one_hot
    classes
except NameError:
    train_x_raw, train_y, train_y_one_hot = cifar10.load_training_data()
    classes = cifar10.load_class_names()


m = 20

train_x = train_x_raw[0:m]


print("Displaying random training example")
plt.imshow(train_x_raw[np.random.randint(m),:,:,:])
plt.show()


#Normalize training data
train_x  = (train_x -np.mean(train_x ))/(np.max(train_x )-np.min(train_x ))

                    


def check_gradients(weights,key,four_dimensional=False,inline_checker=False):
    
    epsilon = 0.0001
    
    parameters1 = copy.deepcopy(weights)
    parameters2 = copy.deepcopy(weights)
    
    if inline_checker:
        activation_caches_1,cost1 = forward_propagate(parameters1,1)
        activation_caches_2,cost2 = forward_propagate(parameters2,2)

    else:
        if four_dimensional == True:
            parameters1[key][0,0,0,0] += epsilon
            parameters2[key][0,0,0,0] -= epsilon
        else:
            parameters1[key][0,0] += epsilon
            parameters2[key][0,0] -= epsilon
    
        activation_caches_1,cost1 = forward_propagate(parameters1)
        activation_caches_2,cost2 = forward_propagate(parameters2)

    return (cost1 - cost2) / (2. *epsilon)


def forward_propagate(_weights,debug=True):
    
    activation_caches = {}
    
    activation_caches["conv1"] = conv_fast(train_x,_weights["W1"],2,1)
    activation_caches["A1"] = relu(activation_caches["conv1"])
    activation_caches["pool1"] = max_pooling(activation_caches["A1"],2)

    # Sanity check to make sure that our convolution vectorization is correct
    if debug:
        conv1_verify = conv_forward_naive(train_x,_weights["W1"],2,1)
        assert np.mean(np.isclose(activation_caches["conv1"],conv1_verify)) == 1.0
    
    
    activation_caches["conv2"] = conv_fast(activation_caches["pool1"],_weights["W2"],2,1)   
    activation_caches["A2"] = relu(activation_caches["conv2"])
    activation_caches["pool2"]=max_pooling(activation_caches["A2"],2)       
    activation_caches["Ar2"] = activation_caches["pool2"].reshape((m, activation_caches["pool2"].shape[1] * 
                     activation_caches["pool2"].shape[2] * activation_caches["pool2"].shape[3]))
    
    
    if debug:
        conv2_verify = conv_forward_naive(activation_caches["pool1"],_weights["W2"],2,1)
        assert np.mean(np.isclose(activation_caches["conv2"],conv2_verify)) == 1.0


    activation_caches["Z3"] = fully_connected(activation_caches["Ar2"], _weights["W3"])
    activation_caches["A3"] = relu(activation_caches["Z3"])
    
    
    activation_caches["Z4"] = fully_connected(activation_caches["A3"],_weights["W4"])
    activation_caches["A4"] = softmax(activation_caches["Z4"])
    
    cost = np.mean(softmax_cost(train_y_one_hot[0:m,...], activation_caches["A4"]))
    
    return activation_caches,cost

"""
Initialize Weights
"""

np.random.seed(3)
        
weights = {}
weights["W1"] = np.random.rand(5,5,3,4) * 0.01
weights["W2"] = np.random.rand(5,5,4,4) * 0.01
weights["W3"] = np.random.rand(256,10)
weights["W4"] = np.random.rand(10,10)


"""
Back Propagation
"""

def backward_propagate_check(df1,df2,dw3,dw4, weights):
    grad_W4 = check_gradients(weights,"W4")
    print("Gradient Approx W4:     " + str(grad_W4))
    print("Gradient Calculated W4: " + str(dw4[0,0]) + "\n")
    
    grad_W3 = check_gradients(weights,"W3")
    print("Gradient Approx W3:     " + str(grad_W3))
    print("Gradient Calculated W3: " + str(dw3[0,0]) + "\n")
    
    grad_W2 = check_gradients(weights,"W2",True,False)
    print("Gradient Approx W2:     " + str(grad_W2))
    print("Gradient Calculated W2: " + str(df2[0,0,0,0]) + "\n")

    grad_W1 = check_gradients(weights,"W1",True,False)
    print("Gradient Approx W1:     " + str(grad_W1))
    print("Gradient Calculated W1: " + str(df1[0,0,0,0]))
    
    
def backward_propagate(weights,caches,debug=True):
    
    if debug:
        print("Testing Backprop")
    
    softmax_grad = softmax_back(caches["A4"], train_y_one_hot[0:m,...], m)
    dw4 = caches["A3"].T.dot(softmax_grad)
        
    da3 = (weights["W4"].dot(softmax_grad.T)).T
    dz3 = relu_back(caches["Z3"]) * da3
    dw3 = caches["Ar2"].T.dot(dz3)
        
    dpool2 = weights["W3"].dot(dz3.T).T
    dpool2_reshape = dpool2.reshape(caches["pool2"].shape)
    
    da2 = max_pooling_back(caches["A2"], caches["pool2"],dpool2_reshape)
    dz2 = relu_back(caches["conv2"]) * da2
    
    df2,dpool1 = conv_fast_back(caches["pool1"],weights["W2"],dz2,2,1)        

    if debug:
        df2_naive,dpool1_naive= conv_back_naive(caches["pool1"],weights["W2"],2,1,dz2)   
        assert np.mean(np.isclose(df2,df2_naive)) == 1.0
        assert np.mean(np.isclose(dpool1,dpool1_naive)) == 1.0

    da1 = max_pooling_back(caches["A1"], caches["pool1"],dpool1)
    dz1 = relu_back(caches["conv1"]) * da1
    df1,dinput = conv_fast_back(train_x,weights["W1"],dz1,2,1) 


    
    if debug:
        df1_naive,dinput_naive= conv_back_naive(train_x,weights["W1"],2,1,dz1)   
        assert np.mean(np.isclose(df1,df1_naive)) == 1.0
        assert np.mean(np.isclose(dinput,dinput_naive)) == 1.0

    return df1,df2,dw3,dw4

"""
caches,cost = forward_propagate(weights,debug=True)

# vectorized
start = time.clock()
df1,df2,dw3,dw4= backward_propagate(weights,caches,debug=False)
print(df1)
end = time.clock()
print(end-start)

#backward_propagate_check(df1,df2,dw3,dw4,weights)
"""


iterations = 1000

for i in range(iterations):
    
    caches,cost = forward_propagate(weights,debug=False)
    df1,df2,dw3,dw4= backward_propagate(weights,caches,debug=False)
    
    
    weights["W1"] -= 0.001 * df1
    weights["W2"] -= 0.001 * df2
    weights["W3"] -= 0.01 * dw3
    weights["W4"] -= 0.01 * dw4

    print(cost)




