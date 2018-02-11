# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cifar10
import copy


# Load in MNIST data and convert to one hot encoding. Only load in data if needed.

try:
    train_x_raw
    train_y
    train_y_one_hot
    classes
except NameError:
    train_x_raw, train_y, train_y_one_hot = cifar10.load_training_data()
    classes = cifar10.load_class_names()

#m=train_y.shape[0]
m = 50

train_x = train_x_raw[0:m]


print("Displaying random training example")
plt.imshow(train_x_raw[np.random.randint(m),:,:,:])
plt.show()

train_x  = (train_x -np.mean(train_x ))/(np.max(train_x )-np.min(train_x ))



#Activation Functions:
    
def relu(x):
    return np.maximum(0, x)

def relu_back(Z):
    return np.int64(Z > 0)

def conv_forward(a_prev, _filter, parameters):
    
    stride = parameters['stride']
    pad = parameters['pad']
    
    (m, n_H_prev, n_W_prev, channels) = a_prev.shape
    (f,f, n_C_prev, n_C) = _filter.shape
   
    n_H = int(((n_H_prev - f + 2 * pad)/stride)+1)
    n_W = int(((n_W_prev - f + 2 * pad)/stride)+1)
    
    a_prev_pad = np.pad(a_prev, ((0,0),(pad,pad),(pad,pad),(0,0)), 'constant', constant_values=0)
   
    Z = np.zeros((m,n_H,n_W,n_C))
    
    for i in range(m):  
        #print('\r{0}\r'.format(i), end='', flush=True)
        a_prev_pad_i = a_prev_pad[i] 
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    
                    vert_start = h*stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    
                    a_slice = a_prev_pad_i[vert_start:vert_end,horiz_start:horiz_end,:]
                    
                    Z[i,h,w,c] = np.sum(np.multiply(a_slice, _filter[...,c]))
                    
    return Z


def max_pooling(prev_layer, filter_size=2):
    
    (m, n_H_prev, n_W_prev, channels) = prev_layer.shape
    
    n_H = int((n_H_prev - filter_size)/filter_size + 1)
    n_W = int((n_W_prev - filter_size)/filter_size + 1)
    
    pooling = np.zeros((m,n_H,n_W,channels))
    
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(channels):
                    vert_start = h*filter_size
                    vert_end = vert_start + filter_size
                    horiz_start = w*filter_size
                    horiz_end = horiz_start + filter_size
                
                    prev_slice = prev_layer[i,vert_start:vert_end,horiz_start:horiz_end,c]
                    
                    pooling[i,w,h,c] = np.max(prev_slice)
                    
    return pooling


def fully_connected(prev_layer, w):
    return prev_layer.dot(w)

def softmax(z):
    return np.exp(z)/np.sum(np.exp(z),axis=1)[:,None]

def softmax_back(softmax, Y,m):    
    return (softmax-Y)/m

def softmax_cost(y, y_hat):
    return -np.sum(y * np.log(y_hat),axis=1)

def check_gradients(Y,weights,key):
    
    epsilon = 0.0001
    
    parameters1 = copy.deepcopy(weights)
    parameters2 = copy.deepcopy(weights)
    

    
    parameters1[key][0,0] += epsilon
    parameters2[key][0,0] -= epsilon
    
    weight_dict_1,activation_caches_1,cost1 = forward_propagate(parameters1)
    weight_dict_2,activation_caches_1,cost2 = forward_propagate(parameters2)
            

    return (cost1 - cost2) / (2. *epsilon)



def forward_propagate(weight_dict):

    parameters={'stride':1,'pad':2 }
    Z1 = conv_forward(train_x,weight_dict["W1"],parameters)
    activation_caches["A1"] = relu(Z1)
    pool1=max_pooling(activation_caches["A1"],2)
    
    Z2 = conv_forward(pool1,weight_dict["W2"],parameters)
    activation_caches["A2"] = relu(Z2)
    pool2=max_pooling(activation_caches["A2"],2)
    
    pool2_flat = pool2.reshape((m, pool2.shape[1] * pool2.shape[2] * pool2.shape[3]))
    
    Z3 = fully_connected(pool2_flat, weight_dict["W3"])
    activation_caches["A3"] = relu(Z3)
    
    activation_caches["Z4"] = fully_connected(activation_caches["A3"],weight_dict["W4"])
    
    activation_caches["A4"] = softmax(activation_caches["Z4"])
    
    
    cost = np.mean(softmax_cost(train_y_one_hot[0:m,...], activation_caches["A4"]))
    
    return weight_dict,activation_caches,cost

np.random.seed(3)
        
weights = {}
weights["W1"] = np.random.rand(5,5,3,3) * 0.01
weights["W2"] = np.random.rand(5,5,3,3) * 0.01
weights["W3"] = np.random.rand(192,10)
weights["W4"] = np.random.rand(10,10)

weight_dict,activation_caches,cost = forward_propagate(weights)

"""Backpropagaton"""

softmax_grad = softmax_back(activation_caches["A4"], train_y_one_hot[0:m,...], m)
dz4 = activation_caches["A3"].T.dot(softmax_grad)

grad_W4 = check_gradients(train_y_one_hot[0:m,...], weight_dict,"W4")
print("First Gradient Approx W4: " + str(grad_W4))
print("Calculated First Gradient W4: " + str(dz4[0,0]) + "\n")


grad_W3 = check_gradients(train_y_one_hot[0:m,...], weights,"W3")
print("Gradient W3: " + str(grad_W3))

grad_W2 = check_gradients(train_y_one_hot[0:m,...], weights,"W2")
print("Gradient W2: " + str(grad_W2))

grad_W1 = check_gradients(train_y_one_hot[0:m,...], weights,"W1")
print("Gradient W1: " + str(grad_W1))

