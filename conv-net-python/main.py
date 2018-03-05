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


m = 50

train_x = train_x_raw[0:m]


print("Displaying random training example")
plt.imshow(train_x_raw[np.random.randint(m),:,:,:])
plt.show()

train_x  = (train_x -np.mean(train_x ))/(np.max(train_x )-np.min(train_x ))

    
def relu(x):
    return np.maximum(0, x)

def relu_back(Z):
    return np.int64(Z > 0)

def conv_forward_naive(_input,_filter,pad,stride):

    
    (m, n_h, n_w, n_C_prev) = _input.shape
    (f,f, n_C_prev, n_C) = _filter.shape
   

    n_H = int(1 + (n_h + 2 * pad - f) / stride)
    n_W = int(1 + (n_w + 2 * pad - f) / stride)
        
    a_prev_pad = np.pad(_input, ((0,0),(pad,pad),(pad,pad),(0,0)), 'constant', constant_values=0)
   
    Z = np.zeros((m, n_H,n_W,n_C))
            
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):

                            
                        vert_start = h*stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f
                        
    
                        a_slice = a_prev_pad[i,vert_start:vert_end,horiz_start:horiz_end,:]
                        
                        Z[i,h,w,c] = np.sum(np.multiply(a_slice, _filter[:,:,:,c]))
    
    return Z



def conv_back_naive(_input,_filter,pad,stride,dout):
 
    (m, n_h, n_w, n_C_prev) = _input.shape
    (f,f, n_C_prev, n_C) = _filter.shape
   
    n_H = int(1 + (n_h + 2 * pad - f) / stride)
    n_W = int(1 + (n_w + 2 * pad - f) / stride)
    
    a_prev_pad = np.pad(_input, ((0,0),(pad,pad),(pad,pad),(0,0)), 'constant', constant_values=0)
   
    dw = np.zeros(_filter.shape,dtype=np.float32)
    dx = np.zeros(_input.shape,dtype=np.float32)

    for h in range(f):
        for w in range(f):
            for p in range(n_C_prev):
                for c in range(n_C):


                    a_slice = a_prev_pad[:,h:h + n_H * stride:stride,w:w + n_W * stride:stride,p]

                    dw[h,w,p,c] = np.sum(a_slice * dout[:,:,:,c])
    
    
    dx_pad = np.pad(dx, ((0,0),(pad,pad),(pad,pad),(0,0)), 'constant', constant_values=0)
    
    for i in range(m):        
        for h_output in range(n_H):
            for w_output in range(n_W):
                for g in range(n_C):
                            
                    vert_start = h_output*stride
                    vert_end = vert_start + f
                    horiz_start = w_output * stride
                    horiz_end = horiz_start + f
                    
                    dx_pad[i,vert_start:vert_end,horiz_start:horiz_end,:] += _filter[:,:,:,g] * dout[i,h_output,w_output,g]
                                    
                
    dx = dx_pad[:,pad:pad+n_h,pad:pad+n_w,:]
                
    return dw,dx


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
                    
                    pooling[i,h,w,c] = np.max(prev_slice)
                    
    return pooling


def max_pooling_back(prev, pool, dout, filter_size=2):
        
    (m, n_H, n_W, channels) = pool.shape
    (m_prev, n_prev_H, n_prev_W, channels_prev) = prev.shape
    
    empty = np.zeros((m, n_prev_H, n_prev_W, channels))
    
    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(channels):
                    
                    vert_start = h*filter_size
                    vert_end = vert_start + filter_size
                    horiz_start = w*filter_size
                    horiz_end = horiz_start + filter_size
                                        
                    mask = prev[i,vert_start:vert_end,horiz_start:horiz_end,c] == pool[i,h,w,c]
     
                    empty[i,vert_start:vert_end,horiz_start:horiz_end,c] = mask * dout[i,h,w,c]
                    
    return empty
                    
def fully_connected(prev_layer, w):
    return prev_layer.dot(w)

def softmax(z):
    return np.exp(z)/np.sum(np.exp(z),axis=1)[:,None]

def softmax_back(softmax, Y,m):    
    return (softmax-Y)/m

def softmax_cost(y, y_hat):
    return -np.sum(y * np.log(y_hat),axis=1)


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


def forward_propagate(weights):
    
    activation_caches = {}

    activation_caches["conv1"] = conv_forward_naive(train_x,weights["W1"],2,1)
    activation_caches["A1"] = relu(activation_caches["conv1"])
    activation_caches["pool1"] = max_pooling(activation_caches["A1"],2)

    
    activation_caches["conv2"] = conv_forward_naive(activation_caches["pool1"],weights["W2"],2,1)   
    activation_caches["A2"] = relu(activation_caches["conv2"])
    activation_caches["pool2"]=max_pooling(activation_caches["A2"],2)       
    activation_caches["Ar2"] = activation_caches["pool2"].reshape((m, activation_caches["pool2"].shape[1] * 
                     activation_caches["pool2"].shape[2] * activation_caches["pool2"].shape[3]))

    activation_caches["Z3"] = fully_connected(activation_caches["Ar2"], weights["W3"])
    activation_caches["A3"] = relu(activation_caches["Z3"])
    
    
    activation_caches["Z4"] = fully_connected(activation_caches["A3"],weights["W4"])
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
    
    
def backward_propagate(weights,caches):
    
    print("\nTesting Backprop \n")
    
    softmax_grad = softmax_back(caches["A4"], train_y_one_hot[0:m,...], m)
    dw4 = caches["A3"].T.dot(softmax_grad)
    
    
    da3 = (weights["W4"].dot(softmax_grad.T)).T
    dz3 = relu_back(caches["Z3"]) * da3
    dw3 = caches["Ar2"].T.dot(dz3)
    
    dpool2 = weights["W3"].dot(dz3.T).T
    dpool2_reshape = dpool2.reshape(caches["pool2"].shape)
    
    da2 = max_pooling_back(caches["A2"], caches["pool2"],dpool2_reshape)
    dz2 = relu_back(caches["conv2"]) * da2
    df2,dpool1 = conv_back_naive(caches["pool1"],weights["W2"],2,1,dz2) 
    
    da1 = max_pooling_back(caches["A1"], caches["pool1"],dpool1)
    dz1 = relu_back(caches["conv1"]) * da1
    df1,dinput = conv_back_naive(train_x,weights["W1"],2,1,dz1) 
    
    return df1,df2,dw3,dw4


caches,cost = forward_propagate(weights)
df1,df2,dw3,dw4 = backward_propagate(weights,caches)
backward_propagate_check(df1,df2,dw3,dw4,weights)

