#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 00:33:13 2018

@author: samkirkiles
"""

# import libraries
import matplotlib.pyplot as plt
import numpy as np
import copy


# Load the cats and dogs
import load_data
import initialize_weights as weights
import forwardpropagate as fp
import activation as a
import backprop as bp


train_x_orig, train_y, test_x_orig, test_y, classes = load_data.loadDatah5()

# Normalize Data-- Very important
train_x_orig = (train_x_orig-np.mean(train_x_orig))/(np.max(train_x_orig)-np.min(train_x_orig))
test_x_orig = (test_x_orig-np.mean(test_x_orig))/(np.max(test_x_orig)-np.min(test_x_orig))


# Create features and training exmples
m = train_x_orig.shape[1]
n = train_x_orig.shape[0]

# Number of hidden layers and Number of nodes on each layer
n_h = 2
n_n = 2

# Initalize weights

dims = [n,7,7,1]

parameters = weights.init_weights(dims)

iterations = 2500

# Run forward and backprop
fw, caches = fp.forward_propagate(parameters,train_x_orig,len(dims))
grads = bp.back_propagation(train_x_orig, train_y, fw, len(dims), parameters, caches)

first_grads = copy.deepcopy(grads)
# Run check gradients method
approx = bp.check_gradients(train_x_orig,train_y,parameters,len(dims))

print("----------------------------------------------------")

print("\n Calculated First Gradients...\n")

print("W1: " + str(float(grads["W1"][0,0])))
print("b1: " + str(float(grads["b1"][0,0]))  + "\n")

print("W2: " + str(float(grads["W2"][0,0])))
print("b2: " + str(float(grads["b2"][0,0]))  + "\n")

print("W3: " + str(float(grads["W3"][:,0])))
print("b3: " + str(float(grads["b3"][:,0])))

print("\n Approximated First Gradients of First 3 Layers...\n")

print("W1: " + str(approx["W1"]))
print("b1: " + str(approx["b1"]) + "\n")

print("W2: " + str(approx["W2"]))
print("b2: " + str(approx["b2"]) + "\n")

print("W3: " + str(approx["W3"]))
print("b3: " + str(approx["b3"]))

# We want this relative error to be very small but it is quite large at 1
print("\n Relative Error of third element of W3...\n")

print(np.abs(approx["W3"]-float(grads["W3"][:,0]))/(np.abs(approx["W3"]) + np.abs(float(grads["W3"][:,0]))))

print("----------------------------------------------------")



print("\n Printing Cost...\n")

learning_rate = 0.01;

j = 0
cost = np.zeros((10,1))

for i in range(0,iterations):    
    fw, caches = fp.forward_propagate(parameters,train_x_orig,len(dims),dropout=False)
    
    grads = bp.back_propagation(train_x_orig, train_y, fw, len(dims), parameters, caches)

    parameters["W3"] -= learning_rate * grads["W3"]
    parameters["b3"] -= learning_rate * grads["b3"]
    
    parameters["W2"] -= learning_rate * grads["W2"]
    parameters["b2"] -= learning_rate* grads["b2"]
    
    parameters["W1"] -= learning_rate * grads["W1"]
    parameters["b1"] -= learning_rate * grads["b1"]
    
    



    # Cost
    if i%(iterations/10) == 0:
        cost[j] = fp.cost(train_y,fw)
        print(cost[j])
        j += 1
        
        approx = bp.check_gradients(train_x_orig,train_y,parameters,len(dims))




plt.subplot()
plt.plot(cost)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.show()

# This is how we train the model

fw, caches = fp.forward_propagate(parameters,train_x_orig,len(dims))


print("Final Cost")
print(fp.cost(train_y,fw))  

print("Accuracy on training set: ", np.mean((fw>=0.5) == train_y) * 100)


fwt, cachest = fp.forward_propagate(parameters,test_x_orig,len(dims))

print("Accuracy on test set: ", np.mean((fwt>=0.5) == test_y) * 100)

