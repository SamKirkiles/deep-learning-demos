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

X_train, y_train = load_data.loadData(_set='train',batch_iter=0,batch_size=100)

#plt.imshow(X_train[:,0].reshape((100, 100, 3)))

# Create features and training exmples
m = X_train.shape[1]
n = X_train.shape[0]

# Number of hidden layers and Number of nodes on each layer
n_h = 2
n_n = 2

# Initalize weights

dims = [n,20,20,1]

parameters = weights.init_weights(dims)

iterations = 10000

# Run forward and backprop
fw, caches = fp.forward_propagate(parameters,X_train,len(dims))
grads = bp.back_propagation(X_train, y_train, fw, len(dims), parameters, caches)

first_grads = copy.deepcopy(grads)
# Run check gradients method
approx = bp.check_gradients(X_train,y_train,parameters,len(dims))

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


batches = 1;

for b in range(0,batches):
    X_train, y_train = load_data.loadData(_set='train',batch_iter=b,batch_size=100)
    j = 0
    cost = np.zeros((10,1))

    for i in range(0,iterations):    
        fw, caches = fp.forward_propagate(parameters,X_train,len(dims),dropout=True)
        
        grads = bp.back_propagation(X_train, y_train, fw, len(dims), parameters, caches)
    
        parameters["W3"] -= 0.01 * grads["W3"]
        parameters["b3"] -= 0.01 * grads["b3"]
        
        parameters["W2"] -= 0.01 * grads["W2"]
        parameters["b2"] -= 0.01 * grads["b2"]
        
        parameters["W1"] -= 0.01 * grads["W1"]
        parameters["b1"] -= 0.01 * grads["b1"]
    
    
    
        # Cost
        if i%(iterations/10) == 0:
            cost[j] = fp.cost(y_train,fw)
            print(cost[j])
            j += 1


    plt.subplot()
    plt.plot(cost)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.show()

# This is how we train the model

fw, caches = fp.forward_propagate(parameters,X_train,len(dims))


print("Final Cost")
print(fp.cost(y_train,fw))  

print("Accuracy on training set: ", np.mean((fw>=0.5) == y_train) * 100)

X_test, y_test = load_data.loadData(32,_set='test')

fwt, cachest = fp.forward_propagate(parameters,X_test,len(dims))

print("Accuracy on test set: ", np.mean((fwt>=0.5) == y_test) * 100)

