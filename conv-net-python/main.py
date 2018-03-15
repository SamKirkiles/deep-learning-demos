# -*- coding: utf-8 -*-t
import matplotlib.pyplot as plt
import cifar10
from terminaltables import AsciiTable
import os
import time
import atexit
import copy
import numpy as np
import tensorflow as tf

from network import forward_propagate, backward_propagate, backward_propagate_check, get_weights, get_weights_from_file

try:
    train_x_raw
    train_y
    train_y_one_hot
    classes
    
    test_x_raw
    test_y
    test_y_one_hot
except NameError:
    cifar10.maybe_download_and_extract()
    train_x_raw, train_y, train_y_one_hot = cifar10.load_training_data()
    test_x_raw, test_y, test_y_one_hot = cifar10.load_test_data()
    classes = cifar10.load_class_names()


# use tensorboard to visualize loss
# run tensorboard --logdir ./out.graph
run_id = str(np.random.randint(100))
writer = tf.summary.FileWriter('out.graph/run_' + run_id,flush_secs=10, graph=tf.get_default_graph())
        
print("Starting training...")

train_mean = np.mean(train_x_raw )
train_max = np.max(train_x_raw )
train_min = np.min(train_x_raw )

train_x_raw  = (train_x_raw - train_mean)/np.std(train_x_raw)
print(np.std(train_x_raw))
test_x_raw  = (test_x_raw - train_mean)/(train_max-train_min)

train_x_raw = train_x_raw[0:200,...]
train_y_one_hot = train_y_one_hot[0:200,...]
train_y = train_y[0:200,...]

BATCH_SIZE = 16
LEARNING_RATE = 0.01
EPOCHS = 5

epoch_size = int(train_x_raw.shape[0]/BATCH_SIZE)



#weights = get_weights_from_file('train_weights.npy')
weights = get_weights()


def exit_handler():
    print("Saving weights...")
    print(weights["W1"][0,0,0,0])
    np.save('train_weights.npy',weights)

atexit.register(exit_handler)

test_full_accuracy = "not caclulated"

steps = 0
"""
for i in range(1000):
    caches,cost = forward_propagate(train_x_raw[0:2,...],train_y_one_hot[0:2,...],weights,debug=False)
    df1,df2,dw3,dw4,db1,db2,db3,db4= backward_propagate(train_x_raw[0:2,...],train_y_one_hot[0:2,...],copy.deepcopy(weights),caches,debug=False)
    weights["W1"] -= LEARNING_RATE * df1
    weights["W2"] -= LEARNING_RATE * df2
    weights["W3"] -= LEARNING_RATE * dw3
    weights["W4"] -= LEARNING_RATE * dw4

    weights["B1"] -= LEARNING_RATE * db1
    weights["B2"] -= LEARNING_RATE * db2
    weights["B3"] -= LEARNING_RATE * db3
    weights["B4"] -= LEARNING_RATE * db4

    summary = tf.Summary(value=[tf.Summary.Value(tag='cost',simple_value=cost)])
    writer.add_summary(summary,i)

"""
for e in range(EPOCHS):
    for b in range(epoch_size):
    
        start = time.time()

        batch_x = train_x_raw[BATCH_SIZE*b:BATCH_SIZE*(b+1),...]
        batch_y = train_y_one_hot[BATCH_SIZE*b:BATCH_SIZE*(b+1),...]

    
        caches,cost = forward_propagate(batch_x,batch_y,weights,debug=False)
        df1,df2,dw3,dw4,db1,db2,db3,db4= backward_propagate(batch_x,batch_y,copy.deepcopy(weights),caches,debug=False)
        
        # Update tensorboard
        summary = tf.Summary(value=[tf.Summary.Value(tag='cost',simple_value=cost)])
        writer.add_summary(summary,steps)
        
        steps +=1 

        backward_propagate_check(df1,df2,dw3,dw4,db1,db2,db3,db4,batch_x,batch_y,weights)
        
        weights["W1"] -= LEARNING_RATE * df1
        weights["W2"] -= LEARNING_RATE * df2
        weights["W3"] -= LEARNING_RATE * dw3
        weights["W4"] -= LEARNING_RATE * dw4
    
        weights["B1"] -= LEARNING_RATE * db1
        weights["B2"] -= LEARNING_RATE * db2
        weights["B3"] -= LEARNING_RATE * db3
        weights["B4"] -= LEARNING_RATE * db4

        end = time.time()
                
        duration = end - start
        duration *= epoch_size - b
        
        duration_hours = int(duration/(3600)) 
        duration_minutes = int((duration - duration_hours*3600)/60)
        duration_seconds = int(duration - duration_hours*3600 - duration_minutes*60)
        
        duration_format = str(duration_hours) + " H " + str(duration_minutes) + " M " + str(duration_seconds) + " S" 
        
        
        accuracy = str(int(np.mean(batch_y.argmax(axis=1) == caches["A4"].argmax(axis=1)) * 100)) + "%"
        
        print(batch_y.argmax(axis=1))
        print(caches["A4"].argmax(axis=1))


        sampl = np.random.randint(test_x_raw.shape[0]-1, size=(200,))

        
        test_caches,test_cost = forward_propagate(test_x_raw[sampl,...],test_y_one_hot[sampl,...],copy.deepcopy(weights),debug=False)
        test_accuracy = str(int(np.mean(test_y_one_hot[sampl,...].argmax(axis=1) == test_caches["A4"].argmax(axis=1)) * 100)) + "%"
        
        data = [["Epoch","~ Time Remaining","Cost","Batch Accuracy","Test Accuracy"],
                [str(int(b/float(epoch_size)*100)) + "% " + str(e) + "/" + str(EPOCHS),duration_format,cost,accuracy,test_accuracy]]

        table = AsciiTable(data)
        
        table.title = "Stats run_" + run_id
        
        os.system('clear') 
        print(table.table)

print("Done.")
print("Saving weights...")
np.save('train_weights.npy',weights)


