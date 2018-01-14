# -*- coding: utf-8 -*-

import os
import matplotlib.image as mpimg
import numpy as np
from skimage.transform import resize
import h5py


def loadData(im_size=32, batch_size=100,batch_iter = 0,_set='train'):
    
    print('Loading data for batch ' + str(batch_iter) + '...')

    
    if _set == 'train':
        path_1 = 'train/cat/'
        path_0 = 'train/dog/'
    else:
        path_1 = 'test/cat/'
        path_0 = 'test/dog/'

        
    
    positive_examples = [f for f in os.listdir(path_1) if f.endswith('.jpg')][batch_size*batch_iter:batch_size*batch_iter+batch_size]
    negative_examples = [f for f in os.listdir(path_0) if f.endswith('.jpg')][batch_size*batch_iter:batch_size*batch_iter+batch_size]

    m = len(positive_examples) + len(negative_examples)
    
    X_pos = np.zeros((im_size * im_size * 3,len(positive_examples)))
    y_pos = np.ones((1,X_pos.shape[1]))
    X_neg = np.zeros((im_size * im_size * 3,len(negative_examples)))
    y_neg = np.zeros((1,X_neg.shape[1]))

    for i in range(0,len(positive_examples)):
        temp = mpimg.imread(path_1 + positive_examples[i])
        temp = resize(temp, (im_size, im_size),mode='constant')
        temp = temp.reshape((temp.shape[0] * temp.shape[1] * temp.shape[2],1))
        X_pos[:,i:i+1] = temp
        
    for i in range(0,len(negative_examples)):
        temp = mpimg.imread(path_0 + negative_examples[i])
        temp = resize(temp, (im_size, im_size),mode='constant')
        temp = temp.reshape((temp.shape[0] * temp.shape[1] * temp.shape[2],1))
        X_neg[:,i:i+1] = temp

    X = np.hstack((X_pos, X_neg))
    Y = np.hstack((y_pos,y_neg))
    
    
    rand_ind = np.array(range(0,m))
    np.random.shuffle(rand_ind)
    
    X = X[:, rand_ind]
    Y = Y[:,rand_ind]
    
    print('Done.')
    
    return X, Y

def loadDatah5():
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    train_set_x_orig = train_set_x_orig.reshape((train_set_x_orig.shape[0],train_set_x_orig.shape[1]*train_set_x_orig.shape[2]*train_set_x_orig.shape[3])).T

    test_set_x_orig = test_set_x_orig.reshape((test_set_x_orig.shape[0],test_set_x_orig.shape[1]*test_set_x_orig.shape[2]*test_set_x_orig.shape[3])).T


    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

