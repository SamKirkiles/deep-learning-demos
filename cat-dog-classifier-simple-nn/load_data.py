# -*- coding: utf-8 -*-

import os
import matplotlib.image as mpimg
import numpy as np
from skimage.transform import resize
import h5py


def load_data():
    train_dataset = h5py.File('train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    original_shape = train_set_x_orig.shape

    train_set_x_orig = train_set_x_orig.reshape((train_set_x_orig.shape[0],train_set_x_orig.shape[1]*train_set_x_orig.shape[2]*train_set_x_orig.shape[3])).T

    test_set_x_orig = test_set_x_orig.reshape((test_set_x_orig.shape[0],test_set_x_orig.shape[1]*test_set_x_orig.shape[2]*test_set_x_orig.shape[3])).T
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes, original_shape

