# -*- coding: utf-8 -*-

import numpy as np

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

                    # go through all the individual positions that this filter affected and multiply by their dout
                    a_slice = a_prev_pad[:,h:h + n_H * stride:stride,w:w + n_W * stride:stride,p]
                     
                    dw[h,w,p,c] = np.sum(a_slice * dout[:,:,:,c])
                    
    # TODO: put back in dout to get correct gradient
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
