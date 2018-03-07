# -*- coding: utf-8 -*-

import numpy as np
import copy
import time

x = np.array([[[[1,1,2,2,1],[0,2,1,1,0],[1,2,1,1,1],[2,0,0,2,2],[0,2,1,1,0]],
                         [[1,1,0,2,1],[2,0,0,1,0],[0,1,2,1,1],[2,0,2,1,0],[2,2,2,0,1]],
                         [[2,2,0,1,2],[0,0,0,2,2],[1,2,1,1,2],[1,1,0,2,2],[1,2,1,0,1]]],[[[1,1,2,2,1],[0,2,1,1,0],[1,2,1,1,1],[2,0,0,2,2],[0,2,1,1,0]],
                         [[1,1,0,2,1],[2,0,0,1,0],[0,1,2,1,1],[2,0,2,1,0],[2,2,2,0,1]],
                         [[2,2,0,1,2],[0,0,0,2,2],[1,2,1,1,2],[1,1,0,2,2],[1,2,1,0,1]]]])
x = np.moveaxis(x,1,-1)

w0 = np.array([[[-1,0,0],[1,1,0],[1,1,-1]],
               [[1,-1,-1],[0,-1,1],[1,-1,1]],
               [[-1,-1,1],[-1,0,0],[-1,-1,-1]]])

w1 = np.array([[[-1,-1,0],[0,1,1],[-1,1,1]],
               [[0,0,1],[1,1,0],[0,1,0]],
               [[0,0,1],[-1,0,0],[-1,0,0]]])

w_filter = np.array([w0,w1])
w_filter = np.moveaxis(w_filter,0,-1)
w_filter = np.moveaxis(w_filter,0,-2)


def im2col_flat(x,field_height,field_width,padding,stride):
    
    N,H, W, C = x.shape
    
    n_H = int(((H - field_height + 2 * padding)/stride)+1)
    n_W = int(((W - field_width + 2 * padding)/stride)+1)
    
    p = padding
    x_padded = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)), mode='constant')

    #Find the shape of the filters across n_H
        
    filters_h = np.repeat(np.arange(field_height),field_width)
    
    height = stride * np.repeat(np.arange(n_H),n_W).reshape(-1,1)
    height_index = filters_h + height
    
    i = np.tile(height_index,C)
    
    filters_w = np.tile(np.arange(field_width),field_height)
    width = stride * np.tile(np.arange(n_W),n_H).reshape(-1,1)
    width_index = filters_w + width

    j = np.tile(width_index,C)
    
    k = np.repeat(np.arange(C),field_height*field_width)
    
    
    return x_padded[:,i,j,k]
    
                        
def conv_fast(x,w_filter,padding=1,stride=1):
    
    
    N,H, W, C = x.shape
    
    n_H = int(((H - w_filter.shape[0]+ 2 * padding)/stride)+1)
    n_W = int(((W - w_filter.shape[1] + 2 * padding)/stride)+1)

    
    flat = im2col_flat(x,w_filter.shape[0],w_filter.shape[1],padding,stride)
    
    filter_flat = w_filter[:,:,:,:].reshape(-1,w_filter.shape[1],w_filter.shape[3]).T.reshape(w_filter.shape[3],-1).T
    
    conv =   flat.dot(filter_flat)
                 
    #now the final reshape
    conv = conv.reshape(N,n_H,n_W,w_filter.shape[3])
    return conv,flat

def conv_fast_back(x,w_filter,padding=1,stride=1):
    
    f_h, f_w, c, f =  w_filter.shape

    dw = np.zeros(w_filter.shape)    
    flat = im2col_flat(x,f_h,f_w,padding,stride)    
    flat = np.sum(flat,axis=(0,1)).reshape(f_h,-1)
    flat = flat.reshape(c,f_w,f_h)
    
    flat = np.moveaxis(flat,0,-1)
    dw = np.repeat(flat[:,:,:,None],w_filter.shape[3],axis=3)
    
    return dw,flat


def check_gradients(_input,_filter,pad,stride):
    
    epsilon = 0.0001
    
    parameters1 = copy.deepcopy(_input)
    parameters2 = copy.deepcopy(_input)
    
    parameters1[0,0,0,0] += epsilon
    parameters2[0,0,0,0] -= epsilon


    out1 = conv(parameters1,_filter,pad,stride)
    out2 = conv(parameters2,_filter,pad,stride)

    return (np.sum(out1)- np.sum(out2)) / (2. *epsilon)

def conv(_input,_filter,pad,stride):
    # With filter size (f,f,c_prev,f_c)
    #and input matrix size (m,w,h,c)
    
    (m, n_h, n_w, n_C_prev) = _input.shape
    (f,f, n_C_prev, n_C) = _filter.shape
   
    n_H = int(((n_h - f + 2 * pad)/stride)+1)
    n_W = int(((n_w - f + 2 * pad)/stride)+1)
    
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



def conv_back(_input,_filter,pad,stride):
    #how does the filter effect the output
    # we know we need an output the size of the filter
    # let's do some gradient checking to make sure we have this right
    (m, n_h, n_w, n_C_prev) = _input.shape
    (f,f, n_C_prev, n_C) = _filter.shape
   
    n_H = int(((n_h - f + 2 * pad)/stride)+1)
    n_W = int(((n_w - f + 2 * pad)/stride)+1)
    
    a_prev_pad = np.pad(_input, ((0,0),(pad,pad),(pad,pad),(0,0)), 'constant', constant_values=0)
   
    Z = np.zeros(_filter.shape,dtype=np.float32)
    dx = np.zeros(_input.shape,dtype=np.float32)

    for h in range(f):
        for w in range(f):
            for p in range(n_C_prev):
                for c in range(n_C):
                    #for channel in range(n_C_prev):
                        #Take a slice of  a and put it in the new volume
                    a_slice = a_prev_pad[:,h:h + n_H * stride:stride,w:w + n_W * stride:stride,p]

                    Z[h,w,p,c] = np.sum(a_slice)
                    
                    
    dx_pad = np.pad(dx, ((0,0),(pad,pad),(pad,pad),(0,0)), 'constant', constant_values=0)

    
    for i in range(m):        
        for h_output in range(n_H):
            for w_output in range(n_W):
                for g in range(n_C):
                            
                    vert_start = h_output*stride
                    vert_end = vert_start + f
                    horiz_start = w_output * stride
                    horiz_end = horiz_start + f
                    
                    dx_pad[i,vert_start:vert_end,horiz_start:horiz_end,:] += _filter[:,:,:,g]
                                    
                
    dx = dx_pad[:,pad:pad+n_h,pad:pad+n_w,:]
    return Z,dx

start = time.clock()                
fastconv,test = conv_fast(x,w_filter,1,2)
end = time.clock()

print(end-start)

start = time.clock()
naiveconv = conv(x,w_filter,1,2)
end= time.clock()

print(end-start)
assert np.mean(fastconv == naiveconv) == 1

dw_fast,flat = conv_fast_back(x,w_filter,1,2)
dw,dx= conv_back(x,w_filter,1,2)

assert np.mean(dw_fast == dw) == 1

