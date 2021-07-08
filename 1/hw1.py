# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 12:45:55 2019

@author: Junz
"""

import numpy as np
import h5py
import time
import matplotlib.pyplot as plt

def minibatch_sgd(epoch, w1, w2, b1, b2, x_train, y_train):
    losses =[]
    for e in range(epoch):
        print(e)
        index = [i for i in range(x_train.shape[0])]
        np.random.shuffle(index)
        x_train = x_train[index,:]
        y_train = y_train[index]
        loss = 0
        for i in range(int(x_train.shape[0]/200)):
            x = x_train[i*200:(i+1)*200,:]
            y = y_train[i*200:(i+1)*200]
            loss += two_nn(w1, w2, b1, b2, x, y, False)
        losses += [loss]  
    return w1, w2, b1, b2, losses

def test_nn(w1, w2, b1, b2, x_test, y_test):
    correct_rate = 0.0
    
    classification = two_nn(w1, w2, b1, b2, x_test, y_test, True)
    n = y_test.shape[0]
    
    for i in range(n):
        if classification[i] == y_test[i]:
            correct_rate += 1.0
    correct_rate = correct_rate/n
    
    return correct_rate

def two_nn(w1, w2, b1, b2, x, y, test):
    Z1, acache1 = affine_forward(x, w1, b1)
    A1, rcache1 = relu_forward(Z1)
    F, acache2 = affine_forward(A1, w2, b2)
    
    if test == True:
        classification = np.argmax(F, axis = 1)
        return classification
    loss, dF = cross_entropy(F, y)
    dA1, dw2, db2 = affine_backward(dF, acache2)
    dZ1 = relu_backward(dA1, rcache1)
    dx, dw1, db1 = affine_backward(dZ1, acache1)
    
    w1 -= 0.1*dw1
    w2 -= 0.1*dw2
    
    return loss

def affine_forward(A, W, b):
    cache = (A, W, b)
    Z = np.dot(A, W)
    Z = Z+b
    return Z, cache

def affine_backward(dZ, cache):
    A, W, b = cache
    
    WT = W.T
    dA = np.dot(dZ, WT)
    
    AT = A.T
    dW = np.dot(AT, dZ)
    
    dB = np.sum(dZ, axis = 0)
    return dA, dW, dB

def relu_forward(Z):
    cache = Z
    A = np.maximum(Z,0)
    
    return A, cache

def relu_backward(dA, cache):
    Z = cache
    X = np.where(Z<=0)
    dZ = dA
    dZ[X] = 0
    return dZ

def cross_entropy(F, y):
    #print(F.shape)
    n = F.shape[0]
    C = F.shape[1]
    loss = 0 
    for i in range(n):
        loss += F[i,int(y[i])]
        e = np.exp(F[i,:])
        e = np.sum(e)
        loss -= np.log(e)
    loss = -1*loss/n
    
    dF = np.zeros((n,C))
    for i in range(n):
        for j in range(C):
            if j == y[i]:
                dF[i,j] += 1
            e = np.exp(F[i,:])
            e = np.sum(e)
            dF[i,j] -= np.exp(F[i,j])/e
            dF[i,j] = -1*dF[i,j]/n
    return loss, dF

def init_weights(i, o):
    return 0.01 * np.random.uniform(0.0, 1.0, (i, o)), np.zeros(o)

if __name__ == '__main__':
    time_start=time.time()
    #load MNIST data
    MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
    x_train = np.float32(MNIST_data['x_train'][:] )
    y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
    x_test = np.float32( MNIST_data['x_test'][:] )
    y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
    MNIST_data.close()
    
    w1, b1 = init_weights(784, 128)
    w2, b2 = init_weights(128, 10)
    
    w1, w2, b1, b2, losses = minibatch_sgd(50, w1, w2, b1, b2, x_train, y_train)
    
    correct_rate = test_nn(w1, w2, b1, b2, x_test, y_test)
    print(correct_rate)
    
    epoch = []
    for i in range(50):
        epoch += [i+1]
    plt.plot(epoch,losses)
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.show()
    
    time_end=time.time()
    print('time cost',time_end-time_start,'s')