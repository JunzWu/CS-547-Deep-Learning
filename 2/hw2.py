# -*- coding: utf-8 -*-

import numpy as np
import h5py
import matplotlib.pyplot as plt

def minibatch_sgd(epoch, K, w1, b1, x_train, y_train, x_test, y_test):
    losses =[]
    for e in range(epoch):
        print(e)
        index = [i for i in range(x_train.shape[0])]
        np.random.shuffle(index)
        x_train = x_train[index,:]
        y_train = y_train[index]
        loss = 0
        for i in range(int(x_train.shape[0]/100)):
            print(e, i*100, (i+1)*100)
            x = x_train[i*100:(i+1)*100,:]
            y = y_train[i*100:(i+1)*100]
            loss += CNN(K, w1, b1, x, y, False)
        losses += [loss]
        
        correct_rate = test_nn(K, w1, b1, x_test, y_test)
        np.save('K', K)
        np.save('w1', w1)
    
        np.save('b1', b1)
        print(correct_rate)
        
    return K, w1, b1, losses

def test_nn(w1, w2, b1, x_test, y_test):
    correct_rate = 0.0
    
    classification = CNN(w1, w2, b1, x_test, y_test, True)
    n = y_test.shape[0]
    
    for i in range(n):
        if classification[i] == y_test[i]:
            correct_rate += 1.0
    correct_rate = correct_rate/n
    
    return correct_rate

def CNN(K, w1, b1, x, y, test):
    Z1, acache1 = conv_forward(x, K)
    A1, rcache1 = relu_forward(Z1)
    A1 = np.reshape(A1, [-1, 1875])
    F, acache2 = fc_forward(A1, w1, b1)
    
    if test == True:
        classification = np.argmax(F, axis = 1)
        return classification
    loss, dF = cross_entropy(F, y)
    dA1, dw1, db1 = fc_backward(dF, acache2)
    dA1 = np.reshape(dA1, [-1, 25, 25, 3])
    dZ1 = relu_backward(dA1, rcache1)
    dK = conv_backward(dZ1, acache1)
    
    K -= 0.01*dK
    
    w1 -= 0.01*dw1
    b1 -= 0.01*db1
    
    return loss

def conv_forward(x, K, stride = 1):
    cache = (x, K)
    Z = np.zeros((x.shape[0], 25, 25, K.shape[1]))
    for i in range(x.shape[0]):
        for j in range(K.shape[1]):
            kernal = K[:,j,:,:]
            for l in range(25):
                for m in range(25):
                    Z[i, l, m, j] = np.sum(x[i,l:l+4,m:m+4]*kernal)                 
    
    return Z, cache

def conv_backward(dZ, cache):
    x, K = cache
    dK = np.zeros(K.shape)
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            kernal = dZ[:,:,:,j]
            for l in range(K.shape[2]):
                for m in range(K.shape[3]):
                    dK[i, j, l, m] = np.sum(x[:,l:l+25,m:m+25]*kernal) 
    return dK

def fc_forward(A, W, b):
    cache = (A, W, b)
    Z = np.dot(A, W)
    Z = Z+b
    return Z, cache

def fc_backward(dZ, cache):
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

def init_linear(i, o):
    return 0.01 * np.random.uniform(0.0, 1.0, (i, o)), np.zeros(o)

def init_conv(i, o, s):
    return 0.01 * np.random.uniform(0.0, 1.0, (i, o, s, s))

if __name__ == '__main__':
    #load MNIST data
    MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
    x_train = np.float32( MNIST_data['x_train'][:] )
    x_train = np.reshape(x_train, [-1, 28, 28])
    y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
    x_test = np.float32( MNIST_data['x_test'][:] )
    x_test = np.reshape(x_test, [-1, 28, 28])
    y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
    MNIST_data.close()
    
    K = init_conv(1, 3, 4)
    w1, b1 = init_linear(1875, 10)
    '''
    K = np.load('K_2.npy')
    w1 = np.load('w1_2.npy')
    b1 = np.load('b1_2.npy')
    '''
    K, w1, b1, losses = minibatch_sgd(5, K, w1, b1, x_train, y_train, x_test, y_test)
    
    np.save('K', K)
    np.save('w1', w1)

    np.save('b1', b1)
    
    correct_rate = test_nn(K, w1, b1, x_test, y_test)
    print(correct_rate)