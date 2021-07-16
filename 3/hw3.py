# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as F1
import torch.optim
import torch.utils.data
import matplotlib.pyplot as plt
from random import shuffle
import h5py
import os
import random

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, 1, 2)
        self.conv1_BN = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 4, 1, 2)
        self.conv2_dp = nn.Dropout(0.5)
        self.conv3 = nn.Conv2d(64, 64, 4, 1, 2)
        self.conv3_BN = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, 4, 1, 2)
        self.conv4_dp = nn.Dropout(0.5)
        self.conv5 = nn.Conv2d(64, 64, 4, 1, 2)
        self.conv5_BN = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv6_dp = nn.Dropout(0.5)
        self.conv7 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv7_BN = nn.BatchNorm2d(64)
        self.conv8 = nn.Conv2d(64, 64, 3, 1, 0)
        self.conv8_BN = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(1024, 500)
        self.fc2 = nn.Linear(500, 500)
        self.fc3 = nn.Linear(500, 10)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, xb):
        '''
        N, W, H = xb.shape
        xb = xb.view(N, 1, W, H)
        '''
        out = F.relu(self.conv1_BN(self.conv1(xb)))
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2, 2)
        out = self.conv2_dp(out)
        out = F.relu(self.conv3_BN(self.conv3(out)))
        out = F.relu(self.conv4(out))
        out = F.max_pool2d(out, 2, 2)
        out = self.conv4_dp(out)
        out = F.relu(self.conv5_BN(self.conv5(out)))
        out = F.relu(self.conv6(out))
        out = self.conv6_dp(out)
        out = F.relu(self.conv7_BN(self.conv7(out)))
        out = F.relu(self.conv8_BN(self.conv8(out)))
        
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.softmax(self.fc3(out))
        
        return out

def fit(net, optimizer, loss_func, n_epochs, x, y, x_val, y_val, batch_size=50):
    
    x = x.cuda()
    y = y.cuda()
    x_val = x_val.cuda()
    y_val = y_val.cuda()
    
    net.cuda()
    
    net.eval() #put the net in evaluation mode
    with torch.no_grad():
        # compute the mean loss on the training set at the beginning of iteration
        train_epoch_loss = []
        val_epoch_loss = []
        losses = torch.zeros(1000)
        for i in range(1000):
            xb = x[i*batch_size:(i+1)*batch_size,:]
            yb = y[i*batch_size:(i+1)*batch_size]
            loss = loss_batch(net, loss_func, xb, yb)
            losses[i] = loss
        #print(losses)
        train_epoch_loss.append(torch.sum(losses))
    
        losses_val = torch.zeros(200)
        for i in range(200):
            xb = x_val[i*batch_size:(i+1)*batch_size,:]
            yb = y_val[i*batch_size:(i+1)*batch_size]
            loss = loss_batch(net, loss_func, xb, yb)
            losses_val[i] = loss
        #print(losses)
        val_epoch_loss.append(torch.sum(losses_val))
        
        
    for e in range(n_epochs):
        net.train() #put the net in train mode
        index = [i for i in range(x.shape[0])]
        shuffle(index)

        x = x[index, :, :]
        y = y[index]
        for i in range(1000):
            #print(e, i*100, (i+1)*100)
            xb = x[i*batch_size:(i+1)*batch_size,:]
            yb = y[i*batch_size:(i+1)*batch_size]
            loss = loss_batch(net, loss_func, xb, yb, optimizer)
        with torch.no_grad():
            net.eval() #put the net in evaluation mode
            losses = torch.zeros(1000)
            for i in range(1000):
                xb = x[i*batch_size:(i+1)*batch_size,:]
                yb = y[i*batch_size:(i+1)*batch_size]
                loss = loss_batch(net, loss_func, xb, yb)
                losses[i] = loss
            #print(losses)
            train_epoch_loss.append(torch.sum(losses))
            
            losses_val = torch.zeros(200)
            for i in range(200):
                xb = x_val[i*batch_size:(i+1)*batch_size,:]
                yb = y_val[i*batch_size:(i+1)*batch_size]
                loss = loss_batch(net, loss_func, xb, yb)
                losses_val[i] = loss
            #print(losses)
            val_epoch_loss.append(torch.sum(losses_val))
            #print("this epoch is done!")

            rate = torch.zeros(200)
            for i in range(200):
                xb = x_val[i*batch_size:(i+1)*batch_size,:]
                yb = y_val[i*batch_size:(i+1)*batch_size]

                yb_pre = net(xb)
                yb_pre = yb_pre.detach()
                yb_pre = torch.argmax(yb_pre, dim = 1)
                rate[i] = accuracy(yb_pre, yb)

            avg_rate = torch.sum(rate)/200
            print(avg_rate)

    return train_epoch_loss, val_epoch_loss, avg_rate

def loss_batch(model, loss_func, xb, yb, opt=None):
    """ Compute the loss of the model on a batch of data, or do a step of optimization.

    @param model: the neural network
    @param loss_func: the loss function (can be applied to model(xb), yb)
    @param xb: a batch of the training data to input to the model
    @param yb: a batch of the training labels to input to the model
    @param opt: a torch.optimizer.Optimizer.  If not None, use the Optimizer to improve the model. Otherwise, just compute the loss.
    @return a numpy array of the loss of the minibatch, and the length of the minibatch
    """
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item()

def accuracy(y_pre, y):
    avg_class_rate = 0.0    
    n = y.shape[0]
    
    for i in range(n):
        if y_pre[i] == y[i]:
            avg_class_rate += 1.0
    avg_class_rate = avg_class_rate/n
    
    return avg_class_rate

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    CIFAR10_data = h5py.File('CIFAR10.hdf5', 'r')
    x_train = torch.tensor(CIFAR10_data['X_train'][:], dtype=torch.float)
    y_train = torch.tensor(CIFAR10_data['Y_train'][:], dtype=torch.long)
    print(x_train.shape)
    
    x_test = torch.tensor(CIFAR10_data['X_test'][:], dtype=torch.float)
    y_test = torch.tensor(CIFAR10_data['Y_test'][:], dtype=torch.long)
    CIFAR10_data.close()
    print("done！")
    
    net=ConvNet()
    net.load_state_dict(torch.load("conv.pb"))
    
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(),lr=0.0002)
    
    train_el1 = []
    val_el1 = []
    number = 0
    for i in range(60):
        print(i)
        # data augmentation
        for k in range(500):
            xb = x_train[k*100:(k+1)*100,:]
            if random.random() < 0.5:
                for j in range(100):
                    xb1 = T.ToPILImage(mode='RGB')(xb[j,:])
                    xb1 = F1.hflip(xb1)
                    xb[j,:] = T.ToTensor()(xb1)
            x_train[k*100:(k+1)*100,:] = xb
        
        if i > 20:
            opt = torch.optim.Adam(net.parameters(),lr=0.0001)
        if i > 30:
            opt = torch.optim.Adam(net.parameters(),lr=0.00005)
        
        train_el, val_el, rate = fit(net, opt, loss_fn, 1, x_train, y_train, x_test, y_test, batch_size = 50)
        if i == 0:
            train_el1 += [train_el[0]]
            train_el1 += [train_el[1]]
            val_el1 += [val_el[0]]
            val_el1 += [val_el[1]]
        if i != 0:
            train_el1 += [train_el[1]]
            val_el1 += [val_el[1]]
        number = i
        torch.save(net.state_dict(), "conv.pb")
        #if rate > 0.81:
            #break
            
    print(train_el1,val_el1)

    #torch.save(net.state_dict(), "conv.pb")

    x=np.arange(number + 2)
    plt.plot(x,train_el1,'r',label='train_epoch_loss')
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel('loss')
    plt.show()
    
    plt.plot(x,val_el1,'b',label='val_epoch_loss')
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel('loss')
    plt.show()
    
    # Monte Carlo simulation (after training)
    
    x_test = x_test.cuda()
    y_test = y_test.cuda()
    net.cuda()
    net.train()

    with torch.no_grad():
        y_pre = torch.zeros(10000,10).cuda()
        for i in range(50):
            rate = torch.zeros(200)
            for j in range(200):
                batch_size = 50
                xb = x_test[j*batch_size:(j+1)*batch_size,:]
                yb = y_test[j*batch_size:(j+1)*batch_size]

                yb_pre = net(xb)
                yb_pre = yb_pre.detach()
                y_pre[j*batch_size:(j+1)*batch_size, :] += yb_pre
                
        y_pre = y_pre/50
        y_pre = torch.argmax(y_pre, dim = 1)
        rate = accuracy(y_pre, y_test)
        print(rate)
    


    
    