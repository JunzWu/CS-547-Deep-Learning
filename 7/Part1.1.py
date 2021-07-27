# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import torchvision.transforms as T
import torchvision.transforms.functional as F1
import torchvision
import torch.optim
import torch.utils.data
import matplotlib.pyplot as plt
from random import shuffle
import h5py
import os
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.autograd import Variable

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 3, 1, 1)
        self.conv1_LN = nn.LayerNorm([128, 32, 32])
        self.conv2 = nn.Conv2d(128, 128, 3, 2, 1)
        self.conv2_LN = nn.LayerNorm([128, 16, 16])
        self.conv3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv3_LN = nn.LayerNorm([128, 16, 16])
        self.conv4 = nn.Conv2d(128, 128, 3, 2, 1)
        self.conv4_LN = nn.LayerNorm([128, 8, 8])
        self.conv5 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv5_LN = nn.LayerNorm([128, 8, 8])
        self.conv6 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv6_LN = nn.LayerNorm([128, 8, 8])
        self.conv7 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv7_LN = nn.LayerNorm([128, 8, 8])
        self.conv8 = nn.Conv2d(128, 128, 3, 2, 1)
        self.conv8_LN = nn.LayerNorm([128, 4, 4])
        self.MP = nn.MaxPool2d(4)
        self.fc1 = nn.Linear(128, 1)
        self.fc2 = nn.Linear(128, 10)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, xb):
        '''
        N, W, H = xb.shape
        xb = xb.view(N, 1, W, H)
        '''
        out = F.leaky_relu(self.conv1_LN(self.conv1(xb)))
        out = F.leaky_relu(self.conv2_LN(self.conv2(out)))
        out = F.leaky_relu(self.conv3_LN(self.conv3(out)))
        out = F.leaky_relu(self.conv4_LN(self.conv4(out)))
        out = F.leaky_relu(self.conv5_LN(self.conv5(out)))
        out = F.leaky_relu(self.conv6_LN(self.conv6(out)))
        out = F.leaky_relu(self.conv7_LN(self.conv7(out)))
        out = F.leaky_relu(self.conv8_LN(self.conv8(out)))
        out = self.MP(out)
        
        out = out.view(out.size(0), -1)
        out1 = self.fc1(out)
        out2 = self.fc2(out)
        out2 = self.softmax(out2)
        
        return out1, out2

def fit(net, optimizer, loss_func, n_epochs, train, val,):
    loss_train = []
    avg_rate_test = []       
    for e in range(n_epochs):
        net.train() #put the net in train mode
        loss = 0
        i = 0
        for x, y in train:
            x = x.cuda()
            y = y.cuda()
            loss += loss_batch(net, loss_func, x, y, optimizer)*y.shape[0]
            i += y.shape[0]
        loss_train.append(loss/i)
        print('train loss is ' + str(loss/i))
        with torch.no_grad():
            net.eval() #put the net in evaluation mod
            rate_test = 0
            i = 0
            for xb, yb in val:
                xb = xb.cuda()
                yb = yb.cuda()

                _, yb_pre = net(xb)
                yb_pre = yb_pre.detach()
                yb_pre = torch.argmax(yb_pre, dim = 1)
                rate_test += accuracy(yb_pre, yb)*yb.shape[0]
                i += yb.shape[0]
            #print(rate_test)
            avg = rate_test/i
            print('test accuracy is ' + str(rate_test/i))
            avg_rate_test.append(avg)

    return loss_train, avg_rate_test

def loss_batch(model, loss_func, xb, yb, opt=None):
    """ Compute the loss of the model on a batch of data, or do a step of optimization.

    @param model: the neural network
    @param loss_func: the loss function (can be applied to model(xb), yb)
    @param xb: a batch of the training data to input to the model
    @param yb: a batch of the training labels to input to the model
    @param opt: a torch.optimizer.Optimizer.  If not None, use the Optimizer to improve the model. Otherwise, just compute the loss.
    @return a numpy array of the loss of the minibatch, and the length of the minibatch
    """
    _, output = net(xb)
    loss = loss_func(output, yb)

    if opt is not None:
        loss.backward()
        for group in opt.param_groups:
            for p in group['params']:
                state = opt.state[p]
                if 'step' in state.keys():
                    if(state['step']>=1024):
                        state['step'] = 1000
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

def plot(samples):
    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
    transform_train = T.Compose([
    T.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0,1.0)),
    T.ColorJitter(
            brightness=0.1*torch.randn(1),
            contrast=0.1*torch.randn(1),
            saturation=0.1*torch.randn(1),
            hue=0.1*torch.randn(1)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = T.Compose([
        T.CenterCrop(32),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    batch_size = 128
    trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)
    print("done")
    
    net=Discriminator()
    #net.load_state_dict(torch.load("conv.pb"))
    
    net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    
    train_el1 = []
    val_el1 = []
    for i in range(100):
        print(i)
        if(i==50):
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0001/10.0
        if(i==75):
            for param_group in optimizer.param_groups:
                param_group['lr'] = 0.0001/100.0
        
        train_el, val_el = fit(net, optimizer, criterion, 1, trainloader, testloader)
        train_el1 += [train_el[0]]
        val_el1 += [val_el[0]]
        torch.save(net.state_dict(), "Dis"+str(i)+"_"+str(val_el[0])+".pb")
        print(train_el1)
        print(val_el1)

    torch.save(net,'cifar10.model')
    

    
    