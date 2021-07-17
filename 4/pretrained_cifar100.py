# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torch.optim
import torch.utils.data
import matplotlib.pyplot as plt
import os
import torch.utils.model_zoo as model_zoo

def fit(net, optimizer, loss_func, n_epochs, train, val, batch_size=50):    
    net.cuda()
    loss_train = []
    avg_rate_test = []        
    for e in range(n_epochs):
        net.train() #put the net in train mode
        loss = 0
        for x, y in train:
            x = x.cuda()
            y = y.cuda()
            x = F.interpolate(x, scale_factor=7, mode='bilinear', align_corners=True)
            loss += loss_batch(net, loss_func, x, y, optimizer)
        loss_train.append(loss/1000)
        print('train loss is ' + str(loss/1000))
        with torch.no_grad():
            net.eval() #put the net in evaluation mod
            rate_test = torch.zeros(200)
            i = 0
            for xb, yb in val:
                xb = xb.cuda()
                yb = yb.cuda()
                xb = F.interpolate(xb, scale_factor=7, mode='bilinear', align_corners=True)

                yb_pre = net(xb)
                yb_pre = yb_pre.detach()
                yb_pre = torch.argmax(yb_pre, dim = 1)
                rate_test[i] = accuracy(yb_pre, yb)
                i += 1
            avg = torch.sum(rate_test)/200
            print('test accuracy is ' + str(avg))
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
    loss = loss_func(model(xb), yb)

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

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def resnet18(pretrained=True):
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18'],model_dir='./'))
    return model
if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    transform_train = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(32, padding = 3),
    T.ToTensor()
    ])

    transform_test = T.Compose([
    T.ToTensor()
    ])

    batch_size = 50
    # For trainning data 
    trainset = torchvision.datasets.CIFAR100(root= '/mnt/c/scratch/training/tra217/HW4/', train=True,download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0) 
    # For testing data 
    testset = torchvision.datasets.CIFAR100(root='/mnt/c/scratch/training/tra217/HW4/', train=False,download=False, transform=transform_test) 
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    print("done")    
    
    net = resnet18(pretrained=True)
    net.fc = nn.Linear(net.fc.in_features, 100)

    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(),lr=0.0005)
    
    train_el1 = []
    val_el1 = []
    number = 0
    for i in range(20):
        print(i)
        if i >= 10:
            opt = torch.optim.Adam(net.parameters(),lr=0.00025)
        if i >= 15:
            opt = torch.optim.Adam(net.parameters(),lr=0.0001)
        
        train_el, val_el = fit(net, opt, loss_fn, 1, trainloader, testloader, batch_size = 50)
        train_el1 += [train_el[0]]
        val_el1 += [val_el[0]]
        number = i
        torch.save(net.state_dict(), "pre"+str(i)+"_"+str(val_el[0])+".pb")
        print(train_el1,val_el1)
        #if rate > 0.81:
            #break
            
    print(train_el1,val_el1)
