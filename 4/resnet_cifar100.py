# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torch.optim
import torch.utils.data
import os


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias = False)
        self.conv1_BN = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False)
        self.conv2_BN = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 1, 2, 0, bias = False)
        
    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.conv1_BN(x1)
        x1 = F.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.conv2_BN(x1)
        if x1.size(2) != x.size(2):
            x = self.conv3(x)
        x = F.relu(x+x1)
        return x


class ResNet(nn.Module):
    def __init__(self, num_classes = 100): 
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1, bias = False)
        self.conv1_BN = nn.BatchNorm2d(32)
        self.conv1_dp = nn.Dropout(0.5)
        self.Block2_1 = BasicBlock(32, 32, stride=1)
        self.Block2_2 = BasicBlock(32, 32, stride=1)
        self.Block3_1 = BasicBlock(32, 64, stride=2)
        self.Block3_2 = BasicBlock(64, 64, stride=1)
        self.Block3_3 = BasicBlock(64, 64, stride=1)
        self.Block3_4 = BasicBlock(64, 64, stride=1)
        self.Block4_1 = BasicBlock(64, 128, stride=2)
        self.Block4_2 = BasicBlock(128, 128, stride=1)
        self.Block4_3 = BasicBlock(128, 128, stride=1)
        self.Block4_4 = BasicBlock(128, 128, stride=1)
        self.Block5_1 = BasicBlock(128, 256, stride=2)
        self.Block5_2 = BasicBlock(256, 256, stride=1)
        self.MP = nn.MaxPool2d(2)
        self.fc = nn.Linear(1024, 100)
        
    def forward(self, x):
        """
        The input will have shape (N, 1, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have shape (N, 10).
        """
        x = self.conv1(x)
        x = self.conv1_BN(x)
        x = F.relu(x)
        x = self.conv1_dp(x)
        x = self.Block2_1(x)
        x = self.Block2_2(x)
        x = self.Block3_1(x)
        x = self.Block3_2(x)
        x = self.Block3_3(x)
        x = self.Block3_4(x)
        x = self.Block4_1(x)
        x = self.Block4_2(x)
        x = self.Block4_3(x)
        x = self.Block4_4(x)
        x = self.Block5_1(x)
        x = self.Block5_2(x)
        x = self.MP(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        
        return x

def fit(net, optimizer, loss_func, n_epochs, train, val, batch_size=50):    
    net.cuda()
    avg_rate_train = []
    avg_rate_test = []       
    for e in range(n_epochs):
        net.train() #put the net in train mode
        for x, y in train:
            x = x.cuda()
            y = y.cuda()
            loss = loss_batch(net, loss_func, x, y, optimizer)
        with torch.no_grad():
            net.eval() #put the net in evaluation mod
            rate_train = torch.zeros(1000)
            i = 0
            for xb, yb in train:
                xb = xb.cuda()
                yb = yb.cuda()
                yb_pre = net(xb)
                yb_pre = yb_pre.detach()
                yb_pre = torch.argmax(yb_pre, dim = 1)
                rate_train[i] = accuracy(yb_pre, yb)
                i += 1
            avg = torch.sum(rate_train)/1000
            print('train accuracy is ' + str(avg))
            avg_rate_train.append(avg)

            rate_test = torch.zeros(200)
            i = 0
            for xb, yb in val:
                xb = xb.cuda()
                yb = yb.cuda()

                yb_pre = net(xb)
                yb_pre = yb_pre.detach()
                yb_pre = torch.argmax(yb_pre, dim = 1)
                rate_test[i] = accuracy(yb_pre, yb)
                i += 1
            avg = torch.sum(rate_test)/200
            print('test accuracy is ' + str(avg))
            avg_rate_test.append(avg)

    return avg_rate_train, avg_rate_test

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

    transform_train = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomCrop(32, padding = 3, padding_mode='edge'),
    T.ToTensor()
    ])

    transform_test = T.Compose([
    T.ToTensor()
    ])

    batch_size = 50
    # For trainning data 
    trainset = torchvision.datasets.CIFAR100(root= 'C:/Users/Administrator/Desktop/Deep_learning/HW/4/', train=True,download=False, transform=transform_train) 
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0) 
    # For testing data 
    testset = torchvision.datasets.CIFAR100(root='C:/Users/Administrator/Desktop/Deep_learning/HW/4/', train=False,download=False, transform=transform_test) 
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    print("done！")    
    
    net=ResNet()
    #net.load_state_dict(torch.load("Res_15_5905.pb"))
    
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(),lr=0.0005)
    
    train_el1 = []
    val_el1 = []
    number = 0
    for i in range(60):
        print(i)
        if i >= 10:
            opt = torch.optim.Adam(net.parameters(),lr=0.00025)
        if i >= 15:
            opt = torch.optim.Adam(net.parameters(),lr=0.0001)
        
        train_el, val_el = fit(net, opt, loss_fn, 1, trainloader, testloader, batch_size = 50)
        train_el1 += [train_el[0]]
        val_el1 += [val_el[0]]
        number = i
        torch.save(net.state_dict(), "Res.pb")
        print(train_el1,val_el1)

            
    print(train_el1,val_el1)

    x=np.arange(17)+1
    my_x_ticks = np.arange(1, 18, 1)
    plt.xticks(my_x_ticks)
    plt.plot(x,val_el1,'b',label='test accuracy')
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel('accuracy')
    plt.show()
