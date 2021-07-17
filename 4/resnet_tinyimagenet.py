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
    def __init__(self, num_classes = 200): 
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
        self.fc = nn.Linear(4096, 200)
        
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
    loss_train = []
    avg_rate_test = []       
    for e in range(n_epochs):
        net.train() #put the net in train mode
        loss = torch.zeros(2000)
        i = 0
        for x, y in train:
            x = x.cuda()
            y = y.cuda()
            loss[i] = loss_batch(net, loss_func, x, y, optimizer)
            i += 1
        loss_train.append(torch.sum(loss)/2000)
        print('train loss is ' + str(torch.sum(loss)/2000))
        with torch.no_grad():
            net.eval() #put the net in evaluation mod
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
            #print(rate_test)
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

def create_val_folder(val_dir): 
    """ This method is responsible for separating validation 
    images into separate sub folders """ 
    # path where validation data is present now 
    path = os.path.join(val_dir, 'images') 
    # file where image2class mapping is present
    filename = os.path.join(val_dir, 'val_annotations.txt') 
    fp = open(filename, "r") # open file in read mode 
    data = fp.readlines() # read line by line
    '''
    Create a dictionary with image names as key and 
    corresponding classes as values 
    '''
    val_img_dict = {} 
    for line in data: 
        words = line.split("\t") 
        val_img_dict[words[0]] = words[1] 
    fp.close() 
    # Create folder if not present, and move image into proper folder 
    for img, folder in val_img_dict.items(): 
        newpath = (os.path.join(path, folder)) 
        if not os.path.exists(newpath): # check if folder exists 
            os.makedirs(newpath) 
        # Check if image exists in default directory 
        if os.path.exists(os.path.join(path, img)): 
            os.rename(os.path.join(path, img), os.path.join(newpath, img)) 
    return

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    transform_train = T.Compose([
    T.RandomHorizontalFlip(),
    #T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
    T.RandomCrop(64, padding = 6),
    T.ToTensor(),
    #T.Normalize(mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5))
    ])

    transform_test = T.Compose([
    T.ToTensor(),
    #T.Normalize(mean = (0.5,0.5,0.5), std = (0.5,0.5,0.5))
    ])

    batch_size = 50
    train_dir = '/mnt/c/scratch/training/tra217/HW4/tiny-imagenet-200/train/'
    train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
    #print(train_dataset.class_to_idx)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dir1 = '/mnt/c/scratch/training/tra217/HW4/tiny-imagenet-200/val/'
    val_dir2 = '/mnt/c/scratch/training/tra217/HW4/tiny-imagenet-200/val/images'
    if 'val_' in os.listdir(val_dir2)[0]:
        create_val_folder(val_dir1)
    else:
        pass
    val_dataset = torchvision.datasets.ImageFolder(val_dir2,transform=transform_test)
    #print(val_dataset.class_to_idx)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print("done")    
    
    net=ResNet()
    #net.load_state_dict(torch.load("50Res3_tensor(0.3856).pb"))
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(net.parameters(),lr=0.0005)
    
    train_el1 = []
    val_el1 = []
    number = 0
    for i in range(20):
        print(i)
        if i >= 10:
            opt = torch.optim.Adam(net.parameters(),lr=0.0003)
        if i >= 15:
            opt = torch.optim.Adam(net.parameters(),lr=0.0001)
        train_el, val_el = fit(net, opt, loss_fn, 1, train_loader, val_loader, batch_size = 50)
        train_el1 += [train_el[0]]
        val_el1 += [val_el[0]]
        number = i
        torch.save(net.state_dict(), "50Res"+str(i)+"_"+str(val_el[0])+".pb")
        print(train_el1,val_el1)
        if val_el[0] > 0.5:
            break
            
    print(train_el1,val_el1)

