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
import torch.distributed as dist
import subprocess
from mpi4py import MPI
from torch.autograd import Variable
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from random import Random

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

def fit(net, loss_func, n_epochs, trainset, val, rank, world_size):    
    net = net.cuda()
    weight_initial(net, rank, world_size)
    opt = torch.optim.Adam(net.parameters(),lr=0.0005)

    loss_train = []
    avg_rate_test = [] 
    for e in range(n_epochs):
        print(e)
        
        if e >= 7:
            opt = torch.optim.Adam(net.parameters(),lr=0.00025)
        if e >= 14:
            opt = torch.optim.Adam(net.parameters(),lr=0.0001)
        
        partition_sizes = [1.0 / world_size for _ in range(world_size)]
        partition = DataPartitioner(trainset, partition_sizes)
        partition = partition.use(dist.get_rank())
        train = torch.utils.data.DataLoader(trainset, batch_size=50, shuffle=True, num_workers=0) 

        net.train() #put the net in train mode
        train_loss = 0
        for x, y in train:
            opt.zero_grad()
            x = Variable(x.cuda())
            y = Variable(y.cuda())
            loss = loss_func(net(x), y)
            train_loss += float(loss)
            loss.backward()
            if(e > 0):
                for group in opt.param_groups:
                    for p in group['params']:
                        state = opt.state[p]
                        if 'step' in state.keys():
                            if(state['step']>=1024):
                                state['step'] = 1000
            for param in net.parameters():
                if param.grad is not None:
                    tensor0 = param.grad.data.cpu()
                    dist.all_reduce(tensor0, op=dist.reduce_op.SUM)
                    tensor0 /= float(num_nodes)
                    param.grad.data = tensor0.cuda()
            opt.step()
        loss_train.append(train_loss/1000)
        print('Train loss of Rank ', dist.get_rank(), ' epoch ', e, ': ', train_loss/1000)

        net.eval() #put the net in evaluation mod
        rate_test = 0
        for xb, yb in val:
            xb = Variable(xb.cuda(), volatile=True)
            yb = Variable(yb.cuda(), volatile=True)

            yb_pre = net(xb)
            yb_pre = yb_pre.detach()
            value, index = torch.max(yb_pre, dim =1)
            rate_test += accuracy(index, yb)
        avg = rate_test/200
        print('Test accuracy of Rank ', dist.get_rank(), ' epoch ', e, ': ', avg)
        avg_rate_test.append(avg)
        torch.save(net.state_dict(), "epoch"+str(e)+"_rank"+str(dist.get_rank())+"_"+str(avg)+".pb")

    return loss_train, avg_rate_test

def accuracy(y_pre, y):
    n = y.shape[0]
    avg_class_rate = int((y_pre == y).sum())/n
    
    return avg_class_rate

def weight_initial(net, rank, world_size):
    for i in net.parameters():
        if rank == 0:
            for j in range(1, world_size):
                dist.send(i.data, dst=j)
        else:
            dist.recv(i.data, src=0)
    return

class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1]):
        self.data = data
        self.partitions = []
        rng = Random()
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

if __name__ == '__main__':
    cmd = "/sbin/ifconfig"
    out, err = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,stderr=subprocess.PIPE).communicate()
    ip = str(out).split("inet addr:")[1].split()[0]
    name = MPI.Get_processor_name()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_nodes = int(comm.Get_size())
    ip = comm.gather(ip)
    if rank != 0:
        ip = None
    ip = comm.bcast(ip, root=0)
    os.environ['MASTER_ADDR'] = ip[0]
    os.environ['MASTER_PORT'] = '2222'
    backend = 'mpi'
    dist.init_process_group(backend, rank=rank, world_size=num_nodes)
    dtype = torch.FloatTensor

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

    # For testing data 
    testset = torchvision.datasets.CIFAR100(root='/mnt/c/scratch/training/tra217/HW4/', train=False,download=False, transform=transform_test) 
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    print("done")    
    
    net=ResNet()
    #net.load_state_dict(torch.load("epoch19_rank1_0.5832.pb"))
    
    loss_fn = nn.CrossEntropyLoss()
        
    train_el, val_el = fit(net, loss_fn, 20, trainset, testloader, rank, num_nodes)
    print(train_el)
    print(val_el)
