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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from random import shuffle
import h5py
import os
import random
import time
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
        
    def forward(self, xb, extract_features=0):
        '''
        N, W, H = xb.shape
        xb = xb.view(N, 1, W, H)
        '''
        out = F.leaky_relu(self.conv1_LN(self.conv1(xb)))
        out = F.leaky_relu(self.conv2_LN(self.conv2(out)))
        out = F.leaky_relu(self.conv3_LN(self.conv3(out)))
        out = F.leaky_relu(self.conv4_LN(self.conv4(out)))
        if(extract_features==4):
            out = F.max_pool2d(out,8,8)
            out = out.view(-1, 128)
            return out
        out = F.leaky_relu(self.conv5_LN(self.conv5(out)))
        out = F.leaky_relu(self.conv6_LN(self.conv6(out)))
        out = F.leaky_relu(self.conv7_LN(self.conv7(out)))
        out = F.leaky_relu(self.conv8_LN(self.conv8(out)))
        if(extract_features==8):
            out = F.max_pool2d(out,4,4)
            out = out.view(-1, 128)
            return out
        out = self.MP(out)
        
        out = out.view(out.size(0), -1)
        out1 = self.fc1(out)
        out2 = self.fc2(out)

        
        
        return out1, out2

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
    
    transform_test = T.Compose([
        T.CenterCrop(32),
        T.ToTensor(),
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    batch_size = 128
    testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)
    testloader = enumerate(testloader)

    model = torch.load('cifar10.model')
    model.cuda()
    model.eval()

    batch_idx, (X_batch, Y_batch) = testloader.__next__()
    X = X_batch.mean(dim=0)
    X = X.repeat(batch_size,1,1,1)
    X = Variable(X,requires_grad=True).cuda()

    Y = torch.arange(batch_size).type(torch.int64)
    Y = Variable(Y).cuda()

    lr = 0.1
    weight_decay = 0.001
    for i in range(200):
        output = model(X, extract_features=4)

        loss = -output[torch.arange(batch_size).type(torch.int64),torch.arange(batch_size).type(torch.int64)]
        gradients = torch.autograd.grad(outputs=loss, inputs=X,
                                grad_outputs=torch.ones(loss.size()).cuda(),
                                create_graph=True, retain_graph=False, only_inputs=True)[0]

        prediction = output.data.max(1)[1] # first column has actual prob.
        accuracy = ( float( prediction.eq(Y.data).sum() ) /float(batch_size))*100.0
        print(i,accuracy,-loss)

        X = X - lr*gradients.data - weight_decay*X.data*torch.abs(X.data)
        X[X>1.0] = 1.0
        X[X<-1.0] = -1.0

    ## save new images
    samples = X.data.cpu().numpy()
    samples += 1.0
    samples /= 2.0
    samples = samples.transpose(0,2,3,1)

    fig = plot(samples[0:100])
    plt.savefig('visualization/max_features.png', bbox_inches='tight')
    plt.close(fig)