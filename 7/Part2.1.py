# -*- coding: utf-8 -*-
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
from random import shuffle
import h5py
import os
import random
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
    _, output = model(xb)
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
    X_batch = Variable(X_batch,requires_grad=True).cuda()
    Y_batch_alternate = (Y_batch + 1)%10
    Y_batch_alternate = Variable(Y_batch_alternate).cuda()
    Y_batch = Variable(Y_batch).cuda()

    ## save real images
    samples = X_batch.data.cpu().numpy()
    samples += 1.0
    samples /= 2.0
    samples = samples.transpose(0,2,3,1)

    fig = plot(samples[0:100])
    plt.savefig('visualization/real_images.png', bbox_inches='tight')
    plt.close(fig)

    _, output = model(X_batch)
    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(Y_batch.data).sum() ) /float(batch_size))*100.0
    print(accuracy)

    ## slightly jitter all input images
    criterion = nn.CrossEntropyLoss(reduce=False)
    loss = criterion(output, Y_batch_alternate)

    gradients = torch.autograd.grad(outputs=loss, inputs=X_batch,
                            grad_outputs=torch.ones(loss.size()).cuda(),
                            create_graph=True, retain_graph=False, only_inputs=True)[0]

    # save gradient jitter
    gradient_image = gradients.data.cpu().numpy()
    gradient_image = (gradient_image - np.min(gradient_image))/(np.max(gradient_image)-np.min(gradient_image))
    gradient_image = gradient_image.transpose(0,2,3,1)
    fig = plot(gradient_image[0:100])
    plt.savefig('visualization/gradient_image.png', bbox_inches='tight')
    plt.close(fig)

    # jitter input image
    gradients[gradients>0.0] = 1.0
    gradients[gradients<0.0] = -1.0

    gain = 8.0
    X_batch_modified = X_batch - gain*0.007843137*gradients
    X_batch_modified[X_batch_modified>1.0] = 1.0
    X_batch_modified[X_batch_modified<-1.0] = -1.0

    ## evaluate new fake images
    _, output = model(X_batch_modified)
    prediction = output.data.max(1)[1] # first column has actual prob.
    accuracy = ( float( prediction.eq(Y_batch.data).sum() ) /float(batch_size))*100.0
    print(accuracy)

    ## save fake images
    samples = X_batch_modified.data.cpu().numpy()
    samples += 1.0
    samples /= 2.0
    samples = samples.transpose(0,2,3,1)

    fig = plot(samples[0:100])
    plt.savefig('visualization/jittered_images.png', bbox_inches='tight')
    plt.close(fig)