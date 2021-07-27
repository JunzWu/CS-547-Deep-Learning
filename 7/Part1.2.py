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
        out = F.leaky_relu(self.conv1_LN(self.conv1(xb)), 0.1)
        out = F.leaky_relu(self.conv2_LN(self.conv2(out)), 0.1)
        out = F.leaky_relu(self.conv3_LN(self.conv3(out)), 0.1)
        out = F.leaky_relu(self.conv4_LN(self.conv4(out)), 0.1)
        out = F.leaky_relu(self.conv5_LN(self.conv5(out)), 0.1)
        out = F.leaky_relu(self.conv6_LN(self.conv6(out)), 0.1)
        out = F.leaky_relu(self.conv7_LN(self.conv7(out)), 0.1)
        out = F.leaky_relu(self.conv8_LN(self.conv8(out)), 0.1)
        out = self.MP(out)
        
        out = out.view(out.size(0), -1)
        out1 = self.fc1(out)
        out2 = self.fc2(out)
        out2 = self.softmax(out2)
        
        return out1, out2

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 2048)
        self.fc1_BN = nn.BatchNorm1d(2048)
        self.convT1 = nn.ConvTranspose2d(128, 128, 4, 2, 1)
        self.convT1_BN = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv2_BN = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv3_BN = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv4_BN = nn.BatchNorm2d(128)
        self.convT5 = nn.ConvTranspose2d(128, 128, 4, 2, 1)
        self.convT5_BN = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv6_BN = nn.BatchNorm2d(128)
        self.convT7 = nn.ConvTranspose2d(128, 128, 4, 2, 1)
        self.convT7_BN = nn.BatchNorm2d(128)
        self.conv8 = nn.Conv2d(128, 3, 3, 1, 1)
        
    def forward(self, xb):
        '''
        N, W, H = xb.shape
        xb = xb.view(N, 1, W, H)
        '''
        out = F.relu(self.fc1_BN(self.fc1(xb)))
        out = out.view(out.size(0), 128, 4, 4)
        out = F.relu(self.convT1_BN(self.convT1(out)))
        out = F.relu(self.conv2_BN(self.conv2(out)))
        out = F.relu(self.conv3_BN(self.conv3(out)))
        out = F.relu(self.conv4_BN(self.conv4(out)))
        out = F.relu(self.convT5_BN(self.convT5(out)))
        out = F.relu(self.conv6_BN(self.conv6(out)))
        out = F.relu(self.convT7_BN(self.convT7(out)))
        out = self.conv8(out)
        out = torch.tanh(out)
        
        return out

def fit(aD, optimizer_d, aG, optimizer_g, loss_func, n_epochs, train, val, batch_size, save_noise):
    start_time = time.time()
    n_z = 100
    n_classes = 10
    gen_train = 1
    # Train the model
    for e in range(n_epochs):
        print(e)

        for group in optimizer_d.param_groups:
            for p in group['params']:
                state = optimizer_d.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000

        for group in optimizer_g.param_groups:
            for p in group['params']:
                state = optimizer_g.state[p]
                if('step' in state and state['step']>=1024):
                    state['step'] = 1000

        aG.train()
        aD.train()

        loss1 = []
        loss2 = []
        loss3 = []
        loss4 = []
        loss5 = []
        acc1 = []

        for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):
            if(Y_train_batch.shape[0] < batch_size):
                continue
            # train G
            if((batch_idx%gen_train)==0):
                for p in aD.parameters():
                    p.requires_grad_(False)

                aG.zero_grad()

                label = np.random.randint(0,n_classes,batch_size)
                noise = np.random.normal(0,1,(batch_size,n_z))
                label_onehot = np.zeros((batch_size,n_classes))
                label_onehot[np.arange(batch_size), label] = 1
                noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
                noise = noise.astype(np.float32)
                noise = torch.from_numpy(noise)
                noise = noise.cuda()
                fake_label = torch.from_numpy(label).cuda()

                fake_data = aG(noise)
                gen_source, gen_class  = aD(fake_data)

                gen_source = gen_source.mean()
                gen_class = criterion(gen_class, fake_label)

                gen_cost = -gen_source + gen_class
                gen_cost.backward()

                optimizer_g.step()
            
            # train D
            for p in aD.parameters():
                p.requires_grad_(True)

            aD.zero_grad()

            # train discriminator with input from generator
            label = np.random.randint(0,n_classes,batch_size)
            noise = np.random.normal(0,1,(batch_size,n_z))
            label_onehot = np.zeros((batch_size,n_classes))
            label_onehot[np.arange(batch_size), label] = 1
            noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
            noise = noise.astype(np.float32)
            noise = torch.from_numpy(noise)
            noise = noise.cuda()
            fake_label = torch.from_numpy(label).cuda()
            with torch.no_grad():
                fake_data = aG(noise)

            disc_fake_source, disc_fake_class = aD(fake_data)

            disc_fake_source = disc_fake_source.mean()
            disc_fake_class = criterion(disc_fake_class, fake_label)

            # train discriminator with input from the discriminator
            real_data = X_train_batch.cuda()
            real_label = Y_train_batch.cuda()

            disc_real_source, disc_real_class = aD(real_data)

            prediction = disc_real_class.data.max(1)[1]
            accuracy = ( float( prediction.eq(real_label.data).sum() ) /float(batch_size))*100.0

            disc_real_source = disc_real_source.mean()
            disc_real_class = criterion(disc_real_class, real_label)

            gradient_penalty = calc_gradient_penalty(aD,real_data,fake_data)

            disc_cost = disc_fake_source - disc_real_source + disc_real_class + disc_fake_class + gradient_penalty
            disc_cost.backward()

            optimizer_d.step()

            loss1.append(gradient_penalty.item())
            loss2.append(disc_fake_source.item())
            loss3.append(disc_real_source.item())
            loss4.append(disc_real_class.item())
            loss5.append(disc_fake_class.item())
            acc1.append(accuracy)
            if((batch_idx%50)==0):
                print(e, batch_idx, "%.2f" % np.mean(loss1), 
                                        "%.2f" % np.mean(loss2), 
                                        "%.2f" % np.mean(loss3), 
                                        "%.2f" % np.mean(loss4), 
                                        "%.2f" % np.mean(loss5), 
                                        "%.2f" % np.mean(acc1))
        
        # Test the model
        aD.eval()
        with torch.no_grad():
            test_accu = []
            for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader):
                X_test_batch, Y_test_batch= X_test_batch.cuda(),Y_test_batch.cuda()

                with torch.no_grad():
                    _, output = aD(X_test_batch)

                prediction = output.data.max(1)[1] # first column has actual prob.
                accuracy = ( float( prediction.eq(Y_test_batch.data).sum() ) /float(batch_size))*100.0
                test_accu.append(accuracy)
                accuracy_test = np.mean(test_accu)
        print('Testing',accuracy_test, time.time()-start_time)

        ### save output
        with torch.no_grad():
            aG.eval()
            samples = aG(save_noise)
            samples = samples.data.cpu().numpy()
            samples += 1.0
            samples /= 2.0
            samples = samples.transpose(0,2,3,1)
            aG.train()

        fig = plot(samples)
        plt.savefig('output/%s.png' % str(e+400).zfill(3), bbox_inches='tight')
        plt.close(fig)

        if(((e+1)%1)==0):
            torch.save(aG,'tempG.model')
            torch.save(aD,'tempD.model')

    return

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

def calc_gradient_penalty(netD, real_data, fake_data):
    DIM = 32
    LAMBDA = 10
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
    alpha = alpha.view(batch_size, 3, DIM, DIM)
    alpha = alpha.cuda()
    
    fake_data = fake_data.view(batch_size, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)

    disc_interpolates, _ = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)                              
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

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
    
    #aD =  Discriminator()
    aD = torch.load('discriminator.model')
    aD.cuda()

    #aG = Generator()
    aG = torch.load('generator.model')
    aG.cuda()

    optimizer_g = torch.optim.Adam(aG.parameters(), lr=0.0001, betas=(0,0.9))
    optimizer_d = torch.optim.Adam(aD.parameters(), lr=0.0001, betas=(0,0.9))

    criterion = nn.CrossEntropyLoss()

    n_z = 100
    n_classes = 10
    np.random.seed(352)
    label = np.asarray(list(range(10))*10)
    noise = np.random.normal(0,1,(100,n_z))
    label_onehot = np.zeros((100,n_classes))
    label_onehot[np.arange(100), label] = 1
    noise[np.arange(100), :n_classes] = label_onehot[np.arange(100)]
    noise = noise.astype(np.float32)

    save_noise = torch.from_numpy(noise)
    save_noise = save_noise.cuda()

    fit(aD, optimizer_d, aG, optimizer_g, criterion, 100, trainloader, testloader, batch_size, save_noise)

    torch.save(aG,'generator.model')
    torch.save(aD,'discriminator.model')


    
    