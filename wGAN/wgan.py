# -*- coding: utf-8 -*-
"""WGAN.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/188ihu7oaX40hFq5of2OF7TzHStYJdLlb
"""

pip install -U https://github.com/PythonOT/POT/archive/master.zip # with --user for user install (no root)

# import necessary packages
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
import torch
import torch.nn as nn
from torch.autograd import Variable
from time import time
import ot

# make a plot for target distribution
mu,sigma = 2,5
num_samples = 10000
xs = np.linspace(-3, 7, num_samples) # interval = 10/1000=0.01
samples = norm.pdf(xs, mu, sigma)
plt.plot(xs, samples)

"""# Ganerative Adversarial Network Framework
1.Generate Target and random distribution.
2.Define Network Architecture.
"""

# Real data distribution
class RealDistribution:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def sample(self, num_samples):
        samples = np.random.normal(self.mu, self.sigma, num_samples)
        samples.sort()

        return samples

# Noise data distribution as inputs for the generator
class NoiseDistribution:
    def __init__(self, data_range):
        self.data_range = data_range

    def sample(self, num_samples):
        offset = np.random.uniform(-0.1,0.1,num_samples)*10 # Random floats with uniform dist. in the interval [0.0, 0.01)
        samples = np.linspace(-self.data_range, self.data_range, num_samples) + offset

        return samples

# three layer NN
class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Generator, self).__init__()

        # Fully-connected layer
        fc = nn.Linear(input_dim, hidden_dim, bias=True)
        # initializer
        nn.init.normal_(fc.weight)
        nn.init.constant_(fc.bias, 0.0)
        # Hidden layer
        self.hidden_layer = nn.Sequential(fc, nn.ReLU())
        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim, bias=True)
        # initializer
        nn.init.normal_(self.output_layer.weight)
        nn.init.constant_(self.output_layer.bias, 0.0)

    def forward(self, x):
        h = self.hidden_layer(x)
        out = self.output_layer(h)

        return out

# three layer NN
class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Discriminator, self).__init__()

        # Fully-connected layer
        fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        # initializer
        nn.init.normal_(fc1.weight)
        nn.init.constant_(fc1.bias, 0.0)
        # Hidden layer
        self.hidden_layer = nn.Sequential(fc1, nn.ReLU())
        # Fully-connected layer
        fc2 = nn.Linear(hidden_dim, output_dim, bias=True)
        # initializer
        nn.init.normal_(fc2.weight)
        nn.init.constant_(fc2.bias, 0.0)
        # Output layer
        self.output_layer = nn.Sequential(fc2, nn.Sigmoid()) ## binary classification

    def forward(self, x):
        h = self.hidden_layer(x)
        out = self.output_layer(h)

        return out

"""# Network Train
1.Initialize Hyper-parameter
2.Use RMSPOP in wGAN paper to train
3.plot the result
"""

# Hyper-parameters
# mu, sigma are target distribution mean and variance.
mu = 20.0
sigma = 100.0
# data range is used for plotting.
data_range = 100
batch_size = 150

input_dim = 1
hidden_dim = 128
output_dim = 1

num_epochs = 1000
learning_rate = 0.0005
clip_value=0.01 # lower and upper clip value for disc. weights
n_critic = 5
cuda = True if torch.cuda.is_available() else False
num_samples = 10000
num_bins = 30

# Samples
realData = RealDistribution(mu, sigma)     # via np.random.normal
noiseData = NoiseDistribution(data_range)  # unfiorm plus some variations

# Create models
G = Generator(input_dim, hidden_dim, output_dim)
D = Discriminator(input_dim, hidden_dim, output_dim)
# Loss function (WGAN loss)

# Optimizers
g_optimizer = torch.optim.RMSprop(G.parameters(), lr=learning_rate)
d_optimizer = torch.optim.RMSprop(D.parameters(), lr=learning_rate)

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

g_loss_list = []
d_loss_list = []

wass_list = []
approx = []
batches_done = 0
saved_imgs = []
for epoch in range(num_epochs):
    #print('Epoch ' + str(epoch) + ' training...' , end=' \n')

    x_ = realData.sample(batch_size)
    x_ = Variable(torch.FloatTensor(np.reshape(x_, [batch_size,input_dim])))
    y_real_ = Variable(torch.ones([batch_size,input_dim]))
    y_fake_ =  Variable(torch.zeros([batch_size, input_dim]))
    # Train discriminator with real data
    d_real_decision = D(x_)
    d_real_loss = torch.mean(d_real_decision)


    z_ = noiseData.sample(batch_size)
    z_ = Variable(torch.FloatTensor(np.reshape(z_,[batch_size, input_dim])))
    z_ = G(z_)
    d_fake_decision = D(z_)
    d_fake_loss = torch.mean(d_fake_decision)

    #fake_data = d_fake_decision.detach()
    # train Discriminator
    # sample noise as generator input

    # generate a batch of images
    # Adversarial loss train
    d_loss = d_fake_loss - d_real_loss
    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()
    # clip weights of discriminator
    for p in D.parameters():
        p.data.clamp_(-1*clip_value,clip_value )

    z_ = noiseData.sample(batch_size)
    z_ = Variable(torch.FloatTensor(np.reshape(z_,[batch_size,input_dim])))
    z_ = G(z_)
    # record batch of fake data

    d_fake_decision = D(z_)
    # loss function
    g_loss = -torch.mean(d_fake_decision)

    # train Generator
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()

    g_loss.backward()
    g_optimizer.step()
    #batches_done += 1
    end = time()
    d_loss_list.append(d_loss.item())
    g_loss_list.append(g_loss.item())
    # compute Wasserstein distance and loss here.
    if epoch % 5 == 0:
        with torch.no_grad():
            critics_fake_data_ = z_.detach().numpy()
            critics_fake_data,_ = np.histogram(critics_fake_data_, num_bins, density=True)
            #print(critics_fake_data.shape)
            true_data_ = x_.detach().numpy()
            true_data,_ = np.histogram(true_data_ , num_bins, density=True)

            #plt.hist(critics_fake_data.numpy(),bins=50, density=True, alpha=0.6, color='g')
            #plt.title(f"Epoch {epoch}")
            #plt.xlabel("Generated Data Value")
            #plt.ylabel("Density")
            #plt.show()
            bins = np.arange(num_bins, dtype=np.float64)
            # loss matrix
            M = ot.dist(bins.reshape((num_bins, 1)), bins.reshape((num_bins, 1)))
            M /= M.max()
            reg = 0.01
            ot_plan = ot.sinkhorn(true_data,critics_fake_data,M,reg)
            wass_dist = np.sum(np.multiply(M,ot_plan))
            wass_list.append(wass_dist)
            approx.append(-1*d_loss.item())
            print('epoch:{}, D loss:{}, G loss:{}\n'.format(epoch, d_loss.item(), d_loss.item()))
            print('epoch:{}, approximate_wass:{}, wass:{}\n'.format(epoch, np.abs(d_loss.item()), wass_dist))

# train loss plot
fig, ax = plt.subplots()
D_losses = np.array(d_loss_list)
G_losses = np.array(g_loss_list)
plt.plot(D_losses, label='Discriminator')
plt.plot(G_losses, label='Generator')
#
plt.title("Training Losses")
plt.legend()
plt.show()

# wasserstein vs.loss
fig, ax = plt.subplots()
plt.plot(wass_list,'-+',label = "Wasserstein distance",linewidth = 1.5)
plt.plot(approx, '-*',label = 'loss value',linewidth = 1)
plt.title("loss vs. wasserstein distance")
plt.legend()
plt.show()
