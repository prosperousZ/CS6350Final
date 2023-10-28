#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 15:03:33 2023

@author: ivanyang
"""

import numpy as np
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss
from scipy.stats import norm
from scipy.stats import uniform
from scipy.linalg import expm

def sinkhorn_comp(a,b,C,lambda0,max_iter):
    n = C.shape[0]
    m = C.shape[1]
    a = a.reshape((n,))
    b = b.reshape((m,))
    v = np.ones((m,))
    K = expm(-(lambda0)*C)
    u = np.ones((n,1))
    for i in range(max_iter):
        u = np.divide(a,np.matmul(K,v))
        
        v = np.divide(b,np.matmul(np.transpose(K),u))
        
    P = np.matmul(np.diag(u),np.matmul(K,np.diag(v)))
    return P


n = 100  # nb bins
n_target = 20  # nb target distributions


#bin positions
x = np.arange(n, dtype=np.float64)

lst_m = np.linspace(20, 90, n_target)

# 1-D Gaussian distributions
# a,b are p.d.f values.
a = gauss(n, m=20, s=5)  # n: # of sampled points, m= mean, s= std

b = np.array([1/n for i in range(n)])

M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)),  'euclidean', p=2)

M /= M.max()

pl.figure(1, figsize=(6.4, 3))
pl.plot(x, a, 'b', label='Source distribution')
pl.plot(x, b, 'r', label='Target distribution')
pl.legend()
pl.figure(2, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, M, 'Cost matrix M')

# use fast 1D solver
G0 = ot.emd_1d(x, x, a, b)
wass1 = ot.emd2(a,b,M)
# Equivalent to
# G0 = ot.emd(a, b, M)

pl.figure(3, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, G0, 'OT matrix G0')


lambd = 0.005
Gs = ot.sinkhorn(a, b, M, lambd, verbose=True)

pl.figure(4, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, Gs, 'OT matrix Sinkhorn')

pl.show()

print('sinkhorn distance is',np.trace(np.matmul(M,np.transpose(Gs))))
G_comp = sinkhorn_comp(a,b,M,lambd,13)
sinkhorn_simu = np.trace(np.matmul(M,np.transpose(G_comp)))
print(sinkhorn_simu)
ot.plot.plot1D_mat(a, b, G_comp, 'sinkhorn matrix by algorithm')