#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 10:59:09 2023

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
#parameters

n = 100  # nb bins
n_target = 20  # nb target distributions
def normal_pdf(x,mean,sigma):
    m = x.shape[0]
    pdf_val = np.zeros((m,1))
    for i in range(m):
        pdf_val[i] = 1/(sigma * np.sqrt(2 * np.pi)) *np.exp( - (x[i] - mean)**2 /(2 * sigma**2))
    return pdf_val
                                                            
def wass_comp(m1,m2,sigma1,sigma2):
    run1 = sigma1**2
    run2 = sigma2**2
    term1 = np.linalg.norm(m1-m2)**2
    term2 = np.matmul(np.sqrt(run1),np.matmul(run2,np.sqrt(run1)))
    return term1 + np.trace(run1+run2-2*np.sqrt(term2))



#sam1 = np.random.normal(30,5,n)
#sam2 = np.random.normal(60,10,n)
#sam1 = np.sort(sam1)
#sam2 = np.sort(sam2)

#C /= C.max()
alpha_a = gauss(n,m=30, s=5)
beta_b = gauss(n,m=60, s = 10)

#alpha_a,ignore= np.histogram(sam1,bins=n) 
#alpha_a = alpha_a/alpha_a.sum()
#beta_b,ignore = np.histogram(sam2,bins = n)
#beta_b = beta_b/beta_b.sum()
cdf_a = np.cumsum(alpha_a)
cdf_b = np.cumsum(beta_b)
alpha = norm.ppf(cdf_a,loc=30,scale =5)
data_true = alpha[~np.isnan(alpha)]
alpha = np.nan_to_num(alpha, copy=True, nan= data_true.max(), posinf=data_true.max(), neginf=data_true.min())
beta = norm.ppf(cdf_b,loc=60,scale = 10)
data_true = beta[~np.isnan(beta)]
beta = np.nan_to_num(beta, copy=True, nan= data_true.max(), posinf=data_true.max(), neginf= data_true.min())
C = ot.dist(alpha.reshape((n,1)),beta.reshape((n,1)),'euclidean',p=2)
#alpha_a = normal_pdf(alpha,3,1.5)
#beta_b = normal_pdf(beta,6,1.2)
#alpha_a = alpha_a/np.sum(alpha_a)
#alpha_b = beta_b/np.sum(beta_b)
pl.figure(2, figsize=(6.4, 3))
pl.plot(alpha,alpha_a, 'b', label='Source distribution')
pl.plot(alpha,beta_b, 'r', label='Target distribution')
pl.legend()
pl.figure(2, figsize=(5, 5))
ot.plot.plot1D_mat(alpha_a, beta_b, C, 'Cost matrix M')

P = ot.emd(alpha_a,beta_b,C)
wass2 = ot.emd2(alpha_a,beta_b,C)
pl.figure(3, figsize=(5, 5))
ot.plot.plot1D_mat(alpha_a, beta_b,P, 'OT matrix G0')

m_1 = np.array([30])
m_2 = np.array([60])
sigma_1 = np.array([5])
sigma_2 = np.array([10])
sigma_1 = sigma_1.reshape((1,1))
sigma_2 = sigma_2.reshape((1,1))

result = wass_comp(m_1,m_2,sigma_1,sigma_2)
wass_simu = np.sqrt(result)