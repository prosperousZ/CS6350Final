#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 15:03:11 2023

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
import cv2


gray_img = cv2.imread('flower1.jpg', cv2.IMREAD_GRAYSCALE)
#cv2.imshow('flower1',gray_img)
gray_img2 = cv2.imread('goldgate.jpg', cv2.IMREAD_GRAYSCALE)

#cv2.imshow('flower2',gray_img)
hist,bins = np.histogram(gray_img,256,[1,256])
hist1,bins = np.histogram(gray_img2,256,[1,256])
plt.hist(gray_img.ravel(),256,[1,256],label='flower')
plt.title('Histogram for gray scale picture')
plt.hist(gray_img2.ravel(),256,[1,256],label = 'sf_gate')
plt.title('Histogram for gray scale picture2')
plt.legend()
plt.show()

hist = hist/hist.sum()
hist1 = hist1/hist1.sum()

bins = bins[0:256]
n = bins.shape[0]

M = ot.dist(bins.reshape((n, 1)), bins.reshape((n, 1)))
M /= M.max()
pl.figure(2, figsize=(5, 5))
ot.plot.plot1D_mat(hist, hist1, M, 'Cost matrix M')
# use fast 1D solver
G0 = ot.emd(hist, hist1,M)

# Equivalent to
# G0 = ot.emd(a, b, M)
dist = ot.emd2(hist,hist1,M)
pl.figure(3, figsize=(5, 5))
ot.plot.plot1D_mat(hist, hist1, G0, 'OT matrix G0')

# while True:
#     k = cv2.waitKey(0) & 0xFF     
#     if k == 27: break             # ESC key to exit 
# cv2.destroyAllWindows()
