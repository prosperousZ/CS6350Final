
#OT distances in 1D

#Shows how to compute multiple Wasserstein and Sinkhorn with two different
#ground metrics and plot their values for different distributions.

import numpy as np
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss
from scipy.stats import norm
#parameters



n = 100  # nb bins
n_target = 20  # nb target distributions


#bin positions
x = np.arange(n, dtype=np.float64)

lst_m = np.linspace(20, 90, n_target)

# 1-D Gaussian distributions
# a,b are p.d.f values.
a = gauss(n, m=20, s=5)  # n: # of sampled points, m= mean, s= std

b = np.array([1/n for i in range(n)])

M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)),  'euclidean')

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

# Verification starts here.

# B = np.zeros((n, n_target))

# for i, m in enumerate(lst_m):
#     B[:, i] = gauss(n, m=m, s=5)

# # loss matrix and normalization

# M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)),  'euclidean')
# M /= M.max() * 0.1
# M2 = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)), 'sqeuclidean')
# M2 /= M2.max() * 0.1

# #plot the distributions

# pl.figure(1)
# pl.subplot(2, 1, 1)
# pl.plot(x, a, 'r', label='Source distribution')
# pl.title('Source distribution')
# pl.subplot(2, 1, 2)
# for i in range(n_target):
#     pl.plot(x, B[:, i], 'b', alpha=i / n_target)
# pl.plot(x, B[:, -1], 'b', label='Target distributions')
# pl.title('Target distributions')
# pl.tight_layout()

# # Compute EMD for the different losses
# # Compute and plot distributions and loss matrix


# #************This is where compute W1 Wasserstein distance
# d_emd = ot.emd2(a, B, M)  # direct computation of OT loss
# #********end computer W1**********

# #********This is where computer W2, M2 is square root of M1********
# d_emd2 = ot.emd2(a, B, M2)  # direct computation of OT loss with metric M2
# #*************end computer W2**************

# d_tv = [np.sum(abs(a - B[:, i])) for i in range(n_target)]

# pl.figure(2)
# pl.subplot(2, 1, 1)
# pl.plot(x, a, 'r', label='Source distribution')
# pl.title('Distributions')
# for i in range(n_target):
#     pl.plot(x, B[:, i], 'b', alpha=i / n_target)
# pl.plot(x, B[:, -1], 'b', label='Target distributions')
# pl.ylim((-.01, 0.13))
# pl.xticks(())
# pl.legend()
# pl.subplot(2, 1, 2)
# pl.plot(d_emd, label='Euclidean OT')
# pl.plot(d_emd2, label='Squared Euclidean OT')
# pl.plot(d_tv, label='Total Variation (TV)')
# pl.xlabel('Displacement')
# pl.title('Divergences')
# pl.legend()


# #Try to plot Euclidean distance vs Was distance
# pl.figure(3)
# pl.clf()
# pl.plot(M)
# pl.plot(d_emd, label='Euclidean OT')

# pl.legend()


# pl.show()
