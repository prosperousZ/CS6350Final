
#OT distances in 1D

#Shows how to compute multiple Wasserstein and Sinkhorn with two different
#ground metrics and plot their values for different distributions.

import numpy as np
import matplotlib.pylab as pl
import ot
import math
from ot.datasets import make_1D_gauss as gauss

#parameters

n = 100  # nb bins
n_target = 20  # nb target distributions


# bin positions
x = np.arange(n, dtype=np.float64)

lst_m = np.linspace(20, 90, n_target)

# Gaussian distributions
a = gauss(n, m=20, s=5)  # m= mean, s= std

B = np.zeros((n, n_target))

for i, m in enumerate(lst_m):
    B[:, i] = gauss(n, m=m, s=5)

# loss matrix and normalization

M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)),  'euclidean')
M /= M.max() * 0.1
M2 = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)), 'sqeuclidean')
M2 /= M2.max() * 0.1

#plot the distributions

pl.figure(1)
pl.subplot(2, 1, 1)
pl.plot(x, a, 'r', label='Source distribution')
pl.title('Source distribution')
pl.subplot(2, 1, 2)
for i in range(n_target):
    pl.plot(x, B[:, i], 'b', alpha=i / n_target)
pl.plot(x, B[:, -1], 'b', label='Target distributions')
pl.title('Target distributions')
pl.tight_layout()





#Euclidean distance


d_emd = ot.emd2(a, B,M)
#********This is where computer W2, M2 is square root of M1********
d_emd2 = ot.emd2(a, B, M2) 
#*************end computer W2**************

d_tv = [np.sum(abs(a - B[:, i])) for i in range(n_target)]


#Try to plot Euclidean distance vs Was distance
pl.figure(2)
pl.clf()
pl.plot(d_emd, label = 'Euclidean distance')
pl.plot(d_emd2, label='Wasserstein distance')

pl.xlabel('Displacement')

pl.legend()


pl.show()
