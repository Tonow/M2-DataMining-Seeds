'''
Fichier:	 python 3
Ecrit par :	 Tonow
Le :		 Date
Sujet:		 TODO
'''

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import sklearn
from sklearn import cluster

# %matplotlib gtk

faithful = pd.read_csv('seeds_dataset_classe-String.csv')
print(faithful.head())
print(faithful.describe())

faithful.columns = ['area',
                    'perimeter',
                    'compactness',
                    'length_of_kerne',
                    'width_of_kernel',
                    'asymmetry_coefficient',
                    'length_of_kernel_groove',
                    'classe',]

# plt.scatter(faithful.length_of_kernel_groove, faithful.length_of_kerne, faithful.perimeter)

#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')



# plt.title('length_of_kerne en fonction de length_of_kernel_groove Data Scatterplot')
# plt.xlabel('length_of_kernel_groove')
# plt.ylabel('length_of_kerne')
# plt.label('length_of_kerne')




fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax = Axes3D(fig)

Xc = faithful.length_of_kernel_groove
Yc = faithful.length_of_kerne
Zc = faithful.perimeter

for X, Y, Z, clas in (Xc, Yc, Zc, faithful.classe):
    if clas == 'classe_1':
        (c, m) = ('red', 'o')
    elif clas == 'classe_2':
        (c, m) = ('blue', '^')
    else:
        (c, m) = ('green', (5, 2))
    import pdb; pdb.set_trace()
    ax.scatter(X, Y, Z, c=c, marker=m)


plt.show()













#
#
# # Fixing random state for reproducibility
# np.random.seed(19680801)
#
#
# def randrange(n, vmin, vmax):
#     '''
#     Helper function to make an array of random numbers having shape (n, )
#     with each number distributed Uniform(vmin, vmax).
#     '''
#     return (vmax - vmin)*np.random.rand(n) + vmin
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# n = 100
#
# # For each set of style and range settings, plot n random points in the box
# # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
# for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
#     xs = randrange(n, 23, 32)
#     ys = randrange(n, 0, 100)
#     zs = randrange(n, zlow, zhigh)
#     ax.scatter(xs, ys, zs, c=c, marker=m)
#
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
#
# plt.show()
