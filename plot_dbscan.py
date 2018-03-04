# -*- coding: utf-8 -*-
"""
===========================
DBSCAN clustering algorithm
===========================

Finds core samples of high density and expands clusters from them.

"""
print(__doc__)

import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


# #############################################################################
# Generate sample data
# serie test
# centers = [[1, 1], [-1, -1], [1, -1]]
# X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
#                             random_state=0)
#
# X = StandardScaler().fit_transform(X)





dataframe = pd.read_csv('seeds_dataset_classe-String.csv')
i = 0
for col in dataframe.columns:
    print(f'colonne {i}: {col}')
    i = i+1


df = pd.read_csv('seeds_dataset.csv')
# col_utile = df.as_matrix(columns=['area', 'perimeter', 'length of kerne', 'length of kernel groove'])
col_utile = df.as_matrix(columns=['perimeter', 'length of kerne', 'length of kernel groove'])


df.columns = ['area',
              'perimeter',
              'compactness',
              'length_of_kerne',
              'width_of_kernel',
              'asymmetry_coefficient',
              'length_of_kernel_groove',
              'classe',]

labels_true = df.classe
X = col_utile

# #############################################################################
# Generate sample data

# #############################################################################
# Compute DBSCAN
def db_scan(X, labels_true):
    best_adjusted_rand_index = 0
    best_v_measure_score = 0
    for eps in np.arange(0.1, 4, 0.001):
        for min_samples in range(2, 12):
            # print(f"esp: {eps}  |  min point {min_samples}")
            # db = DBSCAN(eps=0.3, min_samples=10).fit(X)
            db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
            # import pdb; pdb.set_trace()

            # labels = DBSCAN().fit_predict(X)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_

            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

            adjusted_rand_index = metrics.adjusted_rand_score(labels_true, labels)
            v_measure_score = metrics.v_measure_score(labels_true, labels)
            best_adjusted_index = adjusted_rand_index > best_adjusted_rand_index
            best_v_measure = v_measure_score > best_v_measure_score
            if best_adjusted_index and best_v_measure:
                print("#"*30)
                print(f"esp: {round(eps, 3)}  |  min point {min_samples}")
                # print("Adjusted Rand Index: %0.3f"
                #       % metrics.adjusted_rand_score(labels_true, labels))
                print(f"Adjusted Rand Index: {round(adjusted_rand_index, 3)}")
                print(f"V-measure: {round(v_measure_score, 3)}")
                best_labels = labels
                best_core_samples_mask = core_samples_mask
                best_n_clusters_ = n_clusters_
                best_adjusted_rand_index = adjusted_rand_index
                best_v_measure_score = v_measure_score
                best_eps = eps
                best_min_samples = min_samples

                homogeneity = metrics.homogeneity_score(labels_true, labels)
                completeness = metrics.completeness_score(labels_true, labels)

                adjusted_mutual_information = metrics.adjusted_mutual_info_score(labels_true, labels)
                silhouette_coefficient = metrics.silhouette_score(X, labels)

    print("#"*50 +"\n")
    print("#"*15 + "  Rapport:  " + "#"*15 + "\n")
    print(f"Nombre de clusters estimer: {best_n_clusters_}")
    print(f"esp: {round(best_eps, 3)}  |  min point {best_min_samples} \n")
    print(f"Homogeneity: {round(homogeneity, 3)}")
    print(f"Completeness: {round(completeness, 3)}")
    print(f"Adjusted Mutual Information: {round(adjusted_mutual_information, 3)}")
    print(f"Silhouette Coefficient: {round(silhouette_coefficient, 3)}")
    print(f"Adjusted Rand Index: {round(best_adjusted_rand_index, 3)}")
    print(f"V-measure: {round(best_v_measure_score, 3)}")
    print("#"*50 +"\n")
    return (best_labels, best_core_samples_mask, best_n_clusters_)

(labels, core_samples_mask, n_clusters_) = db_scan(X, labels_true)

# #############################################################################
# Plot result
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# Black removed and is used for noise instead.
unique_labels = set(labels)
unique, counts = np.unique(labels, return_counts=True)
nb_val_in_cluster = dict(zip(unique, counts))

for cluster, nombre_element in nb_val_in_cluster.items():
    if cluster == -1:
        cluster = 'noise'
    print(f" Le cluster {cluster} a {nombre_element} element")

colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Noir pour noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

# serie test
    # xy = X[class_member_mask & core_samples_mask]
    # ax.scatter(xy[:, 0], xy[:, 1], c=tuple(col), marker='o')
    #
    # xy = X[class_member_mask & ~core_samples_mask]
    # ax.scatter(xy[:, 0], xy[:, 1], c=tuple(col), marker='+', s=40)

# + area
    # xyz = X[class_member_mask & core_samples_mask]
    # ax.scatter(xyz[:, 1], xyz[:, 2], xyz[:, 3], c=tuple(col), marker='o')
    #
    # xyz = X[class_member_mask & ~core_samples_mask]
    # ax.scatter(xyz[:, 1], xyz[:, 2], xyz[:, 3], c=tuple(col), marker='+', s=40)


# trois colonne
    xyz = X[class_member_mask & core_samples_mask]
    ax.scatter(xyz[:, 1], xyz[:, 2], xyz[:, 0], c=tuple(col), marker='o')

    xyz = X[class_member_mask & ~core_samples_mask]
    # lst = list(tuple(col))
    # lst[0] = 1
    ax.scatter(xyz[:, 1], xyz[:, 2], xyz[:, 0], c=tuple(col), marker='+', s=180)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()
