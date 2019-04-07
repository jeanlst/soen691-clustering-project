# -*- coding: utf-8 -*-

"""Main module."""

import glob
import os
from collections import defaultdict

from cure import Cure
from kmeans import KMeans
from hierarchical_agglomerative import HierarchicalAgglomerative
from visualizer import ClusteringVisualizer

import numpy as np


def read_2dshaped(filename):
    with open(filename) as f:
        data_dict = defaultdict(list)
        for line in f.readlines():
            split = line.split('\t')
            data_dict[split[2].strip()].append([float(split[0]), float(split[1])])
        return data_dict


def read_all_2dshaped(directory, extension, pattern='*'):
    data_dict = {}
    for path in glob.glob(directory + pattern + extension):
        data_dict[os.path.basename(path).strip(extension)] = read_2dshaped(path)
    return data_dict


def plot_clustering(fig, data, clusters, x_label, y_label, title, ax=None):
    if not ax:
        ax = fig.add_subplot(111)

    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)


if __name__ == '__main__':
    all_data = read_all_2dshaped('./data/2d/shaped/', '.dat')
    data_name = 'spiral'
    data = []
    for cluster in all_data[data_name].values():
        data.extend(cluster)

    len_data = [len(cluster) for cluster in all_data[data_name].values()]

    number_of_clusters = len(all_data[data_name].keys())

    X_1 = np.random.multivariate_normal(mean=[4, 0], cov=[[1, 0], [0, 1]], size=75)
    X_2 = np.random.multivariate_normal(mean=[6, 6], cov=[[2, 0], [0, 2]], size=250)
    X_3 = np.random.multivariate_normal(mean=[1, 5], cov=[[1, 0], [0, 2]], size=20)
    df = np.concatenate([X_1, X_2, X_3])

    kmeans = KMeans(df.tolist(), 3)
    kmeans.clustering()

    hac = HierarchicalAgglomerative(df.tolist(), 3)
    hac.clustering()

    cure = Cure(df.tolist(), 3, 0.3, 10)
    cure.clustering()

    visualizer = ClusteringVisualizer(number_canvas=3, number_columns=1, titles=['KMeans', 'HAC', 'CURE'])
    visualizer.add_clustering(kmeans.get_indexes(), df.tolist(), canvas=0)
    visualizer.add_clustering(hac.get_indexes(), df.tolist(), canvas=1)
    visualizer.add_clustering(cure.get_indexes(), df.tolist(), canvas=2)
    visualizer.plot(invisible_axis=True)
