# -*- coding: utf-8 -*-

"""Main module."""

import glob
import os
from collections import defaultdict

from cure import Cure
from kmeans import KMeans
from bfr import BFR
from hierarchical_agglomerative import HierarchicalAgglomerative
from visualizer import ClusteringVisualizer

import numpy as np
from sklearn.datasets.samples_generator import (make_blobs,
                                                make_circles,
                                                make_moons)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                             Helper Functions                                 """
"""                                                                              """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


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


def get_data(data_name):
    data = []
    for cluster in all_data[data_name].values():
        data.extend(cluster)
    return data


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                               Main Method                                    """
"""                                                                              """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

if __name__ == '__main__':
    all_data = read_all_2dshaped('./data/2d/shaped/', '.dat')
    dataset = get_data('spiral')

    len_data = [len(cluster) for cluster in all_data['spiral'].values()]
    number_of_clusters = len(all_data['spiral'].keys())

    from pyclustering.samples.definitions import FCPS_SAMPLES
    from pyclustering.utils import read_sample

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """                               Spherical 2D                                   """
    """                                                                              """
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    df_list = read_sample(FCPS_SAMPLES.SAMPLE_LSUN)
    n_clusters = 3

    kmeans = KMeans(df_list, n_clusters)
    kmeans.clustering()

    hac = HierarchicalAgglomerative(df_list, n_clusters)
    hac.clustering()

    cure = Cure(df_list, n_clusters, 0.2, 10)
    cure.clustering()

    bfr = BFR(data=df_list, k=n_clusters)
    bfr.cluster_noPart()

    visualizer = ClusteringVisualizer(number_canvas=4, number_columns=2, number_clusters=number_of_clusters,
                                      titles=['KMEANS', 'HAC', 'CURE', 'BFR'])
    visualizer.add_clustering(kmeans.get_indexes(), df_list, canvas=0)
    visualizer.add_clustering(hac.get_indexes(), df_list, canvas=1)
    visualizer.add_clustering(cure.get_indexes(), df_list, canvas=2)
    visualizer.add_clustering(bfr.get_indexes(), df_list, canvas=3)
    visualizer.plot(invisible_axis=True)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """                               Non-Spherical 2D                               """
    """                                                                              """
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
