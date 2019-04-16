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

from sklearn.datasets.samples_generator import make_circles
import numpy as np

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                             Helper Functions                                 """
"""                                                                              """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def read_all_data_labeled(directory, extension, pattern='*', dimension='2d', all_data_dict=None):
    data_dict = defaultdict(dict) if all_data_dict is None else all_data_dict
    for path in glob.glob(directory + pattern + extension):
        data_dict[dimension][os.path.basename(path).replace(extension, '')] = read_data_labeled(path)
    return data_dict


def read_data_labeled(filename):
    with open(filename) as f:
        data_dict = defaultdict(list)
        for line in f.readlines():
            split = line.split('\t')
            size = len(split)
            data_dict[split[size - 1].strip()].append([float(coord) for coord in split[:size - 1]])
        return data_dict


def get_data(data_name, data_dict):
    data = []
    for cluster in data_dict[data_name].values():
        data.extend(cluster)
    return data


def compare_clustering(data, clusters):
    data_to_analyze = {}
    clustering_to_analyze = {}
    for label, points_list in enumerate(sorted([sorted(cluster) for cluster in data.values()])):
        for point in sorted(points_list):
            data_to_analyze[tuple(point)] = int(label)
    sorted_clusters = sorted([sorted(cluster.points) for cluster in clusters]) if not type(
        clusters[0]) == list else sorted([sorted(cluster) for cluster in clusters])
    for label, cluster in enumerate(sorted_clusters):
        for point in cluster:
            clustering_to_analyze[tuple(point)] = int(label)

    total = len(data_to_analyze)
    miss = 0.0
    for point, label in clustering_to_analyze.items():
        if data_to_analyze[point] != label:
            miss += 1

    return (1 - miss / total) * 100


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                               Main Method                                    """
"""                                                                              """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

if __name__ == '__main__':
    all_data = read_all_data_labeled('./data/3d/shaped/', extension='.dat', dimension='3d',
                                     all_data_dict=read_all_data_labeled('./data/2d/shaped/', '.dat'))

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """                               Non-Spherical 2D                               """
    """                                                                              """
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    results_nonspherical_2d = []

    dataset_name = 'spiral'
    dataset = get_data(dataset_name, all_data['2d'])
    real_clusters = all_data['2d'][dataset_name]
    len_data = sorted([len(cluster) for cluster in all_data['2d'][dataset_name].values()])
    n_clusters = len(all_data['2d'][dataset_name].keys())
    print('---------{} DATASET---------'.format(dataset_name.upper()), len_data, n_clusters)
    df_list = dataset

    kmeans = KMeans(df_list, n_clusters)
    kmeans.clustering()
    kmeans_clusters = []
    for indexes in kmeans.get_indexes():
        cluster = []
        for index in indexes:
            cluster.append(df_list[index])
        kmeans_clusters.append(cluster)

    hac = HierarchicalAgglomerative(df_list, n_clusters)
    hac.clustering()

    cure = Cure(df_list, n_clusters, 0.15, 10)
    cure.clustering()

    bfr = BFR(data=df_list, k=n_clusters)
    bfr.cluster_noPart()

    results_nonspherical_2d.append([compare_clustering(real_clusters, kmeans_clusters),
                                    compare_clustering(real_clusters, hac.get_clusters()),
                                    compare_clustering(real_clusters, cure.get_clusters()),
                                    compare_clustering(real_clusters, bfr.get_clusters())])

    visualizer = ClusteringVisualizer(number_canvas=4, number_columns=2, number_clusters=n_clusters,
                                      titles=['KMEANS', 'HAC', 'CURE', 'BFR'], fig_title='Path-based2: SPIRAL')
    visualizer.add_clustering(kmeans.get_indexes(), df_list, canvas=0)
    visualizer.add_clustering(hac.get_indexes(), df_list, canvas=1)
    visualizer.add_clustering(cure.get_indexes(), df_list, canvas=2)
    visualizer.add_clustering(bfr.get_indexes(), df_list, canvas=3)
    visualizer.plot(invisible_axis=True)

    dataset_name = 'jain'
    dataset = get_data(dataset_name, all_data['2d'])
    real_clusters = all_data['2d'][dataset_name]
    len_data = sorted([len(cluster) for cluster in all_data['2d'][dataset_name].values()])
    n_clusters = len(all_data['2d'][dataset_name].keys())
    print('---------{} DATASET---------'.format(dataset_name.upper()), len_data, n_clusters)
    df_list = dataset

    kmeans = KMeans(df_list, n_clusters)
    kmeans.clustering()
    kmeans_clusters = []
    for indexes in kmeans.get_indexes():
        cluster = []
        for index in indexes:
            cluster.append(df_list[index])
        kmeans_clusters.append(cluster)

    hac = HierarchicalAgglomerative(df_list, n_clusters)
    hac.clustering()

    cure = Cure(df_list, n_clusters, 0.3, 5)
    cure.clustering()

    bfr = BFR(data=df_list, k=n_clusters)
    bfr.cluster_noPart()

    results_nonspherical_2d.append([compare_clustering(real_clusters, kmeans_clusters),
                                    compare_clustering(real_clusters, hac.get_clusters()),
                                    compare_clustering(real_clusters, cure.get_clusters()),
                                    compare_clustering(real_clusters, bfr.get_clusters())])

    visualizer = ClusteringVisualizer(number_canvas=4, number_columns=2, number_clusters=n_clusters,
                                      titles=['KMEANS', 'HAC', 'CURE', 'BFR'], fig_title='Jain')
    visualizer.add_clustering(kmeans.get_indexes(), df_list, canvas=0)
    visualizer.add_clustering(hac.get_indexes(), df_list, canvas=1)
    visualizer.add_clustering(cure.get_indexes(), df_list, canvas=2)
    visualizer.add_clustering(bfr.get_indexes(), df_list, canvas=3)
    visualizer.plot(invisible_axis=True)

    circles = make_circles(factor=0.5, noise=0.05, n_samples=700)
    circles_clusters = defaultdict(list)
    real_clusters = defaultdict(list)
    for k, v in list(zip(circles[0], circles[1])):
        circles_clusters[v].append(k)
        real_clusters[v].append(list(k))
    circles_clusters = list(circles_clusters.values())
    df = np.concatenate(circles_clusters)
    df_list = df.tolist()
    n_clusters = 2
    # dataset_name = 'r15'
    # dataset = get_data(dataset_name, all_data['2d'])
    # real_clusters = all_data['2d'][dataset_name]
    # len_data = sorted([len(cluster) for cluster in all_data['2d'][dataset_name].values()])
    # n_clusters = len(all_data['2d'][dataset_name].keys())
    # print('---------{} DATASET---------'.format(dataset_name.upper()), len_data, n_clusters)
    # df_list = dataset

    kmeans = KMeans(df_list, n_clusters)
    kmeans.clustering()
    kmeans_clusters = []
    for indexes in kmeans.get_indexes():
        cluster = []
        for index in indexes:
            cluster.append(df_list[index])
        kmeans_clusters.append(cluster)

    hac = HierarchicalAgglomerative(df_list, n_clusters)
    hac.clustering()

    cure = Cure(df_list, n_clusters, 0.1, 10)
    cure.clustering()

    bfr = BFR(data=df_list, k=n_clusters)
    bfr.cluster_noPart()

    results_nonspherical_2d.append([compare_clustering(real_clusters, kmeans_clusters),
                                    compare_clustering(real_clusters, hac.get_clusters()),
                                    compare_clustering(real_clusters, cure.get_clusters()),
                                    compare_clustering(real_clusters, bfr.get_clusters())])

    visualizer = ClusteringVisualizer(number_canvas=4, number_columns=2, number_clusters=n_clusters,
                                      titles=['KMEANS', 'HAC', 'CURE', 'BFR'], fig_title='Circles')
    visualizer.add_clustering(kmeans.get_indexes(), df_list, canvas=0)
    visualizer.add_clustering(hac.get_indexes(), df_list, canvas=1)
    visualizer.add_clustering(cure.get_indexes(), df_list, canvas=2)
    visualizer.add_clustering(bfr.get_indexes(), df_list, canvas=3)
    visualizer.plot(invisible_axis=True)

    dataset_name = 'pathbased'
    dataset = get_data(dataset_name, all_data['2d'])
    real_clusters = all_data['2d'][dataset_name]
    len_data = sorted([len(cluster) for cluster in all_data['2d'][dataset_name].values()])
    n_clusters = len(all_data['2d'][dataset_name].keys())
    print('---------{} DATASET---------'.format(dataset_name.upper()), len_data, n_clusters)
    df_list = dataset

    kmeans = KMeans(df_list, n_clusters)
    kmeans.clustering()
    kmeans_clusters = []
    for indexes in kmeans.get_indexes():
        cluster = []
        for index in indexes:
            cluster.append(df_list[index])
        kmeans_clusters.append(cluster)

    hac = HierarchicalAgglomerative(df_list, n_clusters)
    hac.clustering()

    cure = Cure(df_list, n_clusters, 0.1, 10)
    cure.clustering()

    bfr = BFR(data=df_list, k=n_clusters)
    bfr.cluster_noPart()

    results_nonspherical_2d.append([compare_clustering(real_clusters, kmeans_clusters),
                                    compare_clustering(real_clusters, hac.get_clusters()),
                                    compare_clustering(real_clusters, cure.get_clusters()),
                                    compare_clustering(real_clusters, bfr.get_clusters())])

    visualizer = ClusteringVisualizer(number_canvas=4, number_columns=2, number_clusters=n_clusters,
                                      titles=['KMEANS', 'HAC', 'CURE', 'BFR'], fig_title='Path-based1')
    visualizer.add_clustering(kmeans.get_indexes(), df_list, canvas=0)
    visualizer.add_clustering(hac.get_indexes(), df_list, canvas=1)
    visualizer.add_clustering(cure.get_indexes(), canvas=2)
    visualizer.add_clustering(bfr.get_indexes(), df_list, canvas=3)
    visualizer.plot(invisible_axis=True)

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    """                               Non-Spherical 3D                               """
    """                                                                              """
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    dataset_name = 'fcps_atom'
    dataset = get_data(dataset_name, all_data['3d'])
    real_clusters = all_data['3d'][dataset_name]
    len_data = sorted([len(cluster) for cluster in all_data['3d'][dataset_name].values()])
    n_clusters = len(all_data['3d'][dataset_name].keys())
    print('---------{} DATASET---------'.format(dataset_name.upper()), len_data, n_clusters)
    df_list = dataset

    kmeans = KMeans(df_list, n_clusters)
    kmeans.clustering()
    kmeans_clusters = []
    for indexes in kmeans.get_indexes():
        cluster = []
        for index in indexes:
            cluster.append(df_list[index])
        kmeans_clusters.append(cluster)

    hac = HierarchicalAgglomerative(df_list, n_clusters)
    hac.clustering()

    cure = Cure(df_list, n_clusters, 0.1, 10)
    cure.clustering()

    bfr = BFR(data=df_list, k=n_clusters)
    bfr.cluster_noPart()

    print([compare_clustering(real_clusters, kmeans_clusters),
          compare_clustering(real_clusters, hac.get_clusters()),
          compare_clustering(real_clusters, cure.get_clusters()),
          compare_clustering(real_clusters, bfr.get_clusters())])

    visualizer = ClusteringVisualizer(number_canvas=4, number_columns=2, number_clusters=n_clusters,
                                      titles=['KMEANS', 'HAC', 'CURE', 'BFR'], fig_title='FCPS Atom')
    visualizer.add_clustering(kmeans.get_indexes(), df_list, canvas=0)
    # visualizer.add_clustering(hac.get_indexes(), df_list, canvas=1)
    # visualizer.add_clustering(cure.get_indexes(), df_list, canvas=2)
    # visualizer.add_clustering(bfr.get_indexes(), df_list, canvas=3)
    visualizer.plot(invisible_axis=True)

    dataset_name = 'fcps_chainlink'
    dataset = get_data(dataset_name, all_data['3d'])
    real_clusters = all_data['3d'][dataset_name]
    len_data = sorted([len(cluster) for cluster in all_data['3d'][dataset_name].values()])
    n_clusters = len(all_data['3d'][dataset_name].keys())
    print('---------{} DATASET---------'.format(dataset_name.upper()), len_data, n_clusters)
    df_list = dataset

    kmeans = KMeans(df_list, n_clusters)
    kmeans.clustering()
    kmeans_clusters = []
    for indexes in kmeans.get_indexes():
        cluster = []
        for index in indexes:
            cluster.append(df_list[index])
        kmeans_clusters.append(cluster)

    hac = HierarchicalAgglomerative(df_list, n_clusters)
    hac.clustering()

    cure = Cure(df_list, n_clusters, 0.1, 10)
    cure.clustering()

    bfr = BFR(data=df_list, k=n_clusters)
    bfr.cluster_noPart()

    print([compare_clustering(real_clusters, kmeans_clusters),
           compare_clustering(real_clusters, hac.get_clusters()),
           compare_clustering(real_clusters, cure.get_clusters()),
           compare_clustering(real_clusters, bfr.get_clusters())])

    visualizer = ClusteringVisualizer(number_canvas=4, number_columns=2, number_clusters=n_clusters,
                                      titles=['KMEANS', 'HAC', 'CURE', 'BFR'], fig_title='FCPS Chainlink')
    visualizer.add_clustering(kmeans.get_indexes(), df_list, canvas=0)
    visualizer.add_clustering(hac.get_indexes(), df_list, canvas=1)
    visualizer.add_clustering(cure.get_indexes(), df_list, canvas=2)
    visualizer.add_clustering(bfr.get_indexes(), df_list, canvas=3)
    visualizer.plot(invisible_axis=True)
