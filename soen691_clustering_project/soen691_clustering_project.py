# -*- coding: utf-8 -*-

"""Main module."""
from pyclustering.utils import read_sample
from pyclustering.samples.definitions import FCPS_SAMPLES
from pyclustering.cluster import cluster_visualizer
from pyclustering.cluster.cure import cure as p_cure
from cure import Cure

from collections import defaultdict
import glob, os


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


if __name__ == '__main__':
    all_data = read_all_2dshaped('./data/2d/shaped/', '.dat')
    data_name = 'spiral'
    data = []
    for cluster in all_data[data_name].values():
        data.extend(cluster)

    len_data = [len(cluster) for cluster in all_data[data_name].values()]

    number_of_clusters = len(all_data[data_name].keys())
    alpha = 0.15
    c = 10

    cure = Cure(data, number_of_clusters, alpha, c)
    cure.clustering()

    len_clusters = sorted([len(cluster) for cluster in cure.get_clusters()])

    print('------------------------------------------------------------------------')

    cure_instance = p_cure(data, number_of_clusters, compression=alpha, number_represent_points=c, ccore=False)
    cure_instance.process()

    len_clusters_2 = sorted([len(cluster) for cluster in cure.get_clusters()])

    print('------------------------------------------------------------------------')
    print(sorted(len_data), len_clusters, len_clusters_2)

    visualizer = cluster_visualizer()
    visualizer.append_clusters([cluster.indexes for cluster in cure.get_clusters()], data)
    visualizer.show()

    visualizer2 = cluster_visualizer()
    visualizer2.append_clusters([cluster for cluster in cure_instance.get_clusters()], data)
    visualizer2.show()
