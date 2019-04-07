# -*- coding: utf-8 -*-

""" Hierarchical Agglomerative class."""
from cluster import Cluster
from clustering import Clustering


class HierarchicalAgglomerative(Clustering):
    def __init__(self, data, number_of_clusters):
        self.__data = data
        self.__number_of_clusters = number_of_clusters
        self.__clusters = [Cluster(point, index) for index, point in enumerate(self.__data)]
        self.__dimension = len(data[0]) if len(data) > 0 else 0

        self.__validate_arguments()

    def clustering(self):
        while len(self.__clusters) > self.__number_of_clusters:
            self.__merge_clusters()

    def __merge_clusters(self):
        min_distance = float('inf')
        closest_clusters = None

        for idx_a, cluster_a in enumerate(self.__clusters):
            for cluster_b in self.__clusters[idx_a+1:]:
                distance = cluster_a.distance(cluster_b)
                if distance < min_distance:
                    min_distance = distance
                    closest_clusters = [cluster_a, cluster_b]

        merged_cluster = Cluster(None, None)
        merged_cluster.points = closest_clusters[0].points + closest_clusters[1].points
        merged_cluster.indexes = closest_clusters[0].indexes + closest_clusters[1].indexes

        merged_cluster.center = [0] * self.__dimension
        for point in merged_cluster.points:
            for idx_coord, coord in enumerate(point):
                merged_cluster.center[idx_coord] += coord

        merged_cluster.center = [coord/len(merged_cluster) for coord in merged_cluster.center]

        self.__clusters.remove(closest_clusters[0])
        self.__clusters.remove(closest_clusters[1])
        self.__clusters.append(merged_cluster)

    def get_clusters(self):
        return self.__clusters

    def get_indexes(self):
        return [cluster.indexes for cluster in self.__clusters]

    def __validate_arguments(self):
        if len(self.__data) == 0:
            raise ValueError("Empty input data. Data should contain at least one point.")

        if self.__number_of_clusters <= 0:
            raise ValueError(
                "Incorrect amount of clusters '{:d}'. Amount of cluster should be greater than 0.".format(
                    self.__number_of_clusters))
        elif not type(self.__number_of_clusters) == int:
            raise ValueError(
                "Incorret type for amount of clusters '{:d}'. Amount of cluster should be an integer.".format(
                    self.__number_of_clusters))


if __name__ == '__main__':
    from pyclustering.utils import read_sample
    from pyclustering.samples.definitions import FCPS_SAMPLES
    from pyclustering.cluster.agglomerative import agglomerative, type_link
    from pyclustering.cluster import cluster_visualizer

    fcps_data = read_sample(FCPS_SAMPLES.SAMPLE_LSUN)
    hierarchical_agglomerative = HierarchicalAgglomerative(fcps_data, 3)
    hierarchical_agglomerative.clustering()

    clusters = hierarchical_agglomerative.get_clusters()
    len_clusters = sorted([len(cluster) for cluster in clusters])
    print('-------------------')
    print(clusters)
    print(len_clusters)

    agglomerative = agglomerative(fcps_data, 3, type_link.CENTROID_LINK)
    agglomerative.process()

    clusters2 = agglomerative.get_clusters()
    len_clusters2 = sorted([len(cluster) for cluster in clusters2])

    print('-------------------')
    print(clusters2)
    print(len_clusters2)

    visualizer = cluster_visualizer()
    visualizer.append_clusters([cluster.indexes for cluster in clusters], fcps_data)
    visualizer.show()

    visualizer2 = cluster_visualizer()
    visualizer2.append_clusters([cluster for cluster in clusters2], fcps_data)
    visualizer2.show()
