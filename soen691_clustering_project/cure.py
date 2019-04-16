# -*- coding: utf-8 -*-

""" Cure class."""
import numpy as np
from cluster import CureCluster
from utils import squared_euclidean_distance, reservoir_sampling
from kd_tree import KDTree
from clustering import Clustering


class Cure(Clustering):
    def __init__(self, data, number_of_clusters, alpha, c, sample_size=None):
        self.__data = data.tolist() if isinstance(data, np.ndarray) else data
        self.__k = number_of_clusters
        self.__alpha = alpha
        self.__c = c  # Representative points
        self.__sampled_data = None
        self.__sampling_reservoir = reservoir_sampling(sample_size) if sample_size is not None else None
        if self.__sampling_reservoir is not None:
            self.__sample_data()

        self.__dimension = len(data[0]) if len(data) > 0 else 0

        self.__clusters = None
        self.__representors = None
        self.__centers = None

        self.__validate_arguments()

    def clustering(self):
        # Stores representatives for each cluster
        self.__create_heap()
        self.__create_kdtree()

        while len(self.__heap_q) > self.__k:
            cluster_u = self.__heap_q[0]
            cluster_v = cluster_u.closest

            self.__heap_q.remove(cluster_u)
            self.__heap_q.remove(cluster_v)

            self.__delete_rep(cluster_u)
            self.__delete_rep(cluster_v)

            cluster_w = self.__merge_clusters(cluster_u, cluster_v)
            self.__insert_rep(cluster_w)

            if len(self.__heap_q) > 0:
                cluster_w.closest = self.__heap_q[0]  # arbitrary cluster from heap
                cluster_w.distance_closest = cluster_w.distance(cluster_w.closest)

                for curr_cluster in self.__heap_q:
                    distance = cluster_w.distance(curr_cluster)

                    if distance < cluster_w.distance_closest:
                        cluster_w.closest = curr_cluster
                        cluster_w.distance_closest = distance

                    if curr_cluster.closest is cluster_u or curr_cluster.closest is cluster_v:
                        if curr_cluster.distance_closest < distance:
                            curr_cluster.closest, curr_cluster.distance_closest = self.__closest_cluster(curr_cluster,
                                                                                                         distance)
                            if curr_cluster.closest is None:
                                curr_cluster.closest = cluster_w
                                curr_cluster.distance = distance
                        else:
                            curr_cluster.closest = cluster_w
                            curr_cluster.distance_closest = distance

                    elif curr_cluster.distance_closest > distance:
                        curr_cluster.closest = cluster_w
                        curr_cluster.distance_closest = distance

            self.__heap_q.append(cluster_w)
            self.__heap_q.sort(key=lambda x: x.distance_closest, reverse=False)

        self.__clusters = [cure_cluster for cure_cluster in self.__heap_q]
        self.__representors = [cure_cluster.rep for cure_cluster in self.__heap_q]
        self.__centers = [cure_cluster.center for cure_cluster in self.__heap_q]

    def __closest_cluster(self, x, dist):
        closest_distance = dist
        closest_cluster = None

        euclidean_dist = dist ** 0.5
        for point in x.rep:
            closest_nodes = self.__KDTree_T.find_closest_nodes(point, euclidean_dist)
            for candidate_distance, kdtree_node in closest_nodes:
                if candidate_distance < closest_distance and kdtree_node is not None \
                    and kdtree_node.payload is not x:
                    closest_distance = candidate_distance
                    closest_cluster = kdtree_node.payload

        return closest_cluster, closest_distance

    def __merge_clusters(self, cluster_u, cluster_v):
        # merge clusters
        cluster_w = CureCluster(None, None)
        cluster_w.points = cluster_u.points + cluster_v.points
        cluster_w.indexes = cluster_u.indexes + cluster_v.indexes
        # mean of merged cluster
        cluster_w.center = [0] * self.__dimension
        if cluster_w.points[1:] == cluster_w.points[:-1]:
            cluster_w.center = cluster_w.points[0]
        else:
            for index in range(self.__dimension):
                cluster_w.center[index] = (len(cluster_u.points) * cluster_u.center[index] + len(cluster_v.points) *
                                           cluster_v.center[index]) / (len(cluster_u.points) + len(cluster_v.points))

        temp_set = []
        for index in range(self.__c):
            max_distance = 0
            max_point = None

            for point in cluster_w.points:
                if index == 0:
                    min_distance = squared_euclidean_distance(point, cluster_w.center)
                else:
                    min_distance = min([squared_euclidean_distance(point, p) for p in temp_set])
                if min_distance >= max_distance:
                    max_distance = min_distance
                    max_point = point

            if max_point not in temp_set:
                temp_set.append(max_point)

        cluster_w.rep = [[val + self.__alpha * (cluster_w.center[idx] - val) for idx, val in enumerate(point)] for point
                         in temp_set]

        return cluster_w

    def __insert_rep(self, cluster):
        for p in cluster.rep:
            self.__KDTree_T.insert(p, cluster)

    def __delete_rep(self, cluster):
        for p in cluster.rep:
            self.__KDTree_T.remove(p, payload=cluster)

    def __create_heap(self):
        # Initializes each point as a Cluster object
        self.__heap_q = [CureCluster(point, index) for index, point in
                         enumerate(self.__sampled_data if self.__sampled_data is not None else self.__data)]

        for curr_cluster in self.__heap_q:
            curr_cluster.closest = min([k for k in self.__heap_q if curr_cluster != k],
                                       key=lambda k: curr_cluster.distance(k))
            curr_cluster.distance_closest = curr_cluster.distance(curr_cluster.closest)

        self.__heap_q.sort(key=lambda x: x.distance_closest, reverse=False)

    def __create_kdtree(self):
        self.__KDTree_T = KDTree()
        for curr_cluster in self.__heap_q:
            for rep_point in curr_cluster.rep:
                self.__KDTree_T.insert(rep_point, curr_cluster)

    def __sample_data(self):
        next(self.__sampling_reservoir)
        samples = []
        for idx, sample in enumerate(self.__data):
            samples = self.__sampling_reservoir.send(idx)
        samples.sort()
        self.__sampled_data = []
        for sample in samples:
            self.__sampled_data.append(self.__data[sample])

    def __validate_arguments(self):
        if len(self.__data) == 0:
            raise ValueError("Empty input data. Data should contain at least one point.")

        if self.__k <= 0:
            raise ValueError(
                "Incorrect amount of clusters '{:d}'. Amount of cluster should be greater than 0.".format(self.__k))
        elif not type(self.__k) == int:
            raise ValueError(
                "Incorret type for amount of clusters '{:d}'. Amount of cluster should be an integer.".format(self.__k))

        if self.__alpha < 0:
            raise ValueError(
                "Incorrect compression (k) level '{:f}'. Compression should not be negative.".format(self.__alpha))

        if self.__c <= 0:
            raise ValueError(
                "Incorrect amount of representatives '{:d}'. Amount of representatives should be greater than 0.".format
                (self.__c))

    def get_clusters(self):
        return self.__clusters

    def get_indexes(self):
        return [cluster.indexes for cluster in self.__clusters]

    def get_representors(self):
        return self.__representors

    def get_centers(self):
        return self.__centers
