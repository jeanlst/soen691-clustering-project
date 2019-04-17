# -*- coding: utf-8 -*-

""" Hierarchical Agglomerative class."""
from cluster import Cluster
from clustering import Clustering
import heapq


class HierarchicalAgglomerative(Clustering):
    def __init__(self, data, number_of_clusters):
        self.__data = data
        self.__number_of_clusters = number_of_clusters
        self.__clusters = {str([index]): Cluster(point, index) for index, point in enumerate(self.__data)}
        self.__dimension = len(data[0]) if len(data) > 0 else 0

        self.__validate_arguments()

    def clustering(self):
        self.__build_priority_queue(self.__compute_distances())
        old_clusters = []
        while len(self.__clusters) > self.__number_of_clusters:
            min_distance, min_heap_node = heapq.heappop(self.__heap)
            closest_clusters = min_heap_node[1]

            if not self.__valid_heap_node(min_heap_node, old_clusters):
                continue

            str_closest_clusters = list(map(str, closest_clusters))
            closest_clusters_objs = [self.__clusters[str_closest_clusters[0]], self.__clusters[str_closest_clusters[1]]]

            merged_cluster = Cluster(None, None)
            merged_cluster.points = closest_clusters_objs[0].points + closest_clusters_objs[1].points
            merged_cluster.indexes = closest_clusters_objs[0].indexes + closest_clusters_objs[1].indexes
            merged_cluster.center = self.__compute_centroid(merged_cluster)

            del self.__clusters[str_closest_clusters[0]]
            del self.__clusters[str_closest_clusters[1]]
            old_clusters.extend(closest_clusters)
            self.__update_heap(merged_cluster)
            self.__clusters[str(merged_cluster.indexes)] = merged_cluster

    def __compute_centroid(self, cluster):
        center = [0] * self.__dimension
        for point in cluster.points:
            for idx_coord, coord in enumerate(point):
                center[idx_coord] += coord

        return [coord / len(cluster) for coord in center]

    def __compute_distances(self):
        distances = []
        for idx_n, cluster_n in self.__clusters.items():
            for idx_i, cluster_i in self.__clusters.items():
                if idx_n != idx_i:
                    dist = cluster_n.distance(cluster_i)
                    distances.append((dist, [dist, [cluster_n.indexes, cluster_i.indexes]]))
        return distances

    def __build_priority_queue(self, distances):
        heapq.heapify(distances)
        self.__heap = distances
        return self.__heap

    def __update_heap(self, new_cluster):
        for idx, cluster in self.__clusters.items():
            dist = new_cluster.distance(cluster)
            heapq.heappush(self.__heap, (dist, [dist, [new_cluster.indexes, cluster.indexes]]))

    def __valid_heap_node(self, heap_node, old_clusters):
        for old_cluster in old_clusters:
            if old_cluster in heap_node[1]:
                return False
        return True

    def __merge_clusters(self):
        """ Naive approach"""
        min_distance = float('inf')
        closest_clusters = None

        for idx_a, cluster_a in enumerate(self.__clusters):
            for cluster_b in self.__clusters[idx_a + 1:]:
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

        merged_cluster.center = [coord / len(merged_cluster) for coord in merged_cluster.center]

        self.__clusters.remove(closest_clusters[0])
        self.__clusters.remove(closest_clusters[1])
        self.__clusters.append(merged_cluster)

    def get_clusters(self):
        return list(self.__clusters.values())

    def get_indexes(self):
        return [cluster.indexes for cluster in self.__clusters.values()]

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
