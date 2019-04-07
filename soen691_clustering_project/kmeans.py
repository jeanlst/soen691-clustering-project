# -*- coding: utf-8 -*-

""" KMeans class."""
import random
from utils import squared_euclidean_distance
from clustering import Clustering

import numpy as np


def initialize_centers(data, k):
    available_indexes = set(list(range(len(data))))

    if k == len(data):
        return data[:]

    centers = [0] * k
    for idx in range(k):
        random_index_point = random.randint(0, len(data[0]))
        if random_index_point not in available_indexes:
            random_index_point = available_indexes.pop()
        else:
            available_indexes.remove(random_index_point)

        centers[idx] = data[random_index_point]

    return centers


class KMeans(Clustering):
    def __init__(self, data, k, tolerance=0.001, max_iter=200):
        self.__data = np.array(data)
        self.__clusters = []
        self.__k = k
        self.__centers = np.array(initialize_centers(self.__data, self.__k))
        self.__dimension = len(data[0]) if len(data) > 0 else 0
        self.__tolerance = tolerance
        self.__max_iterations = max_iter
        self.__sse = float('inf')

        self.__validate_arguments()

    def clustering(self):
        it = 0
        old_centers = []
        while not self.__should_stop(old_centers, it):
            self.__clusters = self.__compute_clusters()
            old_centers = self.__centers
            self.__centers = self.__compute_centers()
            it += 1

        self.__sse = self.__compute_sse()

    def __should_stop(self, old_centers, iteration):
        return self.__compute_changes(old_centers) < (
            self.__tolerance * self.__tolerance) or iteration > self.__max_iterations

    def __compute_clusters(self):
        clusts = [[] for _dummy in range(self.__k)]

        distances = self.__compute_distances()
        opt_points = np.argmin(distances, axis=0)
        for idx, point in enumerate(opt_points):
            clusts[point].append(idx)

        return [cluster for cluster in clusts if len(cluster) > 0]

    def __compute_centers(self):
        centers = np.zeros((self.__k, self.__dimension))
        for idx, cluster in enumerate(self.__clusters):
            cluster_points = self.__data[cluster, :]
            centers[idx] = cluster_points.mean(axis=0)

        return centers

    def __compute_distances(self):
        """Calculate distance from each point to each cluster center."""
        return [[squared_euclidean_distance(point, center) for point in self.__data] for center in self.__centers]

    def __compute_changes(self, centers):
        if len(self.__centers) != len(centers):
            maximum_change = float('inf')
        else:
            changes = squared_euclidean_distance(self.__centers, centers)
            maximum_change = np.max(changes)

        return maximum_change

    def __compute_sse(self):
        distances = self.__compute_distances()
        sse = 0
        for idx_cluster, cluster in enumerate(self.__clusters):
            for idx_point in cluster:
                sse += distances[idx_cluster][idx_point]
        return sse

    def get_clusters(self):
        return self.__clusters

    def get_indexes(self):
        return self.__clusters

    def get_centers(self):
        return self.__centers

    def get_sse(self):
        return self.__sse

    def __validate_arguments(self):
        if len(self.__data) == 0:
            raise ValueError("Empty input data. Data should contain at least one point.")

        if self.__k <= 0:
            raise ValueError(
                "Incorrect amount of clusters '{:d}'. Amount of cluster should be greater than 0.".format(self.__k))
        elif not type(self.__k) == int:
            raise ValueError(
                "Incorret type for amount of clusters '{:d}'. Amount of cluster should be an integer.".format(self.__k))

        if self.__tolerance <= 0:
            raise ValueError(
                "Incorrect value for tolerance '{:f}'. Tolerance value should be greater than 0.".format(
                    self.__tolerance))

        if self.__max_iterations <= 0:
            raise ValueError(
                "Incorrect number of iterations '{:d}'. Maximum number of iterations should be greater than 0.".format(
                    self.__max_iterations))
