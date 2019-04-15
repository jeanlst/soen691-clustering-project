# -*- coding: utf-8 -*-

""" Cluster class."""
from utils import squared_euclidean_distance


class Cluster:
    def __init__(self, point, index):
        assert (point is None and index is None) or (point is not None and index is not None)

        if point is not None:
            self.points = [point]
            self.center = point
            self.indexes = [index]
        else:
            self.points = []
            self.center = None
            self.indexes = []

    def distance(self, other_cluster):
        return squared_euclidean_distance(self.center, other_cluster.center)

    def __len__(self):
        return len(self.points)

    def __repr__(self):
        return 'Cluster(Center={}, Points={})'.format(self.center, self.points)

    def __str__(self):
        return self.__repr__()


class CureCluster(Cluster):
    def __init__(self, point, index):
        super().__init__(point, index)

        if point is not None:
            self.rep = [point]
        else:
            self.rep = []

        self.closest = None
        self.distance_closest = float('inf')

    def distance(self, cluster_v):
        distance = float('inf')
        for p in self.rep:
            for q in cluster_v.rep:
                dist = squared_euclidean_distance(p, q)
                if dist < distance:
                    distance = dist
        return distance

    def __repr__(self):
        return 'Cluster(Dist={}, Center={}, Points={})'.format(self.distance_closest, self.center, self.points)
