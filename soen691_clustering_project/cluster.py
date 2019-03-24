# -*- coding: utf-8 -*-

""" Cluster class."""
import numpy as np


class Cluster:
    def __init__(self, shape, point=None):
        if point is not None:
            self.points = np.matrix(point)
            self.center = point
            self.rep = np.matrix(point)
        else:
            self.points = np.empty(shape=(0, shape[1]))
            self.center = None
            self.rep = np.empty(shape=(0, shape[1]))

        self.closest = None
        self.distance_closest = float('inf')
