# -*- coding: utf-8 -*-

""" Utils class."""
import numpy as np


def euclidean_distance(x, y):
    return squared_euclidean_distance(x, y) ** 0.5


def squared_euclidean_distance(x, y):
    if ((type(x) == int) and (type(y) == int)) or ((type(x) == float) and (type(y) == float)):
        return (x - y) ** 2.0
    elif (type(x) == list and type(y) == list) or (type(x) == np.ndarray and type(y) == np.ndarray):
        if len(x) == len(y):
            distance = 0.0
            for i in range(0, len(x)):
                distance += (x[i] - y[i]) ** 2.0
            return distance
        else:
            raise ValueError('len of x ({}) != y ({})'.format(len(x), len(y)))
    else:
        raise ValueError('types of x ({}) and y ({}) are different or not supported'.format(type(x), type(y)))


def reservoir_sampling(size):
    i, sample = 0, []
    while True:
        item = yield i, sample
        i += 1
        k = np.random.randint(0, i)
        if len(sample) < size:
            sample.append(item)
        elif k < size:
            sample[k] = item
