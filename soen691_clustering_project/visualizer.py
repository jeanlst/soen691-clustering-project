# -*- coding: utf-8 -*-

"""Visualizer class."""
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from itertools import cycle
import math

color_list = [('red', '#e6194b'), ('green', '#3cb44b'), ('brown', '#aa6e28'), ('blue', '#0000FF'), ('cyan', '#46f0f0'),
              ('purple', '#911eb4'), ('orange', '#f58231'), ('magenta', '#f032e6'), ('grey', '#808080'),
              ('pink', '#fabebe'), ('teal', '#008080'), ('lavender', '#e6beff'), ('yellow', '#ffe119'),
              ('beige', '#fffac8'), ('maroon', '#800000'), ('mint', '#aaffc3'), ('olive', '#808000'),
              ('coral', '#ffd8b1'), ('navy', '#000080'), ('rosy brown', '#bd8e8c'), ('white', '#FFFFFF'),
              ('sky blue', '#56B4E9'), ('bluish green', '#009E73'), ('vermilion', '#D55E00'),
              ('reddish purple', '#CC79A7'), ('lime', '#d2f53c'), ('blue violet', '#8c28e7'), ('dimgray', '#696969'),
              ('lightslategray', '#778899'), ('slategray', '#708090'), ('darkslategray', '#2F4F4F'),
              ('powder blue', '#9EB9D4'), ('gainsboro', '#DCDCDC'), ('salmon_crayola', '#FF91A4'),
              ('salmon', '#FA8072')]


class ClusterRepresentation:
    def __init__(self, cluster, data, marker, markersize, color):
        self.cluster = cluster
        self.data = data
        self.marker = marker
        self.markersize = markersize
        self.color = color


def draw_canvas_cluster(ax, dimension, cluster_representation):
    cluster = cluster_representation.cluster
    data = cluster_representation.data
    marker = cluster_representation.marker
    markersize = cluster_representation.markersize
    color = cluster_representation.color

    for item in cluster:
        if dimension == 1:
            if data is None:
                ax.plot(item[0], 0.0, color=color, marker=marker, markersize=markersize)
            else:
                ax.plot(data[item][0], 0.0, color=color, marker=marker, markersize=markersize)
        elif dimension == 2:
            if data is None:
                ax.plot(item[0], item[1], color=color, marker=marker, markersize=markersize)
            else:
                ax.plot(data[item][0], data[item][1], color=color, marker=marker, markersize=markersize)
        elif dimension == 3:
            if data is None:
                ax.scatter(item[0], item[1], item[2], c=color, marker=marker, s=markersize)
            else:
                ax.scatter(data[item][0], data[item][1], data[item][2], c=color, marker=marker, s=markersize)


class ClusteringVisualizer:
    def __init__(self, number_canvas=1, number_columns=1, number_clusters=None, titles=None, x_labels=None,
                 y_labels=None):
        self.__number_of_canvas = number_canvas
        self.__number_of_columns = number_columns

        self.__clusters = [[] for _ in range(number_canvas)]
        self.__dimensions = [None for _ in range(number_canvas)]
        self.__titles = [None for _ in range(number_canvas)]
        self.__x_labels = [None for _ in range(number_canvas)]
        self.__y_labels = [None for _ in range(number_canvas)]

        if titles is not None:
            self.__titles = titles
        if x_labels is not None:
            self.__x_labels = x_labels
        if y_labels is not None:
            self.__y_labels = y_labels
        if number_clusters is not None:
            if number_clusters < 0:
                raise ValueError("Number of clusters '{}' should be >= 0".format(number_clusters))

        self.__color_pool = cycle([color for (name, color) in color_list]) if number_clusters is None else cycle(
            [color for (name, color) in color_list[:number_clusters]])

    def add_cluster(self, cluster, data=None, canvas=0, marker='.', markersize=None, color=None):
        if len(cluster) == 0:
            return

        if canvas > self.__number_of_canvas or canvas < 0:
            raise ValueError("Canvas index '{}' is out of range [0; {}].".format(canvas, self.__number_of_canvas))

        if color is None:
            color = next(self.__color_pool)

        cluster_representation = ClusterRepresentation(cluster, data, marker, markersize, color)
        self.__clusters[canvas].append(cluster_representation)

        if data is None:
            dimension = len(cluster[0])
            if self.__dimensions[canvas] is None:
                self.__dimensions[canvas] = dimension
            elif self.__dimensions[canvas] != dimension:
                raise ValueError("Only clusters with the same dimension of objects can be displayed on canvas.")
        else:
            dimension = len(data[0])
            if self.__dimensions[canvas] is None:
                self.__dimensions[canvas] = dimension
            elif self.__dimensions[canvas] != dimension:
                raise ValueError("Only clusters with the same dimension of objects can be displayed on canvas.")

        if (dimension < 1) or (dimension > 3):
            raise ValueError(
                "Only objects with size dimension 1 (1D plot), 2 (2D plot) or 3 (3D plot) can be displayed with this "
                "class.")

        if markersize is None:
            if dimension == 1 or dimension == 2:
                cluster_representation.markersize = 5
            elif dimension == 3:
                cluster_representation.markersize = 30

    def add_clustering(self, clusters, data=None, canvas=0, marker='.', markersize=None):
        for cluster in clusters:
            self.add_cluster(cluster, data, canvas, marker, markersize)

    def plot(self, figure=None, invisible_axis=False, visible_grid=True, display=True, save_path=None, shift=None):
        canvas_shift = shift
        if canvas_shift is None:
            if figure is not None:
                canvas_shift = len(figure.get_axes())
            else:
                canvas_shift = 0

        if figure is not None:
            cluster_figure = figure
        else:
            cluster_figure = plt.figure(figsize=(14, 8))

        maximum_cols = self.__number_of_columns
        maximum_rows = math.ceil((self.__number_of_canvas + canvas_shift) / maximum_cols)
        grid_spec = GridSpec(maximum_rows, maximum_cols)

        for index_canvas in range(len(self.__clusters)):
            canvas_data = self.__clusters[index_canvas]
            if len(canvas_data) == 0:
                continue

            dimension = self.__dimensions[index_canvas]

            # ax = axes[real_index];
            if dimension == 1 or dimension == 2:
                ax = cluster_figure.add_subplot(grid_spec[index_canvas + canvas_shift])
            else:
                ax = cluster_figure.add_subplot(grid_spec[index_canvas + canvas_shift], projection='3d')

            if len(canvas_data) == 0:
                plt.setp(ax, visible=False)

            for cluster_descr in canvas_data:
                draw_canvas_cluster(ax, dimension, cluster_descr)

            if invisible_axis is True:
                ax.xaxis.set_ticklabels([])
                ax.yaxis.set_ticklabels([])

                if dimension == 3:
                    ax.zaxis.set_ticklabels([])

            if self.__titles[index_canvas] is not None:
                ax.set_title(self.__titles[index_canvas])

            if self.__x_labels[index_canvas] is not None:
                ax.set_xlabel(self.__x_labels[index_canvas])

            if self.__y_labels[index_canvas] is not None:
                ax.set_ylabel(self.__y_labels[index_canvas])

            ax.grid(visible_grid)

        if display is True:
            plt.show()
        elif save_path is not None:
            cluster_figure.save_fig(save_path)

        return cluster_figure
