# -*- coding: utf-8 -*-

""" KDTree class."""

import numpy as np
from utils import squared_euclidean_distance


class Node:
    def __init__(self, values=None, payload=None, left=None, right=None, disc=None, parent=None):
        self.values = values  # Data point that is presented as list of coodinates.
        self.payload = payload  # Payload of node that can be used by user for storing specific information in the node.
        self.disc = disc  # Index of dimension.

        self.right = right
        self.left = left
        self.parent = parent

    def find_min(self, discriminator):
        stack = [self]
        candidates = []
        is_finished = False

        head = self.left
        while not is_finished:
            if head is not None:
                stack.append(head)
                head = head.left
            else:
                if len(stack) != 0:
                    head = stack.pop()
                    candidates.append(head)
                    head = head.right
                else:
                    is_finished = True

        return min(candidates, key=lambda curr_node: curr_node.values[discriminator])

    def __repr__(self):
        return "({:s}: [L:'{:s}', R:'{:s}'])".format(self.values, self.left.values if self.left is not None else None,
                                                     self.right.values if self.right is not None else None)

    def __str__(self):
        return self.__repr__()


def create_point_comparator(type_point):
    if type_point == np.ndarray:
        return lambda obj1, obj2: np.array_equal(obj1, obj2)

    return lambda obj1, obj2: obj1 == obj2


def create_rule_search(point_comparator, p, point_payload=None):
    if point_payload:
        return lambda node, point=p, payload=point_payload: point_comparator(node.values,
                                                                             p) and node.payload == payload

    return lambda node, point=p: point_comparator(node.values, p)


class KDTree:
    def __init__(self, data_list=None, payload_list=None):
        self.__root = None
        self.__dimension = None
        self.__point_comparator = None

        self.__fill_tree(data_list, payload_list)

    def insert(self, point, payload):
        if self.__root is None:
            self.__dimension = len(point)
            self.__root = Node(point, payload, None, None, 0)
            self.__point_comparator = create_point_comparator(type(point))
            return self.__root

        curr_node = self.__root
        while True:
            if curr_node.values[curr_node.disc] <= point[curr_node.disc]:
                if curr_node.right is None:
                    discriminator = curr_node.disc + 1
                    if discriminator >= self.__dimension:
                        discriminator = 0

                    curr_node.right = Node(point, payload, None, None, discriminator, curr_node)
                    return curr_node.right
                else:
                    curr_node = curr_node.right
            else:
                if curr_node.left is None:
                    discriminator = curr_node.disc + 1
                    if discriminator >= self.__dimension:
                        discriminator = 0

                    curr_node.left = Node(point, payload, None, None, discriminator, curr_node)
                    return curr_node.left
                else:
                    curr_node = curr_node.left

    def remove(self, point, payload=None):
        node_to_remove = self.find_node(point, payload)
        if node_to_remove is None:
            return None

        parent = node_to_remove.parent
        sub_tree_root = self.__remove(node_to_remove)
        if parent is not None:
            if parent.left is node_to_remove:
                parent.left = sub_tree_root
            elif parent.right is node_to_remove:
                parent.right = sub_tree_root
        else:
            self.__root = sub_tree_root
            if sub_tree_root is not None:
                sub_tree_root.parent = None

        return self.__root

    def __remove(self, node_to_remove):
        if (node_to_remove.right is None) and (node_to_remove.left is None):
            return None
        discriminator = node_to_remove.disc
        if node_to_remove.right is None:
            node_to_remove.right = node_to_remove.left
            node_to_remove.left = None

        min_node = self.find_min(discriminator, node_to_remove.right)
        parent = min_node.parent

        if parent.left is min_node:
            parent.left = self.__remove(min_node)
        elif parent.right is min_node:
            parent.right = self.__remove(min_node)

        min_node.parent = node_to_remove.parent
        min_node.disc = node_to_remove.disc
        min_node.right = node_to_remove.right
        min_node.left = node_to_remove.left

        # Update parent for successors of previous parent.
        if min_node.right is not None:
            min_node.right.parent = min_node

        if min_node.left is not None:
            min_node.left.parent = min_node

        return min_node

    def find_min(self, discriminator, head=None):
        if head:
            return head.find_min(discriminator)
        return self.__root.find_min(discriminator)

    def find_closest_nodes(self, point, distance):
        nodes = []
        if self.__root is not None:
            self.__find_closest_nodes(point, distance, distance * distance, self.__root, nodes)
        return nodes

    def __find_closest_nodes(self, point, distance, sqrt_distance, head, nodes):
        if head.right is not None:
            if point[head.disc] >= head.values[head.disc] - distance:
                self.__find_closest_nodes(point, distance, sqrt_distance, head.right, nodes)

        if head.left is not None:
            if point[head.disc] < head.values[head.disc] + distance:
                self.__find_closest_nodes(point, distance, sqrt_distance, head.left, nodes)

        candidate_distance = squared_euclidean_distance(point, head.values)
        if candidate_distance <= sqrt_distance:
            nodes.append((candidate_distance, head))

    def __find_node(self, point, search_rule, curr_node):
        node_to_find = None
        if curr_node is None:
            curr_node = self.__root

        while curr_node:
            if curr_node.values[curr_node.disc] <= point[curr_node.disc]:
                if search_rule(curr_node):
                    node_to_find = curr_node
                    break
                curr_node = curr_node.right
            else:
                curr_node = curr_node.left

        return node_to_find

    def find_node(self, point, payload=None, curr_node=None):
        rule_search = create_rule_search(self.__point_comparator, point, payload)
        return self.__find_node(point, rule_search, curr_node)

    def __fill_tree(self, data_list, payload_list):
        if data_list is None or len(data_list) == 0:
            return

        if payload_list is None:
            for index in range(0, len(data_list)):
                self.insert(data_list[index], None)
        else:
            for index in range(0, len(data_list)):
                self.insert(data_list[index], payload_list[index])

        self.__point_comparator = create_point_comparator(type(self.__root.values))
