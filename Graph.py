# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Mohammad Zarei
# Created Date: 16 Aug 2022
# ---------------------------------------------------------------------------
"""Implementation of graph data structure and related algorithms"""
# imports
from collections import defaultdict


class Node():
    def __init__(self, val, name) -> None:
        self.val = val
        self.name = name
    def __str__(self) -> str:
        return f"{self.name}[{self.val}]"

class Graph():
    def __init__(self, directed=True) -> None:
        # Constructor initializes an empty dictionary
        self._nodes = set()
        self._edges = set()
        self._weights = dict()
        self._adjList = defaultdict(set)
        self._directed = directed
        

    def print_graph(self) -> None:
        print("Graph:")
        for node in self._adjList:
            neighbors = []
            for neighbor in self._adjList[node]:
                neighbors.append(f"({neighbor.__str__()}, {self._weights[(node, neighbor)]})")
            print(f"{node.__str__()}: {neighbors}")

    def get_weight(self, start_node, end_node):
        return self._weights((start_node, end_node))

    def add_node(self, node) -> None:
        # check the type of node input
        assert isinstance(node, Node), "input node must be a 'Node' type"

        self._nodes.add(node)
        self._adjList[node] = set()
    
    def add_edge(self, start_node, end_node, weight=1) -> None:
        # check the type of node input
        assert isinstance(start_node, Node), "start_node must be a 'Node' type"
        assert isinstance(end_node, Node), "end_node must be a 'Node' type"

        # check if nodes are already added to the graph
        assert start_node in self._nodes, "start_node is not added to the graph yet"
        assert end_node in self._nodes, "end_node is not added to the graph yet"

        self._edges.add((start_node, end_node))
        self._weights[(start_node, end_node)] = weight
        self._adjList[start_node].add(end_node)
        

        if not self._directed:
            self._edges.add((end_node, start_node))
            self._weights[(end_node, start_node)] = weight
            self._adjList[end_node].add(start_node)

    def remove_node(self, node_to_be_removed):
        """ Remove all references to node """

        # remove from node list
        try:
            self._nodes.remove(node_to_be_removed)
        except KeyError:
                pass

        # remove from edge list and weight list
        for edge in self._edges.copy():
            if edge[0] == node_to_be_removed or edge[1] == node_to_be_removed:
                self._edges.remove(edge)
                del self._weights[edge]
        
        # remove from _adjList
        try:
            self._adjList.pop(node_to_be_removed)
        except KeyError:
                pass
        for node in self._adjList.copy():
            for neigbour in self._adjList[node].copy():
                if neigbour == node_to_be_removed:
                    self._adjList[node].remove(neigbour)

    def is_connected(self, start_node, end_node):
        """ Check if the geiven start node is connected to the given end node """
        # check the type of node input
        assert isinstance(start_node, Node), "start_node must be a 'Node' type"
        assert isinstance(end_node, Node), "end_node must be a 'Node' type"

        # check if nodes are already added to the graph
        assert start_node in self._nodes, "start_node is not added to the graph yet"
        assert end_node in self._nodes, "end_node is not added to the graph yet"

        return end_node in self._adjList[start_node]


        
        


graph = Graph(directed=True)
a = Node(0, 'a')
b = Node(1, 'b')
c = Node(2, 'c')
d = Node(3, 'd')

# add nodes to graph
graph.add_node(a)
graph.add_node(b)
graph.add_node(c)
graph.add_node(d)

# add edges to graph
graph.add_edge(a,b, 1)
graph.add_edge(d,b, 2)
graph.add_edge(c,b, 4)

# print the graph
graph.print_graph()

# remove a node
graph.remove_node(a)
graph.remove_node(a)

graph.print_graph()

# check if two nodes are connected
# print(graph.is_connected(a, b))


