# Implementation of Graph Structure and Related Algorithms
Node class:
Each node has a value and name.

Graph class:
- `add_node()`: add a new node to the graph
- `add_edge()`: add a new edge to the graph
- `remove_node()`: remove a node from the graph
- `remove_edge()`: remove an edge from the graph
- `print_graph()`: prints the adjacency list from of the graph
- `get_wieght()`: get the weight between two nodes in the graph
- `is_connected()`: returns True if the given nodes are connected

## Detect Cycle
To check if an undirected graph has cycle, we run a DFS while keeping the parent node. In case of directed graph, we maintain a recurssion stack. This is implemented in `has_cycle()` method.

## Topological Sort
Topological sorting for Directed Acyclic Graph (DAG) is a linear ordering of nodes such that for every directed edge u->v, node u comes before node v in the ordering. Topological Sorting for a graph is not possible if the graph is not a DAG. This can be achieved using the `topological_sort()` method.

## Number of Strongly Connected Components
A directed graph is strongly connected if there is a path between all pairs of vertices. A strongly connected component (SCC) of a directed graph is a maximal strongly connected subgraph. Number of SCCs in a given graph can be found using DFS for undirected graphs and Kosaraju’s algorithm for directed graphs. Kosaraju’s algorithm includes running DFS on topologically ordered nodes given the transposed graph.

## Minimum Spanning Tree (MST)
 A minimum spanning tree (MST) for a weighted, connected, undirected graph is a subgraph that is a tree and connects all the vertices together with a weight less than or equal to the weight of every other spanning tree. Kruskal's algorithm find MST using union-find data structure (O(E log N)).

## Single Source Shortest Path Problem
Given a graph and a `source` node in graph, find shortest paths from `souce` to all nodes in the given graph.
- Graphs with all weights equal to 1: breadth first search (BFS)
- Directed acyclic graphs: modified Bellman-Ford Algorithm (O(E + N))
- All weights are positive: Dijkstra's Algorithm (O(E log N))
- For other graph cases: Bellman-Ford Algorithm (O(E * N))
