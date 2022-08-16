# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Mohammad Zarei
# Created Date: 16 Aug 2022
# ---------------------------------------------------------------------------
"""Implementation of graph data structure and related algorithms"""
# imports
from collections import defaultdict


class Node():
    def __init__(self, value, name) -> None:
        self.value = value
        self.name = name
    def __str__(self) -> str:
        return f"{self.name}[{self.value}]"
    def __repr__(self) -> str:
        return f"{self.name}[{self.value}]"

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
                neighbors.append(f"({neighbor}, {self._weights[(node, neighbor)]})")
            print(f"{node}: {neighbors}")

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

    def get_weight(self, start_node, end_node):
        assert self.is_connected(start_node, end_node), "Given nodes are not connected"

        return self._weights[(start_node, end_node)]

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

    def topological_sort(self) -> list:
        """ Return the topologically sorted nodes for directed acyclic graph"""
        assert not self.has_cycle() and self._directed, "graph must be directed and acyclic"

        def _dfs(node, visited, stack):
            # add the node to visited set
            visited.add(node)

            for neighbor in self._adjList[node]:
                if neighbor not in visited:
                    _dfs(neighbor, visited, stack)
            stack.append(node)

        stack = []
        visited = set()
        for node in self._nodes:
            if node not in visited:
                _dfs(node, visited, stack)
        return stack[::-1]

    def get_scc_num(self) -> int:
        """ Returns the number of strongly connected components (scc) """
        if self._directed == True:
            return self._get_scc_num_directed()
        else:
            return self._get_scc_num_undirected()

    def has_cycle(self):
        """ Returns true if there is cycle in graph """
        if self._directed == True:
            return self._has_cycle_directed()
        else:
            return self._has_cycle_undirected()


    def get_transpose(self):
        transposed_adjList = defaultdict(set)
        for node in self._nodes:
            transposed_adjList[node] = set()

        for start in self._adjList:
            for end in self._adjList[start]:
                transposed_adjList[end].add(start)
        
        return transposed_adjList


    def Kruskal_MST(self) -> list:
        ''' Returns the minimum spanning tree using Kruskal alg with disjoint sets'''

        assert self.get_scc_num() == 1 and not self._directed, "MST is only defined for connected and undirected graphs"
        
        def _find(node, root):
            if root[node] == node:
                return node
            return _find(root[node], root)
        def _union(node1, node2, root, rank):
            parent1, parent2 = _find(node1, root), _find(node2, root)
            if rank[parent1] > rank[parent2]:
                root[parent2] = parent1
            elif rank[parent1] < rank[parent2]:
                root[parent1] = parent2
            else:
                root[parent1] = parent2
                rank[parent2] += 1
        
        result, root, rank = [], {}, {}
        for node in self._nodes:
            root[node], rank[node] = node, 0
        
        sorted_edges = sorted(self._edges, key=lambda x: self.get_weight(x[0],x[1]))
        for u, v in sorted_edges:
            if _find(u, root) != _find(v, root):
                result.append((u,v))
                _union(u,v, root, rank)
        return result


    def BellmanFord(self, source):
        ''' Returns the minimum distance from given source to all nodes'''

        # check the type of node input
        assert isinstance(source, Node), "source must be a 'Node' type" 

        # Initialize distance and parent maps given the single source
        distance, parent = {}, {}
        for node in self._nodes:
            distance[node], parent[node] = float('Inf'), None
        distance[source], parent[source] = 0, source

        # relax all edges n-1 time
        for _ in range(len(self._nodes)-1):
            for u, v in self._edges:
                self._relax(u, v, distance, parent)

        # if more relaxation is possible, there is a negative weight cycle
        for u, v in self._edges:
            if distance[v] > distance[u] + self._weights[(u,v)]:
                print(' There is a negative weight cycle in the graph')
                return
        
        return distance

    def modified_BellmanFord(self, source):
        ''' Returns the minimum distance from given source to all nodes in directed acyclic graphs'''
        
        assert isinstance(source, Node), "source must be a 'Node' type"  
        assert not self.has_cycle() and self._directed, "graph must be directed and acyclic"

        # Initialize distance and parent maps given the single source
        distance, parent = {}, {}
        for node in self._nodes:
            distance[node], parent[node] = float('Inf'), None
        distance[source], parent[source] = 0, source

        topo_sorted = self.topological_sort()
        for node in topo_sorted:
            for neighbor in self._adjList[node]:
                self._relax(node, neighbor, distance, parent)
        return distance


    def Dijkstra(self, source):
        ''' Returns the minimum distance from given source to all nodes in positive weighted graphs'''

        assert isinstance(source, Node), "Source must be a 'Node' type"  
        assert all(w > 0 for w in self._weights.values()), "Weights must be positive"

        def _minDistanceNode(distance, finished):
            # Initialize minimum distance for next node
            min = float('Inf')
    
            for node in self._nodes:
                if distance[node] < min and node not in finished:
                    min = distance[node]
                    min_node = node
    
            return min_node
        
        # Initialize distance and parent maps given the single source
        distance, parent = {}, {}
        for node in self._nodes:
            distance[node], parent[node] = float('Inf'), None
        distance[source], parent[source] = 0, source

        finished = set()

        while len(finished) != len(self._nodes):
            min_node = _minDistanceNode(distance, finished)
            finished.add(min_node)

            for neighbor in self._adjList[min_node]:
                self._relax(min_node, neighbor, distance, parent)
        
        return distance

    def BFS_shortest_path(self, start, end):
        def _backtrace(parent, start, end):
            path = [end]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            return path

        parent = {}
        queue = []
        queue.append(start)
        parent[start] = None
        while queue:
            node = queue.pop(0)
            if node == end:
                return _backtrace(parent, start, end)
            for neighbor in self._adjList[node]:
                if neighbor not in parent:
                    parent[neighbor] = node 
                    queue.append(neighbor)


                







    
    
    # helper methods
    def _has_cycle_directed(self):
        def _dfs(node, rec_stack, visited):
            # add the node to visited set
            visited.add(node)
            rec_stack.add(node)
            for neighbor in self._adjList[node]:
                if neighbor not in visited:
                    if _dfs(neighbor, rec_stack, visited): return True
                elif neighbor in rec_stack: return True
            rec_stack.remove(node)
            return False
        
        visited, rec_stack = set(), set()
        for node in self._nodes:
            if node not in visited:
                if _dfs(node, rec_stack, visited): return True
        return False

    def _has_cycle_undirected(self):
        def _dfs(node, parent, visited):
            # add the node to visited set
            visited.add(node)
            for neighbor in self._adjList[node]:
                if neighbor not in visited:
                    if _dfs(neighbor, node, visited): return True
                elif parent != neighbor: return True
            return False
        
        visited = set()
        for node in self._nodes:
            if node not in visited:
                if _dfs(node, -1, visited): return True
        return False

    def _get_scc_num_undirected(self) -> int:
        def _dfs(node, visited):
            # add the node to visited set
            visited.add(node)
            for neighbor in self._adjList[node]:
                if neighbor not in visited:
                    _dfs(neighbor, visited)

        scc_count = 0
        visited = set()
        for node in self._nodes:
            if node not in visited:
                _dfs(node, visited)
                scc_count += 1
        return scc_count

    def _get_scc_num_directed(self) -> int:
        def _dfs(node, visited):
            # add the node to visited set
            visited.add(node)
            for neighbor in transposed_adjList[node]:
                if neighbor not in visited:
                    _dfs(neighbor, visited)

        scc_count = 0
        visited = set()

        transposed_adjList = self.get_transpose()
        topo_sorted = self.topological_sort()
        for node in topo_sorted:
            if node not in visited:
                _dfs(node, visited)
                scc_count += 1
        return scc_count
    
    def _relax(self, u, v, distance, parent):
            if distance[v] > distance[u] + self._weights[(u,v)]:
                distance[v] = distance[u] + self._weights[(u,v)]
                parent[v] = u
        
        
