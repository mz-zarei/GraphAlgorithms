from Graph import Graph, Node


graph = Graph(directed=True)
a = Node(0, 'a')
b = Node(1, 'b')
c = Node(2, 'c')
d = Node(3, 'd')
e = Node(2, 'e')

# add nodes to graph
graph.add_node(a)
graph.add_node(b)
graph.add_node(c)
graph.add_node(d)
graph.add_node(e)

# add edges to graph
graph.add_edge(a,b, 1)
graph.add_edge(b,c, 2)
graph.add_edge(a,c, 4)

graph.add_edge(c,d, 1)
graph.add_edge(c,e, 3)
graph.add_edge(d,e, 1)

# print the graph
graph.print_graph()

# remove a node
# graph.remove_node(a)
# graph.print_graph()

# topological sort
print(graph.topological_sort())

# get transposed graph
print(graph._adjList)
print(graph.get_transpose())

# Get the number of SCCs
print(graph.get_scc_num())

# Find cycle
print(graph.has_cycle())

# Return MST
# print(graph.Kruskal_MST())

# Return distance from single source to all nodes
print(graph.BellmanFord(a))
print(graph.modified_BellmanFord(a))
print(graph.Dijkstra(a))
print(graph.BFS_shortest_path(a,e))
