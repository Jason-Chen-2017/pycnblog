
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Dijkstra's algorithm is a popular algorithm for finding the shortest path between two nodes (vertices) in a graph with non-negative weights on edges. It was developed by computer scientist Edsger Dijkstra and is still widely used in practical applications such as routing algorithms in networks and computer software route optimization. The algorithm works by maintaining a set of visited vertices that have already been processed, along with their distances from the source vertex. Initially, only the source vertex has a distance of zero from itself and all other distances are infinity. At each step, we select the unvisited vertex with the minimum distance among those whose neighbors haven't yet been visited. We update the distances of its neighbors to be the sum of the current distance plus the weight of the edge connecting them to the selected vertex. This process continues until all vertices have been visited or there is no unvisited vertex left with a smaller distance than the remaining ones. Once finished, we can use the final distances to find the shortest paths between any pair of vertices in the graph. 

In this article, we will implement Dijkstra's algorithm in Python language and apply it to solve the shortest path problem.

# 2.术语定义
## Graph: A graph consists of a finite set of vertices connected by edges. Each edge has a weight associated with it, which may or may not be negative depending upon the application requirements. Edges also have directions assigned to them indicating whether they connect from one vertex to another or vice versa. In this article, we consider directed graphs where the direction of an edge indicates the order in which the vertices should appear in the sequence of vertices traversed during the shortest path search. For example, if there exists an edge from vertex X to vertex Y, then the shortest path from vertex X to vertex Y must always start at X and end at Y. In contrast, undirected graphs have two possible orientations for each edge, making it impossible to determine the absolute direction without additional information provided by the user. 

## Vertex: Vertices represent points in a graph. They are labeled uniquely within the graph. One common way to label vertices is to assign integers starting from 0 to n-1 where n represents the number of vertices in the graph. However, vertices can be represented by any unique identifier of your choice.

## Edge: An edge connects two vertices together. Each edge has a weight associated with it that determines how fast the connection passes through space or time. If the weights are positive, then the edge serves as a road or a river; otherwise, it represents a dead end or an obstacle.

## Source Vertex: The source vertex refers to the initial vertex from which the shortest path is being searched. It does not have any incoming edges because it is the origin point for the traversal. When you calculate the shortest path from the source vertex to every other vertex, you obtain the shortest distance to reach any other vertex from the source vertex.

## Destination Vertex: The destination vertex is the final vertex to which the shortest path is being calculated. It does not have any outgoing edges because once reached, the traversal stops.

## Distance: The length or cost of the shortest path between two vertices is called the distance. It is measured in terms of the number of edges required to traverse. Note that the distance is usually referred to as "cost" or simply "length", but technically, both mean the same thing. Also note that the actual physical distance covered by the edges doesn't matter here since the purpose of the shortest path problem is to identify the quickest/shortest way from one vertex to another regardless of the actual travel distance.

## Adjacency Matrix: An adjacency matrix is a square matrix representing the connectivity or relationships between vertices in a graph. An element in the matrix is either 0 or 1, indicating whether there exists an edge connecting the corresponding vertices or not. There are different ways to represent the adjacency matrices for various types of graphs, including weighted and unweighted. Here, we assume that our graph is represented using an adjacency matrix.

## Priority Queue: A priority queue is a collection of elements sorted based on some criterion. Elements with higher priority values come first. Operations performed on a priority queue include inserting an element into the queue, removing the top element from the queue, and checking the size of the queue. Here, we will use heapq module in Python to implement a min-heap based priority queue.

## Node: A node represents a vertex and contains pointers to its neighboring vertices and their respective edge weights. In addition, it keeps track of its own distance and parent pointer, which are essential components of the dijkstra's algorithm implementation.