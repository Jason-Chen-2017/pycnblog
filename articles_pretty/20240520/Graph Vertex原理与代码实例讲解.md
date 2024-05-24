# Graph Vertex原理与代码实例讲解

## 1.背景介绍

### 1.1 什么是图?

在计算机科学中,图(Graph)是一种非线性数据结构,由一组顶点(Vertex)和连接这些顶点的边(Edge)组成。图可以用来表示和建模许多现实世界中的关系和网络,如社交网络、交通网络、电路网络等。

### 1.2 图的应用

图在许多领域都有广泛的应用,例如:

- 社交网络分析
- 网页排名(PageRank算法)
- 路径规划和寻找最短路径
- 电路设计
- 计算机网络拓扑
- 推荐系统
- 计算机视觉和模式识别

### 1.3 图的表示

常见的图的表示方式有两种:

1. **邻接矩阵(Adjacency Matrix)**: 使用二维数组来表示顶点之间的连接关系。
2. **邻接表(Adjacency List)**: 使用链表或数组列表来存储每个顶点的邻接顶点。

## 2.核心概念与联系  

### 2.1 顶点(Vertex)

顶点是图中的基本单元,用来表示图中的对象或实体。每个顶点通常用一个唯一的标识符(如数字或字符串)来标记。

### 2.2 边(Edge)

边是连接两个顶点的线。根据边的方向性,可分为:

- **无向边(Undirected Edge)**: 边没有方向,可双向遍历。
- **有向边(Directed Edge)**: 边有方向,只能单向遍历。有向边也称为弧(Arc)。

### 2.3 路径(Path)

路径是一系列通过边相连的顶点序列。根据路径的起点和终点是否重合,可分为:

- **简单路径(Simple Path)**: 起点和终点不重合。
- **环(Cycle)**: 起点和终点重合。

### 2.4 连通性

- **连通图(Connected Graph)**: 对于任意两个顶点,都存在一条路径可以到达。
- **非连通图(Disconnected Graph)**: 存在至少一对顶点之间没有路径相连。

### 2.5 权重(Weight)

边可以携带权重(Weight)信息,表示边的某种代价或长度。带权图在很多应用中非常有用,如最短路径算法。

### 2.6 度(Degree)

顶点的度表示与该顶点相连的边的数量。

- 无向图中,顶点度等于所有邻接边的数量。
- 有向图中,有入度(In-Degree)和出度(Out-Degree)之分。

## 3.核心算法原理具体操作步骤

### 3.1 图的遍历

图的遍历是访问图中所有顶点的过程,常用的遍历算法有:

#### 3.1.1 深度优先遍历(Depth-First Search, DFS)

深度优先遍历从一个根顶点开始,沿着一条路径尽可能深入,直到没有其他可访问的顶点为止,然后回溯并转向另一条路径。可以使用递归或栈来实现。

```python
from collections import defaultdict 

class Graph:
    def __init__(self): 
        self.graph = defaultdict(list)

    def addEdge(self,u,v):
        self.graph[u].append(v)

    def DFSUtil(self, v, visited):
        visited.add(v)
        print(v, end=' ')
        for neighbour in self.graph[v]:
            if neighbour not in visited:
                self.DFSUtil(neighbour, visited)

    def DFS(self, v):
        visited = set()
        self.DFSUtil(v, visited)

g = Graph() 
g.addEdge(0, 1) 
g.addEdge(0, 9) 
g.addEdge(1, 2) 
g.addEdge(2, 0) 
g.addEdge(2, 3) 
g.addEdge(9, 3) 

print("Depth First Traversal (starting from vertex 2)")
g.DFS(2)
```

输出:
```
2 0 1 9 3
```

#### 3.1.2 广度优先遍历(Breadth-First Search, BFS)  

广度优先遍历从根顶点开始,先访问所有邻接顶点,然后访问邻接顶点的邻接顶点,以此类推。可以使用队列来实现。

```python
from collections import defaultdict, deque

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)

    def addEdge(self, u, v):
        self.graph[u].append(v)

    def BFS(self, s):
        visited = [False] * (max(self.graph) + 1)
        queue = deque()
        queue.append(s)
        visited[s] = True

        while queue:
            s = queue.popleft()
            print(s, end=" ")

            for i in self.graph[s]:
                if visited[i] == False:
                    queue.append(i)
                    visited[i] = True

g = Graph()
g.addEdge(0, 1)
g.addEdge(0, 2)
g.addEdge(1, 2)
g.addEdge(2, 0)
g.addEdge(2, 3)
g.addEdge(3, 3)

print("Breadth First Traversal (starting from vertex 2)")
g.BFS(2)
```

输出:
```
2 0 3 1
```

### 3.2 最短路径算法

#### 3.2.1 Dijkstra算法

Dijkstra算法用于计算单源最短路径问题,即从源顶点到其他所有顶点的最短路径。适用于带权重的有向或无向图。

1. 创建一个集合 `sptSet` 来跟踪已处理的顶点。
2. 为每个顶点分配一个距离值。最初,将源顶点的距离设为0,其余顶点的距离设为无穷大。
3. 从未处理的顶点中选择距离值最小的顶点 `u`。
4. 更新 `u` 的邻接顶点的距离值。
5. 将 `u` 添加到 `sptSet` 中。
6. 重复步骤3-5,直到所有顶点都被处理。

```python
import sys 

class Graph():
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]
                      for row in range(vertices)]

    def printSolution(self, dist):
        print("Vertex Distance from Source")
        for node in range(self.V):
            print(node, ":\t", dist[node])

    def minDistance(self, dist, sptSet):
        min = sys.maxsize
        for v in range(self.V):
            if dist[v] < min and sptSet[v] == False:
                min = dist[v]
                min_index = v

        return min_index

    def dijkstra(self, src):
        dist = [sys.maxsize] * self.V
        dist[src] = 0
        sptSet = [False] * self.V

        for cout in range(self.V):
            u = self.minDistance(dist, sptSet)
            sptSet[u] = True

            for v in range(self.V):
                if self.graph[u][v] > 0 and sptSet[v] == False and dist[v] > dist[u] + self.graph[u][v]:
                    dist[v] = dist[u] + self.graph[u][v]

        self.printSolution(dist)

g = Graph(9)
g.graph = [[0, 4, 0, 0, 0, 0, 0, 8, 0],
           [4, 0, 8, 0, 0, 0, 0, 11, 0],
           [0, 8, 0, 7, 0, 4, 0, 0, 2],
           [0, 0, 7, 0, 9, 14, 0, 0, 0],
           [0, 0, 0, 9, 0, 10, 0, 0, 0],
           [0, 0, 4, 14, 10, 0, 2, 0, 0],
           [0, 0, 0, 0, 0, 2, 0, 1, 6],
           [8, 11, 0, 0, 0, 0, 1, 0, 7],
           [0, 0, 2, 0, 0, 0, 6, 7, 0]
           ]

g.dijkstra(0)
```

输出:
```
Vertex Distance from Source
0 :  0
1 :  4
2 :  12
3 :  19
4 :  21
5 :  11
6 :  9
7 :  8
8 :  14
```

#### 3.2.2 Bellman-Ford算法

Bellman-Ford算法可以处理包含负权重边的图,用于解决单源最短路径问题。它适用于有向图和无向图。

1. 初始化所有顶点的距离值为无穷大,源顶点的距离值为0。
2. 对所有边进行 `V-1` 次松弛操作,即检查是否可以通过某条边获得更短的路径。
3. 检查是否存在导致负权重循环的边。如果存在,则无法找到最短路径。

```python
class Graph:
    def __init__(self, vertices):
        self.V = vertices  # 顶点数
        self.graph = []  # 边的列表

    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])

    def printArr(self, dist):
        print("Vertex Distance from Source")
        for i in range(self.V):
            print("{0}\t\t{1}".format(i, dist[i]))

    def BellmanFord(self, src):
        dist = [float("Inf")] * self.V
        dist[src] = 0

        for i in range(self.V - 1):
            for u, v, w in self.graph:
                if dist[u] != float("Inf") and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w

        for u, v, w in self.graph:
            if dist[u] != float("Inf") and dist[u] + w < dist[v]:
                print("Graph contains negative weight cycle")
                return

        self.printArr(dist)

g = Graph(5)
g.addEdge(0, 1, -1)
g.addEdge(0, 2, 4)
g.addEdge(1, 2, 3)
g.addEdge(1, 3, 2)
g.addEdge(1, 4, 2)
g.addEdge(3, 2, 5)
g.addEdge(3, 1, 1)
g.addEdge(4, 3, -3)

g.BellmanFord(0)
```

输出:
```
Vertex Distance from Source
0               0
1               -1
2               2
3               -2
4               1
```

### 3.3 最小生成树算法

#### 3.3.1 Kruskal算法

Kruskal算法用于在加权连通无向图中找到最小生成树。

1. 按照边的权重对所有边进行排序。
2. 从权重最小的边开始,添加到最小生成树中。如果添加该边不会形成环,则继续添加。
3. 重复步骤2,直到树中包含了所有顶点。

```python
from collections import defaultdict

class Graph:
    def __init__(self, vertices):
        self.V = vertices  # 顶点数
        self.graph = []  # 边的列表

    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])

    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)

        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    def KruskalMST(self):
        result = []
        i = 0
        e = 0

        self.graph = sorted(self.graph, key=lambda item: item[2])

        parent = []
        rank = []

        for node in range(self.V):
            parent.append(node)
            rank.append(0)

        while e < self.V - 1:
            u, v, w = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)

            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.union(parent, rank, x, y)

        print("Following are the edges in the constructed MST")
        for u, v, weight in result:
            print("%d -- %d == %d" % (u, v, weight))

g = Graph(4)
g.addEdge(0, 1, 10)
g.addEdge(0, 2, 6)
g.addEdge(0, 3, 5)
g.addEdge(1, 3, 15)
g.addEdge(2, 3, 4)

g.KruskalMST()
```

输出:
```
Following are the edges in the constructed MST
2 -- 3 == 4
0 -- 3 == 5
0 -- 1 == 10
```

#### 3.3.2 Prim算法

Prim算法用于在加权连通无向图中找到最小生成树。

1. 选择一个起始顶点,将其加入到最小生成树中。
2. 从与最小生成树相连的边中选择权重最小的边,将该边的