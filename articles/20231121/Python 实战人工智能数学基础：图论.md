                 

# 1.背景介绍


## 1.1 图的定义
图（graph）是由顶点和边组成的一个数学结构。其中的顶点可以看作是一个对象或者实体，而边则代表了两个顶点之间的关系或连接。在图中，任意两顶点之间都存在一条边相连，但是并不一定是无向边。图的种类繁多，如有向图、带权图、环图等。

## 1.2 图的基本术语
- 顶点：一个图中的元素称为顶点，顶点又可以细分为节点或节点。每个节点对应于一个问题域或问题抽象层面的实体。
- 边：边表示两个节点之间的连接，它可以是有向边或无向边。有向边表示从一个顶点指向另一个顶点，无向边则表示两节点间的连接没有方向。
- 路径：路径是通过图中一系列的边，从某个顶点到达另一个顶点的最短的路径。通俗地说，路径就是把顶点按照一定的顺序连起来。
- 简单路径：简单的路径是指不含重复顶点的路径，比如从顶点 A 到顶点 E 的路径 ABCE 和 BCDE 都是简单路径，而 ACBE 不属于简单路径。
- 回路：回路是指一个顶点到自身的路径。
- 森林：森林是指互相互不连接的多个树的集合。

# 2.核心概念与联系
## 2.1 图的存储结构
图的存储结构通常采用两种形式之一：邻接矩阵和邻接表。
### 2.1.1 邻接矩阵
邻接矩阵是一种比较简单的图存储方式。对于给定的图 G = (V,E) ，其中 V 是顶点集，E 是边集，用二维数组 Adj[i][j] 表示 i 号顶点 j 号顶点是否存在边。如果 Adj[i][j] 为 1，则说明 i 号顶点与 j 号顶点之间有边相连；否则，说明 i 号顶点与 j 号顶点之间没有边相连。如下图所示：
```python
+-----+-----+     +-----+-----+     +-----+-----+     +-----+-----+
|  1  |  1  |... | -1  |-1   |     |  0  |  1  |... |  1  | -1  |
+-----+-----+     +-----+-----+     +-----+-----+     +-----+-----+
  \__v___/         \_ _\_ _/_       /      \_ ___\      \_ ___\_ __/
      |              |   |_|_ _ _ _|        ||    |        ||    |_||
      u              v           \_ _ _ _ _||__u__|        ||__v__|||
                    w                ||                  ||    ||
                                            k                  o    p
                                             \_ _ _ _ _ _ _ _ _ _|
                                                 ||              
                                                     m                  
                                                       n                    
                                                          l                      
                                                             q                        
                                                                c                       
                                                                 r                     
                                                                    e                 

# (u,v),(w,k),(e,r),(l,n),q,(m,p),(c,o),...
```

在这种存储方法中，每条边对应于矩阵的一个元素。即如果 G 有一条边 (u,v)，那么 Adj[u][v] 或 Adj[v][u] 的值为 1。这样，对于一个顶点，所有它的邻居就都可以通过 O(1) 的时间复杂度来查询到。

### 2.1.2 邻接表
邻接表是一种更加灵活的数据结构，它利用链表的方式来存储图的顶点与边的关系。对于给定的图 G = (V,E)，其中 V 是顶点集，E 是边集，用一个数组表示各个顶点的邻接表。例如，Adj[v] 保存着顶点 v 的所有邻居。Adjacency List 的实现如下：
```python
from typing import List


class Vertex:
    def __init__(self, value):
        self.value = value
        self.neighbors = []

    def add_neighbor(self, neighbor):
        if neighbor not in self.neighbors:
            self.neighbors.append(neighbor)
    
    # Add other methods as needed for your use case

class Graph:
    def __init__(self):
        self.vertices = {}
        self.num_vertices = 0

    def add_vertex(self, value):
        new_vertex = Vertex(value)
        self.vertices[new_vertex.value] = new_vertex
        self.num_vertices += 1
        return new_vertex

    def get_vertex(self, value):
        return self.vertices.get(value)

    def add_edge(self, vertex1, vertex2):
        if vertex1 not in self.vertices or vertex2 not in self.vertices:
            raise ValueError("Both vertices must be added to the graph first.")

        vertex1.add_neighbor(vertex2)
        if isinstance(vertex1, Vertex) and isinstance(vertex2, Vertex):
            pass
            # TODO Implement this method for directed graphs.

    def has_cycle(self):
        visited = set()
        for vertex in self.vertices.values():
            if vertex not in visited and self._has_cycle(visited, [], vertex):
                return True
        return False

    def _has_cycle(self, visited, path, current_vertex):
        visited.add(current_vertex)
        path.append(current_vertex)
        for neighbor in current_vertex.neighbors:
            if neighbor == path[-1]:
                print('Cycle detected:', path)
                return True
            elif neighbor not in visited and self._has_cycle(visited, path, neighbor):
                return True
        path.pop()
        visited.remove(current_vertex)
        return False
```

图 G 可以建立起一些索引，比如字典类型的 vertices，用来快速找到某个顶点对应的邻居列表。图还可以使用 DFS 算法检测循环。

## 2.2 图的遍历
图的遍历（traversal）是图数据结构中重要的一种操作。它包括对图中所有的顶点访问一次且仅访问一次，以及对图中某些特定顶点访问多次，但在每次访问时，都要保证每个顶点只访问一次。

以下介绍三种经典的图的遍历算法：DFS（Depth First Search），BFS（Breath First Search），和 Dijkstra's Algorithm。

### 2.2.1 Depth First Search （DFS）
DFS 是一种深度优先搜索算法。它沿着图的某条边走，直到走到尽头，然后再回溯到之前的那个顶点，依次尝试其他的路径。

下图展示了 DFS 在一个稀疏图上的运行过程。


如上图所示，DFS 从顶点 S 出发，先访问顶点 S，然后 DFS 选择顶点 U 作为新的探索目标。因为 U 没有直接的邻居，所以 DFS 将回溯到顶点 S，然后继续探索顶点 T。当它发现顶点 T 的邻居均已被探索过，并返回顶点 T 时，顶点 U 可被视为新的探索目标，然后 DFS 开始探索它。顶点 T 还有其他的邻居，因此 DFS 将回溯到顶点 S，继续探索顶点 V。因为 V 沔恩可达，所以 DFS 将会选择顶点 W 作为新的探索目标。当顶点 W 没有直接的邻居，因此 DFS 将回溯到顶点 V，然后继续探索顶点 X。此时顶点 W 的邻居均被探索过，并返回顶点 W，此时顶点 X 成为新的探索目标，然后 DFS 开始探索它。顶点 X 没有可达的邻居，因此 DFS 会回溯到顶点 Y。顶点 Z 也没有邻居，因此 DFS 将回溯到顶点 Y，然后继续探索顶点 B。当它发现顶点 C 没有邻居，然后将回溯到顶点 B，探索顶点 A。顶点 A 没有可达的邻居，因此 DFS 将回溯到顶点 B，探索顶点 H。顶点 F 和 G 均没有邻居，因此 DFS 将回溯到顶点 H，继续探索顶点 I。顶点 K 没有可达的邻居，因此 DFS 将回溯到顶点 I，然后返回顶点 I。顶点 J 没有可达的邻居，因此 DFS 将回溯到顶点 H，然后继续探索顶点 L。顶点 M 也没有可达的邻居，因此 DFS 将回溯到顶点 L，然后返回顶点 L。DFS 将再次回溯到顶点 H，然后探索顶点 N。顶点 P 没有可达的邻居，因此 DFS 将回溯到顶点 N，然后返回顶点 N。顶点 O 也没有可达的邻居，因此 DFS 将回溯到顶点 N，然后返回顶点 N。DFS 最后返回顶点 S。

DFS 使用递归的方式实现，所以它也可以用于求最短路径等应用。同时，由于它以深度优先的方式查找，因此适合处理有向图。

### 2.2.2 Breath First Search （BFS）
BFS 是一种宽度优先搜索算法。它从初始顶点开始，将它的所有直接相邻的顶点加入队列，并标记它们。然后从队列中取出第一个顶点，对它进行探索。当它发现它的所有相邻顶点均已经探索过后，才开始探索它的相邻顶点。这种方式类似于树的广度优先遍历。

下图展示了一个例子。


如上图所示，BFS 从顶点 S 出发，首先访问 S。然后，它将顶点 S 的相邻顶点（B,C）加入队列，并标记它们。当它取出队列中第一个顶点时，它发现它只有 C，并将它标记为已探索。然后它将 C 的相邻顶点（D,F）加入队列，并标记它们。当它取出第二个顶点时，它发现它只有 D 和 F，并将它们标记为已探索。然后它将 D 的相邻顶点（G）加入队列，并标记它。当它取出第三个顶点时，它发现它只有 G，并将它标记为已探索。然后它将 G 的相邻顶点（H）加入队列，并标记它。当它取出第四个顶点时，它发现它只有 H，并将它标记为已探索。当它完成顶点 S 的探索时，整个图便被完全探索完毕。

BFS 使用队列的方式实现，所以效率较高。它适合处理稠密图。

### 2.2.3 Dijkstra's Algorithm
Dijkstra's Algorithm 是一种单源最短路径算法。它可以计算指定源顶点到所有其他顶点的最短路径。

其主要思想是，维护一个开销值数组 dist[]，记录到起始顶点 s 的距离。初始化 dist[s] 为 0，然后开始迭代，每次迭代中，找出 dist[] 中最小的值，然后将该值的顶点标记为已探索。对于当前顶点 v，更新 dist[v]，使得 dist[v] 等于 dist[curr]+weight(curr->v)。然后对于这个新值 v，检查其所有相邻顶点 curr，如果 curr 尚未探索过，则根据 weight(curr->v)+dist[curr]<dist[curr] 更新 dist[curr]。循环往复，直至所有顶点均被探索。

下图展示了一个例子。


如上图所示，图 G = (V={A,B,C,D,E}, E={(a,b,7),(a,d,5),(b,c,8),(b,d,9),(c,d,15),(c,e,6),(d,e,8),(d,f,11)}) 。Dijkstra's Algorithm 根据源顶点 A 计算最短路径。首先初始化 dist[A] 为 0，然后开始迭代。首先，对于顶点 B 来说，dist[B]=min(dist[B],dist[A]+weight(A->B))=7<∞，因此不需要更新，对于顶点 C 来说，dist[C]=min(dist[C],dist[A]+weight(A->C))+weight(C->D)=20+15=35<∞，因此需要更新 dist[C]=35。同样，对于顶点 D 来说，dist[D]=min(dist[D],dist[A]+weight(A->D))+weight(D->E)=20+8=28<∞，因此需要更新 dist[D]=28。以此类推，直至全部顶点均被探索。