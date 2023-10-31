
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


## 一、概述
图（graph）是网络结构中的一种数据结构，由顶点和边组成，通常用来表示复杂的系统或信息的结构。图在科技领域中应用广泛，例如在地图上表示城市之间的道路连接关系，在生物信息学中用于描述基因间的相互作用关系。

图作为基本的数据结构，它所具有的各种性质及其相关的算法，都将对人工智能（AI）的研究和开发产生极大的影响。如图网络的划分方法、特征提取方法、路径搜索算法等，都可以用来解决现实世界中最复杂的问题，包括设计高效的算法、分析复杂的系统等。因此，掌握图论算法对于成为一名合格的机器学习工程师、数据分析师和工程专家至关重要。

本系列教程，是作者通过 Python 编程语言基于图论的算法实现，对常用的图算法进行整理和总结，从而帮助读者更好的理解并运用图论知识。主要涉及到的图论算法有邻接矩阵、邻接表、十字链表、邻接多重列表、边集数组、度序列、马尔科夫链、最小生成树、拓扑排序、最大流、匹配与最大权匹配、哈密顿回路等。除此之外，还会介绍一些其他的经典图算法，比如狄克斯特拉算法、弗洛伊德算法等，这些算法在实际中应用非常广泛。

图论算法最早由 E.W.Dijkstra 提出，他是著名的计算机科学家。1930年代以后，随着人工智能的兴起，图论算法也逐渐被应用到很多重要的领域，如路由选择、生物信息学、网络安全、软件工程等。近几年，随着图论算法在人工智能领域的应用越来越广泛，越来越多的人开始关注并借鉴图论的一些最优性质，而不断创新，提升图论算法的效率和效果。

## 二、定义
在数学中，一个图 $G=(V,E)$ 是由顶点集合 $V$ 和边集合 $E$ 组成的。其中，顶点集合 $V=\{v_1,v_2,\cdots,v_n\}$ 表示图的结点或节点；边集合 $E=\{(u_i,v_j)\}_{(u_i,v_j)\in V\times V}\cup \{(u_k,u_l)\}_{(u_k,u_l)\in V^2}$ 表示图的边。

如果 $G$ 中每个顶点 $v_i$ 均有一个度，则称 $G$ 为带度（degree-labeled）图；否则，$G$ 为不带度（unlabeled）图。一个图的度 $d(v_i)$ 是指与 $v_i$ 直接相连的边数。$G$ 的度分布函数 $\operatorname{deg}(v)=d(v)$ 是映射，它给出了每个顶点的度数。

一个无向图 $G=(V,E)$ 可以用两种方式编码：

1. 邻接矩阵：采用 $|V|\times |V|$ 的矩阵 $A$，$A_{ij}=1$ 表示 $(u_i,v_j)\in E$，$A_{ij}=0$ 表示 $(u_i,v_j)\notin E$。
2. 邻接表：采用 $V$ 个链表，每条链表表示 $v_i$ 指向的其他所有顶点。

# 2.核心概念与联系
## （1）图的表示方法
### 1.1 邻接矩阵
当 $G$ 有序对 $(u_i,v_j)$ 中的顶点个数为非负整数时，可以用邻接矩阵来存储图的信息，即用 $n\times n$ 矩阵 $A=(a_{ij})$ 来表示图 $G=(V,E)$。其中，如果 $(u_i,v_j) \in E$，那么 $a_{ij}=1$；反之，若不存在 $(u_i,v_j)$ 这一条边，则 $a_{ij}=0$。

#### 1.1.1 十字链表
十字链表（Adjacency Cross List，ACL）是一个可节省空间的数据结构。它将图中的边按下列方式组织起来：用一个包含 $n$ 个元素的数组 $E[1..n]$ 来记录边 $(u_i,v_j)$ 的终点 $v_j$；另一个包含 $n$ 个元素的数组 $L[1..n]$ 来记录边 $(u_i,v_j)$ 的前驱边 $(u_{i-1},v_{i-1})\in L$。由于 $E[v_j]=L[u_i]$,因此可以根据一条边的终点找到它的前驱边；而根据一条边的前驱边就可以找到它的所有后继边。

#### 1.1.2 邻接多重表
邻接多重表（Adjacency Multilist，AML）是邻接表的扩展。它将每个顶点对应的邻居建立一个双向链表。顶点的度数可以快速计算出来，而修改邻接表所需的时间为 $O(|E|+|V|)$。

### 1.2 邻接表
邻接表存储图信息的另一种方法是用一个 $n$ 大小的数组来表示所有的顶点，每一个数组元素代表一个顶点所拥有的邻居的数量，以及对应于每个邻居的位置。对于顶点 $v_i$ ，其邻居的位置可以通过下标 $i+1$ 加上 $1$ 至 $E[i]+i$ 得到。而邻居的数量可以通过元素 $i$ 得到。

### 1.3 边集数组
边集数组（Edge Set Array，ESA），是用一个数组来表示图的边。对于 $G$ 的每条边 $(u_i,v_j)$ ，都可以简单地将 $uv_j$ 置为 1，表示该边存在。这样的边集数组可以用于判断某两个顶点是否有边相连，或者求出图的总边数。但是，边集数组占用的空间比较大。因此，如果 $m$ 为边数目，那么边集数组需要的空间为 $O(\frac{mn}{w})$,其中 $w$ 为整数位宽。

### 1.4 度序列
度序列（Degree Sequence）是一个数组，数组元素 $d_i$ 表示第 $i$ 个顶点的度数。度序列可以方便地通过一个数组来进行编码。另外，可以利用度序列来统计图的各种性质。例如，$\sum_{i=1}^n d_i = 2m$ 。其中，$m$ 是图的边数目。

## （2）图的遍历方法
### 2.1 深度优先搜索（DFS）
深度优先搜索（Depth First Search，DFS），是一种用于遍历图的算法。它沿着图的当前分支一直探索下去，直到所有的分支都已经被访问过一次。它采用“深”的方式遍历，所以先处理深的结点，后处理浅的结点。

DFS 在最坏情况下的时间复杂度是 $O(|V|+|E|)$ 。但平均情况下的时间复杂度稍微好一点。因为 DFS 每个顶点只访问一次，并且每个顶点的邻居都被访问一次。

#### 2.1.1 拓扑排序
在应用 DFS 之前，需要先将图变成 DAG（有向无环图）。为了保证DAG，需要满足以下两个条件：

1. 每个顶点只能出现一次。
2. 如果 $v_i$ 和 $v_j$ 有边相连，那么 $v_i$ 应该在 $v_j$ 之前出现。

如果满足以上两点要求，那么可以利用 DFS 对顶点按照拓扑排序进行排序。也就是先访问入度为零的顶点，然后再访问入度为 $1$ 的顶点，依次类推。这种排序结果称为拓扑排序。

#### 2.1.2 路径压缩
如果发现了一个节点的邻居已经被遍历过了，那么就跳过这个邻居。这是因为把节点的深度设为 $\infty$ 会让算法更加快捷。路径压缩（Path Compression）就是通过将父亲节点的引用指针改为子节点来实现的。

### 2.2 宽度优先搜索（BFS）
宽度优先搜索（Breadth First Search，BFS），又称广度优先搜索，是一种用于遍历图的算法。与 DFS 不同的是，它沿着图的宽度来搜索，每次只看相邻的一个分支。

BFS 的时间复杂度在最坏情况下为 $O(|V|+|E|)$，但平均情况下的时间复杂度为 $O(\frac{|V|+|E|}{\sqrt{|V|}})$。

## （3）最小生成树（MST）
最小生成树（Minimum Spanning Tree，MST）是指一颗权值最小的连通子图。通常来讲，求解 MST 的算法可以分为两步：第一步，使用 Prim 或 Kruskal 算法计算出一棵最小生成树；第二步，增加一条边，使得新生成的子图不是 MST，重复执行第二步，直到最后获得一个 MST 为止。

### 3.1 Prim 算法
Prim 算法（Prim's Algorithm）用于计算最小生成树。它首先将图中任意一个顶点作为源点，并找出连接到该源点的最短边，将它加入到 MST 里。之后，便把那些连接到新加入的顶点上的边全部删除掉，并找出连接到那些被删除边的顶点的最短边，再添加到 MST 里。重复执行这个过程，直到所有的顶点都已经在 MST 里。

### 3.2 Kruskal 算法
Kruskal 算法（Kruskal's Algorithm）也是用于计算最小生成树。它首先将图中所有边按照权值的大小排序。然后，从小到大依次考虑每条边，如果这条边连接的两个顶点均不属于同一个联通块（Connected Component），那么就将这条边加入到 MST 里。之后，便删去这条边，并重复执行这个过程，直到 MST 的边数等于 $n-1$ （这意味着有 $n-1$ 条边构成了 MST）。

Kruskal 算法比 Prim 算法更适合计算稠密图的 MST。

# 3.核心算法原理和具体操作步骤以及数学模型公式详细讲解
## （1）稀疏图模型——图的表示方法
### 1.1 随机图模型
假设图 $G=(V,E)$ 有 $n$ 个顶点，每个顶点 $v_i$ 之间都有一个随机的权值，并且满足 $0<=w_{ij}<1$。随机图模型的目标是，通过估计 $w_{ij}$ 的期望，来构造一个真实的图 $G$ 。

#### 1.1.1 模拟退火算法
模拟退火算法（Simulated Annealing）是优化算法的一种。它的基本思想是，在初始状态以一定概率接受一个解，然后在一个低温度温度范围内进行迭代。在每一步迭代中，接受概率由上一次迭代的值衰减，并降低一定的温度。当温度达到一定程度后，停止接受解，转而接受较差的解，继续调整温度。这样，可以达到快速收敛的效果。

模拟退火算法用来估计随机图模型中的权值，如下面的算法所示。

```python
import random

def estimate():
    # randomly generate a graph with the same number of vertices and edges as G
    A = [[random.uniform(0, 1) for j in range(n)] for i in range(n)]

    return A

T = initial_temperature()   # set initial temperature T
while True:
    A = estimate()           # estimate w_{ij} using A
    k = np.argmin([np.mean((A - G)**2) for G in ground_truth])    # find best ground truth graph that fits A most closely
    cost = mst_cost(A)       # calculate minimum spanning tree cost of A
    
    delta = cost - current_mst_cost     # calculate change in minimum spanning tree cost since last iteration
    
    if delta < 0 or math.exp(-delta / T) > random.random():      # accept new estimate with probability exp(-delta/T)
        return A, cost
    
    T *= cooling_factor          # decrease temperature by factor of cooling_factor
```

### 1.2 完全图模型
完全图模型假设 $G$ 的每两个顶点之间都有一条边。模型的目标是在随机图模型的基础上引入约束条件，使得每条边的权值为 $1$ 。简而言之，就是假设边的权值不再服从均匀分布，而是趋于 $1$ 。

#### 1.2.1 模拟退火算法的修正版
修正后的模拟退火算法可以用如下所示的伪代码表示。

```python
import random

def estimate():
    # randomly generate a fully connected graph with the same number of vertices as G
    A = [[1 if (i == j or random.random() <= alpha) else 0 for j in range(n)] for i in range(n)]

    return A

T = initial_temperature()   # set initial temperature T
alpha = anneal_rate        # set annealing rate alpha
while True:
    A = estimate()           # estimate weighted adjacency matrix A based on model assumptions
    k = np.argmin([np.mean((A - G)**2) for G in ground_truth])    # find best ground truth graph that fits A most closely
    cost = mst_cost(A)       # calculate minimum spanning tree cost of A
    
    delta = cost - current_mst_cost     # calculate change in minimum spanning tree cost since last iteration
    
    if delta < 0 or math.exp(-delta / T) > random.random():      # accept new estimate with probability exp(-delta/T)
        return A, cost
    
    T *= cooling_factor          # decrease temperature by factor of cooling_factor
```

### 1.3 二部图模型
二部图模型（Binary Graph Model）假设图 $G$ 只含有两组顶点 $S$ 和 $T$ ，且 $S$ 和 $T$ 不相交。二部图模型的目标是，在假设 $S$ 和 $T$ 不相交的同时，引入边的权值。简而言之，就是假设每个顶点至少有一条边指向另一组的顶点。

#### 1.3.1 模拟退火算法的修正版
修正后的模拟退火算法可以用如下所示的伪代码表示。

```python
import random

def estimate():
    # randomly select S and T such that there are no common neighbors between them
    S = [i for i in range(n//2)]
    while any(set(adj[x]).intersection({y for y in S if x!= y})) or len(S)!= 2*m:
        S = random.sample(range(n), n//2)
        
    T = list(set(range(n)) - set(S))
    while all(any(set(adj[t]).intersection({s}) and s not in adj[t][:pos]) for pos in range(len(adj[t])) for t in S):
        T = random.sample(range(n), n//2)
        
    # randomly assign weights to each edge within S and connect it to nearest vertex in T
    A = [[0]*n for _ in range(n)]
    for u in S:
        for v in {x for x in T if x!= u}:
            dists = sorted([(dist(u, x), x) for x in T if x!= u], key=lambda p: p[0])[::-1]
            
            A[u][dists[0][1]] = random.gauss(1/(max(beta**p * len(T)/2, beta)), 0.01)
            
    # randomly assign weights to remaining edges connecting vertices from both groups
    num_edges = n // 2 + len(S)
    used_vertices = set(S).union(T)
    while sum(sum(row[:pos]) for row in A for pos in range(len(row))) < num_edges:
        # choose two unconnected vertices at random
        u, v = random.sample(used_vertices, 2)
        
        dists = [(dist(u, x), x) for x in T if x not in adj[u]][:-1]
        rand_num = random.random()
        
        total_weight = max(sum(A[u]), sum(A[v]), 1)
        candidate_weights = [sum(A[x]) / total_weight + (rand_num + i) / len(dists)
                             for i, (_, x) in enumerate(sorted(dists))]
        A[u].append(candidate_weights[0])
        A[v].append(candidate_weights[-1])
        
    return A

T = initial_temperature()   # set initial temperature T
beta = min_connection_prob  # set connection probability threshold beta
while True:
    A = estimate()           # estimate weighted adjacency matrix A based on model assumptions
    k = np.argmin([np.mean((A - G)**2) for G in ground_truth])    # find best ground truth graph that fits A most closely
    cost = mst_cost(A)       # calculate minimum spanning tree cost of A
    
    delta = cost - current_mst_cost     # calculate change in minimum spanning tree cost since last iteration
    
    if delta < 0 or math.exp(-delta / T) > random.random():      # accept new estimate with probability exp(-delta/T)
        return A, cost
    
    T *= cooling_factor          # decrease temperature by factor of cooling_factor
```

### 1.4 小世界模型
小世界模型（Small-World Network Model）假设在任意两个相邻顶点间，存在一个概率为 $p$ 的转移连接，其余的边权值均为 $1$ 。

#### 1.4.1 模拟退火算法的修正版
修正后的模拟退热算法可以用如下所示的伪代码表示。

```python
import random

def estimate():
    # generate small world network by adding links with some probability p
    A = [[1 if (random.random() <= p) else 0 for j in range(n)] for i in range(n)]
    
    return A

T = initial_temperature()   # set initial temperature T
p = transition_probability # set transition probability p
while True:
    A = estimate()           # estimate weighted adjacency matrix A based on model assumptions
    k = np.argmin([np.mean((A - G)**2) for G in ground_truth])    # find best ground truth graph that fits A most closely
    cost = mst_cost(A)       # calculate minimum spanning tree cost of A
    
    delta = cost - current_mst_cost     # calculate change in minimum spanning tree cost since last iteration
    
    if delta < 0 or math.exp(-delta / T) > random.random():      # accept new estimate with probability exp(-delta/T)
        return A, cost
    
    T *= cooling_factor          # decrease temperature by factor of cooling_factor
```

## （2）图的基本操作——图的遍历、最小生成树、环检测、最短路径
### 2.1 图的遍历
图的遍历（Graph Traversal）是指遍历图的各个顶点的方法。主要分为深度优先搜索（DFS）和宽度优先搜索（BFS）。

#### 2.1.1 深度优先搜索
深度优先搜索（DFS）算法采用递归的方法，先访问某一顶点，然后再分别访问该顶点的邻居，直到所有顶点都被访问完毕。为了避免循环，可以在访问某个顶点之前，保存该顶点的深度，以此来确保不会进入回路。

#### 2.1.2 宽度优先搜索
宽度优先搜索（BFS）算法采用队列的方法，首先将源点加入队列，然后不停地出队，访问队列头的顶点，然后将该顶点的邻居加入队列，继续如此，直到队列为空。

### 2.2 最小生成树
最小生成树（Minimum Spanning Tree，MST）是指一颗权值最小的连通子图。一般来说，求解 MST 的算法可以分为两步：第一步，使用 Prim 或 Kruskal 算法计算出一棵最小生成树；第二步，增加一条边，使得新生成的子图不是 MST，重复执行第二步，直到最后获得一个 MST 为止。

#### 2.2.1 Prim 算法
Prim 算法（Prim's Algorithm）用于计算最小生成树。它首先将图中任意一个顶点作为源点，并找出连接到该源点的最短边，将它加入到 MST 里。之后，便把那些连接到新加入的顶点上的边全部删除掉，并找出连接到那些被删除边的顶点的最短边，再添加到 MST 里。重复执行这个过程，直到所有的顶点都已经在 MST 里。

#### 2.2.2 Kruskal 算法
Kruskal 算法（Kruskal's Algorithm）也是用于计算最小生成树。它首先将图中所有边按照权值的大小排序。然后，从小到大依次考虑每条边，如果这条边连接的两个顶点均不属于同一个联通块（Connected Component），那么就将这条边加入到 MST 里。之后，便删去这条边，并重复执行这个过程，直到 MST 的边数等于 $n-1$ （这意味着有 $n-1$ 条边构成了 MST）。

Kruskal 算法比 Prim 算法更适合计算稠密图的 MST。

### 2.3 环检测
环（Cycle）是一个图中顶点的集合，其任意两个顶点之间都存在一条边。一个无环图中的所有顶点可形成一个完整的闭环，称为欧拉回路（Eulerian Circuit）。

#### 2.3.1 欧拉回路检测算法
欧拉回路检测算法（Hierholzer’s Algorithm）用于检测图中的欧拉回路。该算法可以分为两个阶段：第一阶段，随机选择一开始点 $v_0$ ，然后顺时针画出该点到每个未访问的顶点的通路，如果有回路，则该图有欧拉回路，结束算法；第二阶段，重复第一个阶段，直到没有更多的回路可以被找到为止。

#### 2.3.2 判断是否为欧拉回路
判定某张图是否是欧拉图，可以使用 Hierholzer’s Algorithm 的第二阶段来做。具体操作如下：

1. 创建一个栈，把 $v_0$ 放进栈中。
2. 从栈中弹出顶点 $u$ ，如果 $u$ 没有未访问的邻居，而且栈中还有其他顶点，那么回溯。
3. 将 $u$ 的邻居中的任意一个点 $v$ 放入栈中。
4. 当栈中只剩下 $v_0$ 时，算法结束。如果发现回路，则返回 true，否则返回 false。

### 2.4 最短路径
最短路径（Shortest Path）是指从源顶点 $s$ 到目的顶点 $t$ 的最少边数，且始终保持每个顶点之间的权值不变。最短路径问题有许多变体，包括单源最短路径、所有 pairs 最短路径、三角剖分等等。

#### 2.4.1 Dijkstra 算法
Dijkstra 算法（Dijkstra's Algorithm）是一种用于计算单源最短路径的算法。算法的基本思想是，在初始化的时候，把源点加入已知距离的集合 $D$ ，并把源点到自己距离为 $0$ ，其他距离设置为正无穷。然后，每次从 $D$ 中选取距离最小的顶点 $u$ ，并更新距离其邻居的距离，如果某条边的权值比新的距离小，则更新。直到所有顶点的距离都更新完毕。

#### 2.4.2 Bellman-Ford 算法
Bellman-Ford 算法（Bellman–Ford algorithm）是一种用于计算单源最短路径的算法。算法的基本思想是，在初始化的时候，把源点加入已知距离的集合 $D$ ，并把源点到自己距离为 $0$ ，其他距离设置为正无穷。然后，每次从 $D$ 中选取距离最小的顶点 $u$ ，并更新距离其邻居的距离，如果某条边的权值比新的距离小，则更新。直到所有顶点的距离都更新完毕。

### 2.5 拓扑排序
拓扑排序（Topological Sort）是指对有向无环图（DAG，Directed Acyclic Graph）的顶点排序，使得其顺序能够决定对所有边的接受或拒绝。

#### 2.5.1 拓扑排序算法
拓扑排序算法（Topological sort algorithm）是一种用于拓扑排序的算法。算法的基本思想是，从图中选取入度为 $0$ 的顶点并输出，然后，从图中删除该顶点，同时，删除它和它的相邻边。然后，重复上述过程，直到图中所有顶点都输出或为空。

## （3）图的连通性
### 3.1 连通性的判定
判定一个图是连通的，可以用两种方式：

1. 使用 BFS 算法检测图中是否存在不连通的区域。
2. 检查图中是否存在奇圈。

#### 3.1.1 BFS 算法的改进版本
改进版的 BFS 算法可以检测图中是否有不连通的区域。它首先随机选择一个顶点 $v_0$ 作为源点，并对 $v_0$ 的相邻顶点 $v$ 进行广度优先搜索，如果在搜索过程中发现已经访问过的顶点，则证明该顶点不连通，返回 false；否则，证明 $v$ 连通，返回 true。

#### 3.1.2 判断是否存在奇圈
对于一个连通图，其唯一的欧拉回路的长度是 $2n-2$ 。如果一个图中不存在奇圈，那么所有点的度数是偶数。

### 3.2 连通子图
一个连通子图是指可以从任意一个点开始，顺时针或逆时针移动，都可以在不经过其他点的情况下，到达图中其余所有点的图。

#### 3.2.1 子图划分算法
子图划分算法（Cut Partitioning Algorithms）是一种用于确定连通图的连通子图的方法。该算法可以采用 DFS 方法来划分图中的连通子图。具体步骤如下：

1. 用 DFS 把图分割成多个连通分量。
2. 对于每一块连通分量，运行 BFS 或 DFS 算法，找出所有的点，并标记这些点作为图的一部分。

#### 3.2.2 Tarjan 算法
Tarjan 算法（Tarjan's Algorithm）是一种用于计算图的割点的方法。该算法采用 dfs 算法，并用两个栈来记录每个顶点的状态。当进入新的连通分量时，将其压入栈底。当当前顶点的邻接边都遍历完时，将其出栈，并从栈顶弹出一个顶点，如果该顶点为 $root$ ，那么说明它是一个割点，否则说明它是一个割边。

### 3.3 强连通分量
强连通分量（Strongly Connected Components，SCC）是一个图中的最大的完全子图。如果将一个无向图改造成有向图，那么每个强连通分量都将是一个环。

#### 3.3.1 并查集算法
并查集算法（Union-Find Algorithm）是一种用于维护连通图的强连通分量的方法。该算法可以从每个顶点开始，标记它们的祖先，并将两个顶点合并为一个集合，称为连通分量。当两个顶点处于不同的连通分量时，说明他们之间存在着一个环。

## （4）最大流与最小费用流
### 4.1 流网络模型
流网络（Flow Network）是一组描述网络流动特性的数据结构。它包括一个点集 $V$ 和一个边集 $E$ ，其中每条边 $(u,v)$ 关联着一个非负容量 $c$ ，以及两个端点 $u$ 和 $v$ 。流网络可以表示一个资源或物品流动的过程。

#### 4.1.1 无源汇点流网络模型
无源汇点流网络模型（Fountain Networks）是指在一个流网络中，不允许存在一个从源点到汇点的方向的边。

#### 4.1.2 有源汇点流网络模型
有源汇点流网络模型（Source-Sink Flow Networks）是指在一个流网络中，允许存在一个从源点到汇点的方向的边。

#### 4.1.3 容量限制流网络模型
容量限制流网络模型（Capacity Limit Flow Networks）是指在一个流网络中，每条边的容量有上下限，不能超过某个预先定义的值。

### 4.2 最大流
最大流（Maxflow）是指在一个流网络中，从源点 $s$ 到汇点 $t$ 的可行流的最大值。假设有一个容量为 $C$ 的流网络，当流 $f$ 发送到边 $(u,v)$ 上时，能够通过该边传输的最多的流量是多少呢？

#### 4.2.1 Ford-Fulkerson 方法
Ford-Fulkerson 方法（Ford–Fulkerson method）是一种用于计算最大流的算法。该算法可以采用 BFS 算法或 DFS 算法，并且每次选择一条有容量的边，尽可能多地发送流。该算法的时间复杂度为 $O(VE^2)$ 。

#### 4.2.2 Edmond-Karp 方法
Edmond-Karp 方法（Edmonds–Karp algorithm）是一种用于计算最大流的算法。该算法可以采用 BFS 算法或 DFS 算法，并且每次选择一条有容量的边，尽可能多地发送流。该算法的时间复杂度为 $O(VE^2)$ 。

### 4.3 最小费用流
最小费用流（Min Cost Maxflow）是指在一个带有货币容量限制的流网络中，从源点 $s$ 到汇点 $t$ 的可行流的最小费用值。假设有一个容量为 $C$ 的流网络，与最大流一样，当流 $f$ 发送到边 $(u,v)$ 上时，能够通过该边传输的最多的流量是多少呢？此外，假设有一个代价函数 $c(u,v)$ ，表示从顶点 $u$ 到顶点 $v$ 需要支付的代价。

#### 4.3.1 Goldberg-Tarjan 算法
Goldberg-Tarjan 算法（Goldberg-Tarjan algorithm）是一种用于计算最小费用流的算法。该算法可以采用 BFS 算法或 DFS 算法，并且每次选择一条有代价的边，尽可能多地发送流，并且该代价尽可能小。该算法的时间复杂度为 $O(VE^2 log V)$ 。

#### 4.3.2 Cholmod 算法
Cholmod 算法（CHold-MAtrix Decomposition algorithm）是一种用于计算最小费用流的算法。该算法可以采用同样的策略，只是采用了线性规划的方法来解决。该算法的时间复杂度为 $O(VE^2)$ 。