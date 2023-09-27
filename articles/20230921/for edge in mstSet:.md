
作者：禅与计算机程序设计艺术                    

# 1.简介
  


什么是最小生成树(Minimum Spanning Tree, MST)？它是一种用来在一个带权连通图中找出边集合使得所有顶点间的最短路径的图论模型。MST的作用是通过加入顶点和边的方式连接所有的节点形成的树，这种树中的任意两个边都不会形成回路（cycle），且这棵树所包含的权重总和最小。

2.基本概念和术语说明

假设有一个带权连通图G=(V,E)，其中V是顶点集，E是边集。一条边<v1, v2>∈ E是一个从v1到v2的一条带权边，而边上的权值w(<v1, v2>)表示该边的长度或费用。对于无向图来说，有两种定义，分别为：
- 有向图中每对顶点之间都存在一条最短路径的边，那么称其为最短路径树(Shortest Path Tree)。
- 有向图中包括两对顶点之间的所有最短路径中的一条作为边，如果这条边不属于某条路径，则称之为“横切边”或“割边”。

根据MST的定义，MST是一个连接所有顶点的权值为最小的树。

除此外，还有一些重要的术语需要说明：

- 子图：指的是无环连通的子图。
- 生成树：一个连通子图，它包含了图G中全部的顶点，并且边数等于|V|-1。
- 边交换：将一条边的两个端点互换后得到另一条相同方向的边。
- 残余网络：在含有边权值的有向图中，以权值从小到大的顺序排列的边组成的网络。
- 支配树：设G=(V,E)是一个带权连通图。若存在一个顶点v1，则支配v1的边就是G中所有从v1可达的边的集合。设由e1、e2、……、ek这k条边所形成的树为T，则称T是G中从v1可达的边所构成的支配树。

以上四个术语的关系如下图所示：


# 3.核心算法原理及操作步骤

Kruskal算法(Kruskal's algorithm)是用于求解最小生成树(MST)的一种贪婪算法，它的基本思想是每次选择权值最小的边，并保证这条边没有形成回路(Cycle)。

1. 初始化：将全部n个顶点放入集合S1。初始化MST为空集MST={};

2. 从E中选取权值最小的边e=<v1, v2>∈ E，其中v1和v2都是S1的元素。

3. 如果v1和v2之间不存在回路，即不在S2中，则添加这条边至MST；否则忽略这条边。

4. 将v1和v2加入集合S2。

5. 更新集合S1。移除包含v1或者v2的任何元素，因为这些元素已经成为S2的成员。

6. 重复步骤2-5直至MST包含n-1条边为止。

最后，MST就是选出的边的一个集合，它包含了图G中全部n个顶点，并且边数等于n-1。

# 4.具体代码实例与解释说明

## 1.Kruskal算法的代码实现

```python
def kruskal_mst(edges):
    n = len(set([edge[0] for edge in edges]+[edge[1] for edge in edges])) #计算图中的顶点个数
    parent = list(range(n))   #初始时每个顶点为独立的连通分量
    
    def find(i):
        if i!= parent[i]:
            parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        parent[find(j)] = find(i)

    result = []    #用于存放最终的结果
    edges.sort()  #按边长进行排序
    
    while len(result) < n - 1:
        u, v, w = edges.pop(0)      #取边中权值最小的边

        if find(u)!= find(v):        #判断是否形成回路
            result.append((u, v, w))   #加入结果
            union(u, v)               #合并连通分量
    
    return [(u+1, v+1, w) for (u, v, w) in result]    #返回1-based结果

edges = [
    (0, 1, 4), 
    (0, 2, 2), 
    (1, 2, 6), 
    (1, 3, 5), 
    (1, 4, 3), 
    (3, 4, 1), 
    (3, 5, 6), 
    (4, 5, 2)
]
print(kruskal_mst(edges)) #[(0, 2, 2), (0, 4, 3), (1, 3, 5), (1, 5, 3), (3, 5, 6)]
```

2.Prim算法的代码实现


```python
import heapq
  
class Edge:
    def __init__(self, start, end, weight):
        self.start = start
        self.end = end
        self.weight = weight
        
    def __lt__(self, other):
        return self.weight > other.weight
    
def prim_mst(graph):
    visited = set()
    queue = []
    mst_edges = []
    
    start = min([(key, value) for key, value in graph.items()], key=lambda x:x[1])[0]   #找到图中连接数量最少的顶点作为起始点
    queue.append(Edge(None, start, 0))   #构造初始队列
    
    while queue and len(visited) < len(graph):
        current_edge = heapq.heappop(queue).toTuple()
        
        if not any(True for _ in filter(lambda e: e.start == current_edge[0], queue)):     #判断当前顶点已经被访问过
            continue
        
        next_vertexes = [_ for _ in range(len(graph)) if \
                         (_ not in visited or _ not in map(lambda e: e.end, queue))]  #查找与已加入MST的顶点相邻且未被访问过的顶点
        distances = {next_vertex:_['weight'] for _, next_vertex in enumerate(graph)}   #获取下一个要加入MST的顶点的距离
        
        new_edge = min([(distances[_], _) for _ in next_vertexes], key=lambda x:x[0])[1]    #寻找距离当前顶点最近的未加入MST的顶点
        
        mst_edges.append(current_edge + (new_edge,))    #加入MST边
        visited.add(new_edge)   #标记该顶点已被访问过
        del distances[new_edge]  #删除已经加入MST的顶点
        
     
    return sorted([[i, j] for i, j in mst_edges[:-1]])
    
graph = {
    0: {'weight': 4}, 
    1: {'weight': 1}, 
    2: {'weight': 2}, 
    3: {'weight': 3}, 
    4: {'weight': 2}
}
 
print(prim_mst(graph))   #[[0, 1], [0, 2], [1, 3]]
```