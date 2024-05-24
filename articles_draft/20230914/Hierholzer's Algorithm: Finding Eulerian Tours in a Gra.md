
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Eulerian tour (游览路线) 是图论中的一个术语，它是一个无回路的图形路径，满足每条边都经过且只经过一次。在现实生活中，许多地区也被称作"回旅路线"或"游览路线"。可以把游览路线视为一段旅程的全貌、完整性、没有重复的一段。
但对于一些特殊的图来说，没有特定的起点和终点，而是存在着多个起始节点和终止节点，同时还可能有环。例如，有时候我们想从不同的城市出发，到达同一目的城市，就需要通过很多条路线。这些图可以说是**多起点、多终点和有向图**。但是，如何找到这样的游览路线却不容易。即使我们已经知道每个城市的出入口，但是仍然无法确定一条通用的游览路线。因此，Eulerian tour (游览路线) 概念的出现就是为了解决这一难题。

# 2.核心概念与术语
## 2.1 图（Graph）
在图论中，图是由顶点（vertex）和边（edge）组成的抽象模型。它用来表示对象之间的各种关系。在这个模型里，图由一系列顶点集合 V 和一系列边集合 E 表示。顶点通常用字母表示，比如 A、B、C、D、E；边则用箭头或者线条表示。如果两个顶点之间有一条边相连，就称这条边连接了这两个顶点。


如上图所示，这是一个表示四个顶点及其连接情况的图。每个顶点代表某个事物，如A、B、C、D；用边表示了它们之间存在的联系，比如AB、BC、CD、DA。

## 2.2 有向图（Directed Graph）
如果图中的边都有方向，那么就称该图为有向图（directed graph）。如果两个顶点之间有一条弧状边相连，表明从第一个顶点指向第二个顶点，而不是反过来。在有向图中，每个顶点的出边集称为出度（out-degree），而每个顶点的入边集称为入度（in-degree）。


如上图所示，这是一个表示两个顶点及其连接情况的有向图。边带有一个箭头指示它的方向。

## 2.3 无回路图（Acyclic Graph）
如果图中不存在任何回路（circuit），那么就称该图为无回路图（acyclic graph）。一个有向图是无回路的，当且仅当它是一个树。树是无回路的图，并且树上的任意两点间都有且只有一条路径相连。如果有一个图既不是树又不是无回路的，那我们就说它是不平衡的（skew）。


如上图所示，这是一个表示四个顶点及其连接情况的图。虽然有些边会造成回路（比如AB和BA），但它还是一个无回路图。另一个例子是表示无向图。

## 2.4 Eulerian tour（游览路线）
如果一个无回路图有欧拉回路，那么它就是一个 Eulerian tour （游览路线）。欧拉回路是在无回路图中具有以下性质的回路：任意选择一个顶点作为起点，然后沿着回路边走，直到返回原点。换句话说，从任一点出发都可以在回路中找到一圈。

欧拉回路还具备以下特性：

- 每次沿着回路边走一步都可到达一个顶点（简单路径），并且最后要回到起点（回路）。
- 如果回路边的数目为偶数，那么最终回到起点时的方向跟初始方向相同；否则，回到起点时的方向跟初始方向相反。

比如，下图是一个二叉树图，它具备欧拉回路：


如上图所示，这是一个表示五个顶点及其连接情况的二叉树。树根（root）A、B、C、D均只有一个出边，所以它是欧拉回路。最简单的欧拉回路是回路上所有顶点都有奇数度，这种情况下，最终回到起点时的方向跟初始方向相同。


如上图所示，这是一个表示七个顶点及其连接情况的有向图。虽然它的欧拉回路有四条边（比如ABCDE），但它不是无回路的，因为图中还有一条边连接C和D，造成了一个环。另外，它也不是二叉树，因为C、D、F、G都是只有一个出边的顶点。因此，图中存在不平衡。

# 3.核心算法
## 3.1 Hierholzer's Algorithm
Hierholzer’s algorithm 是一种计算欧拉回路的算法。该算法利用了一种称之为最小树计数的启发式方法，即以欧拉路径的长度为估计值，按照边长递增的方式构造最小权值的横切边界（crossing edge boundary）。然后再利用上述方法构建欧拉路径。该算法的时间复杂度为 O(VE^2)，其中 E 为边的数量，V 为顶点的数量。

下面给出算法流程图：


1. 将图中的所有顶点标记为未访问，初始化栈和横切边界。
2. 从任意顶点开始，将其标记为已访问，并放入栈中。
3. 当栈为空时结束算法。否则执行步骤4到6。
4. 弹出栈顶元素u，标记它为已访问。
5. 对每个邻接于u的顶点v：
   - 如果 v 未被访问过，则将其入栈，并标记为已访问。
   - 如果 u < v ，则将 (u, v) 加入横切边界。
6. 返回第6步。
7. 从横切边界中选择一条权值最小的边 e 。
8. 把 e 加入欧拉路径。
9. 删除横切边界中 e 的一条端点。
10. 返回第5步。

## 3.2 优化版 Hierholzer's Algorithm
除了原始版本 Hierholzer's Algorithm 外，还有一种常用的优化版算法——**Subset Sum Problem**. Subset Sum Problem 是指给定一个整数数组和目标值 target，求数组中是否存在子集和等于 target 的方案。对于一个大小为 n 的数组 arr，Subset Sum Problem 等价于寻找一个集合 S，使得 |S| ≤ k 且 ∑arr[x]∈ S，其中 x∈ [n], ∑arr[x]=target。

根据这个等价关系，可以看出，Subset Sum Problem 是一个寻找包含和等于 target 的子集的问题，它可以转换成一个充满约束条件的整数规划问题。利用整数规划求解 Subset Sum Problem 的时间复杂度为 O(nk)。

对于 Hierholzer's Algorithm，可以通过转化为 Subset Sum Problem 来求解，具体做法如下：

1. 初始化一个整数变量 sum 为零，一个二维数组 is_included ，其中 is_included[i][j] 表示是否选取第 i 个元素且当前总和小于等于 j 。
2. 设置一个 Boolean 函数 f(i, j)，用于判断是否选取第 i 个元素且当前总和等于 j 。
3. 对每个顶点 v，设置一个整数变量 outdegree(v) 表示顶点 v 的出度。
4. 对每条边 (u, v)，若 u < v ，则计算 is_included[v][i]+is_included[u][j-adj(u)] 是否满足 f(i+1,j) = True。也就是说，判断从 u 到 v 的边是否被包括进来，且当前的总和比之前的少了 adj(u) 。
5. 根据 Step 4 中的信息，可以得到一个子集和等于 target 的候选集。遍历整个图的所有顶点，检查每个顶点是否是候选集的一部分。
6. 从候选集中选择一条权值最小的边作为欧拉路径。
7. 删除此边的端点。
8. 返回第6步。

# 4.具体实现
## 4.1 Python 实现
```python
from typing import List

class Solution:
    def find_eulerian_tour(self, edges: List[List[int]]) -> List[int]:
        n = len(edges)
        
        # step 1
        not_visited = set([i for i in range(n)])
        stack = []
        crossing_boundary = set()

        while len(not_visited) > 0 and len(stack) > 0:
            if len(stack) == 0:
                break
            
            vertex = stack[-1]
            not_visited -= {vertex}

            # step 4 to 6
            adjacent_vertices = [e[1] for e in filter(lambda e: e[0]==vertex, edges)][:2]
            
            for neighbor in adjacent_vertices:
                if neighbor in not_visited:
                    stack.append(neighbor)
                    
                    crossing_edge = tuple(sorted((min(vertex, neighbor), max(vertex, neighbor))))
                    crossing_boundary.add(crossing_edge)
                
        # step 7 to 10        
        path = []
        
        while len(path)!= n*2-2:
            candidates = [(e[0], e[1]) for e in filter(lambda e: e not in crossing_boundary, [(i,j) for i in range(n) for j in range(n) if i<j])]
            candidate = min(candidates, key=lambda c: abs(sum([(1 if j>c[0] else (-1 if j<c[0] else 0)) * w for j,w in enumerate(edges[c[0]][c[1]])])) + ((len(path)+2)//2)*abs(c[0]-path[-2]))
            
            crossing_boundary.remove(tuple(sorted(candidate)))
            path += list(candidate)
            
        return path
```