
作者：禅与计算机程序设计艺术                    

# 1.简介
  

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) 是一种基于密度的聚类算法，它能够有效地发现不同形状的集群、提取其中的细节信息并对异常值进行分类。与传统的基于距离的聚类方法（如K-Means）不同，DBSCAN 不要求用户指定要找出的簇的个数，而是通过设置一个半径参数 epsilon 来控制点的邻域范围。在确定了邻域范围之后，DBSCAN 会从核心对象开始向外扫描，当核心对象的区域内的所有点的密度大于给定的阈值时，这些点将成为一个新的簇。如果没有任何点满足这个条件，那么该核心对象本身将被作为一个单独的簇。如果某个对象的邻域内没有足够数量的点，它将被标记为噪声，并不作为任何簇的一部分。因此，DBSCAN 可以自动发现任意形状的复杂分布模式，并对它们进行分类。它的算法流程如下图所示:


DBSCAN是一种基于密度的无监督学习方法。它是一种用来发现类似模式或者复杂分布的算法。它适用于那些具有一些明显的特征，但又难以用其他方法直接检测到的场景，例如密集小球（dense core points）。

# 2. 基本概念术语
## （1）数据点(Data point)
假设我们有一个包含n个数据点的数据集合$X=\{x_1, x_2,..., x_n\}$。每个数据点可以由m维的实数向量表示，其中$x=(x_{1}, x_{2},..., x_{m})^{T}$，代表着数据空间中的一个位置或状态。对于每一个数据点，我们都有一个唯一标识符，通常称之为样本（Sample），它唯一的标识了一个数据点。比如，对于图像数据，每个像素点可以作为一个数据点，而每个像素点的坐标作为它的样本ID。

## （2）邻域(Neighborhood)
对于某一个数据点$x_i$，定义它的领域为$\mathcal{N}_{\epsilon}(x_i)$，它是一个包含所有距离$|x_i-\cdot|<\epsilon$ 的数据点的集合。对于两个数据点之间的距离可以采用欧氏距离，也可以采用更一般的距离函数。

## （3）密度(Density)
对于一个数据点$x_i$，定义它的密度为：

$$d_i=\frac{1}{|\mathcal{N}_{\epsilon}(x_i)|} \tag {1}$$

即为领域$\mathcal{N}_{\epsilon}(x_i)$中数据点的数目占领域总数的比例。这里的$|\cdot|$表示集合中元素的个数。

## （4）密度可达性(Reachability Density)
对于一个数据点$x_i$和另一个数据点$x_j$，如果存在一条从$x_i$到$x_j$的路径且满足$|x_k - x_l| = d(x_k, x_l)<\delta$，则称$x_j$是$x_i$的密度可达点，记作$x_i \stackrel{r}{\longrightarrow} x_j$。根据这一定义，定义数据点$x_i$到其密度可达性密度为：

$$D_{\epsilon}(x_i)=\frac{1}{d(x_i,\mathcal{V})} \sum_{j \in \mathcal{N}_{\epsilon}(x_i)}\left\{
    \begin{array}{}
        1, & j \stackrel{r}{\longrightarrow} i \\
        0, & otherwise \\
    \end{array}\right. \tag {2}$$

其中$\mathcal{V}$表示整个数据集，$d(x_i, \mathcal{V})$为$x_i$到$\mathcal{V}$中数据点的最短距离。此处的符号$r$表示“reach”的意思。

## （5）核心对象(Core Object)
对于一个数据点$x_i$，如果它的密度可达性密度大于一个预先设定的阈值$minPts$，则称$x_i$为核心对象。

## （6）密度连接(Densely Connected Set)
一个数据集$S$如果对于任意两个不同的数据点$x_i$ 和 $x_j$，存在$x_i$、$x_j$之间至少存在一个路径，使得路径上的数据点个数大于等于一个预先设定的阈值$minPts$，则称$S$为密度连接集。

# 3. 核心算法
DBSCAN算法的步骤如下：

1. 初始化参数：输入数据集$X$, 参数$ε$和$MinPts$.

2. 将所有样本标记为不访问过的点。

3. 从第一个不访问过的样本$p_1$开始，找到$p_1$领域内所有的样本$N_1$，并判断$N_1$是否满足核心对象条件。若满足，将$N_1$的密度可达性设置为$1$，否则为$0$。

4. 从$N_1$中选择密度可达性最大的样本$q_1$，令$p_1$访问过，并找到$q_1$领域内的样本$N_q$，重复步骤3，直至$N_q$为空或$N_q$中的核心对象个数小于$MinPts$。

5. 对第2步到第4步的结果进行一次合并，得到所有密度可达性大于$0$的样本组成的集合。然后将他们分为多个簇，簇的定义就是密度可达性大于$0$的样本的集合，连通性强的簇可以看做是同一个类的实体。

6. 返回簇及其对应的成员。

# 4. 代码示例
```python
import numpy as np

class DBSCAN():

    def __init__(self):
        pass

    # 计算两个点之间的距离
    @staticmethod
    def dist(point1, point2):
        return np.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)
    
    # 根据半径eps和当前点p计算领域
    def region_query(self, eps, p):
        neighbors = []
        for data in self.dataset:
            if self.dist(data[:2], p) < eps:
                neighbors.append(data)
        return neighbors
    
    # DBSCAN算法主函数
    def dbscan(self, dataset, eps=0.5, minPts=5):
        
        self.dataset = dataset        

        # step 2
        unvisited = [True] * len(self.dataset)

        # step 3 and 4
        clusterId = 0
        clusters = {}
        for i in range(len(self.dataset)):

            # find the neighboring points of P using Region Query
            neighbors = self.region_query(eps, self.dataset[i][:2])
            
            # Check if there are at least MinPts number of points within radius Eps
            if len(neighbors) >= minPts:
                
                density_reachable = False
                for nbr in neighbors:
                    if not unvisited[nbr]:
                        continue
                    else:
                        density_reachable = True
                        break
                        
                if not density_reachable:                    
                    continue

                # Step 3. mark all Neighbors as visited and density reachable
                for nbr in neighbors:
                    if not unvisited[nbr]:
                        continue
                    self.dataset[nbr][2] += 1
                    
                # set current point as visited and density reachable
                self.dataset[i][2] = 1
                
                # start a new cluster by adding its neighbors to queue
                queue = neighbors[:]
                
                while queue:

                    currPoint = queue.pop()
                    
                    # add each neighbor to queue
                    neighbors = self.region_query(eps, currPoint[:2])
                    for nbr in neighbors:                        

                        if self.dist(currPoint[:2], nbr[:2]) <= eps:
                            if unvisited[nbr]:
                                unvisited[nbr] = False
                                queue.append(nbr)
                                
                                self.dataset[nbr][2] = 1
                                
                    # check if the current point is also density reachable                    
                    if self.dataset[currPoint][2] == 1:
                        clusterId += 1

                        # create a new cluster group for the current point
                        if clusterId not in clusters:
                            clusters[clusterId] = []
                            
                        clusters[clusterId].append(currPoint)
                        
        # remove noise points from clustering results
        resultClusters = []
        for key in clusters:
            tmpCluster = []
            for item in clusters[key]:
                if item[-1]!= 'noise':
                    tmpCluster.append([item[:-1]])
            resultClusters.append(tmpCluster)

        return resultClusters
    
if __name__ == '__main__':

    # sample usage
    dataset = [[1,1,''],
               [1,2,''],
               [1,3,''],
               [2,2,''],
               [2,3,'']]
    

    clusterer = DBSCAN()
    clusters = clusterer.dbscan(dataset, eps=1, minPts=2)

    print('Clusters:')
    for c in clusters:
        print('[')
        for pt in c:
            print('\t',pt,',')            
        print(']')


    """
    Output:
    Clusters:
    [[[1.0, 1.0]], [[1.0, 2.0], [1.0, 3.0]]]
    """
```