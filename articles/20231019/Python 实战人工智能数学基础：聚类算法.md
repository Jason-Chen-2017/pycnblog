
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


## 1.1 聚类算法简介
聚类算法（clustering algorithm）是利用数据中相似性或相关性信息将数据集划分为多个子集，每个子集内的数据点具有相同的属性或者特征，称为聚类。聚类算法的目标是使得数据的元素不属于同一个簇。
聚类的两种基本方法：
- 分割型聚类：将数据集中的对象划分到几个互斥的、没有重叠的子集中；
- 层次型聚类：按照距离或相似性的递增顺序将对象集分层；
常用聚类算法：
- K-means 算法：K-means 是最简单的无监督聚类算法，通过迭代的方法把 N 个数据点分成 K 个簇，使得簇内数据点之间的距离最小，簇间数据点之间的距离最大，并收敛到全局最优解。
- DBSCAN 算法：DBSCAN （Density-Based Spatial Clustering of Applications with Noise）算法是基于密度的聚类算法，可以发现任意形状的复杂高维空间中的离散数据簇。
- Agglomerative 算法：Agglomerative 算法是层次型的聚类算法，它是一种自下而上的合并算法，一步一步地合并两个相邻的子集直到所有对象都在一个子集中。
## 1.2 Python 中聚类算法的实现
Python 中常用的聚类算法库包括 scikit-learn 和 scipy 中 stats 模块。本文会对这两种聚类算法进行探讨。
### scikit-learn 中的 k-means 算法
scikit-learn 提供了一些用于聚类分析的工具函数，其中包括 `sklearn.cluster.KMeans` 函数。该函数通过最小化各组样本距离均值的方差来确定 K 个初始质心，然后更新这些质心使得各组样本距离的均值最小化，最后将各个样本分配到最近的质心所属的组中。具体流程如下图所示。
下面给出一个例子：
```python
import numpy as np
from sklearn.cluster import KMeans

X = np.array([[1, 2], [1, 4], [1, 0],[4, 2], [4, 4], [4, 0]])
model = KMeans(n_clusters=2) # 指定分成两类
y_pred = model.fit_predict(X)
print("Cluster labels: ", y_pred) # [0 0 0 1 1 1]
print("Cluster centers: ", model.cluster_centers_) # [[ 1.  2.] [ 4.  2.]]
```
### scipy 中的 DBSCAN 算法
scipy 的 stats 模块提供了 DBSCAN 算法，具体流程如下图所示。
下面给出一个例子：
```python
from scipy.spatial.distance import pdist, squareform
from scipy.stats import multivariate_normal
from scipy.stats import dbscan

def make_data():
    """生成数据"""
    n_samples = 100
    X = np.zeros((n_samples, 2))
    X[:n_samples // 3, :] = (np.random.rand(n_samples // 3, 2) -.5) * 2
    X[n_samples // 3:, :] = (np.random.rand(n_samples // 3, 2) +.5) * 2

    return X

def generate_outliers():
    """生成异常点"""
    outliers = []
    for i in range(10):
        x = np.random.uniform(-10, 10)
        y = np.random.uniform(-10, 10)
        z = np.random.uniform(.5, 1.5)
        if abs(x) < 2 and abs(y) < 2:
            continue

        cov = np.diag([z, z])
        rv = multivariate_normal([x, y], cov)
        point = rv.rvs()
        outliers.append(point)

    return np.array(outliers)


if __name__ == '__main__':
    np.random.seed(42)
    data = make_data().tolist() + generate_outliers().tolist()
    eps =.75
    min_samples = 3
    dists = squareform(pdist(data, 'euclidean'))
    clusters, _ = dbscan(dists, eps, min_samples)
    print('Cluster memberships:', clusters)
```