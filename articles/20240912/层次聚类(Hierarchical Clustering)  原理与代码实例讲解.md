                 

# 层次聚类（Hierarchical Clustering）- 原理与代码实例讲解

## 目录

- [1. 层次聚类的概念](#1-层次聚类的概念)
- [2. 层次聚类的原理](#2-层次聚类的原理)
  - [2.1 聚类层次树（Dendrogram）](#21-聚类层次树dendrogram)
  - [2.2 距离度量方法](#22-距离度量方法)
  - [2.3 聚类算法](#23-聚类算法)
- [3. 代码实例](#3-代码实例)
  - [3.1 数据准备](#31-数据准备)
  - [3.2 层次聚类实现](#32-层次聚类实现)
  - [3.3 聚类层次树可视化](#33-聚类层次树可视化)
- [4. 总结](#4-总结)
- [5. 参考文献](#5-参考文献)

## 1. 层次聚类的概念

层次聚类是一种无监督机器学习方法，它通过将数据集中的每个数据点作为单独的簇开始，然后逐步合并距离较近的簇，最终形成一个层次结构。这种方法的优势在于它可以清晰地展示数据之间的相似性和距离关系，便于分析和解释。

## 2. 层次聚类的原理

### 2.1 聚类层次树（Dendrogram）

聚类层次树（也称为Dendrogram）是一个图形化的表示，展示了聚类过程中簇的合并过程。每个节点表示一个簇，节点之间的连线表示簇之间的合并过程。聚类层次树的根部表示整个数据集，叶子节点表示原始数据点。

### 2.2 距离度量方法

距离度量是层次聚类中一个重要的步骤，用于计算簇之间的相似度。常用的距离度量方法包括：

- 欧氏距离（Euclidean distance）
- 曼哈顿距离（Manhattan distance）
- 切比雪夫距离（Chebychev distance）

### 2.3 聚类算法

层次聚类的算法可以分为两种类型：自下而上（凝聚）和自上而下（分裂）。常见的算法包括：

- 单链接（Single Linkage）
- 全链接（Complete Linkage）
- 平均链接（Average Linkage）
- 中位数链接（Median Linkage）

## 3. 代码实例

### 3.1 数据准备

我们使用 sklearn 库中的鸢尾花（Iris）数据集来演示层次聚类的实现。

```python
from sklearn.datasets import load_iris
import numpy as np

iris = load_iris()
X = iris.data
```

### 3.2 层次聚类实现

```python
from sklearn.cluster import AgglomerativeClustering

# 设置链接方法为平均链接
clustering = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='average')

# 拟合聚类模型
clustering.fit(X)

# 获取聚类结果
labels = clustering.labels_

# 获取簇之间的距离
distances = clustering.distances_
```

### 3.3 聚类层次树可视化

```python
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram

# 绘制聚类层次树
plt.figure(figsize=(10, 7))
dendrogram(distances)
plt.title("Iris Dataset Agglomerative Clustering Dendrogram")
plt.xlabel("Index")
plt.ylabel("Distance")
plt.show()
```

## 4. 总结

层次聚类是一种重要的无监督学习方法，适用于探索性数据分析。本文介绍了层次聚类的概念、原理以及实现过程，并通过实例展示了如何使用 Python 中的 sklearn 库进行层次聚类。

## 5. 参考文献

- [1] sklearn 官方文档 - AgglomerativeClustering <https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering>
- [2] Machine Learning - A Probabilistic Perspective, Kevin P. Murphy, Chapter 14.3
- [3] 统计学习方法，李航，第四章

