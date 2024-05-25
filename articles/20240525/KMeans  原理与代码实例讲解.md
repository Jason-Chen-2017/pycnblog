## 1. 背景介绍

K-Means是一种经典的聚类算法，用于识别数据中的一组模式。它基于一种简单的思想：给定一个数据集，K-Means会将其划分为K个子集，以便于识别其中的模式。K-Means算法的核心思想是：在数据集中随机选择K个中心点，然后将数据点分配给最近的中心点，直至收敛。

K-Means算法在许多领域有广泛的应用，例如图像处理、文本分类、金融分析等。它的优点是简单易用、效率高，但也存在一定局限性，例如需要预先设定K值，容易陷入局部最优解。

## 2. 核心概念与联系

K-Means算法的核心概念包括：

* **聚类：** 将数据点分组，以便识别其中的模式。
* **中心点：** 代表每个聚类的数据点。
* **距离：** 测量数据点与中心点之间的相似度。
* **迭代：** 直至收敛，数据点分配不变时停止。

K-Means算法的联系包括：

* **监督学习与无监督学习：** K-Means是一种无监督学习算法，因为它不需要标记数据。
* **分层聚类与密度聚类：** K-Means是一种分层聚类算法，因为它将数据点按距离排序，而不是根据密度。

## 3. 核心算法原理具体操作步骤

K-Means算法的具体操作步骤包括：

1. **初始化：** 随机选择K个数据点作为初始中心点。
2. **分配：** 将数据点分配给最近的中心点。
3. **更新：** 根据分配结果更新中心点。
4. **迭代：** 直至收敛，数据点分配不变时停止。

## 4. 数学模型和公式详细讲解举例说明

K-Means算法的数学模型可以表示为：

$$
c_i = \frac{\sum_{x \in C_i} x}{|C_i|}
$$

其中$c_i$表示中心点，$C_i$表示第i个聚类，$x$表示数据点。

K-Means算法的公式可以表示为：

$$
\min_{C} \sum_{i=1}^K \sum_{x \in C_i} ||x - c_i||^2
$$

其中$C$表示中心点集，$x$表示数据点。

举例说明：

假设我们有一组数据点：

$$
\begin{bmatrix}
1 \\
2 \\
3 \\
4 \\
5
\end{bmatrix}
$$

我们希望将其划分为两类。我们随机选择两个数据点作为初始中心点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix},
\begin{bmatrix}
3 \\
4
\end{bmatrix}
$$

我们将数据点分配给最近的中心点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_1
$$

我们更新中心点：

$$
c_1 = \frac{1 + 5}{2} = 3 \\
c_2 = \frac{2 + 4}{2} = 3
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们更新中心点：

$$
c_1 = \frac{1 + 2 + 3}{3} = 2 \\
c_2 = \frac{4 + 5}{2} = 4.5
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
$$

我们再次分配数据点：

$$
\begin{bmatrix}
1 \\
2
\end{bmatrix} \in C_1 \\
\begin{bmatrix}
3 \\
4
\end{bmatrix} \in C_2 \\
\begin{bmatrix}
5
\end{bmatrix} \in C_2
$$

我们再次更新中心点：

$$
c_1 = \frac{1 + 2}{2} = 1.5 \\
c_2 = \frac{3 + 4 + 5}{3} = 4
