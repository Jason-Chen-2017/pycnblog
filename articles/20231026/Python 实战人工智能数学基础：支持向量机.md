
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


## 1.1 支持向量机（SVM）
支持向量机（support vector machine，SVM），一种监督学习方法，它利用训练数据集对输入空间进行划分（一般是高维空间），将正负样本分开。在二分类问题中，给定一个超平面（hyperplane）将输入空间分成两个区域。每一侧称为间隔边界（margin boundary）。
<center>
</center>


## 1.2 目的
目标是实现能够处理非线性的数据、找到特征之间的相关性并最大化分离两类数据的距离，即使在高维空间下也不失为一种有效的方法。

# 2.核心概念与联系
## 2.1 定义
支持向量机是机器学习的基本模型，由 Vapnik 和 Chervonenkis 在1995年提出。它的提出主要是为了解决的问题如下：
> 给定一个数据集合和一个标签集合，如何求得能够将数据集中的正样本和负样本分开并且尽量少地发生错误？也就是找到最合适的超平面。 

支持向量机的优化目标是:

$$\max_{\alpha}\quad \sum_{i=1}^{m}-\frac{1}{2}{\left[\mathbf{\alpha}_i(\mathbf{w}^T\cdot\mathbf{x}_i + b)+1\right]}+\sum_{j=1}^{n_y}({\alpha}_j^y(2-|\mathbf{\alpha}_j|)\\)
s.t.\quad y_i(\mathbf{\alpha}_i^T\cdot\mathbf{x}_i+b)\geq 1-\xi_i,\forall i \\[2ex]$$

其中，$\alpha$ 是模型参数，$\mathbf{w}$ 和 $b$ 是超平面的参数，$m$ 表示正例的个数，$n_y$ 表示所有正样本的个数。$\mathbf{x}_i$ 是第 $i$ 个训练样本的特征向量，$y_i$ 表示该样本的标签（是否是正样本），$-1$ 表示负样本，$\xi_i$ 表示第 $i$ 个训练样本违反松弛条件的松弛因子。这个问题是凸二次规划问题。

因此，支持向量机通过求解上述凸二次规划问题来找出最优的 $\mathbf{w}$, $b$, $\alpha$ 来完成分类任务。


## 2.2 SMO算法
SMO (Sequential minimal optimization)，即序列最小优化算法，是支持向量机的求解方法之一。SMO 的基本思想是通过启发式的方法从样本中选择两个变量，然后根据更新后的模型来调整另一个变量的值，来对目标函数进行极小化。循环往复的进行这一过程，直至收敛。

具体来说，SMO 算法的步骤如下：
1. 对每个样本，选取一个变量 $\alpha$ ，固定其他变量，计算目标函数的梯度。如果梯度小于零，那么令 $\alpha$ 增加；如果梯度大于零，那么令 $\alpha$ 减小；否则，保持不变。
2. 更新模型参数，包括 $\mathbf{w},b,$ 和 $\alpha_i$.
3. 检验是否满足停机条件。如果满足则退出循环，否则转到步1继续迭代。

<center>
</center>



## 2.3 拉格朗日对偶
拉格朗日对偶法（Lagrange duality），是求解凸二次规划问题的常用方法。具体来说，通过求解下列问题来求解原始问题的最优解：

$$
L(\alpha,\beta)=\sum_{i=1}^{m}\alpha_i-\frac{1}{2}\sum_{i,j=1}^{m}\alpha_i\alpha_jy_iy_j\left<\mathbf{x}_i,\mathbf{x}_j\right>+b^{\top}\left(\begin{array}{ccc}-1 & 1 & \cdots & 1\\y_1&\cdots&y_n&\end{array}\right)\left(\begin{array}{ccccc}\alpha_1\\\vdots\\\alpha_m\end{array}\right)\\[2ex]
\text{subject to } \quad 0\leq\alpha_i\leq C\quad \forall i\quad\text{(约束条件)},C>0,\alpha^\top\left(\begin{array}{c}-1 & -1 & \cdots & -1\\y_1&\cdots&y_n&\end{array}\right)\leq l\\[2ex]\beta=0
$$

这里，$\left<\mathbf{x}_i,\mathbf{x}_j\right>$ 表示 $\mathbf{x}_i$ 和 $\mathbf{x}_j$ 的内积。$\alpha$ 和 $\beta$ 分别表示原始问题中的拉格朗日乘子和对偶变量，$l$ 表示约束条件 $0\leq\alpha_i\leq C$ 中的右端 $C$ 。原始问题的目标函数值是 $\min_\alpha\max_\beta L(\alpha,\beta)$ 。

当 $C=\infty$ 时，等价于原始问题；当 $C$ 为某个有限值时，满足 KKT 条件。