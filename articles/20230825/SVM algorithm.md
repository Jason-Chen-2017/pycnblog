
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Support Vector Machine (SVM) 是一种二类分类模型，它通过设置超平面来最大化决策边界与两个类别间距离的最小值，使得支持向量到决策面的距离最小，即软间隔支持向量机（Soft-margin Support Vector Machine）。在实际应用中，我们需要对训练数据进行特征提取、降维处理后才能将原始数据投影到高维空间中，以便于进行线性可分的数据集的建模。SVM 的优点是解决了训练样本少导致的过拟合现象，以及核函数的非线性映射能力，能够有效地处理高维数据的复杂问题。

SVM 的基本模型由两部分组成：优化目标函数和优化方法。优化目标函数是一个二次规划问题，通过求解目标函数的最优解，得到软间隔支持向量机的最优超平面。优化方法可以是线性搜索法或牛顿法，用于快速找到最优解。

SVM 在工业界非常重要，因为它广泛应用于文本分类、图像识别、生物信息学等领域。SVM 在模式识别、数据挖掘、机器学习、图像处理、自然语言处理等众多领域都有着广泛的应用。其中，图像识别领域 SVM 的应用十分普遍。除此之外，SVM 还被广泛应用于金融市场风险预测、生物信息学分析、推荐系统等领域。

# 2.基本概念术语说明
## 2.1 定义
支持向量机（support vector machine，SVM）是一种二类分类模型，其决策函数为一个分离超平面，把正负两类的样本点完全正确分开。通过找到这样一个超平面，SVM 可以最大化两类样本点之间的“间隔”或者“距离”，并使得越远离分割面的样本点的权重越小。

举个例子，假设有一组线性不可分的数据点，如下图所示：

如果要将这些数据点划分成两个不同的类别，那么显然无法画出一条直线将它们完全分开。但是，如果引入松弛变量 $λ$，则可以做如下变换：
$$
\begin{align*}
&\text{max}_{\phi} & \quad \sum_{i=1}^n \lambda_i - \frac{1}{2}\sum_{i=1}^{n-1} \sum_{j=i+1}^n (\alpha_i-\alpha_j)^2 \\
&s.t.&\quad y_i(\vec x_i^T\phi+\xi_i)=1,\forall i\\
&\quad & 0\leq\alpha_i\leq C,\forall i\\
&\quad & \sum_{i=1}^n\alpha_iy_i=0.\\
\end{align*}
$$
上述约束条件表示：
1. $\phi$ 是超平面方程，$y_i(x_i^T\phi + \xi_i)$ 表示第 $i$ 个数据点到超平面的距离，$\xi_i$ 为松弛变量；
2. 每一个数据点都有一个对应的 $\alpha_i$ 参数，用来确定是否位于分割面的哪一侧；
3. 只允许 $\alpha_i$ 满足 $(0,C)$ 范围内的值，$C$ 为惩罚系数，用于控制误分率。

因此，SVM 通过寻找最优的超平面，来最大化分类间隔，同时满足所有数据点的要求。

## 2.2 支持向量
定义：对于线性可分数据集 $\{(x_1,y_1),(x_2,y_2),...,(x_N,y_N)\}$ ，令 $\hat y = sign(w^Tx + b)$ ，其中 $b$ 是截距项，$w=(w_1,w_2,...w_d)^T$ 为线性可分超平面参数，那么对于任意给定的输入实例 $x$，有：
$$
y(w^Tx+b) = \tilde y = sign(w^Tx+b)
$$

如果 $y(w^Tx+b)$ 为负值，则称该数据点为支持向量（support vector），反之则不属于支持向量。记作 $\supp(x)$ 。支持向量构成了对偶问题的拉格朗日因子。

## 2.3 对偶问题
SVM 的对偶形式的目标函数是：
$$
\min_{\alpha} \frac{1}{2}\left|\|W\alpha-\boldsymbol{b}\right|\|^2 + \sum_{i=1}^m\rho_i\bigg[\xi_i^{(i)}\xi_i^{(i)} - \xi_i^{(i)}\bigg] + \mu ||\alpha||_1.
$$

其中 $W=[w_1; w_2;... ; w_d]$ 为低秩矩阵，$\boldsymbol{b} = [b]$ 为目标值，$i$ 代表第 $i$ 个数据点，$\xi_i^{(i)}=\hat y_i - y_i$ 为松弛变量，$\rho_i>0$ 为软间隔惩罚系数，$\mu>0$ 为正则化参数。

对偶问题可以用拉格朗日乘子的方法求解。首先定义拉格朗日函数：
$$
L(\alpha,\xi,\beta,\gamma,\zeta) = \frac{1}{2}\left|\|W\alpha-\boldsymbol{b}\right|\|^2 + \sum_{i=1}^m\rho_i\bigg[\xi_i^{(i)}\xi_i^{(i)} - \xi_i^{(i)}\bigg] + \mu\sum_{i=1}^n |\alpha_i| + \sum_{i=1}^m \zeta_i\xi_i^{(i)} + \sum_{j=1}^m\sum_{k=1}^m \eta_{ij}\xi_i^{(i)} \xi_j^{(j)},
$$
其中 $\alpha = [\alpha_1;\alpha_2;...;\alpha_n]^T$, $\beta=[\beta_1^\top,\beta_2^\top,..., \beta_M^\top]^T$ 是拉格朗日乘子，$\gamma_i>0$, $\zeta_i\in\{0,1\}$, $i=1,2,...,m$ 表示约束条件。

定义 KKT 乘子：
$$
\kappa_i = \partial L/\partial\alpha_i = W_{i,:}^\top(\hat y_i - y_i) - \rho_i\xi_i^{(i)} - \mu\cdot\left\{\begin{matrix}1, \alpha_i < C\\0, \alpha_i \geqslant C\end{matrix}\right. - \sum_{j=1}^m \eta_{ij}\xi_i^{(i)} \cdot \xi_j^{(j)}.
$$
其中 $W_{i,:}=w_i$ 和 $\hat y_i = W_{i,:}^\top\alpha + b$. 

约束条件：
$$
K_i = \nabla f_i(a)-\nabla f_i(b) = \nabla_{a}\left[-\frac{1}{2}\left(\alpha^{T}(Ax+b-Y)^{T}(Ax+b-Y)+R(Ax+b-Y)\right)\right] + \frac{\partial R}{\partial a},
$$
其中 $A=[X_1^\top X_2^\top... X_N^\top]$, $Y=[y_1;y_2;...;y_N]$. $R=\begin{pmatrix}1 & 1 &... & 1 \\ 1 & 1 &... & 1 \\... &... &... &... \\ 1 & 1 &... & 1\end{pmatrix}$.

约束 $\xi_i^{(i)}\xi_i^{(i)} - \xi_i^{(i)}\leqslant  0$ 可转为 $\xi_i^{(i)}\leqslant 0$ ，因为 $-\frac{1}{2}\xi_i^{(i)}^2 \leqslant -\xi_i^{(i)}\leqslant 0$.

KKT 乘子的选择：
$$
\begin{aligned}
&\Delta_{a}=-\nabla L(a)-\lambda_1K_1,~\Delta_{b}=-\nabla L(b)-\lambda_2K_2,\\[2ex]
&\frac{\partial F_i}{\partial a_i}&=0,~&\text{for }i=1,2,...,m;\text{ and}\\[1em]
&\delta_i &=\max_{a<a'} F_i(a')-\max_{b<b'}F_i(b'), ~&\text{for }i=1,2,...,m.\text{ and}\\[1em]
&\lambda_j=\frac{1}{C}\left\{u-l\right\}.
\end{aligned}
$$

其中 $\Delta_{a}$,$\Delta_{b}$ 分别表示 $\alpha$ 增减量，$F_i$ 表示某些 $a_i$ 或 $b_i$ 的函数。$\lambda_1$,$\lambda_2$ 表示拉格朗日乘子，它们分别对应于改变 $\alpha_i$ 或 $b$ 的增加或减少量。

令 $\bar y_i=y_i+\epsilon_i$, 其中 $\epsilon_i\in(-\xi_i^{(i)},\xi_i^{(i)})$, 则对偶问题的目标函数成为：
$$
\min_{\alpha} \frac{1}{2}\left|\|W\alpha-\boldsymbol{b}\right|\|^2 + \sum_{i=1}^m\rho_i\bigg[\xi_i^{(i)}\xi_i^{(i)} - \xi_i^{(i)}\bigg] + \mu ||\alpha||_1.
$$
$$
\xi_i^{(i)}\leqslant 0,~i=1,2,...,m,
$$
$$
0\leqslant\alpha_i\leqslant C,~i=1,2,...,m,
$$
$$
\sum_{i=1}^m\alpha_iy_i=0.
$$