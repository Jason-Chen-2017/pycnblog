
作者：禅与计算机程序设计艺术                    

# 1.简介
  

支持向量机（Support Vector Machine，SVM）是一种二类分类模型，其主要目的是寻找一个超平面，使得该平面的距离分割两类样本数据集最远。它是一个基础且广泛应用于机器学习领域的模型，被证明在很多领域都能取得很好的效果。本文首先会对支持向量机相关概念及特点做简单的介绍。然后会详细介绍硬间隔支持向量机（hard margin SVM），并阐述如何通过软间隔支持向量机（soft margin SVM）实现非线性分类。接着，我们会以LIBSVM库为例，介绍该库对SVM的各种实现方法。最后，还会展望SVM在实际工程中的应用方向，如图像处理、文本处理等。

2. 支持向量机概览
## 2.1 什么是支持向量机？
支持向量机（support vector machine，SVM）是一种二类分类模型，其主要目的是寻找一个超平面，使得该平面的距离分割两类样本数据集最远。由此，SVM对已知的数据进行最大间隔分割，把样本划分到不同的类别中。SVM引入了一组称作支持向量的特殊样本，这些样本彼此间隔，而且处于最小化误差函数值最大化的考虑范围之内。这样可以保证找到的超平面能够较好地将训练数据分开。图2-1展示了一个具有三个类的简单二维数据集的示意图。


图2-1 两个类的简单数据集示意图

SVM是一个最大间隔分类器，也就是说，它假设训练数据集中存在着一些“支持向量”，并且这些“支持向量”恰好处于某些边界上。换言之，就是说，SVM要找到一个超平面，这个超平面距离数据的两侧越远越好，这样就可以将不同类的数据正确地分类。另外，SVM还可以通过核函数的方式，将输入空间映射到高维特征空间，从而实现非线性分类。本文中，我们将重点关注如何构造一个硬间隔支持向量机。

## 2.2 支持向量机模型
### 2.2.1 模型定义
给定一个训练数据集$T=\left\{(\mathbf{x}_i,\tilde{y}_i)\right\}_{i=1}^N$，其中$\mathbf{x}_i \in \mathcal{X}=\mathbb{R}^d$, $\tilde{y}_i\in\{-1,+1\}$, $i=1,\cdots,N$，称$y_i=f(\mathbf{x}_i)=sign(\sum_{j=1}^p w_jx_j^Tx_i+\theta)$为决策函数。其中，$\mathbf{w}$为权重参数，$\theta$为偏置项，$p$为特征的个数。对于某个给定的输入$\mathbf{x}$，$sign(\cdot)$表示符号函数，即$\operatorname{sign}(a)=\left\{ \begin{array}{ll}-1 & a<0 \\ 0 & a=0 \\ +1 & a>0 \end{array}\right.$。显然，如果所有样本点满足约束条件$y_i(\mathbf{w}^{(t)}^\top\mathbf{x}_i+\theta^{(t)})\geq M-\xi_i, i=1,\cdots, N,$，那么$y(\mathbf{x})=\operatorname{sign}(\sum_{i=1}^N y_i(\mathbf{w}^{(t)}^\top\mathbf{x}_i+\theta^{(t)})\xi_i)=f(\mathbf{x})$的确有着恰好$M$个支持向量的最大间隔。在这种情况下，函数$f(\cdot)$有着无穷多个极小值，但只有少数几个极小值的组合才可能构成分离超平面。因此，为了找到合适的超平面$H$以及相应的分类决策函数$f(\cdot)$，我们需要优化目标函数
$$
\min_{\mathbf{w},\theta}\quad \frac{1}{2}\|\mathbf{w}\|^2+\lambda\parallel\xi\parallel_2^2\\
s.t.\quad y_i(\mathbf{w}^\top\mathbf{x}_i+\theta)\geq M-\xi_i, i=1,\cdots, N,\\
\quad \xi_i\geq 0, i=1,\cdots, N.
$$
这是一个凸二次规划问题，可以通过求解其相应的解析解或凸优化算法来求解。

### 2.2.2 最大间隔与几何解释
给定一个训练数据集$T=\{\mathbf{x}_i,y_i\}_{i=1}^N$，其中$\mathbf{x}_i \in \mathcal{X}=\mathbb{R}^n$, $y_i\in\{-1,+1\}$, $i=1,\cdots,N$，已知$\alpha=(\alpha_1,\cdots,\alpha_N)^T\succeq 0$是拉格朗日乘子，则:
$$
L(\alpha,\beta)=\frac{1}{2}\|\beta\|^2+\frac{1}{2}\sum_{i=1}^N\alpha_i[y_i(\mathbf{w}^\top\mathbf{x}_i+\theta)-1+\xi_i],\quad s.t.\quad \alpha_i\geq 0,\forall i=1,\cdots,N,\qquad\beta_i=0,\forall i\not=0
$$
其中，$\beta=\left(b_0,\ldots,b_N\right)$为未知参数，$b_0$为偏移参数，$h(\mathbf{x};\beta)=\sum_{j=1}^nb_jx_j^Tx+\beta_0$为预测函数。显然，如果$\beta_0=0$，则$L(\alpha,\beta)$可简化为$\frac{1}{2}\|\beta\|^2+\frac{1}{2}\sum_{i=1}^N\alpha_i[y_i(\mathbf{w}^\top\mathbf{x}_i+\theta)-1+\xi_i]$. 在最大间隔分割下，满足约束条件$y_i(\mathbf{w}^\top\mathbf{x}_i+\theta)\geq M-\xi_i, i=1,\cdots, N$，若$K=e^{\pm\sqrt{\lambda}}(\mathbf{I}-\mathbf{w}\mathbf{w}^\top)$，则$\alpha=(\alpha_1,\cdots,\alpha_N)^T\succeq 0$，故存在唯一解$$(\beta_0,\beta)=(-\frac{1}{\sqrt{\lambda}},\mathbf{w})\qquad s.t.\quad b_i=-\sum_{j=1}^Nx_jy_jh_j(\mathbf{w}^\top x_i+\theta)+\frac{1}{\sqrt{\lambda}}\alpha_iy_ih_i\qquad i=1,\cdots,N.$$

令$C=\frac{1}{\sqrt{\lambda}}$，则
$$
L(\alpha,\beta)=\frac{1}{2}|\beta|^2+\frac{1}{2}\sum_{i=1}^N\alpha_i[y_i(\mathbf{w}^\top\mathbf{x}_i+\theta)-1+\xi_i]+\frac{1}{C}\sum_{i=1}^N\xi_i-\frac{1}{C}\log\left(\sum_{i=1}^N e^{-y_i(\mathbf{w}^\top\mathbf{x}_i+\theta)}\right).
$$
该形式下的SVM成为“软间隔SVM”。

## 2.3 核技巧

当训练数据集不是线性可分的时，可以使用核函数将原始输入空间映射到更高维的特征空间。举个例子，如果输入空间$\mathcal{X}=\mathbb{R}^2$不可分，但是存在一个径向基函数$\phi:\mathcal{X}\rightarrow\mathcal{H}=span(\phi_1,\cdots,\phi_m)$，满足$\{\phi_k\}$线性无关，则可以通过核函数将$\mathcal{X}$变换到新的特征空间$\mathcal{H}$。具体来说，将样本点$((x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N))$映射到特征空间$\mathcal{H}$的方法如下：

$$
\hat{x}_i=(y_i,x_1\phi_1(x_i)+x_2\phi_2(x_i)+\cdots+x_N\phi_N(x_i)),\quad i=1,\cdots,N,
$$

其中，$\phi_k(x):\mathcal{X}\rightarrow\mathcal{H}, k=1,\cdots,m$是径向基函数。那么，通过映射后的新特征空间$\mathcal{H}$，是否仍然可分呢？答案是否定的。因为存在不止一个超平面能够将$\mathcal{H}$划分为两部分。因此，为了进一步提升模型的分类能力，需要采用核技巧。具体来说，核技巧基于的想法是，如果能够直接将样本点映射到特征空间$\mathcal{H}$中，那么就不需要计算高维空间中的内积。

具体来说，假设训练数据集$T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\}$。记核函数为$k:\mathcal{X}\times\mathcal{X}\rightarrow\mathbb{R}$，那么，核函数将原始输入空间映射到特征空间$\mathcal{H}$的方法为：

$$
\Phi:(x,z)\mapsto k(x,z)=\langle\phi(x),\phi(z)\rangle.
$$

假设$\mathcal{X}$线性无关，径向基函数$\phi_k(x):=\phi(x)\phi_k$，且$\phi_k(x)$与$\phi_l(x)$无关，则：

$$
\langle\phi(x),\phi(z)\rangle=\sum_{k=1}^md_kd_l\phi_k(x)\phi_l(z),
$$

其中，$d_k$是基函数系数，对应于$x$到每个基函数$\phi_k(x)$的投影长度。于是，

$$
\Phi:(x,z)\mapsto k(x,z)=\langle\phi(x),\phi(z)\rangle=\sum_{k=1}^md_kd_l\langle\phi_k(x),\phi_l(z)\rangle.
$$

因此，可以直接计算核函数：

$$
k(x,z)=\langle\phi(x),\phi(z)\rangle=\langle\psi(x),\psi(z)\rangle=\sum_{k=1}^md_kd_l\langle\phi_k(x),\phi_l(z)\rangle=\sum_{j=1}^nd_jd_j\langle\phi(x),\phi(x')\rangle^{d/2}.
$$

其中，$\psi(x)=\sum_{j=1}^n\alpha_j\phi_j(x)$是新的基函数，$\alpha_j$是对应的基函数系数。所以，通过核函数，就可以将输入空间$\mathcal{X}$映射到特征空间$\mathcal{H}$中，从而避免计算高维空间中的内积。

总结一下，核技巧的主要思路是：如果能够直接将样本点映射到特征空间$\mathcal{H}$中，那么就不需要计算高维空间中的内积。通过核函数将样本点映射到特征空间$\mathcal{H}$后，即可通过SVM求解出超平面和分类决策函数。具体的核函数和具体的映射方式依赖于具体情况。