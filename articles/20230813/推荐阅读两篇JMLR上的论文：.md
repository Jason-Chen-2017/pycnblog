
作者：禅与计算机程序设计艺术                    

# 1.简介
  
：使用核方法进行非线性分类
# 作者：<NAME> and <NAME> and <NAME> and <NAME> and M<NAME>
# 摘要：在很多现实任务中，包括文本分类、生物信息分析、预测精度的提升等方面，核方法（Kernel Method）已被广泛应用。然而，对于高维数据的应用及其扩展，仍存在一些困难。本文基于核支持向量机（Kernel Support Vector Machine）进行研究，通过推导核函数形式化地定义数据之间的距离度量，从而实现高维空间中的非线性分类。这项工作对其他核方法的发展起到了积极作用。最后，本文提出了几个新的核函数，并用它们与原有核函数结合，建立了复杂数据的非线性分类模型。
# 报告地址：http://jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf
# 代码：https://github.com/jwcalder/kernel_svm

# 2.相关工作：
许多非线性分类器，如朴素贝叶斯、支持向量机、神经网络、决策树等，都需要对特征进行非线性变换后才能有效地进行分类。但这样做有时会造成严重的计算开销。因此，早期的工作试图在低维空间里找到一个适当的非线性映射，使得目标函数的最优化可以直接在高维空间里进行，如隐马尔科夫模型。但这些方法在处理高维数据上仍然存在很大的困难。另一种方法则是利用核技巧将输入空间映射到高维空间，从而可以直接进行非线性分类，如核感知学习、核线性回归、核随机森林。

# 3.问题描述：给定训练集$T=\{(x_i,y_i)\}_{i=1}^N$，其中$x_i\in \mathcal{X}$ 为特征向量，$\mathcal{Y} = \{ -1, +1\}$, $y_i\in\mathcal{Y}$ 是类标签。希望找到一个非线性分类模型，即求解如下损失函数
$$L(w)=\frac{1}{N}\sum_{i=1}^NL(\delta_i), \quad L(\delta_i)=-\sum_{k=1}^K y_i \delta_ik^Tx_i+\frac{1}{\lambda}\left|\sum_{k=1}^Ky_i\delta_ik\right|,$$
其中 $\delta_i$ 为第 i 个样本对每个类的拉格朗日乘子，$\lambda$ 为正则化参数。$\delta_i=(\delta_i^{(1)},\ldots,\delta_i^{(K)})^{\mathrm T}$ ，且 $\delta_i^{(k)} \geqslant 0$, $k=1,\cdots,K$. $\delta_i$ 表示第 i 个样本对所有类别的贡献度。$\lambda$ 为正则化参数。求解这个问题的目的是为了找到一个最优的权重向量 $w$ 来最小化误差函数 $L(w)$ 。


# 4.算法流程：
首先，引入核函数，定义核矩阵：
$$K(x,z)=\phi(x)^{\mathrm T}\phi(z).$$
其中，$\phi:\mathcal{X}\to \mathbb{R}^{d}$ 是特征映射函数，可以选择不同的核函数 $\phi$ 。

然后，根据拉格朗日乘子法，设 $L(w;\delta)=\frac{1}{N}\sum_{i=1}^NL(\delta_i;\hat{y}_i,f_{\theta}(x_i))-E_D(w), \quad \delta_i=\beta_ky_ix_i, \quad k=1,\cdots,K.$ 

$\beta_k$ 为拉格朗日乘子，约束条件为 $\sum_{k=1}^K\beta_k=0, \forall i$ 。

令 $g_k(x_i)=\delta_iy_if_{\theta}(x_i)+b_k.$ 

根据拉格朗日乘子的性质，可得到：

$$L(w;\delta_i)=-\sum_{k=1}^K g_k(x_i)-\frac{1}{\lambda}\left|\sum_{k=1}^Kg_k(x_i)\right|.$$ 

将拉格朗日函数对 $w$ 和 $b$ 求偏导，即

$$-\nabla_LwL(w;\delta_i)=\frac{1}{N}\sum_{i=1}^N(-y_ig_k(x_i)-\delta_iy_if_{\theta}(x_i)-b_k-\frac{1}{\lambda})e^{y_ig_k(x_i)w}-E_D(w)e^{y_ig_k(x_i)w}.$$ 

由于所有样本点的影响是相同的，所以只需考虑第一个和第二个部分即可，即

$$\begin{array}{ll}
\mathop{min}_{w,b}&\frac{1}{N}\sum_{i=1}^N(-y_ig_k(x_i)-\delta_iy_if_{\theta}(x_i)-b_k-\frac{1}{\lambda})e^{y_ig_k(x_i)w}\\
&\text{s.t.}\\
&-\frac{1}{N}\sum_{i=1}^Ny_ie^{y_ig_k(x_i)w}=0\\
&\sum_{i=1}^Nk_ib_kg_k(x_i)=0.\\
\end{array}$$ 

这里的约束条件要求取到全局最优，因此还需要考虑对偶问题：

$$\max_{\delta_i}\underset{w,b}{\arg\min}_{w,b}L(w;\delta_i).$$ 

对偶问题的一个解就是原始问题的解。因此，在求解原问题时，只需要依据约束条件 $k_ib_k=0$ 来更新约束条件，即 $-k_ib_k\leqslant \epsilon,$ $\epsilon >0,$ $\forall i.$ 

最后，在 $K(x,z)<\sigma$ 时，$k_ib_k=0.$ 