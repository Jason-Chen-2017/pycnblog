
作者：禅与计算机程序设计艺术                    

# 1.简介
  

文档分类是自然语言处理（NLP）的一个重要任务，文本数据通常需要经过多种形式的提取、特征提取、降维等预处理过程后才能送入机器学习模型进行训练，最终得到可用于后续任务的文档表示或表示向量。支持向量机（SVM）是一种典型的机器学习方法，可以有效地对文档进行分类。本文将对SVM在文档分类中的应用做一个介绍。
# 2.基本概念术语说明
## 2.1 定义
支持向量机（Support vector machine，SVM）是一种二类分类方法，其基本想法是找到一个超平面（hyperplane），通过最大化边界间距和最小化支持向量到超平面的距离来对训练样本进行线性分割。换句话说，SVM就是通过寻找能够最大化边界间距并最少使得支持向量远离超平面的超平面，来构建一个“最优”的分割超平面。它的一般形式如下所示：
$$
\begin{equation}
\operatorname*{minimize}_{\omega,b}\quad \frac{1}{2}||w||^2+\rho_i\sum_{i=1}^{m}\xi_i\\
s.t.\quad y_i(w^\top x_i+b)\geq 1-\xi_i,\quad i=1,...,m;
\end{equation}
$$
其中，$\omega=(w_1,w_2,\cdots,w_n)^T$ 为权值向量，$b$ 为偏置项，$x_i=(x_{i1},x_{i2},\cdots,x_{id})^T$ 为输入向量，$y_i$ 为标记类别，$\rho_i>0$ 为松弛变量，$\xi_i\geq 0$ 为规范化因子。
## 2.2 SVM的优化目标
SVM的优化目标是希望找到一个超平面将训练集中的正负实例点划分开来，同时最大化实例点之间的间隔（margin）。实例点到超平面的总距离可以用以下公式表示：
$$
\begin{equation}
\min_{\omega, b, \xi} \frac{1}{2} ||w||^2 + C\sum_{i=1}^m\xi_i \\
s.t.\quad y_i(\omega^\top x_i+b) \geq 1- \xi_i,\forall i=1,...,m\\
\xi_i \geq 0, \forall i = 1,..., m
\end{equation}
\label{eq:svm_obj}
\end{equation}
$$
其中，C是惩罚系数，用来控制约束严格程度。
## 2.3 支持向量
在超平面下方的点都被称为支持向量，它们对于确定分割超平面的选择至关重要。而相应的损失函数的值就会被设为0，因为这些点不参与计算。而且由于它们被包含在了超平面内，所以可以通过添加一个松弛变量来描述它们对总距离的贡献。如果将所有松弛变量之和记为$R(w,b)$，则此时优化目标变为：
$$
\begin{equation}
\begin{aligned}
&\min_{\omega, b, \xi}\\
& \quad s.t. R(w,b)=\frac{1}{2}\left \| w \right \|^2 + C\sum_{i=1}^m\xi_i - \sum_{i=1}^m y_i(\omega^\top x_i+b)\delta_{in}(r_i)\\
&\delta_{in}(z)=\begin{cases}
    1,&z\geq 0\\
    0,& z<0
    \end{cases}\\
&\xi_i \geq 0, \forall i = 1,..., m
\end{aligned}
\label{eq:sv_opt}
\end{equation}
$$
其中，$r_i=-\frac{1}{\rho_i}$。可以看出，当$C$很小时，只允许些许错误率，那么最优分割超平面上就只会出现一些支持向量；而当$C$较大时，允许较大的错误率，那么分割超平面上的所有点都会成为支持向量。
# 3. Core Algorithm and Math Formulas
## 3.1 Probabilistic interpretation of SVM
为了给SVM提供更加直观的解释，引入概率解释。首先假设有一个二分类问题，其中第$k$类的样本由随机变量$X_{ik}$表示，且满足条件
$$
P\{X_{ik}=1\}>P\{X_{ik}=0\}\;\forall k\in \{1,2,\cdots,K\}.
$$
例如，对于邮件分类问题，第1类表示垃圾邮件，第2类表示正常邮件。假设还有一个概率分布$\pi=\{p_k\}_{k=1}^K$，表示每个类的先验概率，即
$$
p_k=\frac{N_k}{N},\;\forall k\in \{1,2,\cdots,K\};
$$
其中，$N_k$ 是训练集中第 $k$ 个类的数量，$N=|X|$ 表示训练集的大小。那么，SVM的问题可以描述成：
$$
\max_{\gamma\in H} L(\gamma)=\mathbb{E}_{q}\left[\log P(\mathcal D|\gamma)\right]
$$
其中，$H$ 是某个函数空间，例如希尔伯特空间；$\gamma$ 是$\theta$ 的函数，$q$ 是贝叶斯分布。求解这个问题等价于：
$$
\max_\gamma\sup_{\theta\in H}\left[\mathbb E_{q}[f(\gamma(\mathbf x))]\right]-\text{KL}\left[q(\theta)||p(\theta)\right],
$$
即最大化后验概率下的期望损失。
### 3.1.1 Maximizing the likelihood function
在给定模型参数 $\theta$ 下，$L(\theta)$ 表示模型生成数据的可能性。假设已知数据 $\mathcal D=\left\{(\mathbf x^{(1)},y^{(1)}),(\mathbf x^{(2)},y^{(2)}),\cdots,(\mathbf x^{(N)},y^{(N)})\right\}$，则模型生成数据的方式为：
$$
P(\mathcal D|\gamma)={1\over Z}\exp\left(-{1\over 2}\sum_{n=1}^N\gamma(\mathbf x^{(n)})^Ty^{(n)}\right).
$$
其中，$Z$ 是归一化常数，是所有可能的模型组合的净值。我们可以使用拉格朗日乘子法最大化 $L(\theta)$，但这非常复杂，通常我们采用如下近似：
$$
L(\theta)\approx {1\over N}\sum_{n=1}^N\underbrace{[-y^{(n)}\log(\gamma(\mathbf x^{(n)}))]}_{\text{cross entropy}}+\lambda R(\theta),
$$
其中，$\lambda$ 是正则化系数，$R(\theta)$ 表示模型复杂度。
### 3.1.2 Prior distribution $p(\theta)$
在实际问题中，我们通常没有得到模型参数的精确值，而是获得了某种类型的分布，例如高斯分布或者混合高斯分布。为了推广到 SVM，需要引入一个先验分布 $p(\theta)$ 来描述模型参数的可能性。根据高斯分布的假设，$p(\theta)=N(0,\Sigma^{-1})$，其中 $\Sigma^{-1}$ 是协方差矩阵。然后，我们可以将模型参数的先验分布写作 $p(\theta)=p(w,b)+p(\Sigma)$，其中 $p(w,b)$ 是高斯分布，而 $p(\Sigma)=\text{Dir}(\alpha)\Sigma^{-\alpha-1}$, 是半正定的狄利克雷分布。
### 3.1.3 Using Bayes rule to derive SVM's objective
考虑 $f(\cdot)$ 和 $\theta$ 的联合分布：
$$
\begin{align*}
f(\mathbf x)&=\mathbb E_{q(\theta)}\left[f(\gamma(\mathbf x))\right]\\
&=\int p(\theta)p(\mathbf x|\theta)f(\mathbf x)d\theta.
\end{align*}
$$
由于 $q(\theta)=p(\theta)/Z$, 因此
$$
\begin{align*}
f(\mathbf x)&=\int q(\theta)p(\mathbf x|\theta)f(\mathbf x)d\theta\\
&\propto \prod_{n=1}^Np(\mathbf x^{(n)};\theta)f(\gamma(\mathbf x^{(n)})).
\end{align*}
$$
为了使模型训练更加鲁棒，我们可以在模型复杂度 $R(\theta)$ 上加入先验信息：
$$
\begin{align*}
&\max_\gamma\sup_{\theta\in H}\left[\mathbb E_{q}[f(\gamma(\mathbf x))]\right]-\text{KL}\left[q(\theta)||p(\theta)\right]\\
&s.t.\quad R(\theta)<\infty.\\
&\text{(dual problem)}.
\end{align*}
$$
### 3.1.4 Simplifying the dual problem
为了求解简化后的双最优问题，可以采用分段线性函数：
$$
g(\gamma)=\log\left({\gamma_+}/{\gamma_-}\right)\ge c-\gamma_+.
$$
其中，${\gamma_+}$ 和 ${\gamma_-}$ 分别表示正例和反例的概率密度。线性划分的误差项为
$$
R(\theta)=\sum_{k=1}^K\text{KL}\left[q(\theta_k)||p(\theta_k)\right].
$$
如果损失函数是凸函数，那么就可以通过梯度上升法或者拟牛顿法求解。然而，这种方法往往收敛速度慢，并且容易陷入局部极小值。另外，如果损失函数不是凸函数，那么就无法直接使用分段线性函数，而需要转用其他方法。