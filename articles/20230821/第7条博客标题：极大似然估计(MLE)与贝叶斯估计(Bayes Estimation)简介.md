
作者：禅与计算机程序设计艺术                    

# 1.简介
  

在概率论与统计学中，极大似然估计（Maximum Likelihood Estimation，简称MLE）和贝叶斯估计（Bayesian Estimation，简称BE）是两种常用的方法，用来估计模型的参数。

极大似然估计（MLE）是一种基于频率统计的方法，通过对已知数据集及其分布进行建模，找出使得观察到的数据出现的可能性最大的那个参数取值。

贝叶斯估计（BE）是另一种参数估计方法，它假设数据的生成过程遵循一个先验分布（prior distribution），然后用贝叶斯公式计算后验分布（posterior distribution）。

由于贝叶斯估计更加强调数据的不确定性，因此通常被认为比极大似然估计更具代表性。

MLE与BE各有特点，本文将分别介绍它们并给出一些实际案例。

# 2.背景介绍
## 2.1 概念阐述

### 极大似然估计（MLE）

极大似然估计（Maximum Likelihood Estimation，简称MLE）是基于频率统计的方法，通过对已知数据集及其分布进行建模，找出使得观察到的数据出现的可能性最大的那个参数取值。

举个例子，某个国家有两个组成成分A和B，我们想知道这些成分的比重。如果有若干个人做了相同的实验，记录下每个人的做事习惯。假设：

1. 每个人的做事习惯服从均值为p1和p2的正态分布；
2. A和B的分布服从独立同分布；

那么，通过MLE可以求出每个人做事习惯的概率分布。

首先，根据独立同分布的假设，分别计算A、B两组人做事习惯的联合分布，也就是pAB = p1*p2。然后，假定独立同分布还存在一个先验分布（prior distribution），即pA、pB均服从均值为0.5的正态分布。那么，可以通过贝叶斯公式计算得到pAB的后验分布，即：

$$p(pAB|D) \propto p(D|pAB)*p(pAB)$$

其中，$p(D|pAB)$表示模型参数pAB下观测到的数据的概率密度，是关于数据集D的函数；$p(pAB)$表示先验分布，也是关于参数pAB的概率密度；$p(pAB|D)$则表示后验分布，也是一个关于参数pAB的概率密度，因为它依赖于已知数据集D的条件概率分布。

所以，为了找到最有可能产生数据的模型参数pAB，我们需要最大化后验概率$p(pAB|D)$中的似然函数，也就是：

$$L(\theta)=\prod_{i=1}^n p(x_i|\theta)$$

其中$\theta$为待求参数。

那么，极大似然估计就是求解关于$L(\theta)$的模型参数的最大值，等价于：

$$arg \max_\theta L(\theta)$$

通过求导计算得到：

$$\frac{\partial L}{\partial \theta} = \frac{1}{L}\frac{\partial}{\partial \theta}\left[ \sum_{i=1}^n log (p(x_i|\theta))\right] \\=\frac{-1}{L}\sum_{i=1}^n \frac{\partial}{\partial \theta}log (p(x_i|\theta))$$

令上式等于零，即可求得$\hat{\theta}$，即：

$$\hat{\theta} = arg \max_{\theta} L(\theta)$$

### 贝叶斯估计（Bayesian estimation）

贝叶斯估计（Bayesian estimation）是另一种参数估计方法，它假设数据的生成过程遵循一个先验分布（prior distribution），然后用贝叶斯公式计算后验分布（posterior distribution）。

比如说，要预测某只股票价格走向的变化情况，我们先假定价格上涨概率为θ，那么就可以计算出每天的收益率。而后，我们可以用贝叶斯公式计算出每天的收益率的先验分布，再结合历史数据拟合出θ，从而得出更准确的预测结果。

具体来说，给定观测数据$D={(x_1,y_1),(x_2,y_2),...,(x_n,y_n)}$，其中的xi代表观测变量，yi代表目标变量。假设目标变量$Y$服从高斯分布，且先验分布为$\pi(θ)=N(\mu,\sigma^2)$。

贝叶斯估计的基本思路是：

1. 用已知的观测数据$D$及其对应的高斯似然函数$\mathcal{L}(Y;\theta;X)$来更新参数$\theta$，即：

   $$\begin{aligned} \theta &\sim \int_{\Theta} p(\theta|D)d\theta \\ &=\frac{1}{Z} \int_{\Theta} \pi(\theta) \mathcal{L}(Y;\theta;X) d\theta \end{aligned}$$

   其中，Z是归一化因子，满足：

   $$Z = \int_{\Theta} \pi(\theta) d\theta.$$
   
2. 根据新的参数$\theta$来重新计算先验分布，即：

   $$\pi'(\theta)=\frac{\pi(\theta)\mathcal{L}(Y;\theta;X)}{\int_{\Theta} \pi(\theta') \mathcal{L}(Y;\theta';X) d\theta'}$$

   这就得到了后验分布$\pi'$。

用数学语言描述：

1. 在当前参数$\theta$的条件下，使用贝叶斯公式：
   
   $$P(\theta|D) \propto P(D|\theta)P(\theta).$$
   
   对参数$\theta$积分，得到后验概率：
   
   $$P(\theta|D) \propto \int_{\Theta} p(\theta'\mid D)p(\theta')d\theta'.$$
   
2. 把后验分布$\pi'(θ)$代入到新的参数$\theta$的表达式，再求和，得到后验期望：
   
   $$\theta | D \sim E[\theta'|D] = \frac{\int_{\Theta} p(\theta'\mid D) p(\theta') d\theta'}{\int_{\Theta} p(\theta\mid D)p(\theta)d\theta}.$$

因此，贝叶斯估计的基本想法是：

1. 从数据中估计出目标变量的期望值。
2. 更新参数的分布，重新计算出后验分布。
3. 根据后验分布，给出更加精准的预测。