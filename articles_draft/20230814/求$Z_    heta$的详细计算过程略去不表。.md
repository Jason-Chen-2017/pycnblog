
作者：禅与计算机程序设计艺术                    

# 1.简介
  


在统计学中，频率分布函数(frequency distribution function)或称概率密度函数(probability density function)，也称为概率质量函数(probability mass function)，即在给定一个随机变量时，其所有可能取值出现的频率，再除以总体数量。它描述了随机变量的概率分布情况。用$\theta$表示某个随机变量的值，则频率分布函数表示为:
$$F_{\theta}(z)=P(\theta \leq z)$$
或者用分布律表示方式:
$$f_{\theta}(x)\equiv P(\theta=x),\forall x,\theta$$

假设随机变量$\theta$服从$N(\mu,\sigma^2)$分布，且有$K$个不同取值，则该分布的频率分布函数可以由下列公式计算:
$$F_{\theta}(z)=\frac{e^{-(\frac{(z-\mu)^2}{2\sigma^2})} }{\sqrt{2\pi}\sigma} $$
其中，$z$是正态累积分布函数的标准化形式，具体计算过程如下:
$$F_{\theta}(z)=\int_{-\infty}^{z}f_{\theta}(x)dx=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^ze^{(-\frac{1}{2}u^2)}du+\mu e^{\frac{-\mu^2}{2}}-e^{\frac{-\mu^2}{2}}e^{\frac{-z^2}{2\sigma^2}}$$
将右边第一项$\frac{1}{\sqrt{2\pi}}$移到积分符号外面即可得到最终结果。


# 2.基本概念术语说明
## (1). 中心极限定理（CLT）
中心极限定理（英语：center limit theorem），又叫“中位数收敛定理”，是指对任意具有独立同分布的随机变量序列，当样本容量趋于无穷大时，它们的平均数收敛于正态分布。也就是说，对于任何随机变量序列$X_1, X_2,..., X_n$, 其均值$\overline{X}_n=\frac{1}{n}\sum_{i=1}^{n}X_i$存在着近似正态分布的关系:

$$Z_n=\frac{\overline{X}-E[\overline{X}]}{\sqrt{\frac{Var[\overline{X}]}{n}}} \stackrel{d}{\longrightarrow} N(0,1)$$

其中，$E[\overline{X}]$表示样本均值的期望，$Var[\overline{X}]$表示样本均值的方差；$\rho$表示样本协方差，$\rho=\frac{Cov[X_i,X_j]}{Var[X_i]Var[X_j]}$。根据中心极限定理，当样本容量趋于无穷大时，$\overline{X}$的分布将趋于正态分布。

## (2). 概率密度函数
频率分布函数描述了随机变量在每一个可能取值处的出现次数与总体次数的比例，称之为概率密度函数。当样本空间离散时，概率密度函数曲线即为频率分布曲线。

## (3). 正态分布
正态分布（Normal Distribution）是一个非常重要的连续型随机变量分布，记作$N(\mu,\sigma^2)$。其概率密度函数为:

$$f(x;\mu,\sigma^2)=\frac{1}{\sqrt{2\pi}\sigma}exp[-\frac{(x-\mu)^2}{2\sigma^2}]$$ 

其期望值（即均值）为$\mu$，方差为$\sigma^2$。当样本容量足够多时，正态分布逐渐接近高斯分布，并逼近$N(0,1)$分布。

## (4). 离散型随机变量及其分布律
离散型随机变量(discrete random variable)是指取值为有限个离散点的随机变量，如抛掷一次骰子有两次头、三次头、四次头、五次头等就是典型的离散型随机变量。

离散型随机变量的分布律表示每个点出现的概率，离散型随机变量的概率分布函数(Probability Mass Function，PMF)可以表示成:

$$P(X=k)=p_k,\quad k=1,2,...,n$$

其中，$k$表示随机变量取值为第几个离散点，$p_k$表示事件“X取值为k”发生的概率。

当随机变量为二元随机变量时，则分布律称为伯努利分布。伯努利分布只有两个值，取值为0或者1，分别表示事件发生和不发生，它的概率分布函数为:

$$p(k;p)=p^k(1-p)^{1-k}$$

其中，$k=0,1$，$p$为事件发生的概率。当事件的独立性较强时，可以考虑联合分布律。

# 3.核心算法原理和具体操作步骤以及数学公式讲解
下面介绍如何利用概率分布函数求得随机变量的期望和方差，并且通过样本数据来估计参数的真实值。

## (1). 概率分布函数与期望
概率分布函数描述了随机变量在每一个可能取值处的出现次数与总体次数的比例，可以表示成:

$$F(z)=P(\theta \leq z)$$

上式表示随机变量$\theta$的分布函数，表示取值小于或等于$z$的概率。通常情况下，我们常用的频率分布函数也可以作为概率分布函数的近似。概率分布函数确定了随机变量的概率质量函数，概率质量函数描述了各个取值的概率大小。

如果随机变量$\theta$服从正态分布$N(\mu,\sigma^2)$，则分布函数为:

$$F_{\theta}(z)=\frac{1}{\sqrt{2\pi}\sigma} exp(-\frac{(z-\mu)^2}{2\sigma^2})$$

如果对$Z=(z-\mu)/\sigma$进行积分，则有:

$$F_{\theta}(z)=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{z} e^{-u^2/2} du+(\mu/\sigma)e^{-(z-\mu)^2/(2\sigma^2)}$$

由于积分区域为正负无穷，因此分母中有一个无穷乘积$(2\pi)^{1/2}$，所以省略掉，得到:

$$F_{\theta}(z)=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{z} e^{-u^2/2} du+(\mu/\sigma)e^{-(z-\mu)^2/(2\sigma^2)}=\frac{1}{\sqrt{2\pi}}\cdot e^{-|z-\mu|^2/(2\sigma^2)}\cdot (\frac{z-\mu}{\sigma})+\frac{1}{\sqrt{2\pi}\sigma}$$

对于概率分布函数，有两个基本性质：

$$P(\theta \geq a)=1-P(\theta < a)$$

$$P(\theta_1 \leq \theta_2)=P(\theta_1<\theta_2)+P(\theta_1=\theta_2)$$

对于离散型随机变量，可以定义相应的分布律:

$$P(X=k)=p_k,\quad k=1,2,...,n$$

其中，$p_k$表示事件“X取值为k”发生的概率。

对于二元随机变量，分布律为：

$$p(k;p)=p^k(1-p)^{1-k},\quad k=0,1$$

其中，$k=0,1$，$p$为事件发生的概率。

## (2). 期望的计算方法
若随机变量$\theta$的概率分布函数为$F_{\theta}(z)$，则其分布的平均值（期望）为:

$$E[\theta]=\int_{-\infty}^{\infty}z F_{\theta}(z) dz$$

因为正态分布是连续型随机变量，不能直接对其积分，而要转换成离散型随机变量才能计算期望。假设$\theta$服从$N(\mu,\sigma^2)$分布，则随机变量$Z$服从$Z=(z-\mu)/\sigma$分布，其分布函数为：

$$F_{Z}(z)=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{z} e^{-u^2/2} du+(\mu/\sigma)e^{-(z-\mu)^2/(2\sigma^2)},\quad -\infty<z<\infty$$

可以看到，$Z$的概率分布函数与正态分布的分布函数相同，只是把积分范围变成了$-\infty<z<\infty$。因此，可以通过查表的方法对$Z$的概率分布函数求出其值域上的积分值，并相加得到期望。

采用数值积分的方法，则有：

$$E[\theta]\approx \lim_{n\to\infty}\frac{1}{n}\sum_{i=1}^n z_i f_{\theta}(z_i)$$

其中，$z_i=\mu+(\sigma Z)_i$，$Z_i$为第$i$个正态分布随机变量。此外，还有其他一些更高级的方法，但这些都需要数值计算。

## (3). 方差的计算方法
方差描述了随机变量$\theta$在任意位置的离散程度，即随机变量$\theta$可能的值与均值的偏离程度，记作：

$$Var[\theta]=E[(z-\mu)^2]=E[z^2]-[E(z)]^2$$

其中，$E(z)$表示随机变量$\theta$的期望。计算方差时，首先计算$z$的期望，然后代入$z$的期望值以及$z$的方差计算公式即可。例如，随机变量$\theta$服从$N(\mu,\sigma^2)$分布，则有:

$$E(z)=E(\mu+\sigma Z)=\mu+E(Z)\sigma+\sigma^2 E(Z)^2$$

所以，方差计算公式为:

$$Var[\theta]=E((z-\mu)^2)=E([z^2]+[\mu^2]+[2\mu z+\sigma^2 Z]^2)-[E(z)]^2=[E(z^2)]-[E(z)]^2-E([2\mu z+\sigma^2 Z])^2$$

利用期望的计算方法即可求得方差。