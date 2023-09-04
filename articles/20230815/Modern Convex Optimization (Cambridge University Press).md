
作者：禅与计算机程序设计艺术                    

# 1.简介
  

近几年，在机器学习、数据分析、图像识别等领域，凸优化（convex optimization）算法越来越受到研究者们的关注。相对于传统的线性规划或整数规划来说，凸优化算法可以找到全局最优解或者局部最优解。凸优化算法经历了多种形式的发展，包括单纯形法、枢轴法、支撑向量法、等式约束方法、序列最小最优逼近法等。近些年来，随着大数据、计算力的增强、应用场景的广泛拓展，以及硬件性能的提升，很多新的凸优化算法被提出。在本书中，作者将带领读者了解最新和最前沿的凸优化算法，并通过现代算法及其实现细节，帮助读者理解、掌握和运用凸优化算法。
# 2.相关阅读
本文对凸优化问题进行了一个简单的介绍。

这个 talk 对线性规划问题进行了深入浅出的介绍。

这是一篇非常好的入门文章，对整数规划问题进行了系统的介绍。

# 3. 基本概念术语说明

## 3.1 优化问题

首先，需要明确什么是优化问题？

> 优化问题是指为了某一个目的或目标，从一个给定的集合中选择一组值或变量，使得某个性能指标最大化或最小化。

通常情况下，优化问题分为无约束和约束两种类型。

### （1）无约束优化问题

无约束优化问题（Unconstrained Optimization Problem）即不涉及任何约束条件，通常是在一个实数维度上寻找全局最优解。

最常见的无约束优化问题有：

1. 求函数$f(x)$的最小值；
2. 求$n$维空间中的点集$\{ x_i\}$的最大间隔（最大范数），其中$x_i \in R^n$；
3. 求$\{ x : f(x)=c_i \forall i=1,2,\cdots,m\}$中的点的集合，其中$c_i \in R$。

### （2）有约束优化问题

有约束优化问题（Constrained Optimization Problem）即描述一类具有以下约束条件的优化问题：

1. 可行性约束：限制解空间中可行解的范围；
2. 边界约束：限制变量的取值范围；
3. 输出约束：限制优化结果的范围。

最常见的有约束优化问题有：

1. 求函数$f(x)$在区间$[a,b]$上的最小值；
2. 求函数$f(x)$的最小值，满足约束条件$g(x)\leq 0$,$h(x)=0$；
3. 求函数$f(x)$的极小值，满足约束条件$A_{eq}x = b_{eq}, Ax \geq b$, $l \leq x \leq u$。

## 3.2 函数

### （1）定义

> 函数（function）是指能够接受输入并产生输出的一类对象。在数学和计算机科学中，函数是关系映射，它把一个或多个自变量映射到一个或多个因变量的值。

例如：

$y = f(x)$: 函数$f$是一个自变量$x$到因变量$y$的映射关系。

### （2）凸函数

> 在函数$f$中，如果存在至少一个常数$k$和某个常比例因子$\lambda$，使得对所有$x$，都有$\lambda_ix + k <= f(x)$。也就是说，对于任意的常数$k$和某个常比例因子$\lambda$，对于所有的$x$，都有一个$x$对应的函数值$\lambda_if(x)+k$，并且该值满足下界线性条件。如果$\lambda_i < 0$则称为严格凹函数；如果$\lambda_i > 0$则称为严格凸函数；否则，称为仲恺函数。

注：若$f$是严格凹函数，则$\lambda$必须为正；若$f$是严格凸函数，则$\lambda$必须为负；若$f$是仲恺函数，则$\lambda$可以是任何非零常数。

#### 2.1.1 单调递减函数

> 如果$f(x)>=f(z),\forall x, z$则$f$为单调递减函数。

#### 2.1.2 严格单调递减函数

> 如果$f(x)>f(z),\forall x, z$且$\exists x',z'$使得$f(x')<f(z'),\forall x', z'$则$f$为严格单调递减函数。

#### 2.1.3 连续函数

> 如果$\forall x \in [a,b],\exists r_1,r_2\in(a,b)$满足$f(x)=f(r_1)+f'(r_1)(x-r_1)+\frac{(x-r_1)^2}{2!}\left|\nabla f(r_1)\right|, \forall x\in [a,b]$, 则称$f$为连续函数。

注：$\nabla f(r_1)$表示$f$在$(r_1)$处的梯度。

#### 2.1.4 严格连续函数

> 如果$\forall x \in [a,b],\exists r_1,r_2\in(a,b)$满足$f(x)=f(r_1)+f'(r_1)(x-r_1)+\frac{(x-r_1)^2}{2!}\left|\nabla f(r_1)\right|+\frac{L(x-r_1)}{2}$, $\forall x\in [a,b]$且$L=\frac{\partial^2}{\partial x^2}|_{\substack{r_1}}, L<0$, 则称$f$为严格连续函数。

注：当且仅当$f(x)\geq 0$时，$f$是严格连续函数。

#### 2.1.5 凸函数

> 定义：如果函数$f:\mathbb{R}^n \rightarrow \mathbb{R}$满足：

1. $f$是连续函数，即：$\forall x_1,x_2 \in \mathbb{R}^n, \eta \in (0,1)$, 有$f(\eta x_1+(1-\eta)x_2)-\eta f(x_1)- (1-\eta)f(x_2)<\epsilon \|x_1-x_2\|$;
2. 局部极值个数$f''(x) \neq 0, \forall x \in \mathbb{R}^n$，则$f$是凸函数。

### （3）极小极大问题

> 当函数$f$在某个点$x^*$处具有极值的必要条件是：$Df(x^*)=0$，但也有可能存在这样的点$x^*$不是极小值点。因此，“极小极大”问题成为研究函数在某点是否为极小值点的问题。

#### 3.1.1 原点的极小值问题

假设$f$是严格连续的，那么函数$g=-f$也是严格连续的，且有$D(-f)(x)=D(f)(x)$，因此函数$g$的梯度向量方向与函数$f$的梯度方向相同。设$g^{+}(t)=e^{tF}(f^{+})$，则$g$为严格单调递减函数。设$\delta t>0$，则对任意$\delta x=(\delta x^{(1)},\delta x^{(2)},..., \delta x^{(n)})^{\mathrm{T}}$，有：

$$\lim_{\delta t \rightarrow 0} g^{+}(\delta t)+g(-\delta x)-\delta t\cdot D(-f)(\delta x)=0.$$

当$t=\inf\{t: g^{+}(t)<0\}$时，有$g^{+}(t)<0$，此时$\delta t$较小，于是：

$$-\delta t D(-f)(\delta x)<\epsilon \|(\delta x)\|$$

因此$D(-f)(\delta x)=0$。由$(\delta x,\delta y,\cdots )^{\mathrm{T}}=(0,-\alpha,\beta,\cdots,0)^{\mathrm{T}}$得到：

$$-\alpha (-f)(\alpha^{-1}-\beta^{-1})\leq -\alpha (-f)(\frac{\alpha^{n-1}}{\beta^{n-1}})=-\frac{\alpha^{-1}-\beta^{-1}}{\beta^{n-1}}\leq\frac{-1}{n-1}$$

由于$n=d$，所以$-1/n\leq -1/(d-1)$。由之前的结论得知，函数$f$在$x^*$处的梯度必须指向$-\nabla f(x^*)$。因此，当$t=\frac{1}{\beta^{\ast}}$时，$\delta x$始终在直线$\frac{1}{\beta^{\ast}}(x^*-\beta^{\ast}x^*)$上，$\delta t$始终等于$\frac{1}{\beta^{\ast}}(x^*-\beta^{\ast}x^*)$，即：

$$x^{*}-\beta^{\ast}x^{*}+\frac{1}{\beta^{\ast}}(x^*-\beta^{\ast}x^*)=(\frac{1}{\beta^{\ast}}(x^*-\beta^{\ast}x^*),0,...\,(0))^{\mathrm{T}}$$

所以，原点$x^*=(-\alpha,0,...,0)^{\mathrm{T}}$是函数$f$的极小值点。

#### 3.1.2 函数的极小值问题

函数$f$的极小值问题即要寻找一个满足$f(x)\leq c$的$x$，使得$f(x)$最小。如果$f(x)$在点$x^*$处是严格单调递减的，那么就得到了最值问题。

如果$f(x)$是凸函数，而且$c=0$，则$f$是凸函数的极小值问题。

设$x^*\in \arg\min_{x} f(x)$，$x$是一个变量，而$f(x)$是一个确定性函数，则有：

$$\begin{array}{ll}\displaystyle \frac{\partial f}{\partial x}=0 & \Rightarrow f(x^*)=\min_{x}f(x)\\\displaystyle (\nabla f(x^*))^{\mathrm{T}}\cdot d& \leq c\\d^\top(\nabla f(x^*))&\leq c\\\end{array}$$

其中，$d$是一个向量。当$c=0$时，只需考虑第一个式子。由$f$的定义得知，当$f(x^*)$是$x$的一个最小值点时，$\nabla f(x^*)=0$，因此只有第二个式子成立。由第二个式子可知，$d^\top(\nabla f(x^*))\leq c$，因此，$d$的长度至多为$c/\Vert\nabla f(x^*)\Vert_\infty$，而$\nabla f(x^*)$的范数又等于$1$，因此，$||d||\leq c/\Vert\nabla f(x^*)\Vert_\infty$。由定理3.5.4可知，当$c=0$时，$\nabla f(x^*)$的范数是最小的。因此，只有当$c>0$时，才有第二个式子成立。

设$d$是$x$的邻域向量，$u,v$是满足约束条件的两个点，则有：

$$\begin{array}{ll}\displaystyle \frac{\partial f}{\partial x}&=0&\text{    }d^\top(\nabla f(x^*))\leq c\\\displaystyle f(u)+\langle d,u-x^*\rangle+\langle v-x^*,d^\top\nabla f(x^*)(u-x^*)\rangle& \leq f(u)+\langle d,u-x^*\rangle+\langle v-x^*,d^\top(\nabla f(x^*)+(\nabla f(x^*)^\top)^{-1}(v-x^*))\rangle\\\displaystyle f(u)&+\langle u-x^*,(\nabla f(x^*)+(\nabla f(x^*)^\top)^{-1}(v-x^*))\rangle \leq f(v)\\\end{array}$$

设$\mu$是任意的常数，则：

$$\begin{array}{ll}\displaystyle f(u)-f(x^*)+\langle u-x^*,d^\top(\nabla f(x^*)+(\nabla f(x^*)^\top)^{-1}(v-x^*))\rangle&\leq f(v)-f(x^*)+\mu\quad (1)\\\displaystyle \frac{\partial f}{\partial x}d^\top(\nabla f(x^*))+d^\top(\nabla f(x^*)+(\nabla f(x^*)^\top)^{-1}(v-x^*))&\leq cd+1\quad (2)\\\displaystyle \frac{\partial f}{\partial x}d^\top(\nabla f(x^*))+d^\top(\nabla f(x^*)+(\nabla f(x^*)^\top)^{-1}(v-x^*))&\leq 0\\d^\top(\nabla f(x^*)+(\nabla f(x^*)^\top)^{-1}(v-x^*))&=0\\\end{array}$$

此时，有：

$$\begin{array}{ll}\displaystyle f(u)-f(x^*)+\mu&\leq f(v)-f(x^*)+\mu\\\displaystyle \langle u-x^*,d^\top(\nabla f(x^*)+(\nabla f(x^*)^\top)^{-1}(v-x^*))\rangle&\leq \langle v-x^*,d^\top(\nabla f(x^*)+(\nabla f(x^*)^\top)^{-1}(v-x^*))\rangle\\d^\top(\nabla f(x^*)+(\nabla f(x^*)^\top)^{-1}(v-x^*))&=0\\\end{array}$$

第(2)式右侧第一项等于零，根据定理3.5.4，可以得到第(2)式右侧第二项为零。故可证$d^\top(\nabla f(x^*)+(\nabla f(x^*)^\top)^{-1}(v-x^*))=0$。由于$d$是$x$的邻域向量，并且$f$是连续的，故存在一个$\rho>0$，使得$\Vert du\Vert_\infty \leqslant \rho$。故：

$$\begin{array}{ll}\displaystyle \frac{\partial f}{\partial x}d^\top(\nabla f(x^*)+(\nabla f(x^*)^\top)^{-1}(v-x^*))&\leq 0\\\displaystyle \sum_{i=1}^{n} \left(\nabla f_i(x^*)+\frac{\nabla f_i(x^*)^\top(\nabla f_i(x^*)+\nabla f_i(x^*)^\top)^{-1}\nabla f_j(x^*)}{(\nabla f_i(x^*)+\nabla f_i(x^*)^\top)^\top(\nabla f_j(x^*)+\nabla f_j(x^*)^\top)}\right)d_i&\leqcd+1\\(\nabla f(x^*)+\nabla f(x^*)^\top)^\top d& \leq c\rho\\||d||_{\infty}\leqslant\rho\\\end{array}$$

第(2)、(3)、(5)式等号成立，第(1)式等号成立当且仅当$\mu=\frac{f(v)-f(x^*)}{d^\top(\nabla f(x^*)+\nabla f(x^*)^\top)^{-1}(v-x^*)}$。但是，当$\rho\rightarrow \infty$时，等号右侧第二项趋于无穷大，因此不能再假设$d$是$x$的邻域向量。

因此，对于$c>0$，函数的极小值问题还没有统一的形式表述，只能依赖于具体的问题具体分析。