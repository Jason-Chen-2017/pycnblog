
作者：禅与计算机程序设计艺术                    

# 1.简介
  

概率论是一门与计算和推理相关的科学学科，研究的是客观世界及其事件发生的概率。概率论包括两个分支：一是**定量统计学（statistical science）**，它研究数量性质的随机现象，如抽样调查、随机试验、实验室测量等。另一个分支则是**随机过程（stochastic process）**，它研究随机运动、随机变化的过程及其规律。概率论是数理统计学的一个分支。概率论的中心理念是描述世界的某些事物在不确定性（uncertainty）或可能性（possibility）下的出现规律，并用数学上的函数来刻画这种规律。概率论可用于对各种现象进行预测、分析、决策和控制。

# 2. 概率分布函数（Probability Distribution Function, PDF）

概率分布函数，也称密度函数（Density function），是随机变量取值的连续分布的曲线图形，能够反映随机变量的概率密度。概率密度是一个概率论中重要的概念，它衡量在一个确定的区间上某个点的取值被多少个单位体积所覆盖。概率分布函数描述了随机变量取值随时间或者空间变化的概率密度。在概率论中，概率分布函数有时也称作分布曲线或概率密度函数。正态分布就是最常用的一种概率分布，也是以钟形曲线为主要特征的概率分布，其概率密度函数公式如下：

$$f(x)=\frac{1}{\sigma \sqrt{2\pi}}e^{-\frac{(x-μ)^2}{2\sigma^2}}$$

其中μ表示正态分布的平均值，σ表示标准差。

# 2.1 CDF：累积分布函数

CDF是指离散型随机变量的累计概率分布。它给出了所有小于等于某一值的随机变量的概率。概率可以理解成从随机变量落入某个值范围内所占的比例。CDF的值是一个非负数，并且当该随机变量大于等于某个特定值时，CDF的值将达到1。显然，CDF仅依赖于概率分布，而不依赖于具体的随机变量采样结果。正态分布CDF的计算公式如下：

$$F(x)=\int_{-\infty}^{x} f(t)\mathrm{d}t=P(X\leq x)=\frac{1}{2}\left[1+\textstyle\int_{-\infty}^{x} e^{-z^2/2} dz\right]$$

# 3. 求解多个随机变量的联合分布

当有多个随机变量的时候，联合分布就产生了。比如，如果两个随机变量分别服从正态分布$N(\mu_1,\sigma_1)$和$N(\mu_2,\sigma_2)$，那么它们的联合分布就是$N((\mu_1+\mu_2), (\sigma_1^2+\sigma_2^2))$。求解联合分布的方法是利用边缘化法。具体来说，首先分别求出每个随机变量的概率密度函数PDF和CDF，然后按照下面这样的方法将两者结合起来得到联合分布：

1. 将每个随机变量的PDF和CDF分别乘以两个随机变量的相应参数，得到两个随机变量的乘积。
2. 对这个乘积的CDF做变换，使得它是两个独立随机变量的联合CDF。

公式化地，假设两个随机变量$X$和$Y$的概率密度函数分别是$p_X(x)$和$p_Y(y)$，且满足以下条件：

* $p_X(x)>0$，$p_X(-\infty)=0$，$p_X(\infty)=1$。
* $p_Y(y)>0$，$p_Y(-\infty)=0$，$p_Y(\infty)=1$。
* $\iint p(x,y)\mathrm{d}x\mathrm{d}y>0$。

那么，对于任意满足$a_1x+b_1y+c_1=0$的实数$(a_1,b_1,c_1)$，有：

$$\iint (ax_0 + by_0 + c)(dx_0dy_0)=(\int dx_0\int dy_0 ax_0by_0)=\int dx_0[\int dy_0 ay_0]=\int dxdy=\frac{\iint dx_0dy_0}{\iint dx_0\iint dy_0}=P(X,Y)$$ 

其中$x_0$是$(x_0,y_0)$的充分统计量。

# 4. 联合分布的应用举例

下面给出两个关于平面中的投掷硬币实验的例子。第一个例子是抛掷一次硬币，另外一个例子是抛掷两次硬币。

## 抛掷一次硬币

在抛掷一次硬币的情况下，只有两种情况：正面和反面。抛掷硬币的可能性是均等的，即有50%的可能性正面朝上，所以正面的概率等于0.5。

先求出正面朝上的概率的分布函数：

$$f(x)=\begin{cases}
    0,&x<0\\
    P(x)&0\leqslant x\leqslant 1\\
    1,&x>1
\end{cases}$$

由此求出正面的CDF：

$$F(x)=\begin{cases}
    0,&x<0\\
    Q(x)&0\leqslant x\leqslant 1\\
    1,&x>1
\end{cases}$$

其中$Q(x)=\int_{-\infty}^{x}f(t)\mathrm{d}t$。

接下来求解$X=U$的联合分布，其中$U$是$[0,1]$上的均匀分布随机变量：

$$F(x)=\begin{cases}
    0,&x<0\\
    Q(x)&0\leqslant x\leqslant 1\\
    1,&x>1
\end{cases}$$

那么，对于任意的$(a,b,c)$，有：

$$\int_{-\infty}^1 \int_{-\infty}^{q} (ax+bq+c) f(q)\mathrm{d}qf(x) =\int_{-\infty}^{1} q f(q) \int_{-\infty}^{q} axf(u)+bqf(u)+cf(u) du\quad (u=x-bq,0\leqslant u\leqslant q)\\
=\frac{1}{2}[uf(u)+(q-u)f(u)] - bq^2 + \frac{1}{2}qf(q)-\frac{1}{2}\int_{-\infty}^{q}(qf(v)-\frac{1}{2}v^2)dv\\
=\frac{1}{2}[af(0)+(bf(0)+c)f(0)+(1-ac-bc)f(1)] - bf(0)^2 \\
= \frac{1}{2} [c+(b+c)f(0)+(1-b-c)f(1)-(a+b)f(0)^2]-bf(0)^2 \\
=-bf(0)^2+a\frac{1}{2}-c\frac{1}{2}+bf(0)^2+cf(0)f(1)$$

因此，对于任意的$(a,b,c)$，有：

$$P(X=a)=\frac{1}{2}[c+(b+c)f(0)+(1-b-c)f(1)-(a+b)f(0)^2]-bf(0)^2 $$

由此可知，$P(X=0)=0.5$。

## 抛掷两次硬币

在抛掷两次硬币的情况下，总共有四种情况：第一轮正面第二轮反面，第一轮反面第二轮正面，两次都正面，两次都反面。而每一种情况的概率都是0.25，所以概率的分布函数为：

$$f(x_1,x_2)=\begin{pmatrix}
0&0&0&\cdots &0&0\\
0&(1-x_1)^{n-1}x_2&0&\cdots &(1-x_1)^{n-1}&0\\
0&0&(1-x_1)^{n-1}x_2&\cdots &0&(1-x_1)^{n}\\
\vdots&\ddots&\vdots&\ddots&\vdots&\vdots\\
0&0&0&\cdots &(1-x_1)^{2}x_2&0\\
0&0&0&\cdots &0&(1-x_1)^{n}
\end{pmatrix}=\delta_{x_1,0}(1-\delta_{x_2,1})^{\ell}M_n(x_2)$$

其中$\delta_{x_1,0}$和$\delta_{x_2,1}$分别是Kronecker符号，$M_n(x)$是第n个多项式，即$M_n(x)=\prod_{j=1}^nx_j^j/(x_1!x_2!\cdots x_n!)$.

由此求出正面的CDF：

$$F(x_1,x_2)=\sum_{\{m_1,m_2:x_1=0,x_2=k\},k=0}^{\min\{n,1\}}\binom{n}{m_1}\binom{n-m_1}{m_2} \delta_{x_2,k}(1-\delta_{x_1,1})^{\ell} M_n(x_2)$$

其中$\binom{n}{m_1}\binom{n-m_1}{m_2}$表示二项系数，$\delta_{x_2,k}(1-\delta_{x_1,1})^{\ell}$是Kronecker符号。

接下来求解$X=U_1, U_2$的联合分布，其中$U_1, U_2$是$[0,1]$上的均匀分布随机变量：

$$F(x_1,x_2)=\sum_{\{m_1,m_2:x_1=0,x_2=k\},k=0}^{\min\{n,1\}}\binom{n}{m_1}\binom{n-m_1}{m_2} \delta_{x_2,k}(1-\delta_{x_1,1})^{\ell} M_n(x_2)$$

那么，对于任意的$(a,b,c)$，有：

$$\int_{0}^{1}\int_{0}^{1} (ax_1^2+bx_1x_2+cx_2^2+dx_1+ex_2+fx_1^2+gx_1x_2+hx_2+ix_1x_2+jx_1^2+kx_1x_2+lx_1+mx_1^2+nx_1+ox_1x_2+px_1^2+qx_1^2+rx_1^2+sx_1+tx_2+ux_1x_2+vx_2^2+wx_1+yx_1^2+zx_1^2+0) f(x_1,x_2) \mathrm{d}x_1\mathrm{d}x_2=\sum_{\{m_1,m_2:x_1=0,x_2=k\},k=0}^{\min\{n,1\}}\binom{n}{m_1}\binom{n-m_1}{m_2} \delta_{x_2,k}(1-\delta_{x_1,1})^{\ell} M_n(x_2)\frac{1}{2}(1+E(x_2)), E(x_2)是第n个多项式$M_n(x_2)/M_{n-1}(x_2)$的形式$$

其中$E(x_2)$表示$x_2$的期望值。由此可知，$P(X=a)=0$, $P(X=b)=0$, $P(X=c)=0$, $P(X=d)=0$, $P(X=e)=1-n(1-x_1)^{\ell}$.