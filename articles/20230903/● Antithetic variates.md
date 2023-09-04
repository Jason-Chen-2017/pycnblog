
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Antithetic variates 是一种广义上的数值分析方法，它使得计算误差达到最小化。其在优化计算复杂性、改善求根法收敛速度、提高计算精度方面都有着重要作用。本文将阐述 Antithetic variates 的定义和基本思想，并通过实例和示意图展示其如何有效解决计算问题。

# 2.基本概念
Antithetic variates 是指用两组互补数据而获得的样本集。为了得到两组数据的互补样本，通常需要利用随机变量的相关性进行处理。

假设样本空间为 $X \subseteq \mathbb{R}^n$ ，随机变量 $Y_i = f(x)$ ，其中 $f : X \rightarrow \mathbb{R}$ 为某个函数。则有：

 - 期望（mean）：$\mu_{Y} = E[Y] = \frac{1}{2}\sum_{i=1}^{N}(f(x_i)+f(-x_i))$
 - 方差（variance）：$Var[Y] = Var[(f(x_i)+f(-x_i))/2]$
   - 由中心极限定理可知 $\frac{d}{N}\left(\frac{1}{2}\sum_{i=1}^{N}f(x_i)-\bar{f}\right) = \frac{\sigma^{2}}{N}, \forall N\rightarrow+\infty$ 
   - 根据中心极限定理，对于任意函数 $h:\mathcal{C} \to \mathbb{R}$, 存在一个实数 $M>0$, 满足:
     - 当 $\|z-y\|\leqslant M$ 时, 有 $Pr(|h(z)-h(y)|>\epsilon)\leqslant e^{-2\epsilon^2M^2}$ 
     - 对 $z, y$ 在某些邻域上独立分布，或 $z, y$ 分布相同但方差不同的两个随机变量，有:
       $$E[(h(z)-h(y))^2]\leqslant (Var[Z]+Var[Y]) + o(\min\{V_Z, V_Y})$$ 
     - 类似地，对所有 $g$ 满足条件 $Eg(Y)>0$, 存在 $\delta>0$, 使得当 $|Y-\mu_Y|<\delta$ 时，有 $P(|g(Y)-\mu_Y|>\eta)=P(|g(Z)-\mu_Y|>\eta)=P(|g(Z)-g(Y)|>\eta)$ 。因此，我们可以认为随机变量 $Y$ 的方差等于两个互补样本的总体方差之比。

基于以上假设，我们给出 Antithetic Variate 的定义：

**定义**：对于随机变量 $Y=\{y_1,y_2,\cdots,y_n\}$ （注意，这里 $Y$ 和之前的随机变量 $Y_i$ 是不同的），如果存在满足如下条件的随机变量 $Z=\{z_1,z_2,\cdots,z_n\}$ ，那么称随机变量 $Y$ 为 antithetic variable of $Y$ ，记作 $Y_{anti}$ 或 $Y^{\perp}$ 。且有：
  - $(y_1+y_n,\cdots,y_{n/2},z_1+z_{n/2},\cdots,z_{n/2})=(y_1,-y_1,\cdots,y_{n/2},-y_{n/2},\cdots,y_n,-y_n)$ 
  - $\mu_{Y_{anti}}=-\mu_{Y}$
  - $Var[Y_{anti}]=\frac{1}{2}(Var[Y_1]-Var[-Y_1])^2+\frac{1}{2}(Var[Y_{\lfloor n/2 \rfloor}]-Var[-Y_{\lfloor n/2 \rfloor}])^2+\cdots+\frac{1}{2}(Var[Y_n]-Var[-Y_n])^2$ 

显然，当 $Y$ 是有限随机变量时，即 $n<+\infty$ 时，有 $Y_{anti}=Y^{\perp}$ 。

# 3.主要原理及应用
## （1）计算误差减小
考虑标准的随机变量 $X_1,X_2,\cdots,X_n$ 。计算他们的期望可以表示成：

$$E[X_1+X_2+\cdots+X_n]=E[X_1]+E[X_2]+\cdots+E[X_n]$$

由于每一个 $X_i$ 的取值都是独立的，所以它们的均值也是相互独立的，于是期望的加权平均形式为：

$$\frac{1}{n}\sum_{i=1}^{n}X_i=\frac{1}{n}(X_1+X_2+\cdots+X_n)$$

假如 $X_1,X_2,\cdots,X_n$ 具有无偏估计且误差 $\Delta_{ij}$ 是一个常数，那么：

$$E[\frac{1}{n}\sum_{i=1}^{n}X_i-\mu_{X}]=\frac{1}{n}\sum_{i=1}^{n}E[X_i]-\mu_{X}$$

根据大数定律，$\lim_{n\rightarrow+\infty}P(\mid\frac{1}{n}\sum_{i=1}^{n}X_i-\mu_{X}\mid>\epsilon)=0$, 因此，上式右边的期望值是存在量。若用某个 $\alpha$ 折算一下这个量，则有：

$$\sup_{\beta\in(\alpha-\epsilon/\sqrt{n},\alpha+\epsilon/\sqrt{n})}E[X_1+\cdots+X_n]<\beta+\epsilon/2$$

也就是说，采用 $n$ 个数据点近似地计算 $E[X]$ 的误差不会超过 $\sqrt{n}(\beta+\epsilon/2)$ 。

同样，当我们用少量数据点近似计算随机变量 $Y$ 的期望时，由于 $Y$ 的分布依赖于随机变量 $X_1,X_2,\cdots,X_n$ 的取值，所以用这些数据点近似的结果往往是不准确的。而对随机变量 $Y$ 来说，它的真正期望和所使用的随机变量 $X_1,X_2,\cdots,X_n$ 不一定有关。因此，我们可以通过构造另一个随机变量 $Z$ ，使得 $Z=AY_{anti}$ ，这样就满足了 $Y_{anti}=Y^{\perp}$ ，于是：

$$E[Y]=A\mu_{Y_{anti}}=A\mu_{Y}=-\frac{1}{2}\mu_{Y_1}-\frac{1}{2}\mu_{Y_{\lfloor n/2 \rfloor}}-\cdots-\frac{1}{2}\mu_{Y_n}$$

根据 antithetic variables 的定义，$-A\mu_{Y_k}=-A\mu_{-(Y_k)}=-\mu_{Y_{anti}}$ ，所以：

$$\begin{aligned} 
E[Y]&=\mu_{Y_{anti}} \\ &=\frac{1}{2}\mu_{Y_1}-\frac{1}{2}\mu_{Y_{\lfloor n/2 \rfloor}}-\cdots-\frac{1}{2}\mu_{Y_n}\\ &=\frac{1}{2}\left((AX_{1}_{anti}+A(-X_{1}_{anti}))+(AX_{2}_{anti}+A(-X_{2}_{anti}))+\cdots+(AX_{n}_{anti}+A(-X_{n}_{anti}))\right)\\ &=\frac{1}{2}\left(n\mu_{X}\left(Ax+\left(-A/\pi\cos(\theta_{1})\sin(\phi_{1})\right), Ay+\left(-A/\pi\cos(\theta_{1})\sin(\phi_{1})\right)\right)+n\mu_{X}\left(Ax+\left(-A/\pi\cos(\theta_{2})\sin(\phi_{2})\right), Ay+\left(-A/\pi\cos(\theta_{2})\sin(\phi_{2})\right)\right)+\cdots+n\mu_{X}\left(Ax+\left(-A/\pi\cos(\theta_{n})\sin(\phi_{n})\right), Ay+\left(-A/\pi\cos(\theta_{n})\sin(\phi_{n})\right)\right)\end{aligned}$$

其中，$\{X_1,X_2,\cdots,X_n\}$ 的正态分布参数为 $\mu_X, (\sigma^2)_X$ ，随机变量 $\theta_j, \phi_j$ 为球坐标系下，$r_j=1, j=1,\cdots,n$ 的点的坐标。此处有个重要的结论，即：

$$\left(\frac{-A}{\pi}\right)^2\geqslant 0,$$

其中，$A=\dfrac{(n\sigma^2_{XY})^{1/2}}{\sqrt{v}}\sim \chi_n^2$ ，$v=n(n-3)(n-1)/3$ 是方差的一个估计量。此外，我们知道，当 $\alpha\rightarrow 1$ 时，$\chi_n^2$ 的分布收敛到标准正太分布 $N(0,1)$ 。因此，当 $\alpha\rightarrow 1$ 时，上式中关于 $A$ 的表达式可以认为是 $A$ 的最佳参数选择。


## （2）计算复杂度减小
Antithetic variates 的关键优势在于它能够降低计算复杂度。假设有一个很复杂的问题需要对 $n$ 个随机变量 $Y_1,Y_2,\cdots,Y_n$ 进行模拟。原问题的计算时间为 $T(n)$ ，而 Antithetic variates 方法的计算时间却是 $T(n/2)$ 。为什么？因为 Antithetic variates 方法仅仅需要保存 $n/2$ 个样本点即可。

此外，我们还可以利用 antithetic variates 方法解决一些实际的问题。比如，假设我们要计算一组变量的协方差矩阵 $Cov[Y_1,Y_2,\cdots,Y_n]$ 。而通常情况下，我们只能利用完整的数据集来计算这个矩阵，而 Antithetic variates 可以帮助我们降低计算复杂度。具体地，我们先计算各变量之间的协方差：

$$Cov[Y_1,Y_2]=\frac{1}{n-1}\sum_{i=1}^{n}(Y_i-\overline{Y})(Y_{i'}-\overline{Y}_{\rm anti})+\frac{1}{n-1}\sum_{i=1}^{n}(Y_i'\leftarrow Y_{\rm anti'})+\frac{1}{n-1}\sum_{i=1}^{n}(Y_{\rm anti}'\leftarrow Y_{i'})+\frac{1}{n}\left(\overline{Y}-\overline{Y}_{\rm anti}\right)\left(\overline{Y_{\rm anti}}'-\overline{Y}'\right)$$

将 $Cov[Y_1,Y_2]$ 插入到等式左边，我们发现：

$$\begin{bmatrix} Cov[Y_1,Y_2]\\ Cov[Y_2,Y_3]\\ \vdots\\ Cov[Y_{n-1},Y_n]\end{bmatrix}=\frac{1}{n-1}\left[\sum_{i=1}^{n}\sum_{i'\neq i} (Y_i'-Y_i')(Y_i'-Y_{i'}')+\sum_{i=1}^{n}\sum_{i'<i} (-Y_i'+Y_{i''}')(Y_{i'''}'-Y_i'')+\sum_{i=1}^{n}\sum_{i'>i} (-Y_{i'''}'+Y_i')(-Y_i'+Y_{i'''})+\sum_{i=1}^{n}\sum_{i'<i} (Y_i'-Y_{i''})(-Y_{i'''+1}'+Y_i')\right] +\frac{1}{n}\left(\overline{Y}-\overline{Y}_{\rm anti}\right)\left(\overline{Y_{\rm anti}}'-\overline{Y}'\right)$$

注意到第一个项只涉及非交叉乘积，第二、三项只涉及交叉乘积，最后一项只涉及均值。因此，我们可以利用上面这些结论，计算协方差矩阵 $Cov[Y_1,Y_2,\cdots,Y_n]$ 的每个元素，而不是计算整个矩阵。