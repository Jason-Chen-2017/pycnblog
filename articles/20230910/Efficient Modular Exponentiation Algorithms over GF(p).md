
作者：禅与计算机程序设计艺术                    

# 1.简介
  

在信息安全、密码学等领域中，大量使用基于数论的模运算。模运算的目的主要是为了解决大整数运算时的快速计算。模运算在RSA，Diffie-Hellman密钥交换等算法中扮演着重要角色，其计算量巨大且十分关键。目前已经提出了很多高效率的模运算方法，比如中国剩余定理（CRT）法和快速模幂法，但还有更进一步的提升空间。本文将介绍一种新的有效的模运算方法——Pomerance–Vaudenay algorithm (PV)，它为GF(p)上的模运算提供了一个快速且有限内存的算法。该算法与中国剩余定理法类似，也是基于整数的算术方程式。但是相比于CRT，它的处理过程更简单，时间复杂度低，所以速度更快。除此之外，它还支持任意维度的模运算，而CRT只支持两次乘法。最后，作者还给出了一些优化算法，如基数限制法、重试策略及哈希表缓存。
# 2.背景介绍
模运算是指用某个正整数模除另一个数，得到的余数。比如对于一个数$a$和$n$，若$b=a\bmod{n}$则称$b$是$a$对$n$取模后的结果。这一过程通常用于加密系统，例如RSA、Diffie-Hellman密钥交换。
在研究模运算时，最常用的数论原理是费马小定理。费马小定理告诉我们，任何一个$p$与$p$互质的整数都可以唯一地表示为两个互质的数的乘积，即$p=\pm ab$。因此，如果$x$和$y$都是$p$的倍数，那么它们的乘积恰好等于$p^k$，其中$k>0$。当$m$是素数时，费马小定理也可以用于求模运算。
然而，实际应用中素数越多，模运算越困难。在这种情况下，有人提出了数论的“蒙哥马利多项式”理论。这个理论认为，任何一个整数都可以由至少两个不同的数与自身的乘积表示出来。因此，根据蒙哥马利多项式，可以构造出无穷多个不同的模数。但是，这样做的代价是无穷多个不同的乘法次数，而且会消耗大量的内存。因此，现实世界中的模运算一般采用基于整数的算术方程式，以减少算术运算的数量。
由于模运算要用到大量的乘法运算，因此，对GF(p)上模运算进行研究，尤其重要。GF(p)是一个有限域，其中每个元素都可以看作是一个长度为$p$的二进制串，但只有$p$个可能的值。这样，模运算的运算规则就变成了“位移”。举例来说，对于$p=7$和$a=(1101\_2)$，求$a\bmod{5}=(1)_2$，因为$a_{7} \equiv a_{5}\pmod{5}$。也就是说，取出的第七位、五位和零位分别等于$1$，而其他位没有意义。GF(p)上的模运算的一个应用就是生成公钥和私钥。所谓公钥就是公开的，任何人都可以知道；私钥则是保密的，只有自己才知道。公钥和私钥之间的关系通常采用模运算，公钥利用私钥加密消息，私钥利用公钥解密消息。
# 3.基本概念术语说明
## 3.1 定义
在开始讨论GF(p)上的模运算之前，先对GF(p)相关的概念和术语进行一个简单的介绍。
### 3.1.1 阶（degree）
定义：设$F$是一个有限域，$\alpha$是$F$上的一个元素，$\deg(\alpha)$记为$\deg_F(\alpha)$，表示$\alpha$关于$F$的阶，也称为$\alpha$在$F$上的degree。$\deg(\alpha)=0$时，称$\alpha$是$F$上的单位元。

举例：设$F$是$\mathbb{Z}/p\mathbb{Z}$，则$\deg_F(f)\leq p-1$，其中$f=\alpha_0+\alpha_1p+...+\alpha_{p-1}p^{p-2}$，$f\in F[x]$，$p$是$F$的素数。

### 3.1.2 次（order）
定义：设$F$是一个有限域，$\beta$是$F$上的一个元素，$ord_F(\beta)$记为$ord(\beta)$或$ord_{\mathfrak{F}}(\beta)$，表示$\beta$关于$F$的阶，也称为$\beta$的order。当$\beta$不是$F$的单位元时，$ord(\beta)\geq 1$，否则$ord(\beta)=0$。

举例：设$F$是$\mathbb{Z}/p\mathbb{Z}$，则$ord_F(\beta)=p$，其中$\beta=x^{\frac{p}{2}}\bmod{p}=g^{\frac{p}{2}}$，$g$是$F$的生成元。

### 3.1.3 加法
定义：设$F$是一个有限域，$\alpha,\beta$是$F$上的元素，$\alpha+\beta$表示$F$上的元素$a$与$b$的加法，记作$\alpha+\beta$。

形式定义：$\forall x\in F:x+x=0$。

结合律：$(\alpha+\beta)+\gamma=(\alpha+\beta')+\gamma$，其中$\beta'=\alpha-\beta$。

分配律：$\alpha+(b\cdot c)=\left\{ {\begin{array}{*{20}{c}}{\alpha + b}&{\text { if } c = 0}\\{\alpha}&{\text { otherwise }}\end{array}} \right.$。

单位元：$e_F=\{\alpha|(\alpha+1)=0\}$。

举例：设$F$是$\mathbb{Z}/p\mathbb{Z}$，则$e_F=\{\omega^\infty | \omega^\infty+\omega^\infty=0\}=\{\omega^p\}$.

### 3.1.4 乘法
定义：设$F$是一个有限域，$\alpha,\beta$是$F$上的元素，$\alpha*\beta$表示$F$上的元素$a$与$b$的乘法，记作$\alpha*\beta$。

形式定义：$\forall x\in F:\exists y\in F:xy=1$。

结合律：$(\alpha*\beta)*\gamma=\alpha*(\beta*\gamma)$。

分配律：$\alpha*(b\cdot c)=(b*\alpha)*(c*\alpha)$。

逆元：存在元素$\alpha^{-1}$使得$\alpha\cdot\alpha^{-1}=1$。

举例：设$F$是$\mathbb{Z}/p\mathbb{Z}$，则$\omega$是$F$上的一个元，$\omega+\omega=0$，且$\exists z\in F:(z*\omega)=1$。$\omega$称为$F$的本原根。

### 3.1.5 上下文
上下文指的是模运算表达式中出现的变量。在GF(p)上模运算中，上下文可能包括$p$值，模数$n$，公钥$n$,$e$值，私钥$d$值等等。

## 3.2 模运算及其性质
模运算是数论中非常重要的运算符。它的作用是在一定范围内进行计算，并避免溢出。本节我们首先介绍模运算的概念以及一些关于GF(p)的性质。

### 3.2.1 模运算的定义及性质
设$F$是一个有限域，$n$是一个整数。那么$F[x]$表示$F$上的多项式环，而$f\in F[x], g\in F[x]$，则$f+g, f-g$表示$F[x]$上的向量加法和减法，而$fg$表示$F[x]$上的向量乘法。设$I=[0,\cdots,n-1]\subseteq\mathbb N$，则$N=\langle I\mid n\rangle$表示$F$的子集$F(\Omega_\text{F}, n)$。

记$F[x]/(f)$为$F[x]$上的$f$除法群，$N/(f)$为$F(\Omega_\text{F}, n)$上$f$的子群。

假设$f(x),g(x)\in F[x]$，且$deg(f)<n$。则$f(x),g(x)$在模$f(x)$下属于同一环，因此，定义$f(x)\equiv g(x)\pmod{f(x)}$。定义$f(x)\equiv h(x)\pmod{(p)}$表示$h(x)$与$p$的最小公倍数。这里，$p$是一个素数。

举例：设$F=\mathbb Z/p\mathbb Z$, $p=5$, $\alpha=\omega^\infty$，则$ord_F(\alpha)=p$. 此外，$F[\omega]=F[x]/(x^5-1), f(x)=x^5-1$. 

### 3.2.2 有限域上的模运算
定义：设$F$是一个有限域，$\alpha,\beta,\gamma\in F$，则$\alpha\equiv\beta\pmod{n}$表示$n$是$F$的某个素数，如果$\beta=\alpha\pmod{n}$。

定理：$\forall n\in\mathbb N_+, (\alpha\equiv\beta\pmod{n})=\neg(\beta\equiv\alpha\pmod{n})$。

定理：设$F$是一个有限域，$\beta\in F$，那么$|\beta|=|\beta^e|=|(d_1\cdot\cdots\cdot d_t)|$,其中$t$是$n$的因子个数，$d_i$是$n$的第$i$个因子。

证明：略。

定理：设$F$是一个有限域，$n$是一个素数，$f\in F[x]$。那么，$f(x)\equiv 0\pmod{n}$当且仅当$f(x)$在模$n$下恒为零。

证明：反证法。若$f\neq 0$，则$f$不能整除$n$。因而，存在整数$\delta$满足$0=\delta\cdot n$。由于$f$的次数一定不超过$n$，故$f$有指数$\delta$。由于$\gcd(\delta,n)=1$, 所以$\delta=u\cdot v$，其中$u,v\in\mathbb N_+$。设$s$是$n$的素因子个数，则$t=\lfloor\frac s2\rfloor$。有$t<\delta$且$s=2t+1$。因为$n$是素数，所以$\delta$是偶数。

因此，$f$恰好是$t$个次数为$-1$的多项式，其系数为$(d_1,-d_2,-\cdots,-d_t)$。而且，$\beta\equiv\sum (-1)^k\beta^e_k\pmod{n}$，其中$e_k=-1$。由于$\beta$是$f$的模，故$f(\beta)=\beta^t=0$，即$f(x)\equiv 0\pmod{n}$. 但$f(x)\neq 0$，矛盾。

综上所述，$f(x)\equiv 0\pmod{n}$当且仅当$f(x)$在模$n$下恒为零。

### 3.2.3 元素$F(\Omega_\text{F}, n)$及其运算
元素$F(\Omega_\text{F}, n)$是一个环，其中元素$w$的乘法有如下形式：
$$
w_1\cdot w_2=(1+\lambda w_1)(1+\mu w_2)-\lambda\mu w_1w_2
$$
其中$\lambda,\mu$是两个非负整数。

定理：设$F$是一个有限域，$n$是一个素数，$F(\Omega_\text{F}, n)$是环。则：
1. $(1+\lambda w_1)(1+\mu w_2)$与$-\lambda\mu w_1w_2$均为$F(\Omega_\text{F}, n)$上的元素。
2. 如果$\beta,\gamma\in F(\Omega_\text{F}, n)$，则$\beta+\gamma,\beta-\gamma,\beta\cdot\gamma\in F(\Omega_\text{F}, n)$。
3. $\beta\in F(\Omega_\text{F}, n)$的$p$次幂是$[(1+\lambda\beta)]^p-(1-\lambda\beta)^p$，其中$p$是素数。

证明：略。