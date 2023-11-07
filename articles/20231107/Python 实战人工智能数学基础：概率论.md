
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


概率论（Probability theory）是数理统计学的主要分支之一。在人工智能、经济学、工程学、管理学等各个领域都有广泛的应用。概率论的核心概念就是事件的发生可能性。概率论的研究对象就是随机现象。例如抛硬币、骰子摇到不同点所产生的结果、机器转动、DNA序列突变等都是随机事件。
# 2.核心概念与联系
## 概率空间（Probability space）
**定义1**：设X是一个非空集合，如果A∈X,B∈X且AB=∅(交集为空)，则称X上关于A的事件空间（event space），并记做$P(A)$或$(\Omega,\mathcal{F},\mathbb{P})$，其中$\Omega$表示样本空间，$\mathcal{F}$表示事件空间，$\mathbb{P}$表示概率函数，满足：

1. $\Omega=\bigcup_{i=1}^n A_i \ (n≥1),A_i \in \mathcal{F}$，即样本空间由一些事件组成；
2. 如果$A_1,A_2,\cdots,A_k$互不相容，则存在$k-1$对不同的$A_i,A_j(\ i\neq j)$，使得$A_i\cap A_j = ∅$；
3. 对所有$A_i\in\mathcal{F}$,有$$\sum_{x\in X} P(A_i) = 1.$$

**定义2**：若A，B是两个事件空间，则它们的笛卡尔积的事件空间记做$A×B$或者$A\times B$,称作乘积事件空间，其样本空间为$X_1\times X_2=\{(x_1,x_2)| x_1\in X_1,x_2\in X_2\}$. 此外，记事件$A$对事件$B$的第i个条件依赖，为$a\to b_i$,意味着当A发生时，B的第i个元素必定发生，即$A→B_i=(\{x|(x\in X_2) \wedge a(x)\}∩\{b_i\}).$

## 概率分布（Probability distribution）
**定义3**：给定一个随机变量X，它的值可以从某个有限或无限多个值中取，这些值的概率分布称为随机变量X的概率分布。随机变量X的概率分布是指随机变量X每一种可能出现的情况及其对应的概率。概率分布用函数$f_X:X\rightarrow [0,1]$表示，其中$f_X(x)$表示在X取值为x时的概率。概率分布的期望值或均值记做$\mu_X$,$E[X]=\mu_X$.

## 随机变量及其分布函数（Random variable and its cumulative distribution function）
**定义4**：随机变量X的分布函数$F_X:\mathbb{R}\rightarrow [0,1]$称为随机变量X的累积分布函数，简记作$F_X(x)=P(X\leq x).$随机变量X的分布函数表示随机变量X在某一值以下的概率等于该值的概率。特别地，对于连续型随机变量，$F_X(x)=P(X<x)=\int_{-\infty}^{x} f_X(t)dt.$

**定义5**：随机变量X的随机性质：

1. $X$是一个非负随机变量，$P(X=0)=1,$而$P(X<0)=0.$
2. $X$是离散型随机变量，则$f_X(x)=p_X(x)$是非负整数的概率质量函数，其累计分布函数为$F_X(x)=\sum_{x_i\leq x} p_X(x_i)$;对于连续型随机变量，$f_X(x)$是概率密度函数，$F_X(x)$是概率密度函数的积分.
3. 当$(X,Y)$是一个联合概率分布时，$(X,Y)$的分布函数为$F_{XY}(x,y)=P((X\leq x)(Y\leq y))=$$P((X\leq x)∧(Y\leq y))+\frac{\partial}{\partial x}P(X\leq x)+\frac{\partial}{\partial y}P(Y\leq y)-\int\int_{\leq x}\int_{\leq y}f_{XY}(u,v)dudvdT(u,v)=$$P(X\leq x)P(Y\leq y)-P(X>x)P(Y>y),$其中$(X,Y)=(X_1+X_2,Y_1+Y_2)=(U_1,V_1)-(U_2,V_2)$。

## 随机变量的独立性和条件概率（Independence of random variables and conditional probability）
**定理6**：设$X$和$Y$是随机变量，若$f_{XY}(x,y)$是关于$X$和$Y$的二元函数，则：

1. 如果$X$和$Y$相互独立，即$P(X,Y)=P(X)P(Y)$,$\forall x\in X,\forall y\in Y,$则$f_{XY}(x,y)=f_X(x)f_Y(y)$,$\forall x\in X,\forall y\in Y,$
2. 如果$Z$是随机变量，且$Z$对$X$和$Y$不显著影响，即$\forall z\in Z,\ P(X|z)=P(X)$,$\forall x\in X,$则$f_{XZ}(x,z)=f_X(x)$,$\forall x\in X$,$\forall z\in Z,$
3. 如果$W$是随机变量，且$W$只与$X$有关，即$\forall w\in W,\ P(X,W)=P(X)P(W)$,$\forall x\in X,$则$f_{WX}(x,w)=f_X(x)g_W(w)$,$\forall x\in X,$$\forall w\in W,$
4. 如果$X$和$Y$相互独立，且$Z$是随机变量，且$Z$仅受$X$和$Y$影响，即$\forall z\in Z,\ P(X,Y,Z)=P(X)P(Y)P(Z)$,$\forall x\in X,\forall y\in Y,\forall z\in Z,$则$f_{XYZ}(x,y,z)=f_X(x)f_Y(y)f_Z(z)$,$\forall x\in X,\forall y\in Y,\forall z\in Z,$
5. 如果$X$和$Y$相互独立，且$W$是随机变量，且$Z$是随机变量，且$Z$仅受$X$和$Y$影响，且$W$也仅受$X$和$Y$影响，即$\forall w\in W,\ P(X,Y,Z,W)=P(X)P(Y)P(Z)P(W)$,$\forall x\in X,\forall y\in Y,\forall z\in Z,\forall w\in W,$则$f_{XYZW}(x,y,z,w)=f_X(x)f_Y(y)f_Z(z)g_W(w)$,$\forall x\in X,\forall y\in Y,\forall z\in Z,\forall w\in W,$
6. 如果$X$是随机变量，且$Y$是另一个随机变量，则$Z=g(X,Y)$也是随机变量，$f_Z(z)=P(Z=z)=\sum_{x\in X,\forall y\in Y} P(Z=g(x,y))(f_X(x)f_Y(y))$,$\forall z\in Z,$。

## 边缘概率（Marginal probabilities）
**定义7**：设$X$和$Y$是随机变量，则随机变量$Z:=X+Y$叫做随机变量$X$和$Y$的和，记做$Z=X+Y$.如果$X$和$Y$相互独立，则称$Z$为$X$和$Y$的独立随机变量。设$Z=X+Y=x+y$,$\forall x\in X,\forall y\in Y,$，则：

1. $X$的边缘概率：$P(X=x):=\sum_{y\in Y} P(X=x,Y=y)$
2. $Y$的边缘概率：$P(Y=y):=\sum_{x\in X} P(X=x,Y=y)$
3. $Z$的边缘概率：$P(Z=z)=P(X+Y=z)=\sum_{x_1\in X,\forall y\in Y} P(Z=x_1+y)\cdot P(Y=y)$,$\forall z\in Z.$