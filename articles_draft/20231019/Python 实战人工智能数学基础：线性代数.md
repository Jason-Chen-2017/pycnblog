
作者：禅与计算机程序设计艺术                    

# 1.背景介绍

  
机器学习、深度学习等人工智能领域中，线性代数在计算向量、矩阵的相关操作上扮演着重要角色。因此，掌握线性代数对于很多机器学习、深度学习的应用非常关键。下面，我们就来学习和了解一些基础的线性代数知识。本文将从以下几个方面对线性代数进行介绍：
- 一维空间上的矢量及其运算
- 二维空间中的线段、平行于坐标轴的直线
- 三维空间中的空间曲面和射影映射
- 对角阵、满秩阵和三对角阵
- 矩阵乘法、转置矩阵、迹、行列式、特征值和特征向量、QR分解等概念及运用方法
- 特殊值和本征值问题、奇异值分解、拉普拉斯方程求根等
- 向量空间、内积、范数、正交化、基变换、矩阵表示、Gram—Schmidt正交化等概念及运用方法
- 求逆矩阵、最小二乘法、投影、正交投影等概念及运用方法
- 线性独立性、列空间、零空间、奇异空间、子空间等概念及运用方法
# 2.核心概念与联系  
## 一维空间上的矢量及其运算
### 1.1.矢量
矢量可以视作数量、方向或者大小都可以变化的量。矢量可以表示几何图形中的线段、平行于坐标轴的直线、某一点到另一点之间的距离等概念。如图所示：  

图中，矢量表示的是一条由坐标轴到两个端点的线段，从一个点到另一个点的距离可以用矢量的模长来表示。比如，图中红色矢量的长度就是该矢量指向另一点的距离。

### 1.2.矢量运算
矢量运算是指对矢量进行加减乘除、单位化、矢量夹角、向量和标量的乘法、反射等运算。下面我们来看一下这些运算的定义和推导过程。
#### 1.2.1.矢量加法
设矢量$a=(a_x, a_y)$、$b=(b_x, b_y)$，则它们的加法$c=a+b$定义为：
$$c=\begin{bmatrix} c_x \\ c_y \end{bmatrix}=a+\begin{bmatrix} b_x \\ b_y \end{bmatrix}$$
其中，$c_x = a_x + b_x$, $c_y = a_y + b_y$. 

证明：  
$$\left(\begin{bmatrix} a_x \\ a_y \end{bmatrix} + \begin{bmatrix} b_x \\ b_y \end{bmatrix}\right)\cdot\left(\begin{bmatrix} -1 \\ 0 \end{bmatrix}\right)=\left(-1\times a_x + 0\times a_y + (-1)\times b_x + 0\times b_y\right)+\left(0\times a_x + 1\times a_y + (0)\times b_x + 1\times b_y\right)\\ = -a_x+a_y+b_x+b_y\\=-ab$$  

即，$(-1\times a_x,0\times a_y,-1\times b_x,0\times b_y)+(0\times a_x,1\times a_y,(0)\times b_x,1\times b_y)=-a_xb_y+a_yb_x$ 。因此，$(c_x,c_y)$满足$(a_x+b_x,a_y+b_y)$。

#### 1.2.2.矢量减法
设矢量$a=(a_x, a_y)$、$b=(b_x, b_y)$，则它们的减法$c=a-b$定义为：
$$c=\begin{bmatrix} c_x \\ c_y \end{bmatrix}=a-\begin{bmatrix} b_x \\ b_y \end{bmatrix}$$
其中，$c_x = a_x - b_x$, $c_y = a_y - b_y$. 

证明：  
$$\left(\begin{bmatrix} a_x \\ a_y \end{bmatrix}-\begin{bmatrix} b_x \\ b_y \end{bmatrix}\right)\cdot\left(\begin{bmatrix} -1 \\ 0 \end{bmatrix}\right)=\left((-1)\times a_x + 0\times a_y + (-1)\times b_x + 0\times b_y\right)-\left(0\times a_x + 1\times a_y + (0)\times b_x + 1\times b_y\right)\\ = -a_x-a_y-b_x+b_y\\=a+b$$  

即，$(-1\times a_x,0\times a_y,-1\times b_x,0\times b_y)-(0\times a_x,1\times a_y,(0)\times b_x,1\times b_y)=(a_x-b_x,a_y-b_y)$。因此，$(c_x,c_y)$满足$(a_x-b_x,a_y-b_y)$。

#### 1.2.3.矢量乘积
设矢量$a=(a_x, a_y)$、$b=(b_x, b_y)$，则它们的乘积$\vec{c}$定义为：
$$\vec{c}=\begin{bmatrix} c_x \\ c_y \end{bmatrix}=a\cdot b$$
其中，$c_x = a_x b_x - a_y b_y$, $c_y = a_x b_y + a_y b_x$. 

证明：  
$$\left(\begin{bmatrix} a_x \\ a_y \end{bmatrix}\cdot\begin{bmatrix} b_x \\ b_y \end{bmatrix}\right)\cdot\left(\begin{bmatrix} -1 \\ 0 \end{bmatrix}\right)=\left((a_x b_x)-(a_y b_y), -(a_x b_y)+(a_y b_x)\right)\cdot\left(-1,0\right)\\=a_yb_x-a_xb_y=-|a||b|\cos\theta_{ab}\\\vec{c}=\begin{pmatrix} a_y & -a_x \\ a_x & a_y \end{pmatrix}\begin{pmatrix} b_x \\ b_y \end{pmatrix}\\\vec{c}=\begin{pmatrix} b_x & -b_y \\ b_y & b_x \end{pmatrix}\begin{pmatrix} a_x \\ a_y \end{pmatrix}\\\vec{c}=\begin{pmatrix} c_x \\ c_y \end{pmatrix}$$  

即，$(c_x,c_y)$满足$(a_x b_x-a_y b_y, a_x b_y+a_y b_x)$。

#### 1.2.4.单位矢量
设矢量$v=(v_x, v_y)^T$，则它的单位矢量$u$可表示为：
$$u=\frac{\vec{v}}{|v|}=\frac{1}{|v|}\begin{bmatrix} u_x \\ u_y \end{bmatrix}$$
其中，$u_x = v_x/\sqrt{v_x^2+v_y^2}$, $u_y = v_y/\sqrt{v_x^2+v_y^2}$.

证明：  
$$u_{\parallel}=\frac{1}{\sqrt{v_x^2+v_y^2}}\begin{bmatrix} u_{\parallel x} \\ u_{\parallel y} \end{bmatrix}$$
证明过程类似矢量加法、减法的证明过程，略去不赘述。

#### 1.2.5.矢量夹角
设矢量$a=(a_x, a_y)$、$b=(b_x, b_y)$，它们的夹角$\theta$可表示为：
$$\theta=\cos^{-1}(\frac{(a\cdot b)/(|a||b|)}{\sqrt{(a\cdot a)(b\cdot b)}})=\cos^{-1}(\frac{a\cdot b}{|a||b|})$$

证明：  
$$\cos^{-1}(x)=\arccos(x)$$
又因为：
$$\cos^{-1}(\frac{a\cdot b}{|a||b|})\approx\frac{a\cdot b}{|a||b|},\quad\text{当}\;|a||b|=1$$
因此：
$$\cos^{-1}(\frac{a\cdot b}{|a||b|})\approx\frac{a\cdot b}{1}$$
再利用：
$$\cos^{2}(\alpha)=1-\sin^{2}(\alpha)$$
得：
$$\theta=\arccos(\frac{a\cdot b}{|a||b|})=\arccos(\frac{a_xa_xb_x+a_ya_yb_y}{|a||b|})$$

#### 1.2.6.$n$维矢量和标量的乘法
设矢量$a=(a_1, a_2,..., a_n)$、$b=b$，则它们的$n$维矢量和标量的乘法可表示为：
$$\vec{c}_n=\begin{pmatrix} c_1 \\ c_2 \\ \vdots \\ c_n \end{pmatrix}=a\cdot b$$

证明：  
$$\left(\begin{pmatrix} a_1 \\ a_2 \\ \vdots \\ a_n \end{pmatrix}\cdot b\right)\cdot\left(\begin{pmatrix} -1 \\ 0 \end{pmatrix}\right)=\left(a_1b+a_2b+\cdots+a_nb\right)\cdot\left(-1,\ldots,0\right)=ba_i$$
即，$(c_1,c_2,\ldots,c_n)$满足$a_1b,a_2b,\ldots,a_nb$。

### 1.3.向量空间、基、基变换
#### 1.3.1.向量空间
向量空间是指能够完成加法、减法、数乘、张成等运算的空间。假设有向量空间$V$，则：
- 非零向量属于$V$；
- 如果$a$、$b$都是非零向量且$a+b$也在$V$，那么$a$和$b$属于$V$；
- 如果存在某个非零向量$0$，使得任意$a+0=a$,$a\cdot 0=0$，那么$V$中必含有零向量。

#### 1.3.2.基
基是$V$中元素个数等于其维数的向量集合。每个基都相互正交。任给向量空间$V$，存在唯一的一组基，这个基称为$V$的标准基。

#### 1.3.3.基变换
如果有$m$个向量$v_1,v_2,\cdots,v_n$，那么可以通过某些基转换得到与$v_1,v_2,\cdots,v_n$同构的新向量。这时，每一个基$e_1,e_2,\cdots,e_n$都对应了一个新的向量$w_1,w_2,\cdots,w_n$，且：
$$w_j=\sum_{i=1}^me_{ij}\cdot v_i,\quad j=1,2,\cdots,n$$
也就是说，通过基$e_1,e_2,\cdots,e_n$的转换，我们可以得到$v_1,v_2,\cdots,v_n$同构的新向量$w_1,w_2,\cdots,w_n$。

### 1.4.线性独立性、列空间、零空间、奇异空间、子空间
#### 1.4.1.线性独立性
向量空间$V$的向量集$S$是否线性无关，如果不依赖于其他向量，则称$S$线性无关，记作$S\perp V$或$S\bot V$。

#### 1.4.2.列空间
设$V$的列向量组为$A=[a_1,a_2,\cdots,a_n]$，那么$span A$，也称$A$的列空间（column space）或基础列空间，表示为：
$$span A=\left\{v\in V:\forall i,av_i=0\right\}.$$

#### 1.4.3.零空间
设$V$的行向量组为$B=[b_1,b_2,\cdots,b_k]$，那么$null B$，也称$B$的零空间（nullspace），表示为：
$$null B=\left\{v\in V:\forall i,bv_i=0\right\}.$$

#### 1.4.4.奇异空间
设$V$的列向量组为$A=[a_1,a_2,\cdots,a_n]$，若$rank A<dim V$，则$A$的秩为$r$，$A$的奇异向量组$U=[u_1,u_2,\cdots,u_r]$，则$U$的集合称为$A$的奇异空间（nullspace）。

#### 1.4.5.子空间
设$W$为向量空间$V$的一个子空间，则$W$叫做$V$的一个真子空间。若$V$中所有真子空间的并集等于$V$，则$V$称为它自己（$V$）的全体真子空间。

## 二维空间中的线段、平行于坐标轴的直线
### 2.1.线段

线段是由两点确定一条直线段，且两点间的直线。它由以下几要素构成：端点、起始点和终止点、方向矢量。端点可以称为端点，起始点和终止点可以称为起点和终点，方向矢量可以称为方向向量。

### 2.2.平行于坐标轴的直线

平行于坐标轴的直线一般可以表示为：$ax+by+c=0$。其中$a$, $b$, $c$为参数，$a$, $b$为直线的斜率，$c$为直线与坐标轴的交点的横坐标。

平行于坐标轴的直线与$Ox$或$Oy$垂直，即：$a^2+b^2=1$。

### 2.3.点、直线的位置关系

判断两个点或直线的位置关系的方法主要有以下五种：

1. 在直线的左边：有一个点在直线的左边，则有一个比它大的另一个点在直线的右边。
2. 在直线的右边：有一个点在直线的右边，则有一个比它小的另一个点在直线的左边。
3. 在直线的上边：有一个点在直线的上边，则有一个比它低的另一个点在直线的下边。
4. 在直线的下边：有一个点在直线的下边，则有一个比它高的另一个点在直线的上边。
5. 在直线上：有一个点在直线上，则它是该直线的延长线段的一部分。

### 2.4.线段的长度

线段的长度由两个端点的连线上两点的距离决定。所以，线段的长度等于其端点间的距离。