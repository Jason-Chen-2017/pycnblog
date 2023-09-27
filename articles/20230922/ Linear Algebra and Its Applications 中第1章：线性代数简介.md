
作者：禅与计算机程序设计艺术                    

# 1.简介
  

线性代数（英语：Linear algebra）是一种数学分支，其研究对象是线性方程组、向量空间及其子空间、张量等。它的基础是矢量空间的概念，包括标量、向量、矩阵、基变换及其矩阵表示。线性代数在计算几何、运动学、信号处理、物理学、工程科学、生物信息学等领域有着广泛应用。

本章简要介绍了线性代数的主要概念和术语，并对如何通过矩阵乘法来解决线性方程组提供了描述。它还探讨了线性变换的概念，以及线性映射的定义及作用。

# 2.基本概念
## 2.1 向量
向量是一个实数序列，通常记作$v=(v_1,\cdots, v_n)$。其中，$v_i$称为向量的第$i$个分量。一个实数向量可以看作具有三个坐标轴（一般分别表示为x、y、z），它可以表示平面中的点或直线上的一点。

## 2.2 线性组合
如果有向量$\vec{a}=(a_1,\cdots, a_m),\vec{b}=(b_1,\cdots, b_m),\vec{c}=(c_1,\cdots, c_m)$，则它们的线性组合$\alpha \vec{a}+\beta \vec{b}$表示为：
$$
(\alpha\vec{a}+\beta\vec{b})=\left[\begin{matrix}\alpha&amp; &amp;&amp;\beta\\\ &amp;\ddots&amp;&amp;\\ &amp;&amp;\alpha&\beta\\\ &amp;&amp;&amp;\end{matrix}\right]\left[
    \begin{matrix}a_1\\\vdots\\a_m\\b_1\\\vdots\\b_m\end{matrix}
  \right]=[\alpha(a_{1}+b_{1}),\ldots,\alpha(a_{m}+b_{m}),\beta(a_{1}+b_{1}),\ldots,\beta(a_{m}+b_{m})].
$$
  
也就是说，将两个向量的各分量按次序相加或相乘得到新的向量。这个过程可以认为是“将两个向量叠加或相互倾斜”，而所得的结果仍然是一组新的向量。

若有向量$\vec{a}_1,\vec{a}_2,\cdots,\vec{a}_k$，这些向量的线性组合$\sum_{i=1}^ka_i\vec{a}_i$表示为：
$$
(\sum_{i=1}^ka_i\vec{a}_i) = [\underbrace{\vec{a}_{1}}_{\text{$\times k$ times}},\underbrace{\vec{a}_{2}}_{\text{$\times k$ times}},\cdots,\underbrace{\vec{a}_{k}}_{\text{$\times k$ times}}]^T.
$$
  
也就是说，先将各向量逆时针排列（即第一组对应的是$\vec{a}_1$，第二组对应的是$\vec{a}_2$，依此类推），然后将各组中的元素合并在一起，形成新向量。这个过程可以认为是“把多个向量合起来”，得到的仍然是一个向量。

## 2.3 零向量
零向量$0$是一个只有零作为分量的向量。它不是任何其他向量的线性组合，也没有自身与任何向量的交集。由于向量的线性组合总是存在，所以只有零向量的情况特殊，并且不能被看作真正的“零”值。不过，零向量有一个重要的性质：加上任意向量都不会改变它。因此，很多时候我们可以通过加入一些噪声来实现某些算法的鲁棒性。比如，某些求导算子可能返回零向量，如果向量为零向量，就可以跳过相关计算。

## 2.4 单位向量
对于长度为$n$的任意向量$\vec{u}$,有一个唯一对应的单位向量$\hat{u}$,使得$\|\vec{u}\|=\|\hat{u}\|=1$.设单位向量为：
$$
\hat{u}_j=\frac{1}{\sqrt{\det\left(\mathbf{e}_j\otimes\mathbf{e}_j\right)}}\left(\begin{array}{cc}-1&amp;1\\\ 1&amp;-1\end{array}\right)\qquad j=1,2,\cdots n
$$
其中，$\mathbf{e}_j$为第$j$个标准正交基的向量。由双目测量定理可知，任意向量$\vec{u}$都可由一组基底和相应的分量构成，即$\vec{u}=c_1\mathbf{e}_1+c_2\mathbf{e}_2+\cdots +c_n\mathbf{e}_n$,其中$\det\left(\mathbf{e}_j\otimes\mathbf{e}_j\right)=(-1)^j$。因此，单位向量$\hat{u}_j$的分量可以由下式计算得出：
$$
\hat{u}_j=\sqrt{-1^{\frac{j-1}{2}}\det\left(\mathbf{e}_j\otimes\mathbf{e}_j\right)}(\begin{cases}-1&amp;j\equiv 1,3,\cdots \\1&amp;j\equiv 2,4,\cdots\end{cases}).
$$
这里用到了Legendre符号$\left(\begin{array}{cc}-1&amp;1\\\ 1&amp;-1\end{array}\right)$。因此，单位向量可以用如下方式构建：
$$
\hat{u}_j=\frac{1}{\sqrt{\det\left(\mathbf{e}_j\otimes\mathbf{e}_j\right)}}\left(\begin{array}{cc}-1&amp;1\\\ 1&amp;-1\end{array}\right)
$$