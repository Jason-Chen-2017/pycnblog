
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Fuzzy sets 是一种模糊集合(也称不完全集)的形式。在模糊集中，元素可以是离散或连续值，取决于所定义的模糊值。模糊集合中包含两类运算符：交集、并集、差分、极限运算等。本文将从几何学、集合论、概率论和代数学的角度介绍模糊集和模糊集运算。

# 2.词汇表
- fuzzy set 模糊集（又称模糊值）
- membership function 成员函数，用于描述集合元素与某种属性的对应关系，由输入变量x到输出值y之间的映射关系。

# 3. 基本概念

## 3.1 向量空间及其内积
模糊集通常由向量空间V及其内积定义。向量空间V是一个集合S，它由向量组成。向量的加法、减法、数乘、点积四则运算构成了向量空间的运算规则。设V=(R^n)为n维欧式空间，$x_1, x_2,..., x_n$ 为V中的n个基向量。则，$\forall v \in V, \forall w\in V,\,v+w=|v|+|w|\cdot\frac{v}{\mid v \mid},\,v-w=|v|-|w|\cdot\frac{w}{\mid w \mid}$。即向量的加法定义为矢量的加法，减法定义为矢量的减法。

若$f:(R^n)\rightarrow (R)$是实值函数，且满足$f(0)=0$,那么对于任意非零向量$v\neq 0\in V$,存在唯一的一个向量$w\in V$，使得$f(v)=f(w)+\nabla f(w)\cdot v$。记$\nabla f=\left(\frac{\partial f}{\partial x_1}(0),\frac{\partial f}{\partial x_2}(0),..., \frac{\partial f}{\partial x_n}(0)\right)^T$，则称向量$v$在点$0$处的方向导数（梯度）为$\nabla f(0)$。根据链式求导法则，可得：
$$f_{i}(t)=\frac{\partial f}{\partial x_i}(t)$$
即，$f(v)$可表示为$f(v)=\sum_{j=1}^n x_jv_j+\nabla f(0)\cdot v=f(0)+\sum_{j=1}^n x_jf_{j}(0)v_j$，其中$f_j(0)=\frac{\partial f}{\partial x_j}(0)$。因此，向量空间V及其内积构成了模糊集的基本概念。


## 3.2 Fuzzy Set
模糊集是一个集合S，由真值函数ψ（membership function）描述。如果向量$x$属于S，则其对应的模糊值f(x)为：
$$f(x)=\phi(x), x\in S$$
其中，$\phi(x)$称为x的membership value或degree of membership。当$\phi(x)=1$时，$x$是S的真子集；当$\phi(x)=0$时，$x$不是S的一部分。显然，$\forall x \in X, \phi(x) \in [0,1]$。用符号表示模糊集：$X_{\phi}:=\{x:\phi(x)>0\}\subseteq X$。即，模糊集是一个满足一定条件的集合。

## 3.3 Complementary Fuzzy Set
对于模糊集$A$, 它的补集$A^{c}=X-A$是一个模糊集。其中，$X$为模糊集A的定义域。
$$A^{c} :=\{x:(x\notin A)\}$$
由于其补集，它具有与模糊集$A$相同的定义域$X$，所以也可以说$A^{c}$也是模糊集，但它不是$A$的真子集。


## 3.4 Disjoint and Overlap Fuzzy Sets
两个模糊集$A$和$B$是独立的，当且仅当它们的补集重叠，即$A\cap B = A^{c} \cap B^{c}=\emptyset$. 两个模糊集是相容的，当且仅当它们的交集等于他们的并集，即$A\cup B = B\cup A$. 

## 3.5 Logical Operators on Fuzzy Sets
模糊集运算包括三种基本运算：交集、并集、差集。设$A$和$B$是两个模糊集，则：

### 3.5.1 Union Operator $\bigcup$ 
$$A\bigcup B := \{x|(x\in A)\vee(x\in B)\}$$
称作A和B的并集。该运算满足结合律，即$(A\bigcup B)\bigcup C=A\bigcup(B\bigcup C)$。设$A_1,A_2,...,A_m$为一系列模糊集，则：
$$\bigcup_{i=1}^mA_i:=A_1\bigcup A_2\bigcup...\bigcup A_m$$
称作A1至Am的并集。该运算满足分配律，即$\bigcup_{i=1}^mA_iB_i=A_i\bigcup(B_i)$。

### 3.5.2 Intersection Operator $\bigcap$
$$A\bigcap B := \{x|(x\in A)\land(x\in B)\}$$
称作A和B的交集。该运算满足结合律，即$(A\bigcap B)\bigcap C=A\bigcap(B\bigcap C)$。设$A_1,A_2,...,A_m$为一系列模糊集，则：
$$\bigcap_{i=1}^mA_i:=A_1\bigcap A_2\bigcap...\bigcap A_m$$
称作A1至Am的交集。该运算满足分配律，即$\bigcap_{i=1}^mA_iB_i=A_i\bigcap(B_i)$。

### 3.5.3 Difference Operator $-$
$$A-B :=\{x|(x\in A)\land(x\notin B)\}$$
称作A和B的差集。该运算满足结合律，即$(A-B)-(C-(D-E))=A-(B-\bigcup_{i=1}^m[C-(D-E)]_i)$。

## 3.6 Continuous Functions and Fuzzy Sets
对于连续函数$f:[a,b]\rightarrow R$，假定$f$在区间上连续。则$f^{-1}(u)$是$u$的“逆函数”。对任意$x\in X_{\phi}$, 有：
$$f(x_{\phi})=\max\{0, u-f^{-1}(\phi(x)), f^{-1}(\Phi(x))\}, x_{\phi}\in X_{\phi}, u\in [0,1]$$
其中，$f^{-1}(\phi(x))$表示$x$的最大特征值。

通过类似的方法，可证明：
$$A_{\phi}^{c} =\{x:f(x_{\phi})<u\}\subset X_{\phi}-A_{\phi},~~A_{\phi}\subset X_{\phi}-A_{\phi}^{c}$$

综上所述，模糊集可以由真值函数定义。两个模糊集的交集、并集、差集，都可以通过上述运算进行计算。

## 3.7 Examples
- $U=(0,1)^2$ and $A=\{(x,y):0\leqslant x\leqslant y\leqslant 1\}$. Then $A_{\phi}(x,y)=1$ for all $(x,y)$, so that $\phi(x,y)=1$ if $x\leqslant y$ and $\phi(x,y)=0$ otherwise. The complementary set is $A_{\phi}^{c}=\{(x,y):\phi(x,y)<0.5\}=\{(0.5,1),(0,0.5)\}$. So the size of intersection between them is zero.
- $A_{\phi}(x)=0$ for all $x\in (0,1)$ except one point at $\frac{1}{2}$, where it takes a maximum value of $\frac{1}{2}$. Therefore, its complementary set is $A_{\phi}^{c}=\{(x):\phi(x)<\frac{1}{2}\}=\{\frac{1}{2}\}$. Its union with any other subset has nonzero measure because there are points outside this subset. However, their intersection can have only one element or be empty due to symmetry property of $\phi$. Therefore, they cannot define a partition of unity over $[0,1]$.