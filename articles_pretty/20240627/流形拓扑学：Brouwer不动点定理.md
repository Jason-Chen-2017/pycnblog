# 流形拓扑学：Brouwer不动点定理

关键词：流形、拓扑学、Brouwer不动点定理、同伦、映射度、Lefschetz数、示性类

## 1. 背景介绍
### 1.1  问题的由来
拓扑学是数学的一个分支,它研究空间在连续变换下保持不变的性质。流形是一类特殊的拓扑空间,局部看起来像欧氏空间。Brouwer不动点定理是拓扑学中一个重要定理,它描述了连续映射在某些条件下必定存在不动点。这个定理在非线性分析、动力系统、博弈论等领域有重要应用。
### 1.2  研究现状
Brouwer不动点定理最初由荷兰数学家 Brouwer 在1911年提出并证明。此后,这个定理得到了广泛的推广和应用。Lefschetz 和 Hopf 分别给出了不动点指数的定义,并用它来刻画不动点的存在性。20世纪50年代, Atiyah 和 Bott 把不动点理论和 K 理论联系起来,开创了新的研究方向。近年来,不动点理论与微分动力系统、偏微分方程、拓扑动力系统等领域的交叉研究十分活跃。
### 1.3  研究意义 
Brouwer不动点定理揭示了拓扑空间的本质特征,是现代拓扑学的一块基石。它不仅在纯数学领域有重要地位,在计算机科学、经济学、博弈论等应用学科也有广泛应用。深入研究 Brouwer 不动点定理,对于拓展拓扑学的理论体系、促进数学与其他学科的交叉融合具有重要意义。
### 1.4  本文结构
本文将从以下几个方面介绍 Brouwer 不动点定理：第2节介绍流形、映射等核心概念；第3节讨论 Brouwer 不动点定理的证明思路和推广形式；第4节利用映射度和 Lefschetz 数给出不动点的代数拓扑刻画；第5节通过具体实例演示定理的应用；第6节总结全文,展望不动点理论的发展前景。

## 2. 核心概念与联系
流形是拓扑空间的一类,它在局部与欧氏空间同胚。n维流形是指每一点都有一个同胚于 $\mathbb{R}^n$ 的开邻域的 Hausdorff 空间。紧致性、连通性、可定向性是流形的重要性质。

连续映射是拓扑学的核心概念。设 $X,Y$ 是两个拓扑空间,如果 $f:X\to Y$ 满足：$\forall V\subseteq Y$ 开集,$f^{-1}(V)\subseteq X$ 也是开集,则称 $f$ 是连续映射。恒等映射、复合映射、逆映射都保持连续性。

同伦是描述连续映射之间变形关系的工具。设 $f_0,f_1:X\to Y$ 是两个连续映射,如果存在连续映射 $F:X\times[0,1]\to Y$ 满足
$$
F(x,0)=f_0(x),\quad F(x,1)=f_1(x),\quad \forall x\in X
$$
则称 $f_0$ 与 $f_1$ 同伦,记作 $f_0\simeq f_1$。同伦是一种等价关系。

Brouwer不动点定理刻画了紧致凸集上的连续自映射的不动点存在性。设 $K$ 是欧氏空间 $\mathbb{R}^n$ 中的紧致凸子集,如果连续映射 $f:K\to K$ 满足 $f(x)=x$,则称 $x$ 是 $f$ 的不动点。Brouwer 不动点定理指出,每一个紧致凸集到其自身的连续映射必有不动点。

## 3. 核心算法原理 & 具体操作步骤
### 3.1  算法原理概述
Brouwer不动点定理的一个经典证明基于单纯复形和重心坐标的概念。对于紧致凸集 $K\subset\mathbb{R}^n$,取其一个单纯复形剖分 $T$。对于 $T$ 中每个 $n$ 维单形 $\sigma^n$,定义其重心坐标 $b_{\sigma^n}:K\to\sigma^n$。利用重心坐标构造映射 $g:K\to K$,并证明 $g$ 必有不动点,从而 $f$ 也必有不动点。
### 3.2  算法步骤详解
(1) 对紧致凸集 $K$ 取单纯复形剖分 $T$,记 $T$ 的 $n$ 维单形全体为 $T^n$。

(2) 对每个 $n$ 维单形 $\sigma^n\in T^n$,其顶点为 $v_0,v_1,\cdots,v_n$。$\forall x\in K$,定义重心坐标 $b_{\sigma^n}(x)=(b_0(x),\cdots,b_n(x))$ 为
$$
b_i(x)=\frac{\mathrm{vol}(\mathrm{conv}(v_0,\cdots,\hat{v}_i,\cdots,v_n,x))}{\mathrm{vol}(\sigma^n)},\quad i=0,1,\cdots,n
$$
其中 $\mathrm{conv}$ 表示凸包,$\hat{v}_i$ 表示去掉 $v_i$ 顶点。

(3) 定义映射 $g:K\to K$ 如下：$\forall x\in K$,设 $x$ 位于某个 $n$ 维单形 $\sigma^n$ 内部,其重心坐标为 $b_{\sigma^n}(x)=(b_0(x),\cdots,b_n(x))$,令
$$
g(x)=b_0(x)f(v_0)+b_1(x)f(v_1)+\cdots+b_n(x)f(v_n)
$$

(4) 记 $S=\{(x,g(x))\,|\,x\in K\}$,$S$ 是 $K\times K$ 的紧致子集。定义映射 $h:K\times K\to\mathbb{R}^n$
$$
h(x,y)=x-y,\quad\forall (x,y)\in K\times K
$$
则 $h$ 连续。若 $f$ 无不动点,则 $h(S)\subset\mathbb{R}^n\setminus\{0\}$。
  
(5) 由 $\mathbb{R}^n\setminus\{0\}$ 在同伦意义下等价于 $\mathbb{S}^{n-1}$,可构造连续映射 $\varphi:h(S)\to\mathbb{S}^{n-1}$。再定义 $\psi=\varphi\circ h|_S:S\to\mathbb{S}^{n-1}$,则 $\psi$ 连续。

(6) 由 $S$ 的构造可知,存在同胚 $\theta:K\to S$。令 $\omega=\psi\circ\theta:K\to\mathbb{S}^{n-1}$,则 $\omega$ 连续。但 $K$ 与 $\mathbb{B}^n$ 同胚,由 Borsuk-Ulam 定理,这样的连续映射 $\omega$ 不存在。矛盾,故 $f$ 必有不动点。

### 3.3  算法优缺点
Brouwer不动点定理的单纯复形证明思路清晰,借助了重心坐标、Borsuk-Ulam定理等工具,证明过程几何直观。但是,该证明没有给出找到不动点的构造性方法。此外,该证明还需要借助同伦理论,在直观性上略有不足。
### 3.4  算法应用领域
Brouwer不动点定理在拓扑学、非线性分析中有重要应用。很多非线性问题可以转化为寻找映射的不动点,利用 Brouwer 不动点定理能证明解的存在性。在博弈论中,Brouwer 不动点定理被用来证明 Nash 均衡的存在性。一些计算方法如 Sperner 引理构造性地找到 Brouwer 不动点。

## 4. 数学模型和公式 & 详细讲解 & 举例说明
### 4.1  数学模型构建
设 $M$ 是 $n$ 维光滑紧致流形,$f:M\to M$ 是连续映射。定义 $f$ 的不动点集 $\mathrm{Fix}(f)$ 和迹 $\mathrm{tr}(f)$ 分别为
$$
\begin{aligned}
\mathrm{Fix}(f)&=\{x\in M\,|\,f(x)=x\} \\
\mathrm{tr}(f)&=\sum_{x\in\mathrm{Fix}(f)}\mathrm{ind}(f,x)
\end{aligned}
$$
其中 $\mathrm{ind}(f,x)$ 是 $f$ 在不动点 $x$ 处的指数。

Lefschetz 不动点定理指出,若 $f$ 同伦于恒等映射 $\mathrm{id}_M$,则 $f$ 的不动点个数(考虑重数)等于 $M$ 的示性标量 $\chi(M)$：
$$
\#\mathrm{Fix}(f)=\mathrm{tr}(f)=\chi(M)
$$

### 4.2  公式推导过程
(1) 考虑流形 $M$ 上的向量场 $v$,若 $v$ 的零点集 $\mathrm{Zero}(v)$ 离散,定义 $v$ 的指标 $\mathrm{Ind}(v)$ 为
$$
\mathrm{Ind}(v)=\sum_{x\in\mathrm{Zero}(v)}\mathrm{ind}(v,x)
$$
其中 $\mathrm{ind}(v,x)$ 是 $v$ 在零点 $x$ 处的指数,可以用 Jacobi 行列式计算。

(2) 设 $f:M\to M$ 是光滑映射,其图像 $\Gamma_f=\{(x,f(x))\,|\,x\in M\}$ 是 $M\times M$ 的子流形。定义向量场 $v_f:M\to TM$ 为
$$
v_f(x)=(x,f(x))\in T_{(x,x)}(M\times M),\quad\forall x\in M
$$
则 $v_f$ 的零点即为 $f$ 的不动点。可以证明
$$
\mathrm{tr}(f)=\mathrm{Ind}(v_f)
$$

(3) 设 $M$ 的切丛为 $TM$,定义 $M$ 上的 Euler 类 $e(M)\in H^n(M;\mathbb{Z})$ 满足
$$
\langle e(M),[M]\rangle=\chi(M)
$$
其中 $[M]\in H_n(M;\mathbb{Z})$ 是 $M$ 的基本同调类。

(4) 若 $f\simeq\mathrm{id}_M$,可以证明切丛 $f^*TM$ 与 $TM$ 同构,从而它们的 Euler 类相等：
$$
e(f^*TM)=e(TM)\in H^n(M;\mathbb{Z})
$$
进而由 Poincaré 对偶得到
$$
\mathrm{tr}(f)=\langle e(f^*TM),[M]\rangle=\langle e(TM),[M]\rangle=\chi(M)
$$

### 4.3  案例分析与讲解
考虑二维球面 $\mathbb{S}^2$,其示性数 $\chi(\mathbb{S}^2)=2$。若 $f:\mathbb{S}^2\to\mathbb{S}^2$ 是连续映射,且 $f$ 同伦于恒等映射,则由 Lefschetz 不动点定理,
$$
\#\mathrm{Fix}(f)=\chi(\mathbb{S}^2)=2
$$
即 $f$ 至少有两个不动点。

再如,考虑二维环面 $\mathbb{T}^2=\mathbb{S}^1\times\mathbb{S}^1$,其示性数 $\chi(\mathbb{T}^2)=0$。若 $f:\mathbb{T}^2\to\mathbb{T}^2$ 是连续映射,且 $f$ 同伦于恒等映射,则
$$
\#\mathrm{Fix}(f)=\chi(\mathbb{T}^2)=0
$$
但这只说明 $f$ 的不动点个数与 $0$ 同余,并不能推出 $f$ 没有不动点。事实上,恒同映射 $\mathrm{id}_{\mathbb{T}^2}$ 的每一点都是不动点。

### 4.4  常见问题解答
Q: Brouwer不动点定理成立的充分必要条件是什么？
A: 设 $K$ 是拓扑空间,$f:K\to K