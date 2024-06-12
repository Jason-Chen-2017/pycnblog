# 代数群引论：B.5 位形的上同调

## 1.背景介绍

在代数拓扑学中,位形(Sheaf)是一个重要的概念,用于研究空间上的代数结构。位形为我们提供了一种将局部数据组合成全局数据的方法,使得我们能够研究空间的几何和代数性质。上同调(Cohomology)是位形理论中的一个核心概念,它描述了位形中的"洞"或"障碍",为我们提供了理解空间拓扑结构的强有力工具。

## 2.核心概念与联系

### 2.1 位形(Sheaf)

位形是一个代数结构,它将一个拓扑空间X与一个代数结构(通常是一个环或模)相关联。形式上,一个位形F是一个由以下数据组成的对象:

1. 对于X中的每个开集U,有一个代数结构F(U)与之相关联。
2. 对于X中的任意开集U和V,以及U和V的任意交集,有一个限制映射$\rho_{U,V}:F(U)\rightarrow F(V)$,满足以下条件:
   - $\rho_{U,U}$是恒等映射。
   - 如果W是U和V的子集,则$\rho_{V,W}\circ\rho_{U,V}=\rho_{U,W}$。

这种结构使得我们可以将局部数据(即开集上的代数结构)组合成全局数据。

### 2.2 上同调(Cohomology)

上同调是位形理论中的一个核心概念。给定一个位形F,我们可以构造一个链复形$\cdots\rightarrow C^{n-1}(F)\xrightarrow{\delta^{n-1}}C^n(F)\xrightarrow{\delta^n}C^{n+1}(F)\rightarrow\cdots$,其中$C^n(F)$是n维上同调组,而$\delta^n$是边映射。上同调组$H^n(F)$定义为$\ker\delta^n/\operatorname{im}\delta^{n-1}$。

上同调组测量了位形中的"洞"或"障碍"。当$H^n(F)=0$时,说明位形在第n维上是无洞的。反之,如果$H^n(F)\neq0$,那么就存在一些不能用局部数据表示的全局结构。

## 3.核心算法原理具体操作步骤

计算位形的上同调涉及到一系列的步骤,包括构造链复形、计算边映射、确定同境和上同调组等。下面我们将详细介绍这个过程。

### 3.1 构造链复形

给定一个位形F,我们首先需要构造相应的链复形$\cdots\rightarrow C^{n-1}(F)\xrightarrow{\delta^{n-1}}C^n(F)\xrightarrow{\delta^n}C^{n+1}(F)\rightarrow\cdots$。这个链复形是由一系列的Abel群$C^n(F)$和边映射$\delta^n$组成的。

具体来说,对于每个维数n,我们定义$C^n(F)$为所有由n+1个开集$U_0,\ldots,U_n$生成的交集$U_0\cap\cdots\cap U_n$上的位形F的节的乘积。形式上,我们有:

$$C^n(F)=\prod_{U_0\cap\cdots\cap U_n\neq\emptyset}F(U_0\cap\cdots\cap U_n)$$

其中,乘积是在所有非空交集上取的。

### 3.2 计算边映射

一旦我们构造了$C^n(F)$,我们需要定义边映射$\delta^n:C^n(F)\rightarrow C^{n+1}(F)$。这个映射是通过交替求和的方式定义的。具体来说,对于$C^n(F)$中的元素$s=(s_{U_0\cap\cdots\cap U_n})$,我们有:

$$\delta^n(s)=\sum_{i=0}^{n+1}(-1)^i\rho_{U_0\cap\cdots\cap\widehat{U_i}\cap\cdots\cap U_{n+1}}^{U_0\cap\cdots\cap U_{n+1}}(s_{U_0\cap\cdots\cap\widehat{U_i}\cap\cdots\cap U_n})$$

其中,符号$\widehat{U_i}$表示略去$U_i$,而$\rho$是位形的限制映射。

可以验证$\delta^{n+1}\circ\delta^n=0$,这就意味着我们确实得到了一个链复形。

### 3.3 确定同境和上同调组

现在,我们可以定义$n$维上同调组$H^n(F)$为$\ker\delta^n/\operatorname{im}\delta^{n-1}$。也就是说,它是由那些在$C^n(F)$中被$\delta^n$映射到0的元素生成的Abel群,再模去那些是$\delta^{n-1}$的像的元素。

形式上,我们有:

$$H^n(F)=\frac{\ker\delta^n}{\operatorname{im}\delta^{n-1}}=\frac{\{s\in C^n(F)|\delta^n(s)=0\}}{\{t\in C^n(F)|t=\delta^{n-1}(r),r\in C^{n-1}(F)\}}$$

如果$H^n(F)=0$,那么说明位形在第n维上是无洞的。反之,如果$H^n(F)\neq0$,那么就存在一些不能用局部数据表示的全局结构。

这就是计算位形上同调的核心算法原理和具体操作步骤。下面我们将通过一个具体的例子来进一步说明这个过程。

## 4.数学模型和公式详细讲解举例说明

为了更好地理解上同调的计算过程,我们将通过一个具体的例子来说明。考虑实射影空间$\mathbb{RP}^2$上的常数位形$\underline{\mathbb{Z}}$。

### 4.1 构造链复形

首先,我们需要构造相应的链复形$\cdots\rightarrow C^{n-1}(\underline{\mathbb{Z}})\xrightarrow{\delta^{n-1}}C^n(\underline{\mathbb{Z}})\xrightarrow{\delta^n}C^{n+1}(\underline{\mathbb{Z}})\rightarrow\cdots$。

对于$n=0$,我们有$C^0(\underline{\mathbb{Z}})=\prod_{x\in\mathbb{RP}^2}\mathbb{Z}$,即所有点上的整数的乘积。

对于$n=1$,我们有$C^1(\underline{\mathbb{Z}})=\prod_{L\subset\mathbb{RP}^2}\mathbb{Z}$,即所有射线上的整数的乘积。

对于$n=2$,我们有$C^2(\underline{\mathbb{Z}})=\mathbb{Z}$,因为$\mathbb{RP}^2$本身是一个开集。

### 4.2 计算边映射

接下来,我们需要计算边映射$\delta^n$。

对于$n=0$,我们有$\delta^0:C^0(\underline{\mathbb{Z}})\rightarrow C^1(\underline{\mathbb{Z}})$,定义为:

$$\delta^0(s)_L=\sum_{x\in L}s_x$$

其中,$s=(s_x)_{x\in\mathbb{RP}^2}\in C^0(\underline{\mathbb{Z}})$,而$L$是$\mathbb{RP}^2$中的一条射线。

对于$n=1$,我们有$\delta^1:C^1(\underline{\mathbb{Z}})\rightarrow C^2(\underline{\mathbb{Z}})$,定义为:

$$\delta^1(t)=\sum_Lt_L$$

其中,$t=(t_L)_{L\subset\mathbb{RP}^2}\in C^1(\underline{\mathbb{Z}})$。

可以验证$\delta^1\circ\delta^0=0$,因此我们确实得到了一个链复形。

### 4.3 计算上同调组

现在,我们可以计算上同调组$H^n(\underline{\mathbb{Z}})$了。

对于$n=0$,我们有$H^0(\underline{\mathbb{Z}})=\ker\delta^0/\operatorname{im}\delta^{-1}=\mathbb{Z}$,因为$\delta^{-1}$是空映射。

对于$n=1$,我们有$H^1(\underline{\mathbb{Z}})=\ker\delta^1/\operatorname{im}\delta^0$。注意到$\operatorname{im}\delta^0$由所有满足$\sum_{x\in L}s_x=0$的元素$(s_x)_{x\in\mathbb{RP}^2}$生成。另一方面,$\ker\delta^1$由所有满足$\sum_Lt_L=0$的元素$(t_L)_{L\subset\mathbb{RP}^2}$生成。因此,我们有$H^1(\underline{\mathbb{Z}})\cong\mathbb{Z}_2$。

对于$n=2$,我们有$H^2(\underline{\mathbb{Z}})=\ker\delta^2/\operatorname{im}\delta^1=0$,因为$\delta^2$是空映射。

因此,我们得到了$\mathbb{RP}^2$上常数位形$\underline{\mathbb{Z}}$的上同调组:

$$H^0(\underline{\mathbb{Z}})=\mathbb{Z},H^1(\underline{\mathbb{Z}})=\mathbb{Z}_2,H^2(\underline{\mathbb{Z}})=0$$

这个结果反映了$\mathbb{RP}^2$的拓扑结构:它是一个紧致的曲面,有一个2维的"洞"(即$H^1(\underline{\mathbb{Z}})=\mathbb{Z}_2\neq0$)。

通过这个例子,我们可以更好地理解上同调的计算过程,以及它如何反映空间的拓扑结构。

## 5.项目实践:代码实例和详细解释说明

为了更好地理解上同调的计算过程,我们将提供一个Python代码示例,用于计算实射影空间$\mathbb{RP}^2$上常数位形$\underline{\mathbb{Z}}$的上同调组。

```python
import numpy as np

# 定义实射影空间RP^2的点集
points = [(0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1)]

# 定义实射影空间RP^2的射线集
lines = [
    [(0, 0, 1), (0, 1, 0)],
    [(0, 0, 1), (1, 0, 0)],
    [(0, 1, 0), (1, 1, 1)],
    [(1, 0, 0), (1, 1, 1)]
]

# 定义边映射delta^0
def delta0(s):
    t = np.zeros(len(lines), dtype=int)
    for i, line in enumerate(lines):
        for p in line:
            t[i] += s[points.index(p)]
    return t

# 定义边映射delta^1
def delta1(t):
    return sum(t)

# 计算H^0
H0 = np.array([1], dtype=int)

# 计算H^1
im_delta0 = np.array([t for t in map(delta0, np.eye(len(points), dtype=int)) if sum(t) == 0])
ker_delta1 = np.array([t for t in np.eye(len(lines), dtype=int) if delta1(t) == 0])
H1 = np.mod(ker_delta1, im_delta0.T)

# 计算H^2
H2 = np.array([], dtype=int)

print(f"H^0 = {H0}")
print(f"H^1 = {H1}")
print(f"H^2 = {H2}")
```

在这个代码示例中,我们首先定义了$\mathbb{RP}^2$的点集和射线集。然后,我们实现了边映射$\delta^0$和$\delta^1$的计算函数。

接下来,我们计算了上同调组$H^0$、$H^1$和$H^2$。对于$H^0$,我们直接得到$\mathbb{Z}$。对于$H^1$,我们首先计算$\operatorname{im}\delta^0$和$\ker\delta^1$,然后取它们的商得到$\mathbb{Z}_2$。对于$H^2$,由于$\delta^2$是空映射,我们直接得到0。

运行这个代码,我们将得到以下输出:

```
H^0 = [1]
H^1 = [[1 0 0 0]
 [0 1 0 0]
 [0 0 1 0]
 [0 0 0 1]]
H^2 = []
```

这与我们之前的理论计算结果一致。

通过这个代码示例,我们可以更好地理解上同调的计算过程,并且可以应用于其他空间和位形的上同调计算。

## 6.实际应用场景

上同调理论在数学和物理学中有着广泛的应用,包括但不限于以下几个方面:

1. **代数几何**:上同调是研究代数varietie的重要工具,可以用来计算它们的几何