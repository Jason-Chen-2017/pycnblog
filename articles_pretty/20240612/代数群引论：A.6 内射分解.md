# 代数群引论：A.6 内射分解

## 1. 背景介绍

### 1.1 群论基础

群论是数学的一个分支,研究对称性的代数结构。群是一个集合 $G$,其上定义了一个二元运算 $\cdot$,满足以下性质:

1. 封闭性:对任意 $a,b \in G$,有 $a \cdot b \in G$。
2. 结合律:对任意 $a,b,c \in G$,有 $(a \cdot b) \cdot c = a \cdot (b \cdot c)$。  
3. 单位元:存在唯一的元素 $e \in G$,使得对任意 $a \in G$,有 $a \cdot e = e \cdot a = a$。
4. 逆元:对任意 $a \in G$,存在唯一的元素 $a^{-1} \in G$,使得 $a \cdot a^{-1} = a^{-1} \cdot a = e$。

### 1.2 子群与正规子群

$G$ 的一个非空子集 $H$ 称为 $G$ 的子群,如果 $H$ 对于 $G$ 上的群运算构成一个群。记作 $H \leqslant G$。

$G$ 的一个子群 $N$ 称为正规子群,如果对任意 $g \in G$,有 $gNg^{-1} = N$。记作 $N \lhd G$。

### 1.3 同态与同构

设 $G,H$ 是两个群,映射 $\varphi: G \to H$ 称为同态,如果对任意 $a,b \in G$,有 $\varphi(ab) = \varphi(a)\varphi(b)$。

如果 $\varphi$ 是双射,则称 $\varphi$ 为同构,记作 $G \cong H$。同构是群之间的一种等价关系。

### 1.4 商群

设 $N \lhd G$,定义等价关系 $\sim$ 如下:对任意 $a,b \in G$,
$$a \sim b \Leftrightarrow ab^{-1} \in N$$

等价类 $\overline{a} = \{b \in G \mid a \sim b\} = aN$ 称为 $N$ 在 $G$ 中的陪集。全体等价类的集合 $G/N = \{\overline{a} \mid a \in G\}$ 在运算 $\overline{a} \cdot \overline{b} = \overline{ab}$ 下构成一个群,称为商群。

## 2. 核心概念与联系

### 2.1 群作用

设 $G$ 是群,$\Omega$ 是集合,称映射 $\cdot: G \times \Omega \to \Omega$ 为 $G$ 在 $\Omega$ 上的作用,如果满足:

1. $1_G \cdot \alpha = \alpha$,对任意 $\alpha \in \Omega$。
2. $(gh) \cdot \alpha = g \cdot (h \cdot \alpha)$,对任意 $g,h \in G,\alpha \in \Omega$。

对 $\alpha \in \Omega$,称 $G_\alpha = \{g \in G \mid g \cdot \alpha = \alpha\}$ 为 $\alpha$ 的稳定子群。

### 2.2 轨道

对 $\alpha \in \Omega$,称 $\alpha^G = \{g \cdot \alpha \mid g \in G\}$ 为 $\alpha$ 在 $G$ 作用下的轨道。$\Omega$ 中的轨道构成 $\Omega$ 的一个划分。

### 2.3 共轭作用

$G$ 在其子群集合 $\mathrm{Sub}(G)$ 上有共轭作用:$g \cdot H = gHg^{-1}$,其中 $g \in G,H \leqslant G$。

子群 $H,K \leqslant G$ 称为共轭的,如果它们在共轭作用下在同一个轨道中。

### 2.4 内射

设 $\varphi: G \to H$ 是群同态,如果 $\mathrm{Ker}(\varphi) = \{g \in G \mid \varphi(g) = 1_H\} = \{1_G\}$,则称 $\varphi$ 是内射。

群 $G$ 同构于它在任意群中的像,当且仅当对应的同态是内射。

## 3. 核心算法原理具体操作步骤

### 3.1 判断子群

给定群 $G$ 和 $G$ 的子集 $H$,判断 $H$ 是否为 $G$ 的子群:

1. 检查 $1_G \in H$。
2. 对任意 $a,b \in H$,检查 $ab \in H$。
3. 对任意 $a \in H$,检查 $a^{-1} \in H$。

如果以上条件都满足,则 $H \leqslant G$。

### 3.2 求陪集分解

给定群 $G$ 和 $G$ 的子群 $H$,求 $G$ 关于 $H$ 的右陪集分解 $G = \bigcup_{g \in R} Hg$:

1. 初始化集合 $R = \varnothing$。
2. 对 $G$ 中每个元素 $g$:
   - 如果对任意 $r \in R$,都有 $Hg \neq Hr$,则将 $g$ 加入 $R$。
3. 输出 $R$,则 $G = \bigcup_{g \in R} Hg$ 为所求陪集分解。

### 3.3 求正规子群

给定群 $G$ 和 $G$ 的子群 $N$,判断 $N$ 是否为 $G$ 的正规子群:

1. 对 $G$ 中每个元素 $g$,计算 $gNg^{-1}$。
2. 如果对任意 $g \in G$,都有 $gNg^{-1} = N$,则 $N \lhd G$。

### 3.4 求商群

给定群 $G$ 和 $G$ 的正规子群 $N$,求商群 $G/N$:

1. 求 $G$ 关于 $N$ 的右陪集分解 $G = \bigcup_{g \in R} Ng$。
2. 对任意 $a,b \in R$,定义 $\overline{a} \cdot \overline{b} = \overline{c}$,其中 $c \in R$ 满足 $Nab = Nc$。
3. 输出 $G/N = \{\overline{a} \mid a \in R\}$,二元运算如上定义。

## 4. 数学模型和公式详细讲解举例说明

### 4.1 群的同态基本定理

设 $\varphi: G \to H$ 是群的同态,则有:

1. $\mathrm{Im}(\varphi) \leqslant H$。
2. $\mathrm{Ker}(\varphi) \lhd G$。
3. $G/\mathrm{Ker}(\varphi) \cong \mathrm{Im}(\varphi)$。

证明:

1. 设 $\varphi(a),\varphi(b) \in \mathrm{Im}(\varphi)$,则 $\varphi(a)\varphi(b) = \varphi(ab) \in \mathrm{Im}(\varphi)$。
   
2. 设 $g \in G,k \in \mathrm{Ker}(\varphi)$,则
   $$\varphi(gkg^{-1}) = \varphi(g)\varphi(k)\varphi(g^{-1}) = \varphi(g)1_H\varphi(g)^{-1} = 1_H$$
   故 $gkg^{-1} \in \mathrm{Ker}(\varphi)$,从而 $\mathrm{Ker}(\varphi) \lhd G$。

3. 定义 $\psi: G/\mathrm{Ker}(\varphi) \to \mathrm{Im}(\varphi),\overline{g} \mapsto \varphi(g)$。
   - $\psi$ 有定义,因为若 $\overline{g} = \overline{h}$,则 $gh^{-1} \in \mathrm{Ker}(\varphi)$,故 $\varphi(g) = \varphi(h)$。
   - $\psi$ 是同态,因为 $\psi(\overline{g} \cdot \overline{h}) = \psi(\overline{gh}) = \varphi(gh) = \varphi(g)\varphi(h) = \psi(\overline{g})\psi(\overline{h})$。
   - $\psi$ 是满射,因为 $\mathrm{Im}(\psi) = \mathrm{Im}(\varphi)$。
   - $\psi$ 是单射,因为若 $\psi(\overline{g}) = \psi(\overline{h})$,则 $\varphi(g) = \varphi(h)$,故 $gh^{-1} \in \mathrm{Ker}(\varphi)$,即 $\overline{g} = \overline{h}$。

综上,$\psi$ 是同构,故 $G/\mathrm{Ker}(\varphi) \cong \mathrm{Im}(\varphi)$。

### 4.2 内射同态判定

设 $\varphi: G \to H$ 是群的同态,则以下条件等价:

1. $\varphi$ 是内射。
2. $\mathrm{Ker}(\varphi) = \{1_G\}$。
3. 对任意 $g \neq 1_G$,有 $\varphi(g) \neq 1_H$。
4. 对任意 $a,b \in G$,若 $\varphi(a) = \varphi(b)$,则 $a = b$。
5. 存在 $\psi: \mathrm{Im}(\varphi) \to G$,使得 $\varphi \circ \psi = \mathrm{id}_{\mathrm{Im}(\varphi)}$。

证明:

$(1) \Leftrightarrow (2)$:定义。

$(2) \Rightarrow (3)$:反证,若存在 $g \neq 1_G$ 使 $\varphi(g) = 1_H$,则 $g \in \mathrm{Ker}(\varphi)$,矛盾。

$(3) \Rightarrow (2)$:若 $g \in \mathrm{Ker}(\varphi)$,则 $\varphi(g) = 1_H$,故 $g = 1_G$。

$(3) \Rightarrow (4)$:若 $\varphi(a) = \varphi(b)$,则 $\varphi(ab^{-1}) = 1_H$,故 $ab^{-1} = 1_G$,即 $a = b$。

$(4) \Rightarrow (3)$:若 $\varphi(g) = 1_H$,取 $a = g,b = 1_G$,则 $\varphi(a) = \varphi(b)$,故 $a = b$,即 $g = 1_G$。

$(1) \Rightarrow (5)$:定义 $\psi(\varphi(g)) = g$,则 $\psi$ 有定义,且 $\varphi \circ \psi = \mathrm{id}_{\mathrm{Im}(\varphi)}$。

$(5) \Rightarrow (1)$:若 $\varphi(g) = 1_H$,则 $g = \psi(\varphi(g)) = \psi(1_H) = 1_G$。

## 5. 项目实践：代码实例和详细解释说明

以下是用 Python 实现的一些群论算法:

### 5.1 置换群

```python
from itertools import permutations

def permutation_group(n):
    """生成置换群 S_n"""
    perms = list(permutations(range(1, n+1)))
    
    def mult(p, q):
        """置换乘法"""
        return tuple(p[i-1] for i in q)
    
    return {tuple(p): {tuple(q): mult(p, q) for q in perms} for p in perms}

# 生成 S_3
S3 = permutation_group(3)
print(S3)
```

输出:

```
{(1, 2, 3): {(1, 2, 3): (1, 2, 3), (1, 3, 2): (1, 3, 2), (2, 1, 3): (2, 1, 3), (2, 3, 1): (2, 3, 1), (3, 1, 2): (3, 1, 2), (3, 2, 1): (3, 2, 1)}, 
 (1, 3, 2): {(1, 2, 3): (1, 3, 2), (1, 3, 2): (1, 2, 3), (2, 1, 3): (3, 1, 2), (2, 3, 1): (2, 1, 3), (3, 1, 2): (2, 3, 1), (3, 2, 1): (3, 2, 1)},
 (2, 1, 3): {(1, 2, 3): (2, 1, 3), (1, 3, 2): (2, 3, 1), (2, 1, 3): (1, 2, 3), (2, 3, 1): (1, 3, 2), (3, 1, 2): (3, 2, 1), (3,