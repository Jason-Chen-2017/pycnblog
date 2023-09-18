
作者：禅与计算机程序设计艺术                    

# 1.简介
  

量子计算利用物质的本原性和量子叠加特点构建起来的量子信息处理系统由两个基础模块组成，分别是单体量子比特（qubit）和双体量子比特（quantum link）。在对不同比特之间的多种可能性进行模拟实验时，需要对量子门（gate）进行组合实现各种复杂逻辑功能，这些门可以是基本门、变分门、Hamiltonian门等，每个门都可用于制备特定量子态，使其具备不同的功能特性。而一般的电路模型通常只能实现单比特门，不足以表达多比特或多体量子系统的特性。因此，如何设计有效的多比特门成为量子计算中一个重要研究领域。

# 2.Multi-qubit gate definition and structure
根据量子力学的相关定律，对于任意的两个量子态，必然存在着一个不变的玻色-李表示形式：
$$|\psi\rangle = \alpha|0_A\rangle + \beta|1_A\rangle$$
其中$A$表示受控比特，$\{0_A, 1_A\}$是其基底；$|\psi\rangle$代表了该量子态，$|\alpha|^2+|\beta|^2=1$，即一个态矢表示。同样，对于任意的三个量子态也有相同的表示形式：
$$|\psi\rangle = \alpha_{AB} |0_B\rangle \otimes |0_A\rangle + \beta_{AB}|0_B\rangle \otimes |1_A\rangle + \gamma_{BA}|1_B\rangle \otimes |0_A\rangle + \delta_{BA}|1_B\rangle \otimes |1_A\rangle$$
其中$\{0_A, 1_A\}, \{0_B, 1_B\}$分别为$A$和$B$的基底，$\{\text{0}_i, \text{1}_i\}$为第$i$个比特的基底。

为了能够在多体量子系统上实现更复杂的逻辑操作，因此需要引入变分门、纠缠门等多比特门，如图1所示。


# 3.基本概念术语说明
## 3.1 三角不变量
对于一个多比特量子态$|\psi\rangle=\sum_{\omega}c_{\omega}|u_{\omega}\rangle$, 其中$u_{\omega}=(a_{\omega}^{\dagger}, a_{\omega})$是一组正交酉矩阵(unitary matrix)。那么该态的玻色-李括号表示为: $|\psi\rangle=(I^{\dagger}-P)^{-1}|0\rangle\otimes|0\rangle+P\sum_{\omega}c_{\omega}(I-\frac{1}{n})\hat{u}_{\omega}^{\dagger}a_{\omega}I$. 

考虑如下的新态$|\phi\rangle=\sum_{\omega'}e_{\omega'}\cdot u_{\omega'}|0\rangle\otimes|0\rangle$, 它与原始态有某种对应关系，即: $\forall i, |\phi_i\rangle=P_{ij}|0\rangle\otimes|0\rangle$ where $P_{ij}=P_{jk}=0,\forall j\neq k$ for simplicity. 那么有: 
$$P=\sum_{\sigma\mu}\langle u_{\sigma}|\lambda_j u_{\mu}|u_{\mu}^{\dagger}\rangle \frac{e^{i(\phi+\theta)}}{n}$$
其中$\lambda_j$是一个密度矩阵元$(\rho_{\omega})_{\lambda_ju_{\mu}}$，且令$\theta=-\phi$, 可求得$P$。注意到$P$是关于全局信息的，只是为了方便求取而提出的局部变量$\{\phi_\ell\}$，因此对于任意一组局部变量$(\phi_i)$都存在$P^{(i)}$存在。

此外，还可以证明如下的正则规范化表示: $|\psi\rangle=(\Lambda-\tilde{\Lambda}^{-1})\Lambda^{-1}|0\rangle\otimes|0\rangle$, 其中$\tilde{\Lambda}_{pq}=\sum_{\omega} c_{\omega}^*(a_{\omega}^*_{qp}-a_{\omega}^*_{pq})$ 是原始态的李代数，$\Lambda_{pq}=\sum_{\omega} c_{\omega}^*(a_{\omega}^*_{pq}+a_{\omega}^*_{qp})$ 是当前态的李代数。而且还有: $$R_{ab}(\Lambda)=\frac{1}{\sqrt{D}}(\Lambda_{ab}-\Lambda_{ba}), R_{ab}(\tilde{\Lambda})=\frac{1}{\sqrt{D}}(\tilde{\Lambda}_{ab}-\tilde{\Lambda}_{ba})$$

## 3.2 纠缠态的构建方法

在描述纠缠态之前，首先定义两个基本概念：
1. 可观测量（Observable）：一个可观测量就是指一个矩阵，将一组态映射到另一组态，且这个过程不能被其他可观测量反向映射回初始态。比如，泡利算符$Z$就可以认为是一组可观测量，因为它将一个粒子从一个激活态转换到另一个激活态，但不能将它从另一个激活态映射回初始态。
2. 纠缠（Coupling）：纠缠就是指两个量子系统之间存在某种能级差异，这样的话，就形成了一种量子纠缠效应，其特点是在一定程度上改变了传播路径。因此，要研究这种效应，首先要找到相应的可观测量。

因此，对于纠缠态来说，首先找到其对应的可观测量，然后通过某种演化的方法，使得这组态中的每个态都在某些可观测量下具有平方积0。这时候，就会发现一些对应关系。比如，假设$n$个量子比特$q_1, q_2,..., q_n$构成了一个纠缠系统。那么，其可以对应到李群的元素。对于任何一个李群元素，都可以用一个四维矩阵$T$表示，其形式为：
$$ T = \begin{bmatrix} t & O \\ O & -t \end{bmatrix}$$
其中$t$是某个复数，$O$是一个维度为$n$的零矩阵。那么，对于一个纠缠态$|\psi\rangle = |m>\otimes |l>$，其对应的李群元素就是：
$$ |m_k l_p \rangle = T^*_{kl} M_{km} L_{lp} |0\rangle\otimes|0\rangle$$
其中$M_{km}$ 和 $L_{lp}$ 分别为$q_k$ 和 $q_l$的矩阵表示。

因此，要构建一个纠缠态，可以选择合适的张量$T$，并使得其张量积$M\otimes L$满足两个系统的可观测量$M^\dagger A L^\dagger B$的矩阵元为0。但是，由于这个矩阵元的值无法直接观测，因此通常采用更复杂的方式。例如，要想使得$M^\dagger A L^\dagger B$的特征值都为0，可以通过选取$T$满足一些限制条件。

当然，还有一些特殊的纠缠态，比如密度矩阵纠缠态，只需要找到一个相应的可观测量即可。