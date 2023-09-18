
作者：禅与计算机程序设计艺术                    

# 1.简介
  

公钥加密算法中最著名的是RSA算法，而DSS（Digital Signature Standard）则是公钥密码算法的代表。DSS基于椭圆曲线，具有公钥匿名性、数字签名的不可伪造、抗篡改性等优点。但是由于DSS依赖椭圆曲ulsession难题，导致计算效率不够高。因此，许多研究人员提出了基于DSS的密钥交换方案，来解决椭圆曲线上椭圆曲面积分困难的问题。其中一种方案就是Minimax Assumption，即假设椭圆曲线在有限域F上的最小作用素数为p。这篇论文从密钥交换的角度，分析了参数选择技术对DSS的影响，并提出了一种新的参数选择策略——Round-Optimal Parameter Selection。

2. Background Introduction and Basic Concepts
2.1 Public-Key Cryptography
公钥密码的基本概念主要有两类：密码学基础及应用层。密码学基础包括：密码编码方式、代数系统、概率模型、椭圆曲线加密、离散对数、同态加密、椭圆曲面积分等。应用层包括：密钥分配、证书管理、认证协议、安全通信、电子货币等。

2.2 DSS (Digital Signature Standard)
DSS是一个公钥密码标准，其工作原理如下：
1. 选择一个椭圆曲线 E over F 椭圆曲面，其中E(F)为多项式形式，且 p 为一个质数。
2. 对消息 M 进行 Hash运算得到消息摘要 h。
3. 随机选择私钥 x ，并计算公钥 y = [x]G 。
4. 用公钥 y 对 h 做签名，得到签名值 r 和 s。其中r为点乘结果G.y 。
5. 如果要验证签名是否正确，需要用公钥y 再次计算出 r 的值，然后比较它与签名中的 r 是否一致。

如果我们假设椭圆曲线 E 在 F 上有限域上的最小作用素数为 p ，则可利用 LLL 算法将 E 拆分成两个部分。由此，可以得到：


```math
\begin{equation*}
    E(F)/E'(F)\cong \mathbb{Z}/(p^{2}-9)(p^2+1)(p^2-16) \oplus \mathbb{Z}/(p^{2}-4)\times F[t]/(\sqrt{-7})
\end{equation*}
```

其中E'为另一个共轭参数。

2.3 Minimax Assumption
Minimax Assumption（MA）是指假定椭圆曲线在有限域 F 上最小作用素数为 p。它的含义是为了计算椭圆曲面的参数，可以忽略掉p对椭圆曲线的影响。MA能够带来一些性能提升，因为对于某些选取的参数而言，它可以使得椭圆曲面积分计算的代价大大降低。另外，MA可以帮助研究人员避免选择一些复杂的椭圆曲面，从而简化计算。然而，MA也存在着一些局限性。例如，MA限制了椭圆曲面的选择范围，不能完全适应实际需求；在生成公钥/私钥对时，仍然需要选择p作为约束条件，但这一步可以在密钥交换阶段完成；椭圆曲面积分可能受到不同攻击的影响，尤其是在F不是伽罗华域的情况下。综合来说，MA并没有带来实质性的性能提升，而且也引入了一些复杂的技术问题。

2.4 Round-Optimal Parameter Selection
Round-Optimal Parameter Selection (ROPS) 是作者提出的一种新颖的密钥交换方案。ROPS可以自动地选择密钥交换过程中使用的椭圆曲线和相邻群的参数，从而保证安全性、隐私性、计算效率和参数空间的有效利用。其具体工作流程如下：

1. 生成密钥对（p、q）。p 是质数，q 是 2*p + 1 中的奇数。
2. 根据 p 和 q 选择椭圆曲线 E 和相邻群 G'，其中 E: F -> E(F), G': E(F) -> F' 。这里，G'是 E 在点 G 的共轭群。
3. 根据公钥发送者私钥 x 和接收者公钥 y，构造共享秘钥 k = H(xG') ^ (yG) 。
4. 加密：消息 m 在 E(F) 上映射为消息点 M = em ，接收方用共享秘钥 k 解密得到明文 m 。
5. 签名：消息摘要 h 在 E(F) 上映射为点 r=eh 。接收方用私钥 x 对 h 做签名，得到签名值 s=(k^(-1))h 。
6. 验证签名：接收方用公钥 y 验证签名，首先计算出签名 r'=(yG)^(-1)rh=((G)^(-1)yh)^(-1)h^eG^er=k^er 。

如此一来，就可以避免复杂的椭圆曲面积分计算，并且可以自动地优化椭圆曲面参数。

3. Core Algorithm and Operational Details
3.1 Generate Elliptic Curve Parameters
首先，选择 p 和 q 为质数。然后，根据 p 和 q 生成椭圆曲线 E over F ，并求解 F' 的 generators。具体方法是选取一组 points P∈E(F)，找到一条包含P的直线L，则有：

```math
\begin{equation*}
\{l(t)|t\in F,|l(t)-P|=|n*\beta|\}=\{m*\alpha+\gamma, |m|\equiv -1\mod n\}\cap \{\gamma,\beta\notin F\}\\
\end{equation*}
``` 

其中 $\alpha$ 和 $\beta$ 分别是 F 的 generator， $n=\frac{p+1}{2}$，$\gamma$ 是关于 F 的任意 point 。为了求解以上方程，只需把 L 的 t 参数化为坐标。当 t≠0 时，方程的解是 $\gamma$；当 t=0 时，方程无解。因此，可以求出 $\alpha$ 和 $\beta$ 后，可求得 F' 的 generators $g_1=\beta$, $g_2=-\alpha$.

之后，生成相邻群 G'，即 G'=[g_1,g_2]G ，其阶为 q+p，定义为 G'=c*[g_1,-g_2]+d*[g_2,-g_1], c, d∈{0,1}，其中有：

```math
\begin{equation*}
    \forall P, Q \in E(F),\\
    \text{if }[Q]=c*P+d*[g_1,-g_2]\\
    \Rightarrow [(P-Q)/(c^2*(d+1))]\bmod {(d+1)}\\
    =[(PQ-cP-dqG_1)/(cd+1)]\bmod {(d+1)}\quad (\because\frac{(PQ-cP-dqG_1)^2}{cd}=1)\\
    =([PQ-(dp+2)G_1]/(cd+1))\bmod {(d+1)}\quad (\because\frac{(PQ-cP-dqG_1)^2}{cd+1}=0)\\
    =(M_{qd})G_{\frac{qp}{2}}\quad (\because\frac{(PQ-cP-dqG_1)^2}{cd+1}=\frac{MP_1^2+MQ_1^2}{C}=\frac{(dp+2)G_1^2}{cd+1})\quad(\because MP_1=MQ_1)
\end{equation*}
``` 

其中 C 是常数，M 是斜率矩阵。

3.2 Compute Shared Secret
计算共享秘钥时，先随机选择私钥 x 和公钥 y，再通过以下方式计算：

```math
\begin{equation*}
    k = H(xG')^{-1}(yG)=H(x^{-1}G',y^{-1}G)\\ 
    &=H(x^{-1}G'+y^{-1}G)\\ 
    &=H(K^{-1}[g_1,g_2]G)\\ 
    &=H(K^{-1}kg_1)\cdot H(K^{-1}kg_2)\\ 
    &=k_1\cdot k_2
\end{equation*}
\end{equation*}
``` 

其中 K=[x^{-1},y^{-1}]G。

3.3 Encryption and Decryption
加密和解密都可以按照以下方式进行：

```math
\begin{equation*}
    \begin{array}{ll}
        EM = em & \\ 
        M = [\lfloor em/(q+p)+1\rfloor](q+p)-em & (\because\lfloor a+b\rfloor=\lfloor a\rfloor\times\lfloor b\rfloor+\lfloor ab\rfloor) \\ 
        M = [(em+(q+p))/\ell][\ell-1]+(em/\ell)[\ell],& (\because\text{gcd}(\ell,pq+1)=1) \\  
        \text{where }\ell=\frac{|EM|+pq+p}{\ell}& (\because\ell,pq+1=\text{gcd}(pq+1,2pq))
    \end{array}
\end{equation*}
``` 

注意，这个过程是模 q+p 的。

3.4 Signing and Verification
签名和验证分别在公钥和私钥的位置上进行。公钥位置上验证签名，私钥位置上签名消息。具体过程如下：

在公钥位置验证签名：

```math
\begin{equation*}
    r = [y]G = g_1^\lambda \cdot H(K^{-1}kg_1) \cdot (zG)^\beta\quad (\lambda,\beta\in F)\\ 
    &= g_1^\lambda \cdot H(K^{-1}kg_1)\quad (\because zG=(z[g_1,-g_2])/c^2, z\in F)\\ 
    &= g_1^\lambda \cdot H(kg_1)\quad (\because \text{gcd}(p,q-p)=1)\\ 
    &= g_1^\lambda \cdot (\alpha^{\lambda}+\delta_{\lambda,p})\\ 
    &= r_\lambda
\end{equation*}
\end{equation*}
``` 

其中 $\lambda = \frac{s}{(q+1)/2}$ ， $\delta_{\lambda,p}=\begin{cases}
    1,&\lambda=p\\ 
    0,&\text{otherwise}
  \end{cases}$ 。

在私钥位置签名消息：

```math
\begin{align*}
    \boxed{em &= [s]\cdot[\lfloor em/(q+p)+1\rfloor](q+p)-em\quad (\text{if } em<q+p)\\
             &= [s]\cdot[(em+(q+p))/\ell][\ell-1]+(em/\ell)[\ell]\quad (\text{otherwise})}\\\\
    \boxed{h &= [s]G\quad}
\end{align*}
``` 

4. Conclusion
本文介绍了 Minimax Assumption 的概念和相关背景。然后，它提出了一个新颖的密钥交换方案 ROPS 来处理椭圆曲线参数选择问题，并给出了该方案的具体实现算法。ROPS 可以自动地选择椭圆曲线和相邻群的参数，从而保证安全性、隐私性、计算效率和参数空间的有效利用。最后，本文讨论了 ROPS 对传统 DSS 签名的影响以及其未来的发展方向。

除此之外，作者还总结了当前椭圆曲面参数选择技术存在的局限性，如：缺少准确评估已知椭圆曲面是否满足 MA 要求；缺少经过充分研究的优化技术；椭圆曲面积分存在很多不确定性；未来仍有很大的发展空间。