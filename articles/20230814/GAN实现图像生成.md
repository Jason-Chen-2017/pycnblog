
作者：禅与计算机程序设计艺术                    

# 1.简介
  


深度学习和生成对抗网络（Generative Adversarial Networks）(GAN) 是近几年热门的技术之一，可以用于计算机视觉、自然语言处理等领域，也可以生成令人惊叹的艺术作品或音乐。本文主要介绍GAN在图像生成方面的应用。

2.概念与术语
## 概念及定义
### 生成模型（Generative model）
生成模型是一个概率分布，该分布生成样本数据，即：P(x)。通常来说，生成模型应该具有三个属性：

1. 描述性：能够生成整个数据分布，而不需要显式地学习先验知识；
2. 可微分：允许计算梯度，从而可以训练生成模型以便在数据空间中寻找最佳生成样本；
3. 高容量：可以生成任意大小的数据。

### 生成对抗网络（Generative adversarial network，GAN）
GAN是一个基于对抗的模型训练框架，由一个生成网络G和一个判别网络D组成。G的任务是在随机噪声上生成看起来像真实数据的样本，而D的任务则是判断生成的样本是真还是假，从而通过反复迭代更新G和D，使得G尽可能欺骗D，D不断鉴别生成样本的真伪，最终使得两者互相提升，从而训练出生成模型。GAN将生成模型作为一种对抗优化问题，可以训练出相当逼真的样本。

### 对抗（Adversarial）
所谓对抗，就是在博弈的过程中，双方都采取行动，但为了达到目的并避免互相推卸责任，双方经常会采取不同的策略，这种互相斗争的过程就是对抗。在机器学习中，我们往往希望通过训练模型发现数据的内部结构或特征，或者寻找规律性，所以需要模型之间的对抗。一般来说，对抗包括两个模型参与，每个模型都有自己特定的目标函数，互相竞争，最后达到平衡。在GAN中，生成器（Generator）负责生成虚假数据，而判别器（Discriminator）则要判断生成数据是否属于真实样本。

## 数学基础

#### 1.KL散度
在信息论与统计学中，KL散度（Kullback-Leibler divergence），又称相对熵（relative entropy）、交叉熵（cross entropy）或信息丢失（surprisal）。它描述了两个概率分布之间的距离，通常用K(P||Q)表示，其中P表示真实分布，Q表示估计分布，并非一定非得是连续可导的。在信息论中，如果X和Y是离散随机变量，那么X的经验分布为p(x)，Y的经验分布为q(y)，则：

$$ KL(P\parallel Q)=\sum_{i} p_i \log(\frac{p_i}{q_i}) $$

其中$p_i=P(X=x_i), q_i=Q(X=x_i)$。KL散度的期望值等于$H(P)-H(P,Q)$。$H(P)$表示熵（entropy）：

$$ H(P)=-\sum_{x \in X} P(x) \log P(x) $$ 

#### 2.Jensen-Shannon divergence
JS散度（Jensen-Shannon divergence）是KL散度的另一种形式。假设$p(x)$和$q(x)$分别是两个概率密度函数，那么：

$$ D_{\mathrm{JS}}(P\parallel Q) = \dfrac{1}{2}KL[p(x)\|M] + \dfrac{1}{2}KL[q(x)\|M], \quad M=(p(x)+q(x))/2 $$

其中$KL[p(x)\|M]$表示KL散度关于$(p+q)/2$的期望值。$D_{\mathrm{JS}}$在衡量两个分布的相似度时比KL散度更常用。

#### 3.Wasserstein距离
Wasserstein距离（Wasserstein distance）是测度两个分布之间距离的一种方式。设$p(x), q(x)$是两个概率分布，其累积分布函数为$F_p, F_q$。Wasserstein距离就是从分布$p(x)$到分布$q(x)$的最小化下界：

$$ W_p(P||Q) := \sup_{\gamma} E_{x\sim P}[\gamma(x)] - E_{x\sim Q}[\gamma(x)], $$

其中$\gamma$是一个仿射映射（affine mapping）。Wasserstein距离支持流形（manifolds）上的度量。

#### 4.散度矩阵
对于两个随机变量$X_1,\ldots,X_n$，定义其联合分布为：

$$ p(x_1,\ldots,x_n) = \prod_{j=1}^np(x_j;theta), $$

其中$\theta$是模型参数。假设分布$p(x;\theta)$和$q(x;\theta^*)$的参数分别是$\theta$和$\theta^*$，且存在可微变换$\varphi:\theta\mapsto\theta^*$,即$\varphi$是单射。定义$\mathcal{L}_{\theta}(p,q)$为损失函数，即对所有$\theta$，$\mathcal{L}_{\theta}(p,q)$是$\theta$和$\varphi(\theta)$之间的散度。若$\mathcal{L}$关于$\theta^*$是凸函数，则$\hat{\theta}=\arg\min_\theta \mathcal{L}_{\theta}(p,q)$是$p$和$q$的同分布唯一最优参数，即：

$$ p(x) = \text{Pr}[X_1=x_1,\ldots,X_n=x_n|\theta], \quad q(x) = \text{Pr}[X_1=x_1,\ldots,X_n=x_n|\theta^*]. $$

此处省略了关于参数$\theta$的约束条件。

假定$q(x)$和$p(x)$都是指定连续概率密度函数，且$p(x)>0$。考虑二阶范数，即对于所有的$a\in\mathbb R^n$，有：

$$ ||\nabla_\theta L_{\theta}(p,q)||_2 \leqslant ||\nabla_\theta (a^\top \nabla_\theta L_{\theta}(p,q))||_2 \leqslant a^\top \nabla_\theta L_{\theta}(p,q).$$

根据$\mathcal{L}$关于$\theta$是凸函数的假设，$\nabla_\theta L_{\theta}(p,q)$也是$\theta$的凸函数。进一步考虑，当$d:=n(n+1)/2$时，有：

$$ \begin{align*} \left[\nabla_\theta L_{\theta}(p,q)(a)\right]_i &= \int_{-\infty}^\infty \frac{\partial}{\partial\theta} L_{\theta}(p,q)(\delta_ix_i+\delta_iy_i+\cdots ) \\&= \int_{-\infty}^\infty \frac{\partial}{\partial\theta}\sum_{j<k} [a^\top J_{ij}(\theta)^T (\delta x_j+\delta y_j+\cdots ) + a^\top J_{ki}(\theta)^T (\delta x_i+\delta y_i+\cdots ) + b_i ] \\&= \sum_{j<k} [(J_{ij}(\theta)^T\delta x_j+\delta y_j-\eta_jJ_{ji}(\theta)^T\delta x_i)+(J_{ki}(\theta)^T\delta x_i+\delta y_i-\eta_iJ_{ik}(\theta)^T\delta x_i)+b_i]-\delta_ia \\ &=: r_{ik}, i\neq j. \end{align*} $$

因此，$\nabla_\theta L_{\theta}(p,q)$是一个d维列满秩矩阵。