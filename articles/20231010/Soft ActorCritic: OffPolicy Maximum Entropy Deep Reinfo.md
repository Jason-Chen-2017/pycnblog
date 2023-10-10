
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


Soft actor critic (SAC) 是一种深度强化学习（Deep reinforcement learning, DRL）方法。它通过最大熵原理，在保证可靠性的同时，提高样本效率。此外，作者提出了一种“软”策略来促进探索。这种策略会减少策略网络预测失误导致的行为动作不稳定，从而提高探索能力。SAC 不像其他基于值函数的方法，只能基于一步前向传播进行学习更新，因此需要采取一些额外的技巧来解决样本效率的问题。
本文主要介绍的是Soft Actor Critic(SAC)的论文原创工作。本文发表于ICML 2019年第227-234页。
# 2.核心概念与联系
## 2.1 Policy Gradient Methods
首先我们回顾一下强化学习中的policy gradient methods。一般情况下，agent在决策过程中要做出动作的选择依赖于一个策略分布$\pi_{\theta}(a|s)$。策略梯度法则是一种优化目标，用于求解策略参数$\theta$，使得期望累积奖励大于等于能够获得更多新知识或信息的损失值的期望，即：
$$J(\theta)=\mathbb{E}_{\tau}\left[R(\tau)\right]=-\int_{t=0}^{\infty}\mathrm{d}\tau R(\tau)\log\pi_{\theta}(a_t|s_t),$$
其中，$\tau$代表策略执行轨迹，即从初始状态到终止状态的一系列行为动作及对应的观察。

为了优化策略参数$\theta$, policy gradient 方法使用梯度上升算法或随机梯度下降算法。具体的，通过采样一批策略执行轨迹$\{\tau^i\}_{i=1}^{n}$（$i=1,\cdots, n$），计算每个$\theta$时刻的梯度方向，并按照梯度方向更新策略参数，直到收敛。例如，如果采用随机梯度下降算法，每一次更新只用到了一个策略执行轨迹。所以，样本效率很低。

## 2.2 Value Function Approximation

另一类RL算法是基于值函数的方法。与策略梯度方法不同，它不是直接对策略参数$\theta$进行优化，而是利用一个值函数$V_{\phi}(s)$来表示状态价值，也称为预测值（prediction value）。值函数$V_{\phi}(s)$可以近似表示状态价值，其表达式为：
$$V_{\phi}(s)=\underset{\pi}{max}\ \mathbb{E}_{\tau\sim\pi}[R(\tau)].$$
值函数近似值函数$V_{\phi}(s)$的好坏依赖于两方面：1、如何估计值函数；2、如何选择近似方法。对于问题1，常用的方法有：离散程度较低（如使用线性方程组）的特征工程方法（如利用图像特征或文本特征）；离散程度较高的神经网络方法（如卷积神经网络）。对于问题2，通常采用基于梯度的方法（如TD方法、MC方法等），或者采用基于自动编码器（AE）的方法。

在SAC中，值函数$V_{\phi}(s)$采用基于神经网络的函数近似方法（NN function approximation method）。NN方法优点在于能够对非凸优化问题进行有效的求解，并且可以采用多层结构。

## 2.3 Reparameterization Trick and Stochastic Actor
在确定策略后，使用策略梯度法来更新策略参数$\theta$，该方法假设策略分布是确定的，即：
$$a=\mu_{\theta}(s)+\epsilon,$$
其中，$\epsilon\sim N(0,\sigma_{\theta})$。但是实际情况下，策略分布往往不是确定的，因为策略是一个黑盒子，无法完全控制。因此，我们需要一个参数化的策略分布，使得能够计算策略分布的期望和方差，进而构造策略梯度。

Reparameterization trick就是将策略分布重写成一个可导的参数化的形式，这样就可以计算它的期望和方差。具体地，令：
$$\rho(z;\mu_\theta,\sigma_\theta)=\frac{1}{\sqrt{2\pi}\sigma_\theta}e^{-\frac{(z-\mu_\theta)^2}{2\sigma_\theta^2}},$$
再令：
$$\theta'=(\mu_\theta,\sigma_\theta).$$
那么，策略分布$\pi_{\theta'}(a|s)=\rho(a;f_{\theta'},b_{\theta'})$就满足reparamterization trick的要求。通过重写这个分布，可以对它进行微分和求期望和方差，从而实现策略梯度的更新。

Stochastic Actor的另一个特点是引入噪声项，使策略变得不确定，以达到探索与执行之间的平衡。具体来说，给策略分布增加一个正态分布噪声项，即：
$$a=\mu_{\theta}(s)+N(0,1)*\sigma_{\theta},$$
其中，$\epsilon\sim N(0,1)$。策略分布变为：
$$\pi_{\theta}(a|s)=\rho(a;f_{\theta'},b_{\theta'}),$$
其中，$(f_{\theta'},b_{\theta'})=\theta'$。

## 2.4 Deterministic Policy Gradient Theorem

为了能够对策略参数进行更新，SAC提出了Deterministic Policy Gradient Theorem（DPGTheorem）。该定理说明了如何利用策略梯度以及证明其收敛性。

DPGTheorem由两步组成：1、最大熵原理；2、策略导数与Q-learning的关系。

### MaxEnt原理
最大熵原理表明，在强化学习中，一个好的策略应该能有效探索所有可能的动作。为此，我们希望找到一个具有最大熵分布的策略，即：
$$\pi_{\theta}(a|s)=\frac{exp(\mu_{\theta}(s)^\top f_{\theta}(s))}{\sum_{a'}exp(\mu_{\theta}(s)^\top f_{\theta}(s))}f_{\theta}(s).$$
其中，$\mu_{\theta}(s)$为参数化的策略分布的均值，$f_{\theta}(s)$为状态相关特征，且有：
$$\pi_{\theta}(a|s) \propto exp(\mu_{\theta}(s)^\top f_{\theta}(s)).$$
因此，我们最大化动作的概率密度乘以动作对应的值（即特征）之和。

MaxEnt原理告诉我们，一个好的策略应当充分利用所有可能的样本。通过最大化动作概率与对应的奖赏之和的乘积，SAC使用最大熵原理来训练策略。

### 策略导数与Q-learning的关系
策略梯度定理说明了如何利用策略梯度更新策略参数。同时，它还可以推广到Q-learning算法。Q-learning是一种在线学习的Off-policy方法。它的基本想法是使用当前策略去评估当前动作的价值，并根据这个评估结果去更新策略。对于一个策略$\pi_{\theta}$，在状态$s$下采取动作$a$，得到奖励$r$以及环境转移到状态$s'$。如果采用Q-learning，即将Q-值改写成动作值函数：
$$Q_{\pi_{\theta}}(s,a)=\mathbb{E}_{\pi_{\theta}}\left[\left(r+\gamma V_{\pi_{\theta'}}(s')\right)\right],$$
其中，$V_{\pi_{\theta'}}(s')$表示基于目标策略$\pi_{\theta'}$的值函数。通过将目标值函数替换为期望值函数，就可以导出策略梯度定理。

给定策略$\pi_{\theta}(a|s)$和一系列行为轨迹$\{\tau\}$, 策略梯度定理表明：
$$\nabla_{\theta} J(\theta)=\sum_{i=1}^{n}\nabla_{\theta} \mathcal{L}(\theta; \tau^{(i)})=\sum_{i=1}^{n}\nabla_{\theta} \sum_{t=0}^T r(s_t,a_t|\theta) \bigg(Q_{\theta}(s_t,a_t)-\alpha\left(\prod_{j=t+1}^T \pi_{\theta}(a_j|s_j)\right)^{\frac{-1}{T-t+1}}\nabla_{\theta} log\pi_{\theta}(a_t|s_t)\bigg),$$
其中，$\mathcal{L}(\theta;\tau)$代表从策略$\theta$生成的轨迹$\tau$的损失。由此可以看到，DPG的损失函数依赖于当前的状态价值函数$Q_{\theta}(s,a)$。而值函数$Q_{\theta}(s,a)$又依赖于当前策略。因此，我们通过更新值函数$Q_{\theta}(s,a)$来达到策略更新目的。