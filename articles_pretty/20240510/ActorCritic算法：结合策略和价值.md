# *Actor-Critic算法：结合策略和价值

作者：禅与计算机程序设计艺术

## 1.背景介绍
   
### 1.1 强化学习概述
   
#### 1.1.1 强化学习的定义与特点
      
强化学习(Reinforcement Learning, RL)是机器学习的一个重要分支,它旨在让智能体(Agent)通过与环境的交互来学习最优策略,以获得最大的累积奖励。与监督学习和非监督学习不同,强化学习不需要预先准备好标注数据,而是通过不断试错和反馈来自主学习和决策。
      
#### 1.1.2 马尔可夫决策过程
   
强化学习问题通常被建模为马尔可夫决策过程(Markov Decision Process, MDP)。一个MDP由状态集合S、动作集合A、状态转移概率P、奖励函数R和折扣因子γ组成。Agent与环境的交互过程可以用$<S_t,A_t,R_{t+1},S_{t+1}>$的轨迹序列来表示。

### 1.2 基于值函数的强化学习方法
   
#### 1.2.1 值函数与贝尔曼方程

在强化学习中,值函数用来估计在某个状态下执行某个策略π所能获得的期望回报。状态值函数$V^π(s)$表示从状态s开始遵循策略π的期望累积回报;动作值函数$Q^π(s,a)$表示在状态s下执行动作a,然后遵循策略π所获得的期望累积回报。它们满足如下贝尔曼方程:

$$V^{\pi}(s)=\sum_{a} \pi(a|s) \sum_{s^{\prime}, r} p\left(s^{\prime}, r | s, a\right)\left[r+\gamma V^{\pi}\left(s^{\prime}\right)\right]$$

$$Q^{\pi}(s, a)=\sum_{s^{\prime}, r} p\left(s^{\prime}, r | s, a\right)\left[r+\gamma \sum_{a^{\prime}} \pi\left(a^{\prime} | s^{\prime}\right) Q^{\pi}\left(s^{\prime}, a^{\prime}\right)\right]$$
   
#### 1.2.2 值迭代与策略迭代

求解MDP的经典方法有值迭代和策略迭代。值迭代通过迭代更新贝尔曼最优方程来求解最优值函数,进而得到最优策略。而策略迭代则交替执行策略评估(求解当前策略的值函数)和策略改进(基于当前值函数得到更优策略),直到策略收敛。

### 1.3 基于策略的强化学习方法
   
策略梯度方法是另一类重要的强化学习算法,它直接对策略函数参数化并沿着提升期望回报的方向上进行梯度上升。常见的策略梯度算法包括REINFORCE、Actor-Critic等。与基于值函数的方法相比,策略梯度通常对状态空间的约束更少,但学习过程的方差较大。

## 2.核心概念与联系

### 2.1 Actor与Critic

Actor-Critic算法的核心思想是将策略函数(Actor)和值函数(Critic)分别用两个独立的网络来表示与学习。Actor网络负责学习最优化的策略,输入状态并输出动作的概率分布;Critic网络负责学习值函数对状态的估值,为Actor网络提供决策指导。

### 2.2 优势函数与策略梯度定理

为了评判在某状态下 actor 所采取的动作相比于平均水平(遵循当前策略)的优劣,一个重要的概念是优势函数(advantage function):
$$A^{\pi}(s,a):=Q^{\pi}(s,a)-V^{\pi}(s)$$
它表示在状态s下选择动作a相较于遵循平均策略π能获得多少额外的回报。 

策略梯度定理给出了期望回报关于策略参数θ的梯度:
$$\nabla_{\theta} J(\theta)=\mathbb{E}_{s \sim \rho^{\pi}, a \sim \pi_{\theta}}\left[\nabla_{\theta} \log \pi_{\theta}(a \mid s) Q^{\pi}(s, a)\right]$$
其中$\rho^π$是使用当前策略的状态分布。实践中, $Q^π$往往被替换为优势函数$A^π$。
   
## 3.核心算法原理具体操作步骤

### 3.1 算法框架

Actor-Critic 算法交替执行两个步骤:

1) Critic更新: 利用TD算法,根据当前策略采样的轨迹数据来更新值函数网络的参数$\omega$,使其逼近真实的值函数 $Q^{\pi_{\theta}}$ 。

2) Actor更新: 固定值函数Q,利用策略梯度定理更新策略网络的参数$\theta$,以提高回报期望。优势函数由值函数近似给出。 

### 3.2 伪代码描述
   
#### 输入:
- 策略网络 $\pi_{\theta}$,值函数网络$Q_{\omega}$
- 一个可微分的策略损失函数 $\mathcal{L}_{\pi}(\theta, \omega)$
- 学习率 $\alpha_{\theta}$ 和 $\alpha_{\omega}$ 

#### 输出:
- 优化后的策略参数 $\theta$ 和值函数参数 $\omega$

#### 算法步骤:

```
Initialize policy parameters $\theta$ and value function parameters $\omega$
for iteration=1,2,... do
    Collect a set of trajectories $\mathcal{D}=\{\tau_i\}$ by executing the current policy $\pi_{\theta}$ 
    Compute rewards-to-go $\hat{R}_t$ 
    Compute advantage estimates $\hat{A}_t$ based on the current value function $V_{\phi}$
    Update the value function by one step of gradient descent using
        $\nabla_{\omega} \frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^{T}\left(V_{\omega}\left(s_{t}\right)-\hat{R}_{t}\right)^{2}$
    Compute policy gradient estimate:
        $\hat{g} \leftarrow \nabla_{\theta} \frac{1}{|\mathcal{D}|} \sum_{\tau \in \mathcal{D}} \sum_{t=0}^{T} \log \pi_{\theta}\left(a_{t} \mid s_{t}\right) \hat{A}_{t}$ 
    Update the policy using:
        $\theta \leftarrow \theta+\alpha_{\theta} \hat{g}$
end for
```

1. 重复以下步骤第3~8行,直到算法收敛。
2. 用当前策略$\pi_{\theta}$在环境中采样一批轨迹数据 $\mathcal{D}=\{\tau_i\}$,其中每条轨迹 $\tau_i={(s_0^i,a_0^i,r_1^i,s_1^i,a_1^i,…,s_{Ti}^i )}$包含了状态、动作和奖励信息。
3. 对每个时间步t,计算从t时刻开始到episode结束的累积折扣回报$\hat{R}_t^i=\sum_{t'=t}^{T_i} \gamma^{t'-t}r_{t'}^i$。
4. 根据当前值函数计算每个$(s_t^i,a_t^i)$的优势函数估计$\hat{A}_t^i=\hat{R}_t^i-V_w(s_t^i)$。
5. 关于$\omega$最小化值函数的均方误差损失 $\mathcal{L}(\omega)=\frac{1}{2|\mathcal{D}|}\sum_i\sum_{t=0}^{T_i}(V_{\omega}(s_t^i)-\hat{R}_t^i)^2$,对$\omega$求梯度并更新一步。
6. 用$\hat{g}=\frac{1}{|\mathcal{D}|}\sum_i\sum_{t=0}^{T_i}\nabla_{\theta}\log\pi_{\theta}(a_t^i|s_t^i)\hat{A}_t^i$估计策略梯度。
7. 沿梯度方向更新策略网络参数$\theta$:$\theta\leftarrow\theta+\alpha_{\theta}\hat{g}$。

通过不断迭代上述过程,Actor-Critic算法最终可以收敛到一个局部最优策略。

## 4.数学模型和公式详细讲解举例说明
   
### 4.1 动作值函数的递推公式
回顾动作值函数$Q^{\pi}(s,a)$满足贝尔曼方程:

$$Q^{\pi}(s, a)=\mathbb{E}_{s^{\prime}, r \sim p}\left[r+\gamma \mathbb{E}_{a^{\prime} \sim \pi}\left[Q^{\pi}\left(s^{\prime}, a^{\prime}\right)\right]\right]$$

其中$r$是当前的奖励,$s'$是执行动作$a$后转移到的下一状态,$a'$是在 $s'$状态下基于策略$\pi$选择的动作。因此可以得到动作值函数的近似递推公式:

$$\begin{aligned}
Q^{\pi}\left(s_{t}, a_{t}\right) & \approx r_{t}+\gamma Q^{\pi}\left(s_{t+1}, a_{t+1}\right) \\
& \approx \frac{1}{N} \sum_{i=1}^{N}\left(r_{t}^{(i)}+\gamma Q^{\pi}\left(s_{t+1}^{(i)}, a_{t+1}^{(i)}\right)\right)
\end{aligned}$$

第2行采用蒙特卡罗方法,基于$N$条采样轨迹$\{(s_t^{(i)},a_t^{(i)},r_t^{(i)},s_{t+1}^{(i)},a_{t+1}^{(i)})\}_{i=1}^{N}$对$Q$函数做无偏估计。

### 4.2 时序差分(TD)误差
   
在时刻$t$上的TD误差定义为:

$$\delta_{t}^{\pi}=r_{t}+\gamma Q^{\pi}\left(s_{t+1}, a_{t+1}\right)-Q^{\pi}\left(s_{t}, a_{t}\right)$$

它表示$Q$函数的一步回报估计值 $r_t+\gamma Q^{\pi}(s_{r+1},a_{t+1})$ 与其当前估计值 $Q^{\pi}(s_t,a_t)$之差。TD误差可以指导价值网络参数$\omega$的更新。考虑如下均方TD误差损失函数:

$$\mathcal{L}(\omega)=\mathbb{E}_{(s, a) \sim \rho^{\pi}}\left[\frac{1}{2}\left(Q_{\omega}(s, a)-\hat{Q}^{\pi}(s, a)\right)^{2}\right]$$

其中 $\rho^{\pi}$是在策略$\pi$下的状态-动作对分布。$\hat{Q}^{\pi}(s_t,a_t)=r_t+\gamma Q_{\omega}(s_{t+1},a_{t+1})$代入可得:

$$=\mathbb{E}_{(s, a) \sim \rho^{\pi}}\left[\frac{1}{2}\left(Q_{\omega}(s, a)-\left(r+\gamma Q_{\omega}\left(s^{\prime}, a^{\prime}\right)\right)\right)^{2}\right]$$
   
对$\omega$求导可得TD误差关于$Q$函数参数的梯度更新公式:

$$\omega \leftarrow \omega+\alpha_{\omega}\left(r+\gamma Q_{\omega}\left(s^{\prime}, a^{\prime}\right)-Q_{\omega}(s, a)\right) \nabla_{\omega} Q_{\omega}(s, a)$$
  
### 4.3 策略梯度定理推导

令$\tau=(s_0,a_0,…,s_{T},a_{T})$表示一条完整的轨迹。轨迹$\tau$出现的概率为:

$$P(\tau \mid \theta)=\rho_{0}\left(s_{0}\right) \prod_{t=0}^{T} \pi_{\theta}\left(a_{t} \mid s_{t}\right) p\left(s_{t+1} \mid s_{t}, a_{t}\right)$$

其中$\rho_0$是初始状态分布。我们要优化的目标函数是累积期望回报:

$$J(\theta)=\mathbb{E}_{\tau \sim P(\tau \mid \theta)}\left[\sum_{t=0}^{T} r\left(s_{t}, a_{t}\right)\right]=\sum_{\tau} P(\tau \mid \theta) R(\tau)$$

其中$R(\tau)=\sum_{t=0}^{T}r(s_t,a_t)$是轨迹$\tau$的总回报。目标函数关于$\theta$的梯度为:

$$\begin{aligned}
\nabla_{\theta} J(\theta) &=\nabla_{\theta} \sum_{\tau} P(\tau \mid \theta) R(\tau) \\
&=\sum_{\tau} \nabla_{\theta} P(\tau \mid \theta) R(\tau) \\
&=\sum_{\tau} P(\tau \mid \theta) \frac{\nabla_{\theta} P(\tau \mid \theta)}{P(\tau \mid \theta)} R(\tau) 