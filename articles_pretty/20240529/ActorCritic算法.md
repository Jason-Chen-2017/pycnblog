# Actor-Critic算法

## 1.背景介绍

### 1.1 强化学习概述

强化学习(Reinforcement Learning, RL)是机器学习的一个重要分支,它研究如何基于环境反馈来学习行为策略,以最大化长期累积奖励。与监督学习不同,强化学习没有给定正确的输入-输出对,而是通过与环境交互来学习。

强化学习问题通常被建模为马尔可夫决策过程(Markov Decision Process, MDP),包括:

- 状态(State) $s \in \mathcal{S}$
- 动作(Action) $a \in \mathcal{A}$  
- 状态转移概率 $\mathcal{P}_{ss'}^a = \Pr(s_{t+1}=s'|s_t=s,a_t=a)$
- 奖励函数(Reward Function) $\mathcal{R}_s^a = \mathbb{E}[r_{t+1}|s_t=s,a_t=a]$

目标是找到一个最优策略(Optimal Policy) $\pi^*$,使得在任何初始状态下,期望的累积奖励都最大化:

$$\pi^* = \arg\max_\pi \mathbb{E}_\pi \left[\sum_{t=0}^\infty \gamma^t r_t\right]$$

其中$\gamma \in [0,1]$是折扣因子,用于权衡即时奖励和未来奖励的重要性。

### 1.2 策略梯度算法

策略梯度(Policy Gradient)方法是解决强化学习问题的一种重要方法,通过直接优化策略参数来最大化期望奖励。Actor-Critic算法就属于策略梯度方法的一种。

传统的策略梯度方法存在一些问题,如高方差、样本效率低等。Actor-Critic算法通过引入值函数(Value Function)评估器(Critic)来减少方差,提高样本效率。

## 2.核心概念与联系  

### 2.1 Actor-Critic架构

Actor-Critic算法由两个组件组成:

- Actor: 根据当前状态选择动作的策略模型,通常是一个深度神经网络。
- Critic: 评估当前状态(或状态-动作对)的值函数,也是一个深度神经网络。

Actor根据Critic提供的值函数估计来更新自身的策略参数,Critic则根据后续的奖励来更新值函数估计。二者相互影响,形成一个闭环系统。

<div class="mermaid">
graph LR
    subgraph Actor-Critic
        Actor("Actor(策略模型)") -->|选择动作| Environment("环境")
        Environment -->|奖励和新状态| Actor
        Environment -->|奖励和新状态| Critic("Critic(值函数评估)")
        Critic -->|值函数估计| Actor
    end
</div>

### 2.2 优势函数

Actor-Critic算法的关键是利用优势函数(Advantage Function)来减少策略梯度估计的方差。优势函数定义为:

$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$

其中$Q^\pi(s,a)$是在策略$\pi$下选择动作$a$时状态$s$的行为值函数(Action-Value Function),表示期望的累积奖励;$V^\pi(s)$是在策略$\pi$下状态$s$的状态值函数(State-Value Function)。

优势函数可以理解为相对于当前状态值函数的改善程度。当优势函数为正时,说明选择该动作比平均水平要好;当为负时,则说明选择该动作比平均水平要差。

通过优势函数,我们可以只更新那些比平均水平好的动作的概率,从而减少策略梯度估计的方差。

### 2.3 Actor-Critic算法流程

Actor-Critic算法的基本流程如下:

1. 初始化Actor和Critic的神经网络参数
2. 获取初始状态$s_0$
3. 对于每个时间步:
    - Actor根据当前状态$s_t$选择动作$a_t$
    - 执行动作$a_t$,获得奖励$r_{t+1}$和新状态$s_{t+1}$
    - Critic根据$(s_t, a_t, r_{t+1}, s_{t+1})$更新值函数估计
    - Actor根据优势函数估计更新策略参数
4. 重复步骤3,直到终止

在实际实现中,通常会采用一些技巧来提高算法的性能和稳定性,如经验回放(Experience Replay)、目标网络(Target Network)、熵正则化(Entropy Regularization)等。

## 3.核心算法原理具体操作步骤

在这一部分,我们将详细介绍Actor-Critic算法的核心原理和具体操作步骤。

### 3.1 策略梯度定理

Actor-Critic算法的核心是基于策略梯度定理(Policy Gradient Theorem)来优化策略参数。策略梯度定理给出了期望累积奖励相对于策略参数的梯度:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^\infty \nabla_\theta \log \pi_\theta(a_t|s_t)Q^{\pi_\theta}(s_t,a_t)\right]$$

其中$\theta$是策略$\pi_\theta$的参数,$Q^{\pi_\theta}(s_t,a_t)$是在策略$\pi_\theta$下状态$s_t$和动作$a_t$的行为值函数。

直接使用上式进行策略梯度估计存在高方差的问题,因为$Q^{\pi_\theta}(s_t,a_t)$的值域可能很大。Actor-Critic算法通过使用优势函数$A^{\pi_\theta}(s_t,a_t)$来代替$Q^{\pi_\theta}(s_t,a_t)$,从而减少方差:

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^\infty \nabla_\theta \log \pi_\theta(a_t|s_t)A^{\pi_\theta}(s_t,a_t)\right]$$

其中$A^{\pi_\theta}(s_t,a_t) = Q^{\pi_\theta}(s_t,a_t) - V^{\pi_\theta}(s_t)$是优势函数。

### 3.2 Actor更新

Actor的目标是最大化期望的累积奖励,即最大化$J(\theta)$。根据策略梯度定理,我们可以通过梯度上升法来更新Actor的参数$\theta$:

$$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$$

其中$\alpha$是学习率。

具体的操作步骤如下:

1. 从当前状态$s_t$出发,根据Actor的策略$\pi_\theta(a|s_t)$采样动作$a_t$
2. 执行动作$a_t$,获得奖励$r_{t+1}$和新状态$s_{t+1}$
3. 计算优势函数估计$\hat{A}^{\pi_\theta}(s_t,a_t)$(后面会介绍如何估计)
4. 计算策略梯度:$\nabla_\theta \log \pi_\theta(a_t|s_t)\hat{A}^{\pi_\theta}(s_t,a_t)$
5. 使用梯度上升法更新Actor参数:$\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(a_t|s_t)\hat{A}^{\pi_\theta}(s_t,a_t)$

### 3.3 Critic更新

Critic的目标是准确估计值函数,包括状态值函数$V^{\pi_\theta}(s)$和行为值函数$Q^{\pi_\theta}(s,a)$。我们可以使用时序差分(Temporal Difference, TD)学习来更新Critic的参数。

#### 3.3.1 状态值函数更新

对于状态值函数$V^{\pi_\theta}(s)$,我们可以使用TD误差来更新:

$$\delta_t = r_{t+1} + \gamma V^{\pi_\theta}(s_{t+1}) - V^{\pi_\theta}(s_t)$$

其中$\gamma$是折扣因子。TD误差$\delta_t$表示实际获得的奖励加上下一状态的估计值,与当前状态的估计值之间的差异。

我们可以使用半梯度TD(Semi-gradient TD)算法来更新Critic的参数$\phi$:

$$\phi \leftarrow \phi + \beta \delta_t \nabla_\phi V^{\pi_\theta}(s_t)$$

其中$\beta$是Critic的学习率。

#### 3.3.2 行为值函数更新

对于行为值函数$Q^{\pi_\theta}(s,a)$,我们可以使用类似的TD误差来更新:

$$\delta_t = r_{t+1} + \gamma Q^{\pi_\theta}(s_{t+1}, a_{t+1}) - Q^{\pi_\theta}(s_t, a_t)$$

其中$a_{t+1}$是在状态$s_{t+1}$下根据Actor的策略$\pi_\theta$采样得到的动作。

我们可以使用半梯度Sarsa算法来更新Critic的参数$\phi$:

$$\phi \leftarrow \phi + \beta \delta_t \nabla_\phi Q^{\pi_\theta}(s_t, a_t)$$

在实践中,通常使用一个神经网络同时估计状态值函数$V^{\pi_\theta}(s)$和行为值函数$Q^{\pi_\theta}(s,a)$,即:

$$Q^{\pi_\theta}(s,a) = V^{\pi_\theta}(s) + A^{\pi_\theta}(s,a)$$

这样可以共享特征提取部分的参数,提高样本效率。

### 3.4 优势函数估计

在Actor-Critic算法中,我们需要估计优势函数$A^{\pi_\theta}(s,a)$。有几种常见的估计方法:

#### 3.4.1 蒙特卡罗估计

蒙特卡罗估计是一种基于完整回报(Return)的无偏估计方法。具体做法是:

1. 从状态$s_t$出发,执行完整的回合,获得奖励序列$r_{t+1}, r_{t+2}, \dots, r_T$
2. 计算回报(Return):$G_t = \sum_{k=t+1}^T \gamma^{k-t-1}r_k$
3. 优势函数估计:$\hat{A}^{\pi_\theta}(s_t,a_t) = G_t - V^{\pi_\theta}(s_t)$

蒙特卡罗估计是无偏的,但存在高方差的问题,尤其是在回合很长的情况下。

#### 3.4.2 时序差分估计

时序差分估计是一种基于增量式更新的方法,可以减少方差。具体做法是:

1. 从状态$s_t$出发,执行一步动作$a_t$,获得奖励$r_{t+1}$和新状态$s_{t+1}$
2. 计算TD误差:$\delta_t = r_{t+1} + \gamma V^{\pi_\theta}(s_{t+1}) - V^{\pi_\theta}(s_t)$
3. 优势函数估计:$\hat{A}^{\pi_\theta}(s_t,a_t) = \delta_t$

时序差分估计虽然有偏,但方差较小,更适合在线更新。

#### 3.4.3 通用优势估计

通用优势估计(Generalized Advantage Estimation, GAE)是一种将蒙特卡罗估计和时序差分估计结合的方法,可以在偏差和方差之间进行权衡。

GAE的优势函数估计公式如下:

$$\hat{A}_t^{GAE}(\gamma, \lambda) = \sum_{l=0}^\infty (\gamma\lambda)^l \delta_{t+l}^V$$

其中$\delta_t^V = r_{t+1} + \gamma V^{\pi_\theta}(s_{t+1}) - V^{\pi_\theta}(s_t)$是TD误差,$\lambda \in [0,1]$是控制偏差-方差权衡的参数。

- 当$\lambda=0$时,GAE等价于时序差分估计,具有最小方差但存在偏差
- 当$\lambda=1$时,GAE等价于蒙特卡罗估计,是无偏估计但方差较大
- 当$\lambda$在0和1之间时,GAE在偏差和方差之间进行权衡

在实践中,通常会使用truncated版本的GAE,即只考虑有限步的TD误差,以限制计算开销。

### 3.5 Actor-Critic算法伪代码

综合上述步骤,我们可以给出Actor-Critic算法的伪代码:

```python
# 初始化Actor和Critic的参数
初始化Actor参数theta
初始化Critic参数phi

# 主循环
for episode in num_episodes:
    # 初始化状态
    s = 环境.reset()
    
    # 执行一个回合
    while not done:
        # Actor选择动作
        a = Actor.sample(s, theta)
        
        # 执行