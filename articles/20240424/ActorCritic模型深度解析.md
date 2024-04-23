## 1. 背景介绍

### 1.1 强化学习概述

强化学习 (Reinforcement Learning, RL) 作为机器学习的一个重要分支，专注于智能体在与环境交互的过程中学习如何做出决策以最大化累积奖励。不同于监督学习和非监督学习，强化学习无需预先提供标签或数据结构，而是通过试错和反馈机制逐步优化策略。

### 1.2 强化学习算法分类

强化学习算法主要分为两大类：基于价值的 (Value-based) 和基于策略的 (Policy-based) 方法。

*   **基于价值的方法**：通过估计状态或状态-动作对的价值函数，间接地学习最优策略。常见的算法包括 Q-learning, SARSA 等。
*   **基于策略的方法**：直接学习策略函数，将状态或观测映射到动作概率分布。常见的算法包括 Policy Gradient, Actor-Critic 等。

### 1.3 Actor-Critic 模型的优势

Actor-Critic 模型结合了基于价值和基于策略方法的优点，利用 Actor 网络学习策略，Critic 网络评估价值函数，两者相互协作，实现更高效的学习过程。

## 2. 核心概念与联系

### 2.1 策略函数 (Policy Function)

策略函数 $\pi(a|s)$ 表示在状态 $s$ 下采取动作 $a$ 的概率。Actor 网络负责学习和优化策略函数，目标是找到能够最大化累积奖励的策略。

### 2.2 价值函数 (Value Function)

价值函数 $V(s)$ 表示在状态 $s$ 下所能获得的期望累积奖励。Critic 网络负责评估价值函数，为 Actor 网络提供学习信号。

### 2.3 Actor 与 Critic 的联系

Actor 和 Critic 相互协作，形成一个闭环反馈系统：

*   Actor 根据当前策略选择动作，与环境交互并获得奖励。
*   Critic 评估当前状态的价值，并根据奖励和价值函数的差异计算时序差分 (TD) 误差。
*   Actor 利用 TD 误差更新策略，使未来更有可能选择产生更高价值的动作。
*   Critic 利用 TD 误差更新价值函数，使其更准确地反映状态的真实价值。

## 3. 核心算法原理

### 3.1 策略梯度 (Policy Gradient)

Actor 网络的更新基于策略梯度方法，目标是最大化期望累积奖励 $J(\theta)$，其中 $\theta$ 是策略网络的参数。

$$
\nabla_{\theta} J(\theta) \approx \mathbb{E}_{\pi_{\theta}}[G_t \nabla_{\theta} \log \pi_{\theta}(A_t|S_t)]
$$

其中 $G_t$ 表示从时间步 $t$ 开始的累积奖励，$\pi_{\theta}(A_t|S_t)$ 表示在状态 $S_t$ 下采取动作 $A_t$ 的概率。

### 3.2 时序差分学习 (Temporal-Difference Learning)

Critic 网络的更新基于时序差分学习，目标是估计状态价值函数 $V(s)$。

$$
V(S_t) \leftarrow V(S_t) + \alpha [R_{t+1} + \gamma V(S_{t+1}) - V(S_t)]
$$

其中 $\alpha$ 是学习率，$\gamma$ 是折扣因子，$R_{t+1}$ 是在时间步 $t$ 采取动作后获得的奖励。

### 3.3 具体操作步骤

1.  初始化 Actor 和 Critic 网络。
2.  循环执行以下步骤：
    *   根据当前策略选择动作，与环境交互并获得奖励和新的状态。
    *   计算 TD 误差：$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$。
    *   更新 Critic 网络参数：$w \leftarrow w + \alpha \delta_t \nabla_w V(S_t)$。
    *   更新 Actor 网络参数：$\theta \leftarrow \theta + \alpha \delta_t \nabla_{\theta} \log \pi_{\theta}(A_t|S_t)$。

## 4. 数学模型和公式详细讲解

### 4.1 策略梯度定理 (Policy Gradient Theorem)

策略梯度定理为 Actor 网络的更新提供了理论基础，它表明策略梯度可以表示为状态-动作值函数 $Q^{\pi}(s,a)$ 和策略函数梯度的乘积的期望。

$$
\nabla_{\theta} J(\theta) = \mathbb{E}_{\pi_{\theta}}[\nabla_{\theta} \log \pi_{\theta}(A_t|S_t) Q^{\pi}(S_t, A_t)]
$$

### 4.2 优势函数 (Advantage Function)

优势函数 $A(s,a)$ 表示在状态 $s$ 下采取动作 $a$ 相比于其他动作的优势，它可以用来改进策略梯度的估计。

$$
A(s,a) = Q(s,a) - V(s)
$$ 

### 4.3 重要性采样 (Importance Sampling)

重要性采样是一种用于在不同策略下估计期望值的技术，它可以用于离线强化学习和策略评估。 
