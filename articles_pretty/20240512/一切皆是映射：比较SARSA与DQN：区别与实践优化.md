## 1. 背景介绍

### 1.1 强化学习概述

强化学习（Reinforcement Learning, RL）是机器学习的一个重要分支，其目标是让智能体（Agent）在与环境的交互中学习最佳策略，从而最大化累积奖励。不同于监督学习，强化学习不依赖于标注数据，而是通过试错和奖励机制来学习。

### 1.2 时间差分学习

时间差分学习（Temporal Difference Learning, TD Learning）是强化学习中一类重要的算法，其核心思想是利用当前状态的价值估计来更新先前状态的价值估计。SARSA 和 DQN 都是基于时间差分学习的算法。

### 1.3 SARSA 与 DQN

SARSA（State-Action-Reward-State-Action）和 DQN（Deep Q-Network）是两种常用的时间差分学习算法，它们在处理价值函数和策略更新方面有所区别。SARSA 是一种 on-policy 算法，而 DQN 是一种 off-policy 算法。

## 2. 核心概念与联系

### 2.1 状态、动作、奖励

*   **状态（State）**:  描述智能体所处环境的信息。
*   **动作（Action）**: 智能体在特定状态下可以采取的操作。
*   **奖励（Reward）**: 智能体执行某个动作后，环境给予的反馈信号，用于评估动作的好坏。

### 2.2 策略、价值函数

*   **策略（Policy）**:  定义智能体在每个状态下应该采取的动作，通常用 $\pi(a|s)$ 表示，表示在状态 $s$ 下采取动作 $a$ 的概率。
*   **价值函数（Value Function）**:  用于评估状态或状态-动作对的长期价值，通常用 $V(s)$ 或 $Q(s, a)$ 表示。

### 2.3 On-policy vs. Off-policy

*   **On-policy**:  智能体根据当前正在执行的策略来学习，例如 SARSA。
*   **Off-policy**:  智能体可以根据其他策略收集的经验来学习，例如 DQN。

## 3. 核心算法原理具体操作步骤

### 3.1 SARSA 算法

SARSA 算法的更新规则如下：

$$
Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma Q(s', a') - Q(s, a)]
$$

其中：

*   $Q(s, a)$: 状态-动作对 $(s, a)$ 的价值估计
*   $\alpha$: 学习率
*   $r$: 在状态 $s$ 下执行动作 $a$ 后获得的奖励
*   $\gamma$: 折扣因子，用于平衡当前奖励和未来奖励
*   $s'$: 执行动作 $a$ 后到达的新状态
*   $a'$: 在新状态 $s'$ 下根据当前策略选择的动作

SARSA 算法的操作步骤如下：

1.  初始化 Q 值表
2.  循环遍历每个 episode：
    *   初始化状态 $s$
    *   根据当前策略选择动作 $a$
    *   执行动作 $a$，观察奖励 $r$ 和新状态 $s'$
    *   根据当前策略选择新动作 $a'$
    *   使用 SARSA 更新规则更新 Q 值表
    *   更新状态 $s \leftarrow s'$，动作 $a \leftarrow a'$
    *   直到 episode 结束

### 3.2 DQN 算法

DQN 算法利用深度神经网络来逼近价值函数，其更新规则如下：

$$
\theta \leftarrow \theta - \alpha \nabla_{\theta} L(\theta)
$$

其中：

*   $\theta$: 神经网络的参数
*   $\alpha$: 学习率
*   $L(\theta)$: 损失函数，通常使用均方误差

DQN 算法的操作步骤如下：

1.  初始化深度神经网络
2.  循环遍历每个 episode：
    *   初始化状态 $s$
    *   循环遍历每个时间步：
        *   根据当前策略选择动作 $a$
        *   执行动作 $a$，观察奖励 $r$ 和新状态 $s'$
        *   将经验 $(s, a, r, s')$ 存储到经验回放池
        *   从经验回放池中随机抽取一批经验
        *   使用 DQN 更新规则更新神经网络参数
        *   更新状态 $s \leftarrow s'$
    *   直到 episode 结束

## 4. 数学模型和公式详细讲解举例说明

### 4.1 Bellman 方程

价值函数满足 Bellman 方程：

$$
V(s) = \max_{a} \sum_{s', r} p(s', r | s, a) [r + \gamma V(s')]
$$

其中：

*   $V(s)$: 状态 $s$ 的价值
*   $\max_{a}$: 表示选择最佳动作
*   $p(s', r | s, a)$: 表示在状态 $s$ 下执行动作 $a$ 后转移到状态 $s'$ 并获得奖励 $r$ 的概率
*   $r$: 奖励
*   $\gamma$: 折扣因子

Bellman 方程描述了状态价值与其后继状态价值之间的关系。

### 4.2 SARSA 更新规则推导

SARSA 更新规则可以从 Bellman 方程推导出来。将 Bellman 方程改写为：

$$
V(s) = \sum_{a} \pi(a|s) \sum_{s', r} p(s', r | s, a) [r + \gamma V(s')]
$$

将 $V(s)$ 替换为 $Q(s, a)$，得到：

$$
Q(s, a) = \sum_{s', r} p(s', r | s, a) [r + \gamma \sum_{a'} \pi(a'|s') Q(s', a')]
$$

将上式中的期望值替换为样本值，得到 SARSA 更新规则：

$$
Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma Q(s', a') - Q(s, a)]
$$

### 4.3 DQN 损失函数

DQN 算法使用均方误差作为损失函数：

$$
L(\theta) = \frac{1}{N} \sum_{i=1}^{N} (y_i - Q(s_i, a_i; \theta))^2
$$

其中：

*   $N$: 经验回放池中样本的数量
*   $y_i$: 目标 Q 值，计算方式为 $y_i = r_i + \gamma \max_{a'} Q(s'_i, a';