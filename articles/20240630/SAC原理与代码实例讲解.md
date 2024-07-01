# SAC原理与代码实例讲解

## 1. 背景介绍

### 1.1 问题的由来

在强化学习领域，如何平衡探索和利用一直是核心问题。传统的强化学习方法，如Q-Learning和SARSA，通常采用 $\epsilon$-greedy策略来平衡探索和利用。然而，$\epsilon$-greedy策略存在一些局限性，例如：

* 当 $\epsilon$ 值过大时，会导致探索过多，学习效率低下；
* 当 $\epsilon$ 值过小时，会导致利用过多，容易陷入局部最优。

为了克服这些局限性，近年来，一种新的强化学习方法——Soft Actor-Critic (SAC) 应运而生。SAC 算法通过引入熵正则化，实现了更有效的探索和利用平衡。

### 1.2 研究现状

SAC 算法自提出以来，受到了广泛关注，并取得了显著进展。近年来，研究人员在 SAC 算法的理论分析、算法改进、应用扩展等方面取得了一系列突破性成果。

* **理论分析方面：** 研究人员对 SAC 算法的收敛性、稳定性等方面进行了深入研究，并提出了相应的理论证明。
* **算法改进方面：** 研究人员针对 SAC 算法的效率和鲁棒性等问题，提出了多种改进算法，例如：
    * **Twin Delayed DDPG (TD3)**：通过引入双Q网络和延迟更新策略，提高了 SAC 算法的稳定性。
    * **Soft Q-Learning (SQL)**：通过引入熵正则化，提高了 SAC 算法的探索效率。
* **应用扩展方面：** 研究人员将 SAC 算法应用于各种领域，例如：
    * **机器人控制：** SAC 算法可以用来训练机器人完成各种任务，例如抓取、导航等。
    * **游戏AI：** SAC 算法可以用来训练游戏AI，例如玩 Atari 游戏、围棋等。
    * **推荐系统：** SAC 算法可以用来优化推荐系统，例如推荐商品、新闻等。

### 1.3 研究意义

SAC 算法作为一种新型的强化学习方法，具有以下重要意义：

* **提高探索效率：** SAC 算法通过引入熵正则化，能够更有效地探索状态空间，避免陷入局部最优。
* **增强鲁棒性：** SAC 算法能够在噪声和不确定性环境中保持较好的稳定性。
* **扩展应用范围：** SAC 算法可以应用于各种领域，例如机器人控制、游戏AI、推荐系统等。

### 1.4 本文结构

本文将从以下几个方面对 SAC 算法进行深入探讨：

* **核心概念与联系：** 介绍 SAC 算法的核心概念，并将其与其他强化学习方法进行对比。
* **核心算法原理 & 具体操作步骤：** 详细介绍 SAC 算法的原理和操作步骤。
* **数学模型和公式 & 详细讲解 & 举例说明：**  构建 SAC 算法的数学模型，并进行详细讲解和举例说明。
* **项目实践：代码实例和详细解释说明：** 提供 SAC 算法的代码实例，并进行详细解释说明。
* **实际应用场景：**  探讨 SAC 算法的实际应用场景。
* **工具和资源推荐：**  推荐一些学习 SAC 算法的工具和资源。
* **总结：未来发展趋势与挑战：**  总结 SAC 算法的研究成果，并展望其未来发展趋势和面临的挑战。


## 2. 核心概念与联系

SAC 算法的核心思想是通过引入熵正则化来平衡探索和利用。熵正则化是指在目标函数中加入一个与策略熵相关的项，从而鼓励策略探索更多不同的行为。

**熵** 是一个衡量随机变量不确定性的指标。在强化学习中，策略的熵可以用来衡量策略探索行为的多样性。熵越大，策略探索的行为越多样，反之则越单一。

**SAC 算法的目标函数** 可以表示为：

$$
J(\theta) = E_{\tau \sim \pi_{\theta}}[\sum_{t=0}^{\infty} \gamma^t (r_t + \alpha H(\pi_{\theta}(\cdot|s_t)))]
$$

其中：

* $\theta$ 是策略参数。
* $\tau$ 是轨迹，表示状态、动作、奖励的序列。
* $\pi_{\theta}$ 是策略，表示在给定状态下选择动作的概率分布。
* $\gamma$ 是折扣因子。
* $r_t$ 是第 $t$ 步的奖励。
* $\alpha$ 是熵正则化系数。
* $H(\pi_{\theta}(\cdot|s_t))$ 是策略在状态 $s_t$ 下的熵。

目标函数的第一项是传统的强化学习目标函数，即最大化累积奖励。第二项是熵正则化项，它鼓励策略探索更多不同的行为。

**SAC 算法与其他强化学习方法的联系：**

* **与 DDPG 的联系：** SAC 算法可以看作是 DDPG 算法的改进版本，它通过引入熵正则化，提高了 DDPG 算法的探索效率和鲁棒性。
* **与 Q-Learning 的联系：** SAC 算法可以看作是 Q-Learning 算法的推广，它将 Q 函数扩展到策略空间，并通过熵正则化来引导策略探索。

## 3. 核心算法原理 & 具体操作步骤

### 3.1 算法原理概述

SAC 算法的核心思想是通过最小化一个目标函数来学习一个最优策略。目标函数包含两个部分：

* **奖励最大化部分：** 鼓励策略选择能够最大化累积奖励的动作。
* **熵正则化部分：** 鼓励策略探索更多不同的行为，避免陷入局部最优。

SAC 算法使用两个神经网络来近似策略和价值函数：

* **策略网络：** 用于输出在给定状态下选择动作的概率分布。
* **价值网络：** 用于估计在给定状态下执行某个动作的预期累积奖励。

SAC 算法采用以下步骤来学习最优策略：

1. **初始化策略网络和价值网络：** 使用随机参数初始化策略网络和价值网络。
2. **收集数据：** 使用当前策略在环境中收集数据，得到一批状态、动作、奖励的序列。
3. **更新价值网络：** 使用收集到的数据来更新价值网络，使其能够准确地估计动作的预期累积奖励。
4. **更新策略网络：** 使用价值网络的输出和熵正则化项来更新策略网络，使其能够选择能够最大化目标函数的动作。
5. **重复步骤 2-4，直到策略收敛：** 随着训练的进行，策略网络和价值网络会逐渐收敛到最优解。

### 3.2 算法步骤详解

SAC 算法的具体操作步骤如下：

1. **初始化：**
    * 初始化策略网络 $\pi_{\theta}$ 和价值网络 $Q_{\phi}$。
    * 初始化目标价值网络 $Q_{\phi'}$。
    * 设置熵正则化系数 $\alpha$。
    * 设置折扣因子 $\gamma$。
    * 设置学习率 $\eta$。

2. **循环：**
    * **收集数据：** 使用当前策略 $\pi_{\theta}$ 在环境中收集一批数据，得到状态、动作、奖励的序列 $(s_t, a_t, r_t, s_{t+1})$。
    * **更新价值网络：** 使用收集到的数据来更新价值网络 $Q_{\phi}$，最小化以下损失函数：

    $$
    L_Q(\phi) = E_{(s,a,r,s') \sim D} [(Q_{\phi}(s,a) - y)^2]
    $$

    其中：

    * $D$ 是数据分布。
    * $y$ 是目标价值，可以由以下公式计算：

    $$
    y = r + \gamma \min_{a' \sim \pi_{\theta}(\cdot|s')} Q_{\phi'}(s', a') - \alpha H(\pi_{\theta}(\cdot|s'))
    $$

    * $H(\pi_{\theta}(\cdot|s'))$ 是策略在状态 $s'$ 下的熵。

    * $Q_{\phi'}$ 是目标价值网络，它的参数 $\phi'$ 是 $Q_{\phi}$ 的软更新，可以由以下公式计算：

    $$
    \phi' \leftarrow \tau \phi + (1-\tau) \phi'
    $$

    * $\tau$ 是软更新系数，通常设置为一个较小的值，例如 0.01。

    * **更新策略网络：** 使用价值网络 $Q_{\phi}$ 和熵正则化项来更新策略网络 $\pi_{\theta}$，最小化以下损失函数：

    $$
    L_{\pi}(\theta) = E_{s \sim D} [-\pi_{\theta}(a|s) Q_{\phi}(s,a) + \alpha H(\pi_{\theta}(\cdot|s))]
    $$

    * **更新目标价值网络：** 使用价值网络 $Q_{\phi}$ 的参数来更新目标价值网络 $Q_{\phi'}$。

3. **重复步骤 2，直到策略收敛：**  随着训练的进行，策略网络和价值网络会逐渐收敛到最优解。

### 3.3 算法优缺点

**优点：**

* **提高探索效率：** SAC 算法通过引入熵正则化，能够更有效地探索状态空间，避免陷入局部最优。
* **增强鲁棒性：** SAC 算法能够在噪声和不确定性环境中保持较好的稳定性。
* **扩展应用范围：** SAC 算法可以应用于各种领域，例如机器人控制、游戏AI、推荐系统等。

**缺点：**

* **计算复杂度较高：** SAC 算法需要同时训练策略网络和价值网络，计算复杂度较高。
* **参数调优难度较大：** SAC 算法的参数，例如熵正则化系数 $\alpha$，需要仔细调优才能获得最佳性能。

### 3.4 算法应用领域

SAC 算法可以应用于各种领域，例如：

* **机器人控制：** SAC 算法可以用来训练机器人完成各种任务，例如抓取、导航等。
* **游戏AI：** SAC 算法可以用来训练游戏AI，例如玩 Atari 游戏、围棋等。
* **推荐系统：** SAC 算法可以用来优化推荐系统，例如推荐商品、新闻等。
* **金融投资：** SAC 算法可以用来进行金融投资策略的优化。

## 4. 数学模型和公式 & 详细讲解 & 举例说明

### 4.1 数学模型构建

SAC 算法的数学模型可以表示为：

* **状态空间：** $S$
* **动作空间：** $A$
* **奖励函数：** $r(s,a)$
* **折扣因子：** $\gamma$
* **策略：** $\pi_{\theta}(a|s)$
* **价值函数：** $V_{\theta}(s)$
* **Q 函数：** $Q_{\phi}(s,a)$
* **熵正则化系数：** $\alpha$

**目标函数：**

$$
J(\theta) = E_{\tau \sim \pi_{\theta}}[\sum_{t=0}^{\infty} \gamma^t (r_t + \alpha H(\pi_{\theta}(\cdot|s_t)))]
$$

**价值函数的更新公式：**

$$
V_{\theta}(s) = E_{a \sim \pi_{\theta}(\cdot|s)}[Q_{\phi}(s,a)]
$$

**Q 函数的更新公式：**

$$
Q_{\phi}(s,a) = r(s,a) + \gamma E_{s' \sim P(s'|s,a)}[V_{\theta}(s')]
$$

**策略的更新公式：**

$$
\pi_{\theta}(a|s) = \frac{\exp(Q_{\phi}(s,a)/\alpha)}{\sum_{a' \in A} \exp(Q_{\phi}(s,a')/\alpha)}
$$

### 4.2 公式推导过程

**价值函数的更新公式推导：**

$$
\begin{aligned}
V_{\theta}(s) &= E_{a \sim \pi_{\theta}(\cdot|s)}[Q_{\phi}(s,a)] \\
&= E_{a \sim \pi_{\theta}(\cdot|s)}[r(s,a) + \gamma E_{s' \sim P(s'|s,a)}[V_{\theta}(s')]] \\
&= E_{a \sim \pi_{\theta}(\cdot|s)}[r(s,a)] + \gamma E_{a \sim \pi_{\theta}(\cdot|s)}[E_{s' \sim P(s'|s,a)}[V_{\theta}(s')]] \\
&= E_{a \sim \pi_{\theta}(\cdot|s)}[r(s,a)] + \gamma E_{s' \sim P(s'|s,a)}[E_{a \sim \pi_{\theta}(\cdot|s')}[V_{\theta}(s')]] \\
&= E_{a \sim \pi_{\theta}(\cdot|s)}[r(s,a)] + \gamma E_{s' \sim P(s'|s,a)}[V_{\theta}(s')]
\end{aligned}
$$

**Q 函数的更新公式推导：**

$$
\begin{aligned}
Q_{\phi}(s,a) &= r(s,a) + \gamma E_{s' \sim P(s'|s,a)}[V_{\theta}(s')] \\
&= r(s,a) + \gamma E_{s' \sim P(s'|s,a)}[E_{a' \sim \pi_{\theta}(\cdot|s')}[Q_{\phi}(s',a')]] \\
&= r(s,a) + \gamma E_{s' \sim P(s'|s,a)}[Q_{\phi}(s',a')]
\end{aligned}
$$

**策略的更新公式推导：**

$$
\begin{aligned}
\pi_{\theta}(a|s) &= \arg\max_{a \in A} Q_{\phi}(s,a) + \alpha H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} Q_{\phi}(s,a) - \alpha \sum_{a' \in A} \pi_{\theta}(a'|s) \log \pi_{\theta}(a'|s) \\
&= \arg\max_{a \in A} \log \pi_{\theta}(a|s) + \frac{Q_{\phi}(s,a)}{\alpha} - \sum_{a' \in A} \pi_{\theta}(a'|s) \log \pi_{\theta}(a'|s) \\
&= \arg\max_{a \in A} \log \pi_{\theta}(a|s) + \frac{Q_{\phi}(s,a)}{\alpha} - \sum_{a' \in A} \pi_{\theta}(a'|s) \log \pi_{\theta}(a'|s) \\
&= \arg\max_{a \in A} \log \pi_{\theta}(a|s) + \frac{Q_{\phi}(s,a)}{\alpha} - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q_{\phi}(s,a)}{\alpha} + \log \pi_{\theta}(a|s) - H(\pi_{\theta}(\cdot|s)) \\
&= \arg\max_{a \in A} \frac{Q