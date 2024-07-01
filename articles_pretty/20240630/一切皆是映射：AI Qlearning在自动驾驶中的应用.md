# 一切皆是映射：AI Q-learning在自动驾驶中的应用

作者：禅与计算机程序设计艺术 / Zen and the Art of Computer Programming 

## 1. 背景介绍

### 1.1 问题的由来

自动驾驶技术作为人工智能领域最具代表性的应用之一，近年来取得了显著进展，但其面临着诸多挑战，其中最关键的挑战之一是如何让车辆在复杂多变的现实世界中做出安全、高效的决策。传统方法通常依赖于预先定义的规则和模型，难以应对突发情况和未知环境。而强化学习 (Reinforcement Learning, RL) 作为一种新型的机器学习方法，为解决这一问题提供了新的思路。

### 1.2 研究现状

近年来，强化学习在自动驾驶领域的研究取得了重大突破，例如：

* **基于 Q-learning 的自动驾驶决策系统:** 利用 Q-learning 算法训练智能体，使其能够根据当前状态选择最佳的动作，从而实现车辆的自动驾驶。
* **基于深度强化学习的自动驾驶控制系统:** 将深度学习与强化学习相结合，构建更强大的模型，能够应对更复杂的驾驶场景。
* **基于多智能体强化学习的自动驾驶协同系统:**  多个智能体之间相互协作，实现更高效的交通管理和车辆控制。

### 1.3 研究意义

自动驾驶技术的应用将带来巨大的社会效益，例如：

* **提高交通安全:** 自动驾驶系统能够消除人为错误，降低交通事故发生率。
* **提升交通效率:** 自动驾驶系统能够优化车辆行驶路线，减少交通拥堵。
* **改善城市环境:** 自动驾驶系统能够减少尾气排放，改善城市空气质量。

### 1.4 本文结构

本文将深入探讨 AI Q-learning 在自动驾驶中的应用，主要内容包括：

* **Q-learning 算法原理:** 介绍 Q-learning 算法的核心概念和工作机制。
* **Q-learning 在自动驾驶中的应用:** 探讨 Q-learning 如何应用于自动驾驶决策系统。
* **案例分析:** 通过实际案例演示 Q-learning 在自动驾驶中的应用效果。
* **未来展望:** 展望 Q-learning 在自动驾驶领域未来的发展趋势。

## 2. 核心概念与联系

### 2.1 强化学习 (Reinforcement Learning, RL)

强化学习是一种机器学习方法，其目标是训练智能体 (Agent) 在与环境交互的过程中学习最佳策略，以最大化累积奖励。强化学习的核心要素包括：

* **智能体 (Agent):** 能够感知环境并做出决策的实体。
* **环境 (Environment):** 智能体所处的外部世界，会根据智能体的动作做出相应的反馈。
* **状态 (State):** 环境在某个时刻的具体情况，例如车辆的位置、速度、周围障碍物等。
* **动作 (Action):** 智能体能够采取的行动，例如加速、转向、刹车等。
* **奖励 (Reward):** 环境对智能体动作的评价，例如到达目的地获得奖励，发生碰撞获得惩罚。
* **策略 (Policy):** 智能体根据当前状态选择动作的规则。

### 2.2 Q-learning 算法

Q-learning 是一种基于价值迭代的强化学习算法，它通过不断学习状态-动作对的价值 (Q-value)，来找到最佳策略。Q-value 表示在某个状态下采取某个动作所能获得的长期累积奖励。Q-learning 算法的核心思想是：

* **状态-动作价值函数 (Q-value):** $Q(s, a)$ 表示在状态 $s$ 下采取动作 $a$ 所能获得的长期累积奖励。
* **贝尔曼方程 (Bellman Equation):** 描述状态-动作价值函数之间的关系，用于更新 Q-value。
* **探索-利用 (Exploration-Exploitation):** 在学习过程中，需要在探索新动作和利用已有知识之间取得平衡。

## 3. 核心算法原理 & 具体操作步骤

### 3.1 算法原理概述

Q-learning 算法的原理是基于贝尔曼方程 (Bellman Equation) 的迭代更新过程，通过不断学习状态-动作对的价值 (Q-value)，最终找到最佳策略。

**贝尔曼方程:**

$$Q(s, a) = R(s, a) + \gamma \max_{a'} Q(s', a')$$

其中：

* $Q(s, a)$: 在状态 $s$ 下采取动作 $a$ 的 Q-value。
* $R(s, a)$: 在状态 $s$ 下采取动作 $a$ 所获得的即时奖励。
* $\gamma$: 折扣因子，用于平衡即时奖励和长期奖励。
* $s'$: 执行动作 $a$ 后到达的下一个状态。
* $\max_{a'} Q(s', a')$: 在下一个状态 $s'$ 下所有动作的 Q-value 中的最大值。

**算法流程:**

1. 初始化 Q-value 表，将所有状态-动作对的 Q-value 初始化为 0。
2. 重复以下步骤，直到收敛：
    * **选择动作:** 根据当前状态 $s$ 和当前的 Q-value 表，选择一个动作 $a$。
    * **执行动作:** 在环境中执行动作 $a$，观察下一个状态 $s'$ 和奖励 $r$。
    * **更新 Q-value:** 根据贝尔曼方程更新 Q-value 表中 $Q(s, a)$ 的值。
3. 最终获得最佳策略，即在每个状态下选择 Q-value 最大的动作。

### 3.2 算法步骤详解

**步骤 1: 初始化 Q-value 表**

将所有状态-动作对的 Q-value 初始化为 0，例如：

```python
Q = {}
for state in states:
    for action in actions:
        Q[(state, action)] = 0
```

**步骤 2: 迭代更新 Q-value**

重复以下步骤，直到 Q-value 收敛：

1. **选择动作:** 根据当前状态 $s$ 和当前的 Q-value 表，选择一个动作 $a$。
    * **ε-贪婪策略:**  以概率 ε 选择随机动作，以概率 1-ε 选择 Q-value 最大的动作。
2. **执行动作:** 在环境中执行动作 $a$，观察下一个状态 $s'$ 和奖励 $r$。
3. **更新 Q-value:** 根据贝尔曼方程更新 Q-value 表中 $Q(s, a)$ 的值。
    * $Q(s, a) = (1 - \alpha) * Q(s, a) + \alpha * (r + \gamma * \max_{a'} Q(s', a'))$
    * $\alpha$: 学习率，控制更新的步长。

**步骤 3: 获得最佳策略**

在 Q-value 收敛后，对于每个状态 $s$，选择 Q-value 最大的动作 $a$，即为最佳策略。

```python
def get_best_action(state):
    best_action = None
    max_q_value = float('-inf')
    for action in actions:
        if Q[(state, action)] > max_q_value:
            best_action = action
            max_q_value = Q[(state, action)]
    return best_action
```

### 3.3 算法优缺点

**优点:**

* **简单易懂:**  Q-learning 算法的原理相对简单，易于理解和实现。
* **通用性强:**  Q-learning 算法可以应用于各种强化学习问题，包括自动驾驶、游戏、机器人控制等。
* **离线学习:**  Q-learning 算法可以离线学习，不需要在线与环境交互。

**缺点:**

* **状态空间大:**  对于状态空间很大的问题，Q-value 表的存储和更新将非常困难。
* **收敛速度慢:**  Q-learning 算法的收敛速度可能很慢，特别是对于复杂的问题。
* **探索-利用困境:**  在学习过程中，需要在探索新动作和利用已有知识之间取得平衡。

### 3.4 算法应用领域

Q-learning 算法广泛应用于各种强化学习问题，例如：

* **自动驾驶:**  自动驾驶决策系统，例如路径规划、车道保持、避障等。
* **游戏:**  游戏 AI，例如棋类游戏、电子游戏等。
* **机器人控制:**  机器人控制系统，例如机械臂控制、移动机器人导航等。
* **推荐系统:**  个性化推荐系统，例如商品推荐、新闻推荐等。

## 4. 数学模型和公式 & 详细讲解 & 举例说明

### 4.1 数学模型构建

Q-learning 算法的数学模型基于马尔可夫决策过程 (Markov Decision Process, MDP)，MDP 描述了智能体与环境之间的交互过程，包括状态、动作、奖励和转移概率。

**MDP 的数学模型:**

* **状态空间 (State Space):** 所有可能状态的集合，记为 $S$。
* **动作空间 (Action Space):** 所有可能动作的集合，记为 $A$。
* **奖励函数 (Reward Function):** $R(s, a)$ 表示在状态 $s$ 下采取动作 $a$ 所获得的即时奖励。
* **转移概率 (Transition Probability):** $P(s'|s, a)$ 表示在状态 $s$ 下采取动作 $a$ 后到达状态 $s'$ 的概率。

### 4.2 公式推导过程

Q-learning 算法的数学模型基于贝尔曼方程 (Bellman Equation)，贝尔曼方程描述了状态-动作价值函数之间的关系，用于更新 Q-value。

**贝尔曼方程:**

$$Q(s, a) = R(s, a) + \gamma \sum_{s'} P(s'|s, a) \max_{a'} Q(s', a')$$

其中：

* $Q(s, a)$: 在状态 $s$ 下采取动作 $a$ 的 Q-value。
* $R(s, a)$: 在状态 $s$ 下采取动作 $a$ 所获得的即时奖励。
* $\gamma$: 折扣因子，用于平衡即时奖励和长期奖励。
* $P(s'|s, a)$: 在状态 $s$ 下采取动作 $a$ 后到达状态 $s'$ 的概率。
* $\max_{a'} Q(s', a')$: 在下一个状态 $s'$ 下所有动作的 Q-value 中的最大值。

**公式推导过程:**

1. 考虑在状态 $s$ 下采取动作 $a$，获得奖励 $R(s, a)$，并转移到下一个状态 $s'$。
2. 在下一个状态 $s'$ 下，智能体可以选择一个动作 $a'$，获得相应的 Q-value $Q(s', a')$。
3. 由于智能体希望最大化长期累积奖励，因此在 $s'$ 下选择 Q-value 最大的动作 $a'$，即 $\max_{a'} Q(s', a')$。
4. 将下一个状态 $s'$ 和其对应的最大 Q-value 折扣后加到当前状态的即时奖励 $R(s, a)$ 上，得到当前状态-动作对的 Q-value $Q(s, a)$。
5. 由于转移概率 $P(s'|s, a)$ 可能有多种，需要将所有可能的 $s'$ 和其对应的最大 Q-value 乘以相应的概率并求和，得到最终的 $Q(s, a)$。

### 4.3 案例分析与讲解

**案例:**  自动驾驶车辆在十字路口遇到红灯，需要决定是停车等待还是继续行驶。

**状态:** 车辆当前位置、速度、周围环境信息（红绿灯状态、其他车辆位置等）。

**动作:** 停车等待、继续行驶。

**奖励:** 

* 停车等待：获得少量负奖励，因为浪费时间。
* 继续行驶：如果闯红灯，获得较大负奖励；如果顺利通过，获得少量正奖励。

**Q-learning 算法流程:**

1. 初始化 Q-value 表，将所有状态-动作对的 Q-value 初始化为 0。
2. 重复以下步骤，直到 Q-value 收敛：
    * **选择动作:**  根据当前状态和 Q-value 表，选择一个动作。
    * **执行动作:**  在环境中执行动作，观察下一个状态和奖励。
    * **更新 Q-value:**  根据贝尔曼方程更新 Q-value 表。
3. 最终获得最佳策略，即在每个状态下选择 Q-value 最大的动作。

**学习过程:**

* 在初始阶段，车辆可能随机选择动作，例如闯红灯，获得负奖励。
* 随着学习的进行，车辆会逐渐学习到在红灯状态下停车等待能够获得更高的长期累积奖励，因此会选择停车等待。
* 最终，车辆会学习到在不同状态下选择最佳动作，以最大化长期累积奖励。

### 4.4 常见问题解答

**问题 1: 如何选择合适的折扣因子 $\gamma$?**

* $\gamma$ 的值越大，表示智能体越重视长期奖励，反之则越重视即时奖励。
* 选择合适的 $\gamma$ 值需要根据具体问题进行调整，通常需要进行实验来找到最佳值。

**问题 2: 如何处理状态空间很大的问题?**

* 可以使用函数逼近方法来近似表示 Q-value 函数，例如神经网络。
* 可以使用经验回放 (Experience Replay) 技术，将历史经验存储起来，并随机抽取经验进行学习，从而提高学习效率。

**问题 3: 如何解决探索-利用困境?**

* 可以使用 ε-贪婪策略，在探索和利用之间取得平衡。
* 可以使用其他探索策略，例如 Boltzmann 探索、UCB 探索等。

## 5. 项目实践：代码实例和详细解释说明

### 5.1 开发环境搭建

**环境:**

* Python 3.x
* TensorFlow 2.x
* NumPy
* Matplotlib

**安装:**

```bash
pip install tensorflow numpy matplotlib
```

### 5.2 源代码详细实现

```python
import numpy as np
import matplotlib.pyplot as plt

# 定义状态空间
states = [0, 1, 2, 3, 4]

# 定义动作空间
actions = [0, 1]  # 0: 停车等待，1: 继续行驶

# 定义奖励函数
def reward_function(state, action):
    if state == 0 and action == 1:
        return -10  # 闯红灯
    elif state == 0 and action == 0:
        return -1  # 停车等待
    elif state == 4 and action == 1:
        return 1  # 顺利通过
    else:
        return 0

# 定义转移概率
def transition_probability(state, action, next_state):
    if state == 0 and action == 1:
        return 0.8 if next_state == 4 else 0.2 if next_state == 0 else 0
    elif state == 0 and action == 0:
        return 1 if next_state == 0 else 0
    elif state == 1 and action == 1:
        return 0.9 if next_state == 2 else 0.1 if next_state == 1 else 0
    elif state == 1 and action == 0:
        return 1 if next_state == 1 else 0
    elif state == 2 and action == 1:
        return 0.9 if next_state == 3 else 0.1 if next_state == 2 else 0
    elif state == 2 and action == 0:
        return 1 if next_state == 2 else 0
    elif state == 3 and action == 1:
        return 1 if next_state == 4 else 0
    elif state == 3 and action == 0:
        return 1 if next_state == 3 else 0
    else:
        return 0

# 定义折扣因子
gamma = 0.9

# 定义学习率
alpha = 0.1

# 初始化 Q-value 表
Q = {}
for state in states:
    for action in actions:
        Q[(state, action)] = 0

# 定义 ε-贪婪策略
def epsilon_greedy_policy(state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(actions)
    else:
        return get_best_action(state)

# 获取最佳动作
def get_best_action(state):
    best_action = None
    max_q_value = float('-inf')
    for action in actions:
        if Q[(state, action)] > max_q_value:
            best_action = action
            max_q_value = Q[(state, action)]
    return best_action

# 训练 Q-learning 算法
def train_q_learning(num_episodes, epsilon):
    for episode in range(num_episodes):
        state = 0  # 初始化状态
        while state != 4:
            action = epsilon_greedy_policy(state, epsilon)
            next_state = np.random.choice(states, p=[transition_probability(state, action, next_state) for next_state in states])
            reward = reward_function(state, action)
            Q[(state, action)] = (1 - alpha) * Q[(state, action)] + alpha * (reward + gamma * max([Q[(next_state, a)] for a in actions]))
            state = next_state
        print(f"Episode {episode+1} completed.")

# 运行训练过程
train_q_learning(1000, 0.1)

# 测试最佳策略
state = 0
while state != 4:
    action = get_best_action(state)
    print(f"State: {state}, Action: {action}")
    state = np.random.choice(states, p=[transition_probability(state, action, next_state) for next_state in states])

# 绘制 Q-value 表
plt.figure(figsize=(10, 5))
for state in states:
    for action in actions:
        plt.bar(f"State: {state}, Action: {action}", Q[(state, action)])
plt.title("Q-value Table")
plt.xlabel("State-Action Pair")
plt.ylabel("Q-value")
plt.show()
```

### 5.3 代码解读与分析

* **代码结构:** 代码首先定义了状态空间、动作空间、奖励函数、转移概率、折扣因子和学习率。
* **Q-value 表:**  使用字典 `Q` 来存储状态-动作对的 Q-value。
* **ε-贪婪策略:**  使用 `epsilon_greedy_policy` 函数实现 ε-贪婪策略，以概率 ε 选择随机动作，以概率 1-ε 选择 Q-value 最大的动作。
* **训练过程:**  使用 `train_q_learning` 函数进行训练，在每个 episode 中，从初始状态开始，根据 ε-贪婪策略选择动作，执行动作，观察下一个状态和奖励，并根据贝尔曼方程更新 Q-value。
* **测试过程:**  使用 `get_best_action` 函数获取最佳策略，并根据最佳策略进行测试，观察车辆的行动轨迹。
* **可视化:**  使用 `matplotlib` 库绘制 Q-value 表，以可视化 Q-learning 算法的学习结果。

### 5.4 运行结果展示

**训练过程:**

```
Episode 1 completed.
Episode 2 completed.
...
Episode 1000 completed.
```

**测试过程:**

```
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0
State: 0, Action: 0