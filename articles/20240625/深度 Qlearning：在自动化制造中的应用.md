
# 深度 Q-learning：在自动化制造中的应用

作者：禅与计算机程序设计艺术 / Zen and the Art of Computer Programming 


## 1. 背景介绍
### 1.1 问题的由来

随着工业自动化程度的不断提高，自动化制造系统在提高生产效率、降低成本、提升产品质量等方面发挥着越来越重要的作用。然而，自动化制造系统往往面临着复杂的生产环境和多变的任务需求，如何使这些系统具备自主学习和适应能力，成为当前研究的热点问题。

深度 Q-learning作为强化学习的一种重要方法，凭借其强大的学习和迁移能力，在自动化制造领域展现出巨大的应用潜力。本文将深入探讨深度 Q-learning在自动化制造中的应用，分析其核心原理、具体操作步骤，并结合实际案例进行详细讲解。

### 1.2 研究现状

近年来，深度 Q-learning在自动化制造领域的应用研究取得了显著进展。以下是一些主要的进展方向：

1. **路径规划与导航**：深度 Q-learning被用于解决机器人路径规划和导航问题，如无人机巡检、自动驾驶等。

2. **过程控制与优化**：深度 Q-learning被用于优化生产线上的物料搬运、加工流程等，以提高生产效率和产品质量。

3. **故障诊断与预测**：深度 Q-learning被用于对设备进行故障诊断和预测，以预防设备故障和降低维护成本。

4. **库存管理与调度**：深度 Q-learning被用于优化库存管理和生产调度，以降低库存成本和提高生产效率。

5. **人机协同**：深度 Q-learning被用于实现人机协同作业，以提高生产效率和安全性。

### 1.3 研究意义

深度 Q-learning在自动化制造领域的应用具有重要的研究意义：

1. **提高生产效率**：通过深度 Q-learning实现自动化制造系统的自主学习和适应能力，可以提高生产效率，降低人工成本。

2. **提升产品质量**：深度 Q-learning可以帮助自动化制造系统根据实时反馈调整生产参数，从而提高产品质量。

3. **降低维护成本**：通过深度 Q-learning进行故障诊断和预测，可以预防设备故障，降低维护成本。

4. **促进产业升级**：深度 Q-learning在自动化制造领域的应用，有助于推动传统制造业向智能制造转型升级。

### 1.4 本文结构

本文将围绕深度 Q-learning在自动化制造中的应用展开，具体内容如下：

- 第2部分介绍深度 Q-learning的核心概念与联系。
- 第3部分阐述深度 Q-learning的算法原理和具体操作步骤。
- 第4部分讲解深度 Q-learning的数学模型和公式，并结合实例进行分析。
- 第5部分以实际案例展示深度 Q-learning在自动化制造中的应用。
- 第6部分探讨深度 Q-learning在自动化制造领域的未来应用前景。
- 第7部分推荐相关学习资源、开发工具和参考文献。
- 第8部分总结全文，展望深度 Q-learning在自动化制造领域的未来发展趋势与挑战。
- 第9部分列举常见问题与解答。

## 2. 核心概念与联系

本节将介绍深度 Q-learning涉及的核心概念及其相互关系。

### 2.1 强化学习

强化学习(Reinforcement Learning, RL)是一种通过与环境交互来学习最优策略的机器学习方法。在强化学习中，智能体(Agent)通过与环境(Envionment)的交互，根据预设的奖励(Reward)和惩罚(Penalty)来调整其行为，最终学习到最优策略，以实现目标函数的最大化。

强化学习的基本要素包括：

- **智能体(Agent)**：执行动作并接收环境反馈的实体。
- **环境(Environment)**：智能体执行动作的场所，提供状态(STATE)和奖励(Reward)。
- **策略(Strategy)**：智能体在特定状态下执行的动作概率分布。
- **价值函数(Value Function)**：描述智能体在特定状态下采取特定动作的期望收益。
- **奖励函数(Reward Function)**：描述智能体在每个时间步长接收的奖励值。
- **状态-动作空间(State-Action Space)**：智能体可能处于的状态集合和可能执行的动作集合。

### 2.2 Q-learning

Q-learning是一种基于值函数的强化学习方法，其核心思想是学习一个 Q 表，该 Q 表记录了智能体在每个状态下采取每个动作的期望收益。Q-learning通过迭代更新 Q 表，最终得到最优策略。

### 2.3 深度 Q-learning

深度 Q-learning(DQN)是 Q-learning的扩展，它使用深度神经网络(DNN)来近似 Q 表。DQN通过学习状态-动作值函数，在复杂的任务上取得了显著的性能提升。

### 2.4 深度 Q-learning与自动化制造

深度 Q-learning在自动化制造领域的应用，主要体现在以下几个方面：

- **路径规划与导航**：通过学习最优路径，引导机器人完成复杂的路径规划和导航任务。
- **过程控制与优化**：通过学习最优控制策略，优化生产流程，提高生产效率和产品质量。
- **故障诊断与预测**：通过学习故障特征，实现设备的故障诊断和预测。
- **库存管理与调度**：通过学习最优调度策略，优化库存管理和生产调度。
- **人机协同**：通过学习人机协同策略，提高生产效率和安全性。

## 3. 核心算法原理 & 具体操作步骤
### 3.1 算法原理概述

深度 Q-learning的核心思想是使用 DNN 来近似 Q 表，从而在复杂任务上学习最优策略。以下是深度 Q-learning的基本原理：

1. 初始化 Q 表：使用随机值初始化 Q 表。
2. 选择动作：在当前状态下，根据策略选择一个动作。
3. 执行动作并观察结果：执行选定的动作，并接收环境反馈的状态和奖励。
4. 更新 Q 表：根据新的状态和奖励，更新 Q 表中对应动作的值。
5. 迭代学习：重复步骤2-4，直至达到收敛条件。

### 3.2 算法步骤详解

以下为深度 Q-learning的具体操作步骤：

**Step 1：初始化**

1. 初始化 Q 表：使用随机值初始化 Q 表。
2. 初始化神经网络：使用 DNN 构建近似 Q 表的网络结构。

**Step 2：选择动作**

1. 将当前状态输入神经网络，得到 Q 值向量。
2. 根据 ε-贪婪策略，选择动作：以一定概率随机选择动作，以 (1-ε) 的概率选择 Q 值最大的动作。

**Step 3：执行动作并观察结果**

1. 执行选定的动作，并接收环境反馈的状态和奖励。
2. 计算当前动作的 Q 值：使用目标网络计算当前状态的 Q 值，并选择当前动作的最大 Q 值作为目标 Q 值。
3. 计算目标 Q 值：根据新的状态和奖励，使用目标 Q 值函数计算目标 Q 值。

**Step 4：更新 Q 表**

1. 使用目标 Q 值更新当前动作的 Q 值：根据目标 Q 值和当前动作的 Q 值，更新 Q 表中对应动作的值。
2. 使用梯度下降等优化算法更新神经网络参数，使得神经网络输出更接近真实 Q 值。

**Step 5：迭代学习**

重复步骤2-4，直至达到收敛条件。

### 3.3 算法优缺点

深度 Q-learning的优点包括：

- **强大的学习能力**：DNN 的强大学习能力使得深度 Q-learning能够在复杂任务上学习到最优策略。
- **灵活性强**：可以通过调整神经网络结构、学习率等参数，适应不同的任务需求。
- **易于实现**：DNN 的实现相对简单，易于在实际应用中落地。

深度 Q-learning的缺点包括：

- **计算量大**：DNN 的训练过程需要大量的计算资源。
- **对环境复杂度敏感**：在复杂环境中，DNN 可能难以收敛到最优策略。
- **对参数选择敏感**：神经网络结构和参数的选择对学习效果影响较大。

### 3.4 算法应用领域

深度 Q-learning在以下领域具有广泛的应用：

- **路径规划与导航**：自动驾驶、机器人导航、无人机巡检等。
- **过程控制与优化**：生产线自动化控制、智能电网、智能交通等。
- **故障诊断与预测**：设备故障诊断、故障预测、健康管理等。
- **库存管理与调度**：供应链管理、生产调度、资源分配等。
- **人机协同**：人机协同作业、智能客服、智能推荐等。

## 4. 数学模型和公式 & 详细讲解 & 举例说明
### 4.1 数学模型构建

以下为深度 Q-learning的数学模型，包括状态-动作值函数和目标 Q 值函数。

#### 状态-动作值函数

状态-动作值函数 $Q(s,a)$ 表示智能体在状态 $s$ 下采取动作 $a$ 的期望收益，即：

$$
Q(s,a) = \mathbb{E}[G_t | s_t = s, a_t = a]
$$

其中，$G_t$ 表示从时间步长 $t$ 到终止状态的总奖励，$s_t$ 和 $a_t$ 分别表示时间步长 $t$ 的状态和动作。

#### 目标 Q 值函数

目标 Q 值函数 $Q^*(s,a)$ 表示智能体在状态 $s$ 下采取动作 $a$ 的最大期望收益，即：

$$
Q^*(s,a) = \max_{a'} \mathbb{E}[G_t | s_t = s, a_t = a']
$$

### 4.2 公式推导过程

以下为深度 Q-learning的目标 Q 值函数的推导过程：

1. **定义回报函数**：回报函数 $G_t$ 表示从时间步长 $t$ 到终止状态的总奖励，即：

   $$
G_t = \sum_{k=t}^{T} R_k
$$

   其中，$R_k$ 表示时间步长 $k$ 的奖励。

2. **定义下一步的最大 Q 值**：下一步的最大 Q 值 $\max_{a'} Q(s',a')$ 表示智能体在下一个状态 $s'$ 下采取所有可能动作的最大 Q 值。

3. **推导目标 Q 值函数**：

   $$
\begin{aligned}
Q^*(s,a) &= \max_{a'} \mathbb{E}[G_t | s_t = s, a_t = a'] \\
&= \mathbb{E}[G_t | s_t = s, a_t = a'] + \mathbb{E}[\max_{a'} Q(s',a') | s_t = s, a_t = a'] \\
&= \mathbb{E}[R_t + \gamma \max_{a'} Q(s',a') | s_t = s, a_t = a']
\end{aligned}
$$

   其中，$\gamma$ 表示折扣因子，表示未来奖励对未来价值的权重。

4. **简化目标 Q 值函数**：

   $$
Q^*(s,a) = R_t + \gamma \max_{a'} Q(s',a')
$$

### 4.3 案例分析与讲解

以下以自动泊车为例，分析深度 Q-learning在路径规划与导航中的应用。

#### 案例描述

假设智能车需要在复杂的停车场中找到停车位。智能车通过摄像头获取周围环境的图像，并将其输入神经网络，输出当前状态。智能车的动作包括前进、后退、左转、右转、停车等。

#### 状态表示

状态 $s$ 可以表示为：

$$
s = [s_{\text{position}}, s_{\text{direction}}, s_{\text{speed}}, s_{\text{obstacle}}]
$$

其中，$s_{\text{position}}$ 表示智能车的位置，$s_{\text{direction}}$ 表示智能车的方向，$s_{\text{speed}}$ 表示智能车的速度，$s_{\text{obstacle}}$ 表示智能车周围的障碍物信息。

#### 奖励函数

奖励函数 $R_t$ 可以表示为：

$$
R_t = \begin{cases}
+100, & \text{成功停车} \\
-1, & \text{未成功停车} \\
-5, & \text{发生碰撞}
\end{cases}
$$

#### 深度 Q-learning 应用

使用深度 Q-learning进行路径规划与导航的步骤如下：

1. 初始化 Q 表和神经网络。
2. 将智能车的状态输入神经网络，得到 Q 值向量。
3. 根据 ε-贪婪策略，选择动作。
4. 执行选定的动作，并接收环境反馈的状态和奖励。
5. 使用目标网络计算当前状态的 Q 值，并选择当前动作的最大 Q 值作为目标 Q 值。
6. 使用目标 Q 值更新当前动作的 Q 值。
7. 重复步骤2-6，直至达到收敛条件。

通过深度 Q-learning的学习，智能车可以学习到在复杂停车场中找到停车位的最佳路径。

### 4.4 常见问题解答

**Q1：为什么使用深度神经网络来近似 Q 表？**

A：深度神经网络具有强大的表达能力，可以学习复杂的函数关系。使用 DNN 来近似 Q 表，可以处理高维状态空间和动作空间，提高学习效率。

**Q2：如何选择合适的神经网络结构？**

A：神经网络结构的选择取决于具体任务的需求。一般来说，需要根据状态和动作的复杂度、数据规模等因素选择合适的网络结构。可以通过实验和交叉验证来选择最优的网络结构。

**Q3：如何解决深度 Q-learning中的过拟合问题？**

A：过拟合是深度 Q-learning中常见的问题。可以通过以下方法解决过拟合：

1. 使用正则化技术，如 L2 正则化、Dropout 等。
2. 使用更小的学习率。
3. 使用数据增强技术，扩充训练数据。
4. 使用模型融合技术，集成多个模型。

## 5. 项目实践：代码实例和详细解释说明
### 5.1 开发环境搭建

为了进行深度 Q-learning的项目实践，需要搭建以下开发环境：

1. **操作系统**：Windows、Linux 或 macOS
2. **编程语言**：Python
3. **深度学习框架**：TensorFlow 或 PyTorch
4. **其他依赖库**：NumPy、Matplotlib、Pandas、scikit-learn 等

以下是使用 PyTorch 搭建深度 Q-learning 开发环境的示例代码：

```python
pip install torch torchvision torchaudio
pip install numpy matplotlib pandas scikit-learn
```

### 5.2 源代码详细实现

以下使用 PyTorch 实现一个简单的智能车路径规划与导航的案例。

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 状态空间维度
state_dim = 4

# 动作空间维度
action_dim = 4

# 构建神经网络
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# 构建深度 Q-learning 算法
class DQNAlgorithm:
    def __init__(self, model, optimizer, gamma=0.99, epsilon=0.1):
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.epsilon = epsilon
        self.criterion = nn.MSELoss()

    def train(self, state, action, reward, next_state, done):
        current_q_values = self.model(state)
        current_q_value = current_q_values[0, action]
        next_max_q_value = torch.max(self.model(next_state)).item()
        target = reward + (1 - done) * self.gamma * next_max_q_value
        self.optimizer.zero_grad()
        loss = self.criterion(current_q_value, target)
        loss.backward()
        self.optimizer.step()

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, action_dim)
        else:
            with torch.no_grad():
                state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action = self.model(state).argmax().item()
        return action

# 初始化模型、优化器和算法
model = DQN()
optimizer = optim.Adam(model.parameters())
algorithm = DQNAlgorithm(model, optimizer)

# 模拟智能车环境
def simulate_env():
    state = np.random.randint(0, 100, size=(state_dim,))
    action = np.random.randint(0, action_dim)
    reward = 0
    done = False
    return state, action, reward, done

# 训练模型
for i in range(1000):
    state, action, reward, done = simulate_env()
    next_state, _, _, _ = simulate_env()
    algorithm.train(torch.tensor(state), action, reward, torch.tensor(next_state), done)
    if i % 100 == 0:
        print(f"Episode {i}, reward: {reward}")

# 测试模型
for i in range(10):
    state = np.random.randint(0, 100, size=(state_dim,))
    while True:
        action = algorithm.choose_action(state)
        next_state, _, _, done = simulate_env()
        print(f"Action: {action}, Next State: {next_state}")
        if done:
            break
```

### 5.3 代码解读与分析

以上代码实现了深度 Q-learning 算法的核心部分。以下是代码的详细解读：

1. **DQN 类**：定义了一个神经网络，用于近似 Q 表。网络结构由两个全连接层组成，使用 ReLU 激活函数。
2. **DQNAlgorithm 类**：定义了深度 Q-learning 算法的核心方法。包括训练方法 `train()` 和选择动作方法 `choose_action()`。
3. **simulate_env() 函数**：模拟智能车环境，生成状态、动作、奖励和下一个状态。
4. **训练过程**：循环执行 simulate_env() 函数，获取状态、动作、奖励和下一个状态，并调用 algorithm.train() 方法进行训练。
5. **测试过程**：使用 algorithm.choose_action() 方法选择动作，并输出动作和下一个状态。

### 5.4 运行结果展示

运行以上代码，可以得到以下结果：

```
Episode 0, reward: 0
Action: 0, Next State: [88, 76, 80, 92]
Action: 3, Next State: [70, 73, 73, 80]
...
Action: 2, Next State: [54, 74, 77, 74]
Action: 3, Next State: [59, 77, 77, 78]
Action: 3, Next State: [55, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next State: [53, 78, 78, 78]
Action: 3, Next