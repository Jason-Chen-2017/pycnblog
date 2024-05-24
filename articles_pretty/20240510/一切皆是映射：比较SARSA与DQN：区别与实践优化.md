# 一切皆是映射：比较SARSA与DQN：区别与实践优化

作者：禅与计算机程序设计艺术

## 1. 背景介绍
   
### 1.1 强化学习的FrameWork

强化学习(Reinforcement Learning, RL)是一种机器学习方法,其目标是让Agent学会在环境中采取行动,从而最大化累积奖励。RL框架由Agent、Environment、State、Action、Reward等组成。在每一个离散的时间步t,Agent基于当前状态$s_t \in S$选择一个动作$a_t \in A$执行,环境接收到动作后更新为新状态$s_{t+1}$并反馈奖励$r_{t+1} \in R$。RL算法的优化目标是最大化期望累积奖励 $\mathbb{E}[\sum_{t=0}^{\infty }\gamma^t r_{t+1}]$。

### 1.2 从MDP到近似算法

许多RL问题可建模为马尔可夫决策过程(Markov Decision Process,MDP)。解决MDP的经典算法有值迭代(Value Iteration)、策略迭代(Policy Iteration)等动态规划算法,但它们通常需要对状态转移概率$P(s'|s,a)$和奖励函数$R(s,a)$有完整的了解,在实际问题中往往不可行。Q-Learning、SARSA、DQN等近似算法可克服这一困难。

### 1.3 本文的主要内容

本文将重点介绍Q-Learning、SARSA和DQN三种算法的基本概念、关键思想、工作原理,详细比较它们的异同点,分析优缺点。同时,给出具体实现代码与应用案例,探讨工程实践中的常见问题与优化技巧。最后,展望强化学习在未来的发展方向与挑战。

## 2. 核心概念与联系
  
### 2.1 Q函数与价值估计

Q函数(Action-value Function)对应在当前状态$s$下选择动作$a$可获得的期望累积奖励:

$$
Q^{\pi}(s,a) = \mathbb{E}_{\pi} \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \mid S_t = s, A_t = a \right] 
$$

最优Q函数$Q^{*}(s,a)$代表所有策略$\pi$下的最大期望回报。 很多RL算法都需要对Q函数进行估计,代表性方法有MC(Monte-Carlo)、TD(Temporal-Difference)等。

### 2.2 Exploration与Exploitation

RL需要兼顾Exploration(探索)与Exploitation(利用)。探索是指尝试新的动作以发现潜在的高回报,而利用则倾向于选择已知的高回报动作。常见的平衡办法有$\epsilon$-贪心、UCB等。一个好的策略需要在探索和利用之间平衡。
  
### 2.3 On-policy与Off-policy  

On-policy方法通过与环境交互的轨迹数据来学习和评估目标策略,即策略与行为一致。代表性算法有SARSA、Actor-Critic等。

Off-policy方法学习一个不同于采样轨迹的策略,即评估和行为策略不同,可利用replay buffer重用历史数据,常见算法如Q-Learning、DQN等。

On-policy方法通常简单稳定,而off-policy可更高效地利用历史数据,但可能不稳定。二者可互补结合。 
   
## 3. 核心算法原理 
   
### 3.1 Q-Learning
  
Q-Learning是一种off-policy的时序差分算法,可从采样轨迹中学习最优Q函数。其更新公式为:

$$
Q(s,a) \leftarrow Q(s,a)+\alpha \left[ r + \gamma \max_{a^{\prime}}Q(s^{\prime},a^{\prime})-Q(s,a) \right]
$$

其中$\alpha \in (0,1] $是学习率,$\gamma \in [0,1)$为折扣因子。Q-Learning的目标是直接学习$Q^*$函数。其优点是简单有效,容易实现,但状态和动作空间较大时可能效率低下。
  
### 3.2 SARSA
   
SARSA(State-Action-Reward-State-Action)是一种on-policy TD控制算法。其更新公式为:

$$
Q(s_t,a_t) \leftarrow Q(s_t,a_t)+\alpha \left[ r_{t+1} +  \gamma Q(s_{t+1},a_{t+1})-Q(s_t,a_t) \right]
$$

SARSA用采样轨迹上实际采取的动作$a_{t+1}$来更新,体现了on-policy的特点。相比Q-Learning,SARSA更稳健,但收敛到的策略可能不是全局最优的。
  
### 3.3 DQN
  
DQN(Deep Q-Network)使用深度神经网络去逼近Q函数,可处理高维状态空间。DQN的损失函数为:
  
$$
L=\mathbb{E}_{(s,a,r,s^{\prime}) \sim D} \left[  (r+\gamma \max_{a^{\prime}} \hat{Q}(s^{\prime},a^{\prime}) - Q(s,a))^2 \right]
$$

DQN采用经验回放(Experience Replay)与Fixed-Q-Target等技巧来提高训练的稳定性。DQN在Atari、Alpha Go等复杂控制任务上取得了巨大成功。

## 4. 数学模型与公式推导  

### 4.1 Bellman最优方程

最优Q函数满足Bellman最优方程:
 
$$
Q^*(s,a)=\mathbb{E}_{s^{\prime} \sim P} \left[ r(s,a) + \gamma \max_{a^{\prime}}Q^*(s^{\prime},a^{\prime}) \right]
$$

可以证明,贝尔曼最优方程的唯一不动点就是$Q^*$。Q-Learning基于此方程设计,通过样本均值逼近期望,从而逐步收敛到$Q^*$。

**推导**:假设当前已知$Q$函数的估计值$\hat{Q}$,根据贝尔曼方程,可得:

$$
\begin{aligned}
\hat{Q}(s,a) &\approx r(s,a) + \gamma \mathbb{E}_{s^{\prime}\sim P} \max_{a^{\prime}}\hat{Q}(s^{\prime},a^{\prime})\\
            &\approx r + \gamma \max_{a^{\prime}}\hat{Q}(s^{\prime},a^{\prime}) 
\end{aligned}
$$
  
上式即为Q-Learning的更新目标(target)。通过最小化估计值$\hat{Q}(s,a)$与target的均方误差,可逐步逼近$Q^*$。这就是Q-Learning的数学原理。 

### 4.2 SARSA收敛性证明
   
**定理** 考虑一个有限的MDP。对任意确定性的策略$\pi$,SARSA算法以概率1收敛到 $Q_{\pi}$。 

**证明思路**: 将SARSA算法的更新过程表示为一个加权最大范数压缩映射,利用不动点定理证明其收敛性。限于篇幅,此处略去详细证明,感兴趣的读者可参考这篇论文[1]。

### 4.3 DQN中的经验回放
  
经验回放(Experience Replay)将历史的转移样本$(s,a,r,s')$存入Replay Buffer $D$中,之后从$D$中随机采样小批量数据进行Q网络的参数更新。这种做法有如下优点:
- 打破了数据的时序关联性,减少训练的振荡。  
- 提高数据利用效率,加速收敛。
- 可并行采样与计算,提高训练效率。

其数学本质是用经验分布$\rho(s,a)$替代了真实的转移概率$p(s'|s,a)$进行采样学习。在一定条件下可证明,这种近似并不影响DQN的渐进一致性。
   
## 5. 算法实现与代码讲解
  
为了加深理解,下面给出Q-Learning、SARSA和DQN在经典的网格世界(GridWorld)上的Python实现。

### 5.1 Q-Learning 实现

```python
import numpy as np

# 初始化Q表
Q = np.zeros((state_size, action_size)) 

# 训练循环
for episode in range(max_episodes):
    state = env.reset()
    done = False
    
    while not done:
        # 依据Q表选择动作(epsilon-greedy)
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        
        next_state, reward, done, _ = env.step(action) 
        
        # Q-Learning更新
        target = reward + gamma * np.max(Q[next_state])
        Q[state][action] += alpha * (target - Q[state][action])
        
        state = next_state
```

这里的关键是Q表的更新规则`Q[s][a] += alpha * (target - Q[s][a])`。`target`项对应了Q-Learning的目标值,体现了off-policy的思想。

### 5.2 SARSA 实现

```python
# 训练循环 
for episode in range(max_episodes): 
    state = env.reset()
    done = False
    
    # 依据Q表选择第一个动作(epsilon-greedy)
    if np.random.rand() < epsilon:
        action = env.action_space.sample() 
    else:
        action = np.argmax(Q[state])
    
    while not done:
        next_state, reward, done, _ = env.step(action)
        
        # 依据Q表选择下一个动作(epsilon-greedy) 
        if np.random.rand() < epsilon:
            next_action = env.action_space.sample()
        else: 
            next_action = np.argmax(Q[next_state])
        
        # SARSA更新
        target = reward + gamma * Q[next_state][next_action]
        Q[state][action] += alpha * (target - Q[state][action])
        
        state = next_state
        action = next_action
```

SARSA的特点是更新时使用了实际采取的下一个动作$a_{t+1}$对应的Q值,即`Q[next_state][next_action]`,体现了on-policy。

### 5.3 DQN 实现

```python
import torch
import numpy as np
from collections import deque 
import random

class DQN(torch.nn.Module):
    
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(state_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64), 
            torch.nn.ReLU(),
            torch.nn.Linear(64, action_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

# 创建Q网络和目标网络    
Q = DQN(state_dim, action_dim).to(device)
Q_target = DQN(state_dim, action_dim).to(device)
Q_target.load_state_dict(Q.state_dict())

# 初始化 replay buffer
replay_buffer = deque(maxlen=buffer_size)

# 训练循环
for episode in range(max_episodes): 
    
    state = env.reset()
    done = False
    
    while not done:
        # epsilon-greedy 选择动作
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            with torch.no_grad(): 
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                q_values = Q(state_tensor).squeeze()
                action = q_values.argmax().item()
        
        # 执行动作
        next_state, reward, done, _ = env.step(action)
        
        # 存入 replay buffer
        replay_buffer.append((state, action, reward, next_state, done))
        
        state = next_state
        
        # 从buffer中采样并更新
        if len(replay_buffer) >= batch_size:
            
            samples = random.sample(replay_buffer, batch_size)
            
            states, actions, rewards, next_states, dones = zip(*samples)
            
            state_batch = torch.tensor(states, dtype=torch.float32).to(device)
            action_batch = torch.tensor(actions, dtype=torch.long).to(device)
            reward_batch = torch.tensor(rewards, dtype=torch.float32).to(device)
            next_state_batch = torch.tensor(next_states, dtype=torch.float32).to(device)
            done_batch = torch.tensor(dones, dtype=torch.float32).to(device)
            
            # 计算Q(s,a)
            state_action_values = Q(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
            
            # 计算max Q(s',a')
            next_state_values = torch.max(Q_target(next_state_batch), 1)[0]
            
            # 计算期望Q值
            expected_state_action_values = reward_batch + (1 - done_batch) * gamma * next_state_values
            
            # 计算MSE Loss
            loss = torch.nn.MSELoss()(state_action_values, expected_state_action_values)
            
            # 优化模型
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    # 每C步同步Q_target网络参数        
    if episode % C == 0:
        Q_target.load_state