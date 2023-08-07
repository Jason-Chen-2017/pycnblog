
作者：禅与计算机程序设计艺术                    

# 1.简介
         

强化学习（Reinforcement Learning，RL）是机器学习领域一个具有革命意义的热门方向。近年来，研究者们在这一领域取得了重大突破，取得了令人惊叹的成就，也给予了许多学习者和工程师以启发、鼓舞和激励。然而，由于强化学习的复杂性和庞大的算法空间，并非所有人都能很好地理解其工作机制、原理、作用及其可能出现的问题。所以，如何更好地传播和利用深度强化学习（Deep Reinforcement Learning，DRL）方面的知识，是一个值得关注的话题。

本文将以2013年乔治华盛顿大学Atari游戏机项目为例，阐述一下DRL在Atari游戏中的应用，讨论DRL在不同阶段的研究进展及其适用场景，探索一下DRL在计算机视觉、机器学习等领域的广阔前景，并给出一些具体方案及建议。希望通过阐述、分析和实践，能够为读者提供一些宝贵参考。

# 2.基本概念术语说明
## 2.1 概念说明
强化学习（Reinforcement Learning，RL）是一类用于解决决策问题的机器学习方法。它依赖于智能体与环境的交互，智能体以某种策略不断探索环境并获取奖励（即所期望的回报），从而根据策略的表现更新策略使之越来越优秀。它的基本想法是建立一个代理系统（agent）来进行自我学习，使得在有限的时间内完成任务。强化学习的一个关键特点是，智能体不需要事先知道环境的完整信息，只需要观察到环境中智能体可以感知到的信息，以及执行动作获得的奖励。与其他机器学习算法相比，强化学习的独特性在于其能够让智能体适应环境并选择最佳的动作。因此，强化学习被认为是一种新型的优化算法。

在强化学习过程中，智能体与环境之间的关系可以分为三个主要方面：状态（state）、动作（action）、奖励（reward）。状态描述的是智能体所处的当前环境状态；动作则是智能体用来影响环境变化的行为，是给出的外部输入；奖励则是智能体在执行某个动作时所获得的奖励。在RL中，每一次迭代，智能体都会采取一个动作并向环境反馈一个奖励，然后基于该反馈更新自己的动作策略，最终达到最优解。

RL的核心问题就是学习如何最大化累计收益（cumulative reward）。在每一步迭代中，智能体都会尝试不同的行动策略以获得最大的奖励。为了提高效率，智能体会在一定的训练周期内采用随机策略（exploration）试错，并逐渐变换到最佳策略（exploitation）。在训练中，智能体需要不断地探索，同时也要保证在已知的情况下获得较好的性能，这两者是相辅相成的。


## 2.2 术语说明
### 2.2.1 DQN
DQN（Deep Q-Network）是DeepMind团队在2013年的一项研究，其核心思想是利用神经网络自动化学习状态转移函数，并用神经网络作为Q-Function，实现端到端的强化学习过程。其特点是在完全的状态下直接预测目标输出（action），而不需要使用函数逼近的方法。此外，DQN通过重用网络结构来减少参数数量，有效地降低计算资源消耗。目前，DQN已经成为Atari游戏中的主流模型，并且在两个连续的大版本（版本7和版本9）上均显示出显著的成果。

### 2.2.2 DPG
DPG（Deterministic Policy Gradient，确定策略梯度）是DeepMind团队在2016年的一项研究，其核心思想是利用强化学习方法来训练能够同时兼顾快速响应和稳定性的智能体。DPG的算法流程与DQN类似，但对策略参数进行了约束，使得智能体只能选择确定的动作。其主要目的是为了克服DQN存在的冷启动问题。目前，DPG已被证明可用于一些复杂的控制问题，如机器人运动规划。

### 2.2.3 DDPG
DDPG（Deep Deterministic Policy Gradient，深层确定策略梯度）是DeepMind团队在2016年的一项研究，其特点是结合DQN和DPG的优点，提出了一个深层的确定策略网络来代替之前的浅层网络。其主要目的是为了克服DQN或DPG遇到的局部极小问题，提升稳定性、加速收敛。目前，DDPG已被证明可用于一些复杂的控制问题，如机器人运动规划。

### 2.2.4 A2C
A2C（Asynchronous Advantage Actor Critic，异步优势演员-评论家）是DeepMind团队在2016年的一项研究，其核心思想是提出异步SGD算法，并将A3C（Asynchronous Methods for Deep Reinforcement Learning，异步深度强化学习的异步方法）的思路引入DQN中。A2C的算法流程包括收集数据、更新策略网络、评估策略网络、更新参数和保存模型，与DQN基本相同。其主要目的也是为了克服DQN的单步样本更新方式导致的局部波动问题，提升性能、加速收敛。目前，A2C已被证明可用于一些复杂的控制问题，如机器人运动规划。

### 2.2.5 PPO
PPO（Proximal Policy Optimization，近端策略优化）是OpenAI团队在2017年的一项研究，其核心思想是通过设定损失函数中的KL散度限制来控制策略搜索范围，以达到平衡 exploration 和 exploitation 的平衡。其基本流程与之前的模型几乎一致，但通过增加一个熵的惩罚项来控制策略的多样性，从而提高学习效率。PPO已被证明可用于一些复杂的控制问题，如机器人运动规划。

### 2.2.6 TRPO
TRPO（Trust Region Policy Optimization，信赖域策略优化）是斯坦福大学团队在2015年的一项研究，其核心思想是通过在策略搜索范围内添加限制条件来避免陷入局部最优解。TRPO利用强化学习中的KL散度来表示不同动作之间的联系，并通过控制参数空间的大小来提高探索效率。其基本流程与之前的模型几乎一致，但增加了一个惩罚项来控制策略的多样性，从而防止模型过拟合。TRPO已被证明可用于一些复杂的控制问题，如机器人运动规划。

### 2.2.7 ACER
ACER（Actor-Critic with Experience Replay，带经验重放的演员-评论家）是斯坦福大学团队在2016年的一项研究，其核心思想是改进基于时间差分的Q-learning的方法，提出经验回放的方法来增强训练样本。ACER的算法流程与DQN基本一致，但引入了经验回放的方法，充分利用了记忆中的信息，减少样本效率下降。目前，ACER已被证明可用于一些复杂的控制问题，如机器人运动规划。

# 3.核心算法原理和具体操作步骤以及数学公式讲解
本节将详细介绍DQN、DPG、DDPG、A2C、PPO、TRPO、ACER的基本原理、特点和设计思路，并结合具体的操作步骤以及数学公式，帮助读者更好地理解其工作机制。

## 3.1 DQN
DQN（Deep Q-Network）是DeepMind团队在2013年的一项研究，其核心思想是利用神经网络自动化学习状态转移函数，并用神经网络作为Q-Function，实现端到端的强化学习过程。其特点是在完全的状态下直接预测目标输出（action），而不需要使用函数逼近的方法。此外，DQN通过重用网络结构来减少参数数量，有效地降低计算资源消耗。目前，DQN已经成为Atari游戏中的主流模型，并且在两个连续的大版本（版本7和版本9）上均显示出显著的成果。

在Atari游戏中，每局游戏都由若干个屏幕（screen）组成，每个屏幕代表一个RGB像素点集合，整个游戏画面可以由多个这样的屏幕叠加得到。而对于一个给定的智能体来说，它接收到的信息只有当前屏幕的图像信息，以及智能体可以执行的动作列表。DQN的目标就是利用这个信息来学习到一个映射，把每张屏幕上的像素点转换为相应的动作，从而使得智能体在游戏中执行出合适的动作。

首先，DQN由两个组件组成：经验池（replay memory）和神经网络。其中，经验池用于存储游戏数据，包括训练数据、游戏状态（screen）、奖励（reward）、是否结束（done）等。神经网络是一个Q-network，它由两部分组成，分别是特征网络（feature network）和动作网络（action network）。特征网络接受当前屏幕图像作为输入，提取出有用的特征，并送至动作网络进行处理。动作网络接收特征网络输出，生成各个动作对应的Q值，再根据Q值选出最优动作。

其次，DQN采取的算法框架是完全基于Q-Learning的。Q-learning是一个值迭代算法，它将问题表示为一个马尔科夫决策过程（Markov Decision Process，MDP），其中智能体与环境的交互定义为状态-动作空间的MDP，目标是找到一个状态动作值函数（State Action Value Function，Q-function），使得智能体可以在后续状态下选择最优动作。DQN借鉴了Q-learning的数学原理，将状态转移概率建模成一个价值网络（value network）和动作选择概率建模成一个策略网络（policy network）。通过最小化Q-learning中的Bellman方程，DQN能够在经验池上自动学习到最优策略。

最后，DQN利用神经网络的自动学习能力来提升学习效率，比如利用重用网络结构来减少参数数量，通过固定目标网络来降低学习风险，利用目标网络的目标来降低更新频率，等等。通过这些技巧，DQN在两个连续的大版本（版本7和版本9）上均显示出显著的成果。

## 3.2 DPG
DPG（Deterministic Policy Gradient，确定策略梯度）是DeepMind团队在2016年的一项研究，其核心思想是利用强化学习方法来训练能够同时兼顾快速响应和稳定性的智能体。DPG的算法流程与DQN类似，但对策略参数进行了约束，使得智能体只能选择确定的动作。其主要目的是为了克服DQN存在的冷启动问题。目前，DPG已被证明可用于一些复杂的控制问题，如机器人运动规划。

首先，DPG与DQN的主要区别在于，DPG中的策略网络不能用Q值来选动作，而是直接输出确定性的动作分布。其次，DPG还加入了正则化项，用来限制策略的偏差。最后，DPG将策略网络的梯度上升步长替换为对抗训练中的梯度上升步长，在一定程度上缓解收敛困难。

## 3.3 DDPG
DDPG（Deep Deterministic Policy Gradient，深层确定策略梯度）是DeepMind团队在2016年的一项研究，其特点是结合DQN和DPG的优点，提出了一个深层的确定策略网络来代替之前的浅层网络。其主要目的是为了克服DQN或DPG遇到的局部极小问题，提升稳定性、加速收敛。目前，DDPG已被证明可用于一些复杂的控制问题，如机器人运动规划。

与DPG一样，DDPG中的策略网络不能用Q值来选动作，而是直接输出确定性的动作分布。其主要区别在于，DDPG构建了一个基于目标网络的目标函数，并结合DQN中经验回放的方法来增强训练样本。最后，DDPG的更新步长设置为基于对抗训练的固定步长，即使得网络始终能够收敛。

## 3.4 A2C
A2C（Asynchronous Advantage Actor Critic，异步优势演员-评论家）是DeepMind团队在2016年的一项研究，其核心思想是提出异步SGD算法，并将A3C（Asynchronous Methods for Deep Reinforcement Learning，异步深度强化学习的异步方法）的思路引入DQN中。A2C的算法流程包括收集数据、更新策略网络、评估策略网络、更新参数和保存模型，与DQN基本相同。其主要目的也是为了克服DQN的单步样本更新方式导致的局部波动问题，提升性能、加速收敛。目前，A2C已被证明可用于一些复杂的控制问题，如机器人运动规划。

与DQN一样，A2C由两个组件组成：经验池（replay memory）和神经网络。经验池用于存储游戏数据，包括训练数据、游戏状态（screen）、奖励（reward）、是否结束（done）等。神经网络由两部分组成，分别是特征网络（feature network）和动作网络（action network）。特征网络接收当前屏幕图像作为输入，提取出有用的特征，并送至动作网络进行处理。动作网络接收特征网络输出，生成各个动作对应的Q值，再根据Q值选出最优动作。

但是，A2C与DQN的不同之处在于，A2C在算法流程上引入了多个线程并行处理的方式，来提升效率。具体来说，A2C在收集数据时，可以并行地收集多条轨迹（trajectory）的数据，并统一整理存入经验池。而在神经网络的更新时，A2C采用异步SGD算法，使得多个智能体可以并行地进行训练，提高算法的并发能力。除此之外，A2C还在更新策略网络时，引入了advantage actor-critic（优势演员-评论家）方法，来帮助模型学习到更好的动作价值函数，从而提升性能。最后，A2C采用贪婪策略（greedy policy）来进行决策，既不完全依据模型预测，也不会完全依赖历史数据，从而最大限度地增加探索因子，以提升模型的鲁棒性。

## 3.5 PPO
PPO（Proximal Policy Optimization，近端策略优化）是OpenAI团队在2017年的一项研究，其核心思想是通过设定损失函数中的KL散度限制来控制策略搜索范围，以达到平衡 exploration 和 exploitation 的平衡。其基本流程与之前的模型几乎一致，但通过增加一个熵的惩罚项来控制策略的多样性，从而提高学习效率。PPO已被证明可用于一些复杂的控制问题，如机器人运动规划。

与DQN、A2C、DDPG不同，PPO没有明确的目标函数，而是利用动态的KL约束来调整策略网络的参数。其具体做法是，PPO利用kl散度来衡量两个策略的相似性，并设置一个超参λ，来控制kl散度的大小。当λ太大时，kl散度限制过宽，导致策略网络收敛缓慢；当λ太小时，kl散度限制过细，导致模型无法泛化到新环境中。因此，PPO通过动态地调整λ的值来找到合适的平衡点。

另外，PPO还引入了一阶动力学损失（first order dynamics loss），从而减少样本效率下降。

## 3.6 TRPO
TRPO（Trust Region Policy Optimization，信赖域策略优化）是斯坦福大学团队在2015年的一项研究，其核心思想是通过在策略搜索范围内添加限制条件来避免陷入局部最优解。TRPO利用强化学习中的KL散度来表示不同动作之间的联系，并通过控制参数空间的大小来提高探索效率。其基本流程与之前的模型几乎一致，但增加了一个惩罚项来控制策略的多样性，从而防止模型过拟合。TRPO已被证明可用于一些复杂的控制问题，如机器人运动规划。

与DQN、A2C、DDPG、PPO不同，TRPO没有明确的目标函数，而是利用控制限制来调整策略网络的参数。其具体做法是，TRPO在每次迭代中，首先计算当前策略的kl散度，并根据kl散度的大小设置一个超参δ，来控制策略网络的参数的更新幅度。当δ太大时，限制过细，导致模型过于保守；当δ太小时，限制过宽，导致模型无法探索到全局最优解。因此，TRPO通过动态地调整δ的值来找到合适的平衡点。

## 3.7 ACER
ACER（Actor-Critic with Experience Replay，带经验重放的演员-评论家）是斯坦福大学团队在2016年的一项研究，其核心思想是改进基于时间差分的Q-learning的方法，提出经验回放的方法来增强训练样本。ACER的算法流程与DQN基本一致，但引入了经验回放的方法，充分利用了记忆中的信息，减少样本效率下降。目前，ACER已被证明可用于一些复杂的控制问题，如机器人运动规划。

与DQN、A2C、DDPG、PPO、TRPO不同，ACER没有明确的目标函数，而是结合DQN的优点，提出了一个优势演员-评论家（Advantage Actor-Critic，A2C）的方案。ACER的优势演员-评论家架构由两部分组成，分别是演员网络（actor network）和评论家网络（critic network）。演员网络接收当前屏幕图像作为输入，生成各个动作对应的概率分布，再根据概率分布选出最优动作。评论家网络接受当前屏幕图像作为输入，与演员网络一起计算每个动作的Q值，并针对Q值进行优化。ACER还利用经验回放的方法来增强训练样本。

# 4.具体代码实例和解释说明
以上所介绍的算法模型只是提供了大纲，实际操作中还涉及很多细节问题，比如算法的超参数选择、数据集的准备、模型参数的加载、样本数据的记录、结果的可视化、训练时的采样模式、推断的实现等。下面是几个具体的代码实例，供大家参考：

## 4.1 DQN
以下是一个DQN的具体代码实例，使用PyTorch编写。

```python
import torch
import torch.nn as nn
import gym
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

class DQN(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DQN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs)
        )

    def forward(self, x):
        return self.fc(x)

env = gym.make('CartPole-v0')

# set up the model
model = DQN(env.observation_space.shape[0], env.action_space.n)
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)
model = model.to(device)

def train(model, optimizer, loss_fn, experience):
    state, action, next_state, reward, done = experience
    
    state = torch.FloatTensor(np.float32(state)).unsqueeze(0).to(device)
    next_state = torch.FloatTensor(np.float32(next_state)).unsqueeze(0).to(device)
    action = torch.LongTensor(action).unsqueeze(0).to(device)
    reward = torch.FloatTensor([reward]).unsqueeze(0).to(device)
    done = torch.FloatTensor([done]).unsqueeze(0).to(device)

    # get Q(s',a) and best action a' using target net
    q_values_next = model(next_state)
    _, actions_next = q_values_next.max(dim=1)
    q_values_next_target = target_net(next_state)
    q_value_next_target = q_values_next_target.gather(1, actions_next.unsqueeze(1)).squeeze(-1)

    # compute target value y
    target = (q_value_next * GAMMA) + (reward + (GAMMA ** N_STEP) * q_value_next_target * (not done))
    
    # predict current Q values
    q_values = model(state)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(-1)
    
    # calculate loss between predicted Q value and actual label
    loss = loss_fn(q_value, target)
    
    # optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
def run_episode():
    episode_rewards = []
    state = env.reset()
    while True:
        # select an action based on epsilon greedy strategy
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        if random.random() < eps_threshold:
            action = env.action_space.sample()
        else:
            state = torch.FloatTensor(np.float32(state)).unsqueeze(0).to(device)
            q_values = model(state)
            _, action = q_values.max(1)
            action = int(action.item())
            
        # perform the selected action in the environment
        next_state, reward, done, _ = env.step(action)
        
        # store the transition in the replay buffer
        exp = (state, action, next_state, reward, done)
        replay_buffer.append(exp)
                
        # update the state and step count
        state = next_state
        steps_done += 1
        
        # train the model after every C steps
        if len(replay_buffer) > BATCH_SIZE and steps_done % C == 0:
            for i in range(TRAIN_FREQ):
                batch = random.sample(replay_buffer, BATCH_SIZE)
                train(model, optimizer, loss_fn, batch)
                
                    
        episode_rewards.append(reward)
        if done or steps_done >= MAX_STEPS:
            break
            
    return sum(episode_rewards)
        
            
if __name__ == '__main__':
    NUM_EPISODES = 200
    REWARDS = []
    
    # initialize target network to same parameters as online network
    target_net = DQN(env.observation_space.shape[0], env.action_space.n)
    target_net.load_state_dict(model.state_dict())
        
    for ep in range(NUM_EPISODES):
        rewards = run_episode()
        REWARDS.append(rewards)
        
        print('[Episode {}/{}] Reward {}'.format(ep+1, NUM_EPISODES, rewards))
        
        if ep % TARGET_UPDATE == 0:
            target_net.load_state_dict(model.state_dict())
                
    # plot the total reward per episode
    plt.plot(REWARDS)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.show()
```

## 4.2 A2C
以下是一个A2C的具体代码实例，使用PyTorch编写。

```python
import os
import time
import torch
import torch.nn as nn
import gym
from tensorboardX import SummaryWriter
import numpy as np
import math

class ActorCritic(nn.Module):
    """
    This class implements the ACTOR CRITIC NETWORK used by the A2C algorithm. It takes as input 
    the size of the observations space and outputs two vectors of length n_actions representing
    the probability distribution over possible actions and the expected value of each action respectively.
    """
    def __init__(self, observation_size, hidden_size, n_actions):
        super().__init__()
        self.hidden_size = hidden_size
        self.actor = nn.Sequential(
                    nn.Linear(observation_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, n_actions)
                )
        self.critic = nn.Sequential(
                    nn.Linear(observation_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, 1)
                )
        
    def forward(self, x):
        """
        Forward pass through both actor and critic networks. Returns tuple consisting of 
        the actor output (action probabilities) and the critic output (expected value of each action).
        """
        probs = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        return probs, value
    
    
class A2CAgent:
    """This is a single agent that interacts with the environment."""
    def __init__(self, name, obs_size, act_size, gamma, lr, entropy_coef, max_steps, log_interval, seed):
        self.name = name
        self.obs_size = obs_size
        self.act_size = act_size
        self.gamma = gamma
        self.lr = lr
        self.entropy_coef = entropy_coef
        self.max_steps = max_steps
        self.log_interval = log_interval
        self.seed = seed
        self.training_mode = False
    
    def choose_action(self, obs, training=True):
        """Choose an action given an observation"""
        self.training_mode = training
        self.model.train(training)
        with torch.no_grad():
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            prob = self.model.actor(obs)[0].detach().cpu().numpy()
            dist = Categorical(prob)
            action = dist.sample().item()
        return action
    
    def learn(self, rollout):
        """Update policy using the given rollout"""
        obs_batch, acts_batch, rews_batch, vals_batch, dones_batch = map(
            lambda x: torch.cat(x, dim=0).to(self.device),
            zip(*rollout)
        )
        advantages = rews_batch - vals_batch[:-1]
        
        probs, vals = self.model(obs_batch)
        m_probs, m_vals = self.model(obs_batch[-1])
        last_val = m_vals.view(-1).item()
        discounted_rews = utils.discount_rewards(rews_batch + [last_val], self.gamma)[:-1]
        
        val_loss = ((discounted_rews - vals)**2).mean()
        entropy_loss = (-(m_probs*torch.log(probs))).sum()
        pol_loss = -(advantages.detach() * torch.log(probs)).mean()
        
        loss = pol_loss + val_loss + (self.entropy_coef * entropy_loss)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
        self.optimizer.step()
        
        return {'pol_loss': pol_loss.item(), 
                'val_loss': val_loss.item(), 
                'entropy_loss': entropy_loss.item()}
    
    
    def init_model(self, env, device='cpu'):
        """Initialize actor-critic neural networks"""
        self.device = device
        self.model = ActorCritic(self.obs_size, HIDDEN_SIZE, self.act_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        
        
class RolloutStorage:
    """Stores rollouts data until they can be used to update a model"""
    def __init__(self, num_steps, num_processes, obs_size, act_size):
        self.observations = torch.zeros(num_steps+1, num_processes, obs_size)
        self.actions = torch.zeros(num_steps, num_processes, 1).long()
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.returns = torch.zeros(num_steps+1, num_processes, 1)
        self.masks = torch.ones(num_steps+1, num_processes, 1)
        self.index = 0
        self.num_steps = num_steps
        self.num_processes = num_processes
        self.obs_size = obs_size
        self.act_size = act_size
        
    def insert(self, current_obs, action, reward, mask):
        """Insert new observation into the storage buffer"""
        self.observations[self.index+1].copy_(current_obs)
        self.actions[self.index].copy_(action)
        self.rewards[self.index].copy_(reward)
        self.masks[self.index+1].copy_(mask)
        self.index = (self.index + 1) % self.num_steps
        
    def after_update(self):
        """Compute returns and clear out the buffers"""
        self._compute_returns()
        self.observations.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.masks.fill_(1)
        self.index = 0
        
    def compute_advantages(self, last_val=0.0):
        """Computes advantage estimates based on the current returns"""
        advs = self.returns[:-1] - self.values + last_val
        advs = (advs - advs.mean())/(advs.std()+1e-8)
        return advs
    
    def feed_forward_generator(self, advantages, mini_batch_size):
        """Generates batches of data from stored rollout"""
        batch_size = self.num_processes * self.num_steps
        assert batch_size >= mini_batch_size, "Batch size should be greater than or equal to sample size"
        indices = torch.randperm(batch_size).tolist()
        for start_idx in range(0, batch_size, mini_batch_size):
            end_idx = start_idx + mini_batch_size
            sampled_indices = indices[start_idx:end_idx]
            yield self._get_samples(sampled_indices, advantages)
            
            
    def _get_samples(self, indices, advantages):
        """Retrieves samples according to the specified indices"""
        obs_batch = self.observations[:-1].view(-1, *self.obs_size)[indices]
        act_batch = self.actions.view(-1, 1)[indices]
        ret_batch = self.returns[:-1].view(-1, 1)[indices]
        adv_batch = advantages.view(-1, 1)[indices]
        old_v_batch = self.values.view(-1, 1)[indices]
        old_p_batch = self.old_log_probs.view(-1, 1)[indices]
        return obs_batch, act_batch, ret_batch, adv_batch, old_v_batch, old_p_batch
    

    def _compute_returns(self):
        """Computes returns recursively from the rewards"""
        R = 0
        self.returns[-1] = self.rewards
        for t in reversed(range(self.rewards.size(0))):
            R = self.gamma * R + self.rewards[t]
            self.returns[t] = R