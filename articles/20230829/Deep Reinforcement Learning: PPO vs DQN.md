
作者：禅与计算机程序设计艺术                    

# 1.简介
  

在人工智能领域，大规模并行计算（如：GPU）成为支撑强化学习系统快速演进、实现高度抽象决策能力的关键技术。此外，还出现了基于神经网络的强化学习方法——DQN(Deep Q Network)及其变体（如：Double DQN），这些模型都应用了深度学习的一些最新成果来解决传统强化学习面临的问题。近年来，研究人员提出了新的深度强化学习方法——Proximal Policy Optimization (PPO)，它可以有效克服DQN存在的很多不足。由于这两种方法都是基于神经网络的强化学习方法，所以文章将通过对比分析PPO与DQN两个方法的设计理念、机制、算法流程等方面，从宏观角度探讨这两者之间差异与联系。最后，文章还将结合实际场景，分析PPO与DQN的优缺点以及何时应该选择哪种方法。希望读者能够从本文受益。

# 2.基本概念术语说明
## 2.1 强化学习简介
强化学习（Reinforcement Learning，RL）是指一个agent（通常是一个智能体或一个机器人）通过与环境的交互来学习到智能行为策略，使之能够最大化长期奖励（reward）。强化学习可以用于训练智能体完成各种复杂任务，例如：游戏、机器人运动控制、自动驾驶等。传统强化学习可以分为监督学习和非监督学习。

- 监督学习：在监督学习中，环境会给予agent一个“正确”的反馈信号，如：图像，表示该图像中的物体是否存在；每一步的action都会得到回报，即agent要最大化预期的回报值。监督学习的代表性算法包括：Q-learning、Sarsa等。

- 非监督学习：非监督学习不需要agent给环境提供反馈，而是在自主发现模式下找到隐藏结构。例如，聚类、模式识别、降维等。非监督学习的代表性算法包括：EM算法、谱聚类算法、核密度估计算法等。

- 强化学习：在强化学习中，环境会给agent一个“不正确”的反馈信号，agent需要自己决定下一步要做什么。强化学习的特点是：agent的反馈不是单一的，而是由环境驱动的，agent需要根据当前状态、历史经验、风险偏好等综合考虑后再作出动作。

## 2.2 相关术语
### 2.2.1 Markov Decision Process（MDP）
马尔可夫决策过程（Markov Decision Process， MDP）是描述马尔可夫随机过程（Markov Random Process， MRP）及其在强化学习中的角色的一组假设。

马尔可夫决策过程包括：

1. **S**：表示环境所处的状态空间。
2. **A**：表示执行动作的动作空间。
3. **P(s'|s,a)**：表示在状态$s$下执行动作$a$之后可能转移到的状态$s'$的概率分布。
4. **R(s,a,s')**：表示在状态$s$下执行动作$a$导致状态$s'$发生的奖赏值。

其中，状态转移函数$T(s,a,s')=P(s'|s,a)$、状态值函数$V_{\pi}(s)=E_\pi[G_t\mid S_t=s]$及策略$\pi$均依赖于动作序列$\{A_t\}$，即agent如何从状态$s$选择动作$a$以便最大化收益。

### 2.2.2 DQN
DQN（Deep Q Network）是一种基于神经网络的强化学习算法，它能够在多项式时间内学习离散动作空间，并有效解决之前基于表格的方法遇到的大量状态和动作组合导致的高维搜索问题。它的特点包括：

1. 使用神经网络拟合状态值函数$V(s; \theta^Q)$和动作价值函数$Q(s, a; \theta^Q)$，其中$\theta^Q$是神经网络的参数。
2. 通过从经验中学习更新神经网络参数来进行训练。

### 2.2.3 Double DQN
双DQN（Double DQN）是DQN的一种变体，它采用两个独立的神经网络进行Q值预测，可以有效避免DQN在某些情况下的过高估计。它的特点包括：

1. 使用一个神经网络拟合目标网络的输出，用另一个网络拟合Q值的期望，来计算DQN损失函数。
2. 每隔一定的训练步数，更新目标网络的权重，从而平滑Q值预测的结果。

### 2.2.4 PPO
PPO（Proximal Policy Optimization）是一种无模型、端到端学习算法，它结合了梯度更新、KL约束和ε-greedy策略调节三个模块。它的特点包括：

1. 不需要手工设计特征函数，直接利用状态和动作之间的关系。
2. 在使用旧策略生成新策略时，将采取紧凑的KL约束。
3. ε-greedy策略调节组件通过自适应调整探索和利用之间trade-off。

# 3.核心算法原理和具体操作步骤以及数学公式讲解
## 3.1 概览
本章首先对强化学习与深度强化学习之间的关系进行了一个简单的介绍，然后详细叙述DQN、Double DQN以及PPO算法各自的设计理念、机制、算法流程以及对应数学公式。接着，针对DQN、Double DQN以及PPO算法的具体操作步骤以及数学公ulator等，提供了代码实例。最后，对以上三种算法的优缺点以及何时应该选择哪种算法进行了阐述。

## 3.2 深度强化学习的发展
在深度强化学习的研究初期，传统强化学习算法的计算效率并不高，且往往需要对大型数据库进行扫描，计算复杂度随着样本数量的增长呈线性增长。因此，研究人员开始寻找更加高效的深度强化学习算法。

深度强化学习的一个重要特点就是具有高度的抽象能力，这意味着它能够处理复杂的任务，同时学习到智能行为策略。例如，AlphaGo和AlphaZero就成功地使用深度强化学习来训练智能体完成围棋和国际象棋这样的复杂任务。

为了达到更好的效果，研究人员提出了许多深度强化学习方法，如DQN、DDQN、PPO等。

## 3.3 算法介绍
### 3.3.1 DQN
DQN（Deep Q Network）是一种基于神经网络的强化学习方法。它的主要特点是通过利用神经网络来拟合状态值函数$V(s;\theta^Q)$和动作价值函数$Q(s, a;\theta^Q)$。其中，$\theta^Q$是神经网络的参数。为了防止网络过拟合，训练过程中引入经验回放和目标网络的概念。

1. 神经网络结构

   DQN采用经典的神经网络结构，其中第一层为卷积层，第二至第四层为全连接层，最后一层为输出层。卷积层有两个卷积核大小为8x8和4x4，激活函数为ReLU。全连接层分别有256、128个神经元。输出层使用一个全连接神经元，并使用tanh激活函数来保证输出范围。

2. 经验回放

   经验回放是DQN中重要的一种技术，它允许智能体收集多个经验数据，并将它们分批送入神经网络进行训练。经验回放能有效减少样本效率上的波动。DQN中的经验回放机制包括：

    - 将经验数据存储在记忆库（memory buffer）中，记忆库维护一定数量的经验数据。
    - 当收集到新的经验数据时，更新记忆库。
    - 当记忆库满时，随机删除一些数据。
    - 从记忆库中随机采样一小批数据作为训练集。
    - 对样本进行标准化处理，使得输入数据的范围一致。

3. 更新目标网络

   为了防止学习过程中的局部最优，DQN使用了目标网络。在训练过程中，目标网络的参数固定不动，仅仅在每次训练时同步更新一次网络参数。目标网络的作用是当Q网络收敛时，它就会复制Q网络的参数，从而在一定程度上抹平训练过程中的震荡。

4. 训练阶段

   训练阶段包括选取样本，计算损失函数，使用梯度下降算法更新网络参数，重复以上步骤直至收敛。
   在选取样本时，DQN按照优先级采样的方式从记忆库中采样数据，优先级指的是如果采样的数据中存在一些误差大的样本，则这些样本的优先级较高。

   下图展示了DQN算法的整体框架。


### 3.3.2 Double DQN
双DQN（Double DQN）是DQN的一种变体。它的主要特点是采用两个独立的神经网络进行Q值预测。其思路是：当我们在进行动作决策时，我们不能只用当前网络进行预测，而应同时使用两个网络：当前网络负责预测当前的Q值，而目标网络则负责预测远期的Q值。

1. 神经网络结构

   和DQN一样，Double DQN也采用经典的神经网络结构。第一层为卷积层，第二至第四层为全连接层，最后一层为输出层。卷积层有两个卷积核大小为8x8和4x4，激活函数为ReLU。全连接层分别有256、128个神经元。输出层使用一个全连接神经元，并使用tanh激活函数来保证输出范围。

2. 更新目标网络

   对于DQN来说，它有一个目标网络，用来提供远期的Q值预测。但是，对于Double DQN来说，它还有一个主网络，负责预测当前的Q值。双网络的目标是使两个网络的参数相等，从而使Q值预测更加稳定。

3. ε-greedy策略

   Double DQN和DQN一样，也是使用ε-greedy策略来进行动作决策。但是，对于Double DQN来说，它对其策略进行了调整，使得它更倾向于使用主网络的预测而不是目标网络的预测。

   下图展示了Double DQN算法的整体框架。


### 3.3.3 PPO
PPO（Proximal Policy Optimization）是一种无模型、端到端学习算法。它的主要特点是结合了梯度更新、KL约束和ε-greedy策略调节三个模块。

1. 模型

   PPO直接基于策略网络进行控制，而不像DQN那样先学习特征函数。具体地，PPO在策略网络的末端加入了两个辅助输出，即正则化项和熵项。正则化项使策略网络更加鲁棒，防止过拟合；熵项用来衡量策略的多样性。

2. KL约束

   KL约束是PPO的另一重要技巧。在策略的更新过程中，我们希望样本分布逼近真实分布。为此，我们在策略网络的输出上加入KL散度，并且让目标网络输出和原网络输出之间的KL散度尽可能小。

3. ε-greedy策略

   PPO使用ε-greedy策略进行动作决策。它在训练阶段使用ε值较小的随机策略进行探索，在测试阶段则使用ε值较大的最优策略。

4. PPO算法流程

   PPO的算法流程如下：

   1. 初始化策略网络θ'和目标网络θ。
   2. 对于每一个episode，初始化环境和初始状态。
   3. 执行策略网络θ'选择动作A_t，或者在ε-greedy策略下选择动作。
   4. 执行动作A_t，获取环境反馈R_t和下一个状态S_{t+1}。
   5. 如果终止，则进入第六步；否则，返回到第三步。
   6. 计算回报值G_t = R_t + γmax_{a}{Q'(S_{t+1},a;θ')}，θ'接受经验<S_t, A_t, G_t>。
   7. 对于第i次迭代，如果batch size为n，则从记忆库里抽取n个随机样本作为训练集。
   8. 计算策略网络θ关于训练集的损失函数L：
      L(θ) = ∑ logπ(a|s,θ) * advantages(s,a),
      其中advantages(s,a) = E[γr + γmax_{a'}Q'(s',a';θ) - Q(s,a;θ)] / δ，δ是任意小的常数。
   9. 用L对θ进行更新，使用梯度下降更新θ。
   10. 每隔一定的训练步数，把θ'的权重更新为θ。
   11. 返回到第3步进行下一个episode的训练。

   下图展示了PPO算法的整体框架。


## 3.4 操作步骤以及代码实例
DQN、Double DQN以及PPO算法都可以应用于具体场景。下面我们以CartPole环境为例，说明如何使用DQN、Double DQN以及PPO算法来解决该问题。

### 3.4.1 CartPole环境介绍
CartPole环境是一个古老的物理游戏。它包含一个铲子，一个移动的杆，一个垂直于杆面的摆球，以及一个墙壁。玩家需要通过不断左右摆动摆球以保持杆子与桥面的夹角不超过 ±12°。每当摆球落入倒立状态，就结束游戏。


该环境的状态共有4个：位置、速度、角度、角速度。动作空间只有两个：左摆动（向左推杆子）、右摆动（向右推杆子）。奖励函数是每一步游戏持续时间的倒立次数。

### 3.4.2 CartPole-v0环境创建
使用gym包创建一个CartPole-v0环境。

```python
import gym
env = gym.make('CartPole-v0')
```

### 3.4.3 DQN算法创建
创建一个DQN agent。

```python
class DQNAgent:
  def __init__(self, state_dim, action_dim):
    self.state_dim = state_dim
    self.action_dim = action_dim
    
    # 创建两个神经网络
    self.model = Sequential()
    self.model.add(Dense(24, input_dim=state_dim, activation='relu'))
    self.model.add(Dense(24, activation='relu'))
    self.model.add(Dense(action_dim, activation='linear'))
    self.target_model = Sequential()
    self.target_model.add(Dense(24, input_dim=state_dim, activation='relu'))
    self.target_model.add(Dense(24, activation='relu'))
    self.target_model.add(Dense(action_dim, activation='linear'))
    
    # 创建优化器
    self.optimizer = Adam(lr=0.001)
    
    # 为两个神经网络的权重设置共享
    self.update_weights()
    
  def update_weights(self):
    self.target_model.set_weights(self.model.get_weights())
  
  def predict(self, state):
    return self.model.predict(state)
    
  def train(self, experience):
    states = np.array([experience[0] for experience in experiences])
    actions = np.array([experience[1] for experience in experiences])
    rewards = np.array([experience[2] for experience in experiences])
    next_states = np.array([experience[3] for experience in experiences])
    dones = np.array([experience[4] for experience in experiences])
    
    targets = self.model.predict(next_states)
    argmax_actions = np.argmax(targets, axis=-1)
    next_qvalues = [self.target_model.predict(ns)[idx] for idx, ns in enumerate(next_states)]
    target_qvalues = [(rewards[i] if dones[i] else rewards[i] + gamma*next_qvals[i]) 
                      for i, next_qval in enumerate(next_qvalues)]
                      
    qvalues = self.model.predict(states)
    onehot_actions = np.eye(self.action_dim)[actions]
    selected_qvalues = np.sum((onehot_actions)*(qvalues), axis=-1)
    loss = mean_squared_error(selected_qvalues, target_qvalues)
    
    self.model.fit(states, onehot_actions, epochs=1, verbose=0)
    
    return loss
    
def run_dqn():
    env = gym.make('CartPole-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim, action_dim)
    
    max_episodes = 1000
    batch_size = 64
    gamma = 0.99
    
    for episode in range(max_episodes):
        done = False
        score = 0
        
        state = env.reset()
        while not done:
            action = agent.act(state)
            
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state
            score += reward
            
            if len(agent.experiences) >= batch_size:
                loss = agent.train(batch_size)
                
            if done:
                print("Episode: {}, Score: {:.2f}".format(episode, score))
                break
            
            agent.update_weights()
            
if __name__ == '__main__':
    run_dqn()
```

### 3.4.4 Double DQN算法创建
创建一个Double DQN agent。

```python
class DoubleDQNAgent:
  def __init__(self, state_dim, action_dim):
    self.state_dim = state_dim
    self.action_dim = action_dim
    
    # 创建两个神经网络
    self.model = Sequential()
    self.model.add(Dense(24, input_dim=state_dim, activation='relu'))
    self.model.add(Dense(24, activation='relu'))
    self.model.add(Dense(action_dim, activation='linear'))
    self.target_model = Sequential()
    self.target_model.add(Dense(24, input_dim=state_dim, activation='relu'))
    self.target_model.add(Dense(24, activation='relu'))
    self.target_model.add(Dense(action_dim, activation='linear'))
    
    # 创建优化器
    self.optimizer = Adam(lr=0.001)
    
    # 为两个神经网络的权重设置共享
    self.update_weights()

  def update_weights(self):
    self.target_model.set_weights(self.model.get_weights())

  def predict(self, state):
    return self.model.predict(state)

  def act(self, state, epsilon):
    if random.random() < epsilon:
      return random.randint(0, self.action_dim-1)
    else:
      q_values = self.model.predict(np.reshape(state, [1, self.state_dim]))[0]
      return np.argmax(q_values)

  def remember(self, state, action, reward, next_state, done):
    self.experiences.append((state, action, reward, next_state, done))

  def replay(self, batch_size):
    minibatch = random.sample(self.experiences, batch_size)
    states = np.array([exp[0] for exp in minibatch])
    actions = np.array([exp[1] for exp in minibatch])
    rewards = np.array([exp[2] for exp in minibatch])
    next_states = np.array([exp[3] for exp in minibatch])
    dones = np.array([exp[4] for exp in minibatch])

    targets = self.model.predict(next_states)
    max_actions = np.argmax(targets, axis=-1)
    target_qvalues = []
    for j in range(len(minibatch)):
      if dones[j]:
        target_qvalues.append(rewards[j])
      else:
        target_qvalue = rewards[j] + gamma*(self.target_model.predict(next_states[[j]])[0][max_actions[j]])
        target_qvalues.append(target_qvalue)
    target_qvalues = np.array(target_qvalues)

    onehot_actions = to_categorical(actions, num_classes=self.action_dim)
    predicted_qvalues = np.sum(onehot_actions*(self.model.predict(states)), axis=-1)

    loss = mean_squared_error(predicted_qvalues, target_qvalues)

    self.model.fit(states, onehot_actions, sample_weight=(gamma**len(predicted_qvalues))*(-1)*loss,
                   epochs=1, verbose=0)


  def train(self, epsilons):
    global total_steps
    total_steps += 1

    losses = []
    for i, epsilon in enumerate(epsilons):
      state = env.reset()
      steps = 0
      score = 0

      while True:
        step_start_time = time.time()

        action = agent.act(state, epsilon)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)

        state = next_state
        steps += 1
        score += reward

        if len(agent.experiences) > batch_size:
          agent.replay(batch_size)
          
        if done or steps >= max_steps:
          losses.append((epsilon, score))
          break

        elapsed_time = int(round(time.time()-step_start_time))
        
        
    best_score = -math.inf
    model_file = "double_dqn_{}.h5".format(total_steps)
    
    with open(os.path.join(".", model_file), 'wb') as f:
      pickle.dump(agent.model, f)
      
    agent.model.save(os.path.join(".", model_file))
    
    print("Best scores:", sorted(losses, key=lambda x: x[1], reverse=True))
```

### 3.4.5 PPO算法创建
创建一个PPO agent。

```python
import torch
from torch import nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from typing import List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def make_env(env_id: str, rank: int, seed: int) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :return: (Callable)
    """
    def _init() -> gym.Env:
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

def set_global_seeds(seed: int) -> None:
    """
    Set the global seed.
    """
    torch.manual_seed(seed)
    if torch.backends.cudnn.enabled:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

class CustomPolicy(nn.Module):
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, lr_schedule: Callable):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.shape[0]),
        )

        self.lr_schedule = lr_schedule
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr_schedule(1))

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return self.net(obs)


class PPOAgent:
    def __init__(self, observation_space: gym.spaces.Space, action_space: gym.spaces.Space, lr_schedule: Callable):
        self.observation_space = observation_space
        self.action_space = action_space
        self.lr_schedule = lr_schedule

        self.env = DummyVecEnv([make_env('CartPole-v0', 0, 0)])
        self.model = PPO(CustomPolicy,
                         self.env,
                         policy_kwargs={"net_arch": [dict(pi=[64, 64], vf=[64, 64])]})

    def learn(self, n_timesteps: int, eval_freq: int = 10000, n_eval_episodes: int = 5):
        model_checkpoint_path = "./ppo_cartpole_"
        model_checkpoint_frequency = eval_freq // 10
        self.model.learn(total_timesteps=n_timesteps, tb_log_name="cart_pole", callback=callback)

        returns = []
        ep_lengths = []
        for e in range(n_eval_episodes):
            obs = self.env.reset()
            done, state = False, None
            episode_return = 0.0
            episode_length = 0

            while not done:
                action, state = self.model.predict(obs, state=state, deterministic=True)

                next_obs, reward, done, info = self.env.step(action)
                episode_return += reward
                episode_length += 1
                obs = next_obs

            returns.append(episode_return)
            ep_lengths.append(episode_length)

        average_return = np.mean(returns)
        std_return = np.std(returns)
        average_length = np.mean(ep_lengths)
        std_length = np.std(ep_lengths)
        print(f" Evaluation using {n_eval_episodes} episodes: mean reward={average_return:.2f}"
              f" +/- {std_return:.2f}")
        print(f" Evaluation using {n_eval_episodes} episodes: mean length={average_length:.2f}"
              f" +/- {std_length:.2f}\n")


def callback(_locals, _globals):
    """
    Callback called at each step (for DQN an others) or after n steps (see ACER or PPO2)
    :param _locals: (dict)
    :param _globals: (dict)
    """
    global N_STEPS, MODEL_NAME
    if (N_STEPS + 1) % MODEL_CHECKPOINT_FREQUENCY == 0:
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        # Save checkpoint every 1M steps
        model_name = os.path.join(MODEL_SAVE_DIR, '{}_{}_steps'.format(EXPERIMENT_ID, N_STEPS))
        _locals['self'].save(model_name)
    N_STEPS += 1


def main():
    experiment_id = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    n_trials = 1
    # Experiment parameters
    TRAIN_TIMESTEPS = 1000000
    N_EVAL_EPISODES = 10
    EVAL_FREQ = 100000
    EPSILON_VALUES = [0.01, 0.02, 0.05, 0.1]
    MODEL_SAVE_FREQUENCY = 1000000
    EXPLORE_STOP = 0.01
    RL_ALGORITHM = PPOAgent
    EXPERIMENT_PARAMETERS = {'TRAIN_TIMESTEPS': TRAIN_TIMESTEPS,
                             'N_EVAL_EPISODES': N_EVAL_EPISODES,
                             'EPSILON_VALUES': EPSILON_VALUES,
                             'EVAL_FREQ': EVAL_FREQ,
                             'MODEL_SAVE_FREQUENCY': MODEL_SAVE_FREQUENCY,
                             'EXPLORE_STOP': EXPLORE_STOP,
                             }
    all_results = {}
    try:
        # Evaluate untrained models over several seeds and save results
        print("Evaluating untrained model...")
        for trial in range(n_trials):
            seed = tune.grid_search([i for i in range(n_trials)])
            rl_algorithm = RL_ALGORITHM(env.observation_space,
                                         env.action_space,
                                         lambda _: 0.001)
            evaluation_result = tune.run(run_or_experiment=rl_algorithm.learn,
                                          config={'env': RL_ENVIRONMENT,
                                                 'seed': seed,
                                                  'eval_freq': EVAL_FREQ,
                                                  'n_eval_episodes': N_EVAL_EPISODES},
                                          stop={'training_iteration': 1})
            if trial == 0:
                best_hyperparams = evaluation_result.best_config
                all_results["Untrained"] = {"Reward": evaluation_result.best_trial.last_result['episode_reward_mean'],
                                            "Length": evaluation_result.best_trial.last_result['episode_len_mean']}

        # Train models on training data with tuned hyperparameters and save checkpoints regularly during training
        print("Training model...")
        rl_algorithm = RL_ALGORITHM(env.observation_space,
                                     env.action_space,
                                     lambda _: 0.001)
        callbacks = []
        callbacks.append(SaveOnBestTrainingRewardCallback(check_freq=MODEL_SAVE_FREQUENCY,
                                                           log_dir="./logs/",
                                                           name_prefix="ppo_{}".format(experiment_id)))
        result = tune.run(run_or_experiment=rl_algorithm.learn,
                          config={'env': RL_ENVIRONMENT,
                                  'eps': tune.grid_search([EPSILON_VALUES]),
                                  **best_hyperparams},
                          stop={'training_iteration': int(TRAIN_TIMESTEPS)},
                          progress_reporter=CLIReporter(metric_columns=["timesteps_total", "episode_reward_mean"]),
                          checkpoint_at_end=True,
                          verbose=1,
                          callbacks=callbacks)
        all_results["Trained"] = {"Reward": result.best_trial.last_result['episode_reward_mean'],
                                  "Length": result.best_trial.last_result['episode_len_mean']}
        tune.report(**all_results)
    except Exception as e:
        logger.exception("Exception occurred while running tune experiments.")

if __name__ == "__main__":
    main()
```