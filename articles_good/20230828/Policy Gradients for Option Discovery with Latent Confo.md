
作者：禅与计算机程序设计艺术                    

# 1.简介
  

在深度强化学习领域,基于策略梯度的方法已被广泛应用。其原理是在强化学习中,通过执行动作序列进行训练,利用Actor-Critic等模型去求取最优策略。然而,这种方法往往存在两个问题:

1. Sample inefficiency: 在真实环境中,每一步行动需要观察到环境状态,而这一过程往往十分耗时。从而导致采样效率低下。
2. Curiosity driven exploration: 在强化学习中,通常采用自助法（Exploration by Random Network Distillation）作为探索策略,即通过对神经网络进行蒸馏得到新的网络参数来增强探索效果。但由于蒸馏过程中生成器网络本身也会出现探索效应,因此蒸馏仍无法完全解决这个问题。另外,在某些情况下,无需真正的探索。比如在一些复杂游戏中,可以直接利用已知规则来指导策略的选择。但目前主流的强化学习框架难以实现这一点。

Hindsight Experience Replay (HER) 与 Value function approximation 方法则可以完美解决上述两个问题。HER 方法能够将过去的经验数据引入到当前的强化学习任务中,以期望能够更好的学习长远规划。而 value function approximation 方法则通过估计价值函数的方式,来帮助Actor去更好地适应当前环境并选择合适的行为。这样做不仅能够有效降低sample inefficiency,还能够提升Curiosity driven exploration能力。

论文作者搭建了一个Option Discovery framework using HER and VFA. 使用HER, 作者将之前的经验记录引入到当前的RL任务中, 使之能够更好地学习长远规划。同时作者通过VFA, 来拟合状态动作价值函数(state-action value function)，从而帮助Actor找到最优的行为策略。

# 2.背景介绍
option discovery 是一种关于利用机器学习解决开放式问题的任务。该任务属于多模态决策问题,包括状态、动作、奖励、奖励遗忘以及其他可能的选项。它的目标是找到一系列动作的组合,能够最大化某个奖励信号,同时满足相关约束条件。例如,在机器人控制领域, option discovery 可以用于找到一个序列的动作,能够让机器人更快、更精确地移动到目标位置。

在深度强化学习领域,基于策略梯度的方法已被广泛应用。其原理是在强化学习中,通过执行动作序列进行训练,利用Actor-Critic等模型去求取最优策略。然而,这种方法往往存在两个问题:

1. Sample inefficiency: 在真实环境中,每一步行动需要观察到环境状态,而这一过程往往十分耗时。从而导致采样效率低下。
2. Curiosity driven exploration: 在强化学习中,通常采用自助法（Exploration by Random Network Distillation）作为探索策略,即通过对神经网络进行蒸馏得到新的网络参数来增强探索效果。但由于蒸馏过程中生成器网络本身也会出现探索效应,因此蒸馏仍无法完全解决这个问题。另外,在某些情况下,无需真正的探索。比如在一些复杂游戏中,可以直接利用已知规则来指导策略的选择。但目前主流的强化学习框架难以实现这一点。

Hindsight Experience Replay (HER) 与 Value function approximation 方法则可以完美解决上述两个问题。HER 方法能够将过去的经验数据引入到当前的强化学习任务中,以期望能够更好的学习长远规划。而 value function approximation 方法则通过估计价值函数的方式,来帮助Actor去更好地适应当前环境并选择合适的行为。这样做不仅能够有效降低sample inefficiency,还能够提升Curiosity driven exploration能力。

论文作者搭建了一个Option Discovery framework using HER and VFA. 使用HER, 作者将之前的经验记录引入到当前的RL任务中, 使之能够更好地学习长远规划。同时作者通过VFA, 来拟合状态动作价值函数(state-action value function)，从而帮助Actor找到最优的行为策略。

# 3.基本概念术语说明
在Option Discovery中，主要涉及以下几个关键词：

1. State: 表示机器人的当前状态，机器人的位置信息、速度信息、姿态信息等都可以视为机器人状态。
2. Action: 表示机器人可执行的动作集合，如移动、保持静止、转向等。
3. Reward: 表示机器人完成特定任务时的奖励。
4. Constraint: 表示限制条件，如机器人所处环境的障碍物、挡板等。
5. Option: 选择的一系列动作组合，称为option。

其中，state-action pair由以下形式表示：

$$s_t^a=\left[s_{t}, a_t\right]$$

其中$s_t$表示机器人在时刻t的状态,$a_t$表示机器人在时刻t的动作。

对于Option Discovery任务来说，每个episode对应于一次执行一个option，option的集合包括所有可能的option组合。即：

$$o_i = \left\{a^{1}_{1},..., a^{1}_{T_1},..., a^{M}_{1},..., a^{M}_{T_m}\right\}$$

其中，$i$表示第i个episode；$\{a^{j}_k\}$表示第j个option对应的动作序列，$|a^{j}_k|$表示动作序列长度，$M$表示option数量。

在进行RL优化的时候，作者将每个option分成多个子option，即将action序列分割成多个子序列。同时，将多个子option组合成多个policy，每个policy对应一个子option，将多个policy组成一个meta policy。最终，meta policy就是整个option的policy。meta policy需要拟合多个子policy的参数，同时为了保证解耦性，meta policy只能看到当前的状态s。

# 4.核心算法原理和具体操作步骤以及数学公式讲解
## 4.1 Hindsight Experience Replay
HER方法的核心思想是：通过重构经验序列,将记忆库中老旧的经验引入到新的数据中，从而提高新数据的学习效果。具体来说，HER方法的做法如下：

1. 首先，遍历记忆库中所有的经验$D$，获取一条$s_t^a$的经验。假设记忆库中的经验都是单步的，即$s_t$与$a_t$构成了一条经验，那么将记忆库中所有$s_t$的经验组成一个新序列$D'$。

2. 对$D'$中的每个元素$d$，构造一个新的经验$r'_t^a$。新的经验由两部分组成：前面跟着之前的经验$s_{t'}^{a'}$，后面接着对应的奖励$r_t$：

   $$r'_t^a=r_{t}+\gamma r_{t+1}+\cdots+\gamma^{n-t}(r_t+\gamma r_{t+1}+\cdots+\gamma^{n-t+1})+\delta^T_{\theta_{old}}\left[\underbrace{\begin{pmatrix}I\\s'^{a'}\end{pmatrix}}_{\text{$n$ timesteps ago}}, \underbrace{\pi_{\theta_{old}}(\cdot | s')}_{\text{current policy}}]\tag{1}$$
   
   上式中，$\gamma$是discount factor，$n$是episode长度，$r_\tau$表示第$\tau$个时间步的奖励，$\theta_{old}$表示早先训练得到的策略参数。$\delta^T_{\theta_{old}}$是一个矩阵，用来计算$n$时间步以前$s'$的经验的特征向量，$\pi_{\theta_{old}}$表示早先训练得到的策略。
   
3. 将上面得到的$(r'_t^a, s_{t}^a)$作为经验存储在记忆库中。注意，这里采用的是完整的经验$s_t^a$，而非去掉reward的经验$(s_t^a, a_t)$。这样做是因为后续要重构reward，所以应该完整保留信息。

## 4.2 Value function approximation
Value function approximation (VFA) 方法使用一个评估函数，来预测动作的奖励值。这一方法通过学习一个状态动作价值函数，来代替传统的状态值函数。值函数的输入是状态$s$和动作$a$,输出是状态$s$下动作$a$的期望奖励。状态动作价值函数的更新方法是：

1. 从记忆库中随机抽取一条经验$s_t^a$。

2. 根据$s_t^a$更新状态动作价值函数：

   
   $$\theta=\theta-\alpha\nabla_{\theta}\log\pi_{\theta}(\cdot | s_t)\frac{(Q^{\pi_{\theta}}(s_t,a)-b(s_t))}{N(s_t)}\nabla_{\theta}Q^{\pi_{\theta}}(s_t,a)\tag{2}$$
   
   其中，$\theta$表示状态动作价值函数的参数，$\alpha$表示学习率，$Q^{\pi_{\theta}}$表示状态动作价值函数，$b(s_t)$表示baseline估计，$N(s_t)$表示经验回报，$\nabla_{\theta}\log\pi_{\theta}(\cdot | s_t)$表示策略梯度。

上面的更新方式使用一阶TD误差，实际上还有其他的更新方式，比如Q-learning，Sarsa等。

## 4.3 Option Discovery Framework
综合以上两种方法，作者搭建了一个Option Discovery Framework，用于解决Option Discovery问题。具体来说， Option Discovery Framework 由以下几个模块组成：

1. **Option sampling module:** 输入当前状态$s_t$，输出候选option集合$O(s_t)$。候选option一般由一条或多条动作序列组成。

2. **Replay buffer:** 存放所有经验，包括完整的$s_t^a$和$r_t^a$，以及重构的$(r'_t^a, s_{t}^a)$。

3. **HER replay wrapper module:** 将经验数据引入到强化学习环境中，通过HER方法学习长远规划。具体来说，在选取子option策略时，需要把当前状态$s_t$带入子option策略的生成概率中，再次进行探索，而不是只用记忆库中的经验。具体的做法是在选取action的地方加入以下操作：
   
   $$(a_t, p_t)=argmax_a Q_{\theta}(s_t, a)+(p(s_t)Z_{\pi_{\theta_{parent}}}((a_t,\delta^T_{\theta_{parent}}\left[\underbrace{\begin{pmatrix}I\\s'^{a'}\end{pmatrix}}_{\text{$n$ timesteps ago}}, \underbrace{\pi_{\theta_{parent}}(\cdot | s')}_{\text{parent policy}}])));\quad Z_{\pi_{\theta_{parent}}}((a_t,\delta^T_{\theta_{parent}}\left[\underbrace{\begin{pmatrix}I\\s'^{a'}\end{pmatrix}}_{\text{$n$ timesteps ago}}, \underbrace{\pi_{\theta_{parent}}(\cdot | s')}_{\text{parent policy}}]))=exp(-c\delta^T_{\theta_{parent}}\left[\underbrace{\begin{pmatrix}I\\s'^{a'}\end{pmatrix}}_{\text{$n$ timesteps ago}}, \underbrace{\pi_{\theta_{parent}}(\cdot | s')}_{\text{parent policy}}} - b_{\pi_{\theta_{parent}}}(s'))\tag{3}$$
   
   其中，$Q_{\theta}$表示meta policy的策略，$Z_{\pi_{\theta_{parent}}}((a_t,\delta^T_{\theta_{parent}}\left[\underbrace{\begin{pmatrix}I\\s'^{a'}\end{pmatrix}}_{\text{$n$ timesteps ago}}, \underbrace{\pi_{\theta_{parent}}(\cdot | s')}_{\text{parent policy}}]))$表示子option的生成概率，$c$表示惩罚系数。
   
4. **Suboption optimizer module:** 通过训练子option，来逐渐调整子option策略，使得子option策略能更准确地预测每个option的奖励。子option策略的参数是固定的，但是会根据上层meta policy给出的动作序列生成概率来调整动作分布。具体的训练方式是：
   
   $$L_p(\theta_p)=E_{\tau\sim D_p}[r(\tau)|s_t;\theta_p]+\lambda E_{\tau\sim D_p}[KL(\pi_{\theta_p}(\cdot|s_t)||\tilde{p}_\theta(a_t|\delta^T_{\theta_{parent}}\left[\underbrace{\begin{pmatrix}I\\s'^{a'}\end{pmatrix}}_{\text{$n$ timesteps ago}}, \underbrace{\pi_{\theta_{parent}}(\cdot | s')}_{\text{parent policy}}\right])|_{\theta_p, \delta^T_{\theta_{parent}}\left[\underbrace{\begin{pmatrix}I\\s'^{a'}\end{pmatrix}}_{\text{$n$ timesteps ago}}, \underbrace{\pi_{\theta_{parent}}(\cdot | s')}_{\text{parent policy}}]\in\Delta(\pi_{\theta_p})}]\tag{4}$$
   
   其中，$\tau$表示经验轨迹，$D_p$表示子option的记忆库，$\pi_{\theta_p}$表示子option的策略，$\tilde{p}_\theta(a_t|\delta^T_{\theta_{parent}}\left[\underbrace{\begin{pmatrix}I\\s'^{a'}\end{pmatrix}}_{\text{$n$ timesteps ago}}, \underbrace{\pi_{\theta_{parent}}(\cdot | s')}_{\text{parent policy}}\right])|_{\theta_p, \delta^T_{\theta_{parent}}\left[\underbrace{\begin{pmatrix}I\\s'^{a'}\end{pmatrix}}_{\text{$n$ timesteps ago}}, \underbrace{\pi_{\theta_{parent}}(\cdot | s')}_{\text{parent policy}}]\in\Delta(\pi_{\theta_p})} $表示根据meta policy给出的动作序列生成概率改动后的分布。
   
5. **Meta policy optimizer module:** meta policy优化器是整体训练过程中使用的策略。它决定了整体策略的更新方向，同时也是前进一步的必要条件。作者使用了随机梯度上升算法来更新meta policy参数。具体的更新方式是：

   $$\theta=\theta+\beta\sum_{p\in O(s_t)}(g(\theta^{(p)})-g(\theta)),\quad g(\theta)=-\mathbb{E}_{s_t\sim p(\cdot|\cdot), a_t\sim \pi_{\theta}^{sub}}[Q_{\theta}(s_t,a_t)]\tag{5}$$

   其中，$O(s_t)$表示状态$s_t$下的所有候选option集合，$\theta^{(p)}$表示子option策略$p$的参数。$\beta$表示学习率，$-g(\theta)$是meta policy对策略梯度的期望。
   
## 4.4 Other modules of the Option Discovery Framework
除了上述的模块外，作者还设计了其它一些模块，比如option samplinig module，history encoding module等。option sampling module负责产生候选option的集合。在实际运行阶段，会将系统状态作为输入，得到候选option的集合。history encoding module负责对历史经验进行编码，方便meta policy进行学习。

# 5.具体代码实例和解释说明
## 5.1 Python implementation
作者在Github上开源了自己的实现。具体的代码放在https://github.com/kevinzakka/pytorch-option-discovery，里面提供了训练脚本train_meta.py和测试脚本test_meta.py。以下是train_meta.py的内容：

```python
import torch
from models import *
from wrappers import *
from optionsampling import *
import numpy as np

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Define environment
    env = gym.make('Pendulum-v0').unwrapped
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Define actor-critic model
    model = ActorCritic(state_dim, action_dim).to(device)
    
    # Define experience buffer
    exp_buffer = Buffer(size=100000)
    
    # Initialize random seeds
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.seed(seed)
    
    # Define meta-optimizer
    meta_optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    # Define option sampler
    option_sampler = OptionSampler()
    
    # Train loop
    num_episodes = 10000
    max_ep_len = 200
    total_timesteps = 0
    ep_rewards = []
    
    for i_episode in range(num_episodes):
        episode_reward = 0
        
        # Initialize environment and get initial observation
        obs = env.reset()
        done = False
        info = None
        
        while not done:
            # Select an option based on current state
            action_sequences = option_sampler.get_actions(obs, [])
            
            # Choose an action from one of the option sequences
            subopt_idx = np.random.choice([i for i in range(len(action_sequences))], size=None)
            selected_seq = action_sequences[subopt_idx]
            action = selected_seq[-1]
            
            # Perform action in environment
            new_obs, reward, done, info = env.step(action)
            
            # Store transition in memory
            exp_buffer.add({'state': obs,
                            'new_state': new_obs,
                            'action': action,
                           'reward': reward,
                            'done': float(done),
                            })
            
            # Update total number of steps taken
            total_timesteps += 1
            episode_reward += reward
            
            # If enough samples are available in memory, update policy parameters
            if len(exp_buffer) >= int(max_ep_len * args.batch_size):
                batch = exp_buffer.sample(args.batch_size)
                
                states = [torch.tensor(transition['state'], dtype=torch.float, device=device).unsqueeze(0) for transition in batch]
                actions = [torch.tensor(transition['action'], dtype=torch.float, device=device).unsqueeze(0) for transition in batch]
                rewards = [torch.tensor(transition['reward'], dtype=torch.float, device=device).unsqueeze(0) for transition in batch]
                dones = [torch.tensor(transition['done'], dtype=torch.float, device=device).unsqueeze(0) for transition in batch]

                _, next_values = model(states[:-1])
                values = [value.detach().unsqueeze(0) for value in next_values]

                returns = compute_returns(rewards, values, dones, discount=args.discount)
                advantage = [(return_t - value_pred_t.squeeze()) for return_t, value_pred_t in zip(returns, values)]

                pi_losses = [-compute_loss_pi(advantage_t, old_probs_act_t, act_dist_t)
                             for advantages_t, old_probs_acts_t, act_dists_t in zip(advantage, old_probs, act_distributions)]

                v_losses = [F.mse_loss(values_t, returns_t) for values_t, returns_t in zip(next_values, returns)]

            meta_update(batch)
            updates += 1

        # Log episode statistics
        ep_rewards.append(episode_reward)
        print("Episode: {}, Total Time Steps: {}, Episode Reward: {:.4f}".format(i_episode, total_timesteps, episode_reward))
        
      ...
```

代码中主要定义了环境env，actor critic模型model，经验缓存exp_buffer，meta优化器meta_optimizer，option采样器option_sampler。

训练循环中，每次从经验缓存exp_buffer中随机采样若干批次经验，然后更新actor critic模型和meta策略。

option sampling module由OptionSampler类实现。在训练循环中，通过调用OptionSampler类的get_actions()方法，输入当前状态obs，得到候选option的集合。在训练模式下，通过不同的策略，可以从候选option中进行采样。测试模式下，可以通过已知的规则，或者机器学习方法，直接进行策略选择。

HER replay wrapper module由her.ReplayWrapper类实现。通过这个wrapper，可以在强化学习环境中，引入HER方法的经验重构。在上面的代码中，通过调用exp_buffer.add()方法，添加完整的经验$(s_t^a, r_t^a)$到缓冲区中，并且调用exp_buffer.sample()方法，从缓冲区中随机采样若干批次经验，传递给actor critic模型进行更新。

meta policy optimizer module的具体更新方法在meta_update()函数中实现。meta policy优化器meta_optimizer根据上层策略给出的动作序列，重新计算子option的动作分布，并且通过子option策略优化器来更新子option参数。

最后，训练日志打印到了终端窗口，并且保存了总奖励列表ep_rewards。

# 6.未来发展趋势与挑战
Option Discovery Framework 提供了一套完整的算法框架，可以解决open-ended decision problems。但是，其目前还是比较初级的算法，很多模块都需要进一步的研究，才能达到更好的效果。

首先，在option sampling module中，目前只有一种简单的方法，即从候选option中随机抽取一个。如何扩展这个模块，或者开发出更好的采样方法，是Option Discovery的关键之一。第二，在experience replay wrapper module中，HER方法只是简单的重构reward，而没有考虑到更细致的reward重构方式。如何改进这部分，使得算法能够更好地学习长远规划，是Option Discovery的一大亮点。第三，在子option optimizer module中，目前采用了简单的损失函数，没有考虑到更复杂的策略学习方式。如何设计出更加鲁棒的损失函数，提升子option策略的学习效果，也是Option Discovery的一个重要挑战。

最后，Option Discovery还有许多细节问题需要解决，比如suboptimality gap的问题，如何最小化策略依赖，等等。这些问题的研究将有助于提升Option Discovery算法的效果，甚至推广到其他decision problem上。