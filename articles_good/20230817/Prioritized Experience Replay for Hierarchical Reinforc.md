
作者：禅与计算机程序设计艺术                    

# 1.简介
  

hierarchical reinforcement learning(HRL)近几年火了一把。其主要的思想是基于agent-environment的联合学习方法。传统的HRL一般都是在离散动作空间上进行的，而一些连续动作的游戏环境的场景也将HRL作为一种有效的方式。但是，由于传统的方法中采取的策略网络往往过于复杂，而连续动作的场景下需要高阶的策略结构才能有效利用多种上下文信息。因此，本文提出的Prioritized Experience Replay (PER)可以结合深层HRL的特点和连续动作的特点，能够有效解决这两个问题。
# 2.相关论文
## Deep reinforcement learning: An introduction 
## Policy Gradient Methods 
## AlphaGo and AlphaZero
## Human-level control through deep reinforcement learning 
## Hindsight Experience Replay 
## Asynchronous Methods for Deep Reinforcement Learning 
# 3. 基本概念和术语说明
hierarchical reinforcement learning：集成强化学习。它不是单纯的多个agent并行执行，而是有父子关系的多个agent之间相互影响。这种关系可以很大程度地模拟人的行为，也就是说，父agent可以影响到它的孩子agent的行为。这种集成学习的最佳实践方法，是先用较低层次的agent学习到一个良好行为策略，然后再用更高层次的agent去继承这个策略，并且与父agent一起共同学习。这样的话，子agent就有机会提高效率，而且能够更加灵活地选择自己的行为方式。hierarchical reinforcement learning的基本模型是马尔可夫决策过程。马尔可夫决策过程是一个Markov decision process，其中每一步agent的动作都依赖于上一步的所有动作，所以称之为"马尔可夫"。 
experience replay：对于RL算法来说，由于agent无法完全的观察环境，所以经验数据是不可得的。为了克服这一困难，可以使用experience replay机制。Experience replay一般分两种，一种是存储记忆库中的经验数据，另一种则是在网络训练时随机抽样经验数据。 
prioritized experience replay：由于存在优先级不均衡的问题，所以提出了prioritized experience replay。它根据episode的重要性来调整样本的权重，使得重要的经验比不重要的经验被更多的反复利用。
model-based RL：基于模型的RL的方法可以认为是一种与环境无关的预测模型，agent可以根据预测的结果做出动作。模型学习的目的是帮助agent快速、准确地预测环境的变化。比如DQN网络里，模型可以学习如何预测Q值，使得Q值快速更新；DDPG网络里，模型可以学习如何控制动作，使得策略快速更新。 
actor-critic method：actor-critic 方法是深度强化学习的核心方法，其中actor负责产生动作指令，critic负责评估actor所产生的动作的效果。可以看到actor-critic方法与传统的DQN、DDPG方法不同，只在actor和critic之间引入模型，actor-critic方法可以在一定程度上减少参数数量，同时也保留了Actor-Critic的优点。
# 4. 核心算法原理及具体操作步骤
Hierarchical Reinforcement Learning与Model-Based Reinforcement Learning的关系非常紧密。前者是一个agent面临复杂任务，后者是一个agent由经验的知识得到启发，发现新可能性。所以，我们首先来看一下Hierarchical Reinforcement Learning。
Hierarchical Reinforcement Learning的核心思想是，在更高层次的agent学习到一个良好行为策略之后，再通过传递信息的方式，引导底层agent学习到一个更好的行为策略。其主要流程如下：
1. 在顶层的agent中，选取一些基本的动作策略，例如最简单的轮流尝试不同的动作。然后，给这些策略分配一个初始的回报。 
2. 低层级的agent根据它们的策略生成对应的奖励，并反馈给顶层的agent。
3. 顶层的agent根据低层级的agent的奖励，通过学习，产生一个最优的整体行为策略。
4. 通过经验回放，高层的agent可以从底层的agent的经验中学习到一些技巧，并将这些技巧传递给低层的agent。
5. 重复以上步骤，直到所有的agent都学习到最优的策略。

对于Perirtdoted Experience Replay，其基本思想是，训练集中经验的优先级较小，并且适当地赋予其更大的权重，这有助于降低学习偏差。具体实现中，我们为每个episode分配了一个优先级，这个优先级越高，说明该episode的重要性越高，在学习过程中应予以更多的关注。同时，PER还可以通过缓冲区的大小来设置优先级。

最后，Hierarchical Reinforcement Learning与Model Based Reinforcement Learning的结合也是提高性能的一个方向。Model-Based Reinforcement Learning是一个与环境无关的预测模型，可以直接用经验数据进行训练。而Hierarchical Reinforcement Learning可以充分利用Model-Based Reinforcement Learning的预测能力。例如，低层级的agent可以根据高层级的agent的预测结果，并结合环境的真实情况，更高效地学习到最优的策略。 

# 5. 代码实例和说明
最后，我们用代码实例和Python语言展示一些关键的操作步骤：

1. 创建两个不同level的agent：

   ```python
   import gym 
   from hrl import create_hrl_env, create_hrl_agents
   
   # define two different level agents 
   env = gym.make("CartPole-v1")
   agent_low = create_hrl_env(env, policy='random', model='linear')
   agent_high = create_hrl_env(env, policy='random', model='linear')
   print(agent_low.policy)   # output random action policy for low level agent
   print(agent_high.policy)  # output random action policy for high level agent
   ```

2. 将低层级agent的动作加入经验池中，并为其分配奖励：

   ```python
   # generate initial actions for the low level agent
   init_actions_low = [agent_low.act() for _ in range(len(agent_low))]
   
   # append initial experiences to memory of low level agent
   for i, a in enumerate(init_actions_low):
       obs, r, done, info = env.step(a)
       if done:
           break
       exp = {'state': np.array([obs]), 'action': np.array([a]),'reward': np.array([r]),
              'next_state': np.array([[np.nan]*4]), 'done': np.array([True])}
       agent_low._memory.append(exp)
       
   print(agent_low._memory[-1]['action'])    # output last added action by low level agent
   
   # assign rewards to initial experiences of low level agent
   for ep in range(len(agent_low)):
       r = sum([-1 if j!= k else -5*(ep+1) for j in range(4) for k in range(j)])
       agent_low._memory[ep]['reward'] += np.array([r]).reshape(-1)
   
   print(agent_low._memory[-1]['reward'][0][0])     # output last assigned reward by low level agent
   ```

3. 用PER训练顶层的agent：

   ```python
   # train top level agent using PER
   from prioritized_replay import ProportionalPERBuffer
   
   buffer = ProportionalPERBuffer(capacity=1000, alpha=0.5, beta=0.5)
   num_episodes = 500
   
   for episode in range(num_episodes):
      state = env.reset()
      while True:
          # select an action based on current state of both levels
          action_top = agent_high.act(state)
          action_low = agent_low.act(state)
          
          # take action in environment and get new state and reward
          next_state, reward, done, info = env.step((action_low, action_top))
          
          # update experience in each level's memory
          exp_low = {'state': np.array([state]), 'action': np.array([action_low]),
                    'reward': np.array([reward]), 'next_state': np.array([[next_state]]), 'done': np.array([done])}
          agent_low._memory.append(exp_low)
          exp_top = {'state': np.array([state]), 'action': np.array([action_top]),
                   'reward': np.array([sum(agent_low._memory[-1]['reward'].tolist())]),
                    'next_state': np.array([[next_state]]), 'done': np.array([False])}
          agent_high._memory.append(exp_top)
          
          # add experience to prioritized replay buffer
          transition = {
                "state": state,
                "action": action_top,
                "reward": reward,
                "next_state": next_state,
                "done": int(done)
            }
          buffer.add(transition, priority=(ep+1)**-beta)
          
          # sample batch from prioritized replay buffer
          transitions, indices, weights = buffer.sample(batch_size=minibatch_size)
          
          states = torch.from_numpy(np.vstack([t['state'] for t in transitions])).float().to(device)
          actions = torch.from_numpy(np.vstack([t["action"] for t in transitions])).long().to(device)
          rewards = torch.from_numpy(np.vstack([t["reward"] for t in transitions])).float().to(device)
          next_states = torch.from_numpy(np.vstack([t["next_state"] for t in transitions])).float().to(device)
          dones = torch.tensor([t["done"] for t in transitions], dtype=torch.int8).unsqueeze_(1).to(device)
          
          optimizer_top.zero_grad()
          
          Q_values = critic_target(next_states)
          _, pred_actions = actor_target(next_states)
          
          target_Q_values = rewards + gamma * Q_values[range(batch_size), pred_actions] * (1-dones)
          
          predicted_Q_values = critic(states)[range(batch_size), actions]
          
          loss_Q = F.mse_loss(predicted_Q_values, target_Q_values.detach())
          
          loss_Q.backward()
          nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
          optimizer_critic.step()
          
          optimizer_top.zero_grad()
          
          pred_actions = actor(states)
          
          act_probs = F.softmax(pred_actions, dim=-1)
          log_prob = F.log_softmax(pred_actions, dim=-1)
          entropy = -(log_prob * act_probs).sum(-1, keepdim=True)
          
          advantage = target_Q_values - predicted_Q_values.detach()
          
          weighted_probs = (weights*advantage.double()).exp()
          weighted_cum_loss = (-weighted_probs*log_prob[:,actions].double()*advantage.double()).mean() / T_horizon
          
          weighted_cum_loss -= c_1*entropy
          
          weighted_cum_loss.backward()
          nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
          optimizer_actor.step()
          
          scheduler_actor.step()
          scheduler_critic.step()
          
          if len(buffer)<batch_size or not learn_flag(): continue
          
          with torch.no_grad():
              
              sampled_indices = np.random.choice(len(buffer), size=batch_size//T_horizon, replace=False)
              sampled_transitions = []
              for index in sampled_indices:
                  prob_sum = sum([sampled_transitions[-1]["weight"]**alpha
                               for sampled_transitions in buffer._priority_queue[:index]])
                  beta = min(1., ((len(buffer)-index)/batch_size)**-beta)
                  weight = (len(buffer)-index)*prob_sum/(batch_size*beta)
                  sampled_transitions.append({"weight": weight})
                  
              new_priorities = np.array([transition["weight"] for transition in sampled_transitions])**alpha + \
                             epsilon*np.max(new_priorities)**(1-alpha)
              
              for index, transition in zip(sampled_indices, sampled_transitions):
                  buffer._update_priorities(index, new_priorities[index-sampled_indices[0]])
          
          # synchronize updated parameters between all agents
          soft_update(actor, actor_target, tau)
          soft_update(critic, critic_target, tau)
   
   print('Training finished.')
   ```

4. 可视化顶层的agent的学习过程：

   ```python
   import matplotlib.pyplot as plt
   
   scores_episode = list(zip(*scores))[0]
   mean_score = [np.mean(scores[i:]) for i in range(num_episodes)]
   
   fig, ax = plt.subplots()
   line_mean, = ax.plot(list(range(num_episodes)), mean_score, label="Mean Score per Episode", color='blue')
   
   xdata, ydata = [], []
   def onclick(event):
       global ix, iy
       if event.inaxes == ax:
           xdata.append(event.xdata)
           ydata.append(event.ydata)
           line, = ax.plot(xdata, ydata, marker='o', markersize=10, color='green')
           line.set_label('Clicked Point')
           textbox.set_val("({:.2f}, {:.2f})".format(event.xdata, event.ydata))
    
   cid = fig.canvas.mpl_connect('button_press_event', onclick)
   textbox = TextBox(ax, 'Selected point:', initial="")
   
   plt.legend()
   plt.show()
   ```

5. 推荐阅读
如果您想进一步了解PER，推荐阅读以下内容：

1. PER: A Study of Sample Complexity and Loss Tradeoffs，这篇文章是PER的第一作者，对PER的基本原理和特性进行了非常好的描述。
2. Efficient Exploration through Optimism in Deep Reinforcement Learning，这篇文章是PER的第二作者，介绍了一种新颖的优化方案Optimism，据称可以提高RL的探索效率。