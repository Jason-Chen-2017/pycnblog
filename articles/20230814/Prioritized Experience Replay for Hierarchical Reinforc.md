
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Reinforcement learning (RL) is a powerful technique for developing intelligent agents that can learn to act in the real world by interacting with environments and receiving rewards based on their actions. In recent years, hierarchical reinforcement learning (HRL), which combines RL algorithms at different levels of abstraction, has been increasingly used to solve challenging problems such as robotic manipulation or complex game playing. However, there have been limited attempts at using priorities in HRL, since it requires careful design of replay memory storage and data processing techniques. 

To address this issue, we propose Prioritized Experience Replay (PER) for HRL, an extension of standard DQN-based RL algorithm designed specifically for HRL scenarios where experience samples are generated at multiple levels of abstraction and need to be combined together before training. PER involves storing a set of transition tuples, each representing an interaction between an agent and its environment. The tuples are ranked according to their importance during training, resulting in more informative updates towards optimal policies. We also provide theoretical analysis justifying the use of PER over other common replay schemes for HRL tasks. Finally, we present experimental results demonstrating how our approach significantly outperforms existing methods on various domains including locomotion control and language understanding. 


# 2.基本概念术语说明
Prioritized experience replay (PER) is a type of off-policy algorithm for deep reinforcement learning (DRL). It was first proposed for discrete action spaces and is now being widely adopted for continuous action spaces in multiagent settings. Similar to other off-policy algorithms like Q-learning, the key idea behind PER is to randomly sample from a replay buffer to update the policy network, but instead of uniform sampling, priority weights are assigned to the transitions in the replay buffer, which determines the likelihood of selecting these transitions during sampling. These weights depend on several factors, such as TD error and magnitude of reward signal, and allows for balancing exploration and exploitation of the agent's behavior during training. Another advantage of PER is that it does not require any adaptation of the target network architecture or hyperparameters, making it easy to plug into most existing DRL frameworks.


In Hierarchical reinforcement learning (HRL), the same RL algorithm is applied at different levels of abstraction, meaning that state representations may differ at each level. This makes it difficult to combine experiences across levels without introducing bias, especially if some levels take longer than others to complete an episode. To overcome this challenge, we propose to apply PER within each individual level separately, then collectively combine them through prioritization. Each level will maintain its own replay buffer and train using only those transitions that were sampled from it. During training, the combined replay buffer will be processed by assigning higher weight to transitions that contributed to good performance across all levels, while decreasing their importance when they did not contribute much. This ensures that the agent focuses on areas that require significant improvements rather than simply exploring in search of better options overall. Overall, this process helps ensure that the agent learns to balance efficient exploration at lower levels with effective learning at higher levels.


We define the following terms:

**Replay Buffer:** A collection of past observations, actions, states, and rewards received by the agent during training. At every time step t, the agent interacts with the environment and receives feedback via its observations and rewards. Some examples of replay buffers include frame stacks, n-step returns, and proportional prioritization.

**Experience Tuple:** A tuple containing one or more consecutive interactions between the agent and its environment, along with additional information such as the corresponding next state and terminal flag. For example, an experience tuple could contain the current state observation S(t), the chosen action A(t), the obtained reward R(t+1), the new state observation S(t+1), and the terminal indicator T.

**Level:** An abstract grouping of related tasks within the task hierarchy. For instance, a high-level goal might consist of multiple subtasks involving navigating a physical space or performing a sequence of actions in a virtual environment. Examples of levels include "navigation" or "language understanding".

**Hierarchy Level:** A specific node in the task hierarchy that corresponds to a particular level of abstraction. This could correspond to a single task or a group of closely related tasks, depending on the complexity of the problem being addressed. For instance, the navigation hierarchy could comprise a local planner, global planner, and controller nodes.