
作者：禅与计算机程序设计艺术                    

# 1.简介
  

AlphaGo Zero，Google Deepmind在2017年开发了一种AI程序，它击败了世界围棋冠军柯洁，成为世界上最先进的围棋AI程序之一，甚至可以说是整个围棋界的巨无霸。AlphaStar，也就是以Go为研究课题的AlphaGo Zero，也使用了自我学习（self-play）的方法，由多个AI之间互相博弈，以提升自己对游戏的把握能力。

作为国际象棋界的顶级冠军，围棋的发展史可称之为“辉煌时代”。几百年前，周恩来和朱德就以孙子李斯，开始制定了一系列围棋法则，并发明出一些规则违例，如“七步杀、十步飞”等，这些规则不仅影响到围棋的风格和技艺，而且直接决定了围棋这个冠绝古老的棋类走向的方向。

那么，自从中日韩联手打败中国棋手将李世石后，围棋进入了一个新的阶段——“亚洲强者时代”。随着中国力量的减弱，围棋界掀起了一场“千年一计、一举多得”的围棋世界大赛，围棋界的许多人才和赢家都要了一张纸，争夺围棋王牌，而AlphaGo就占据了其中一个领先的位置。

虽然围棋界的战局仍然是极其残酷，但这并不是说围棋就应该被遗忘。近些年来，以AlphaGo Zero为代表的深度学习技术带给我们的启示，已经引导着人工智能的发展，并开始重塑现有的产业结构。

以AlphaStar为例，它是围棋AI中最前沿的项目之一，虽然它的算法复杂度很高，但是却能做到依靠计算机自动分析围棋局面，进行有效的搜索和对弈，并通过自我博弈的方式提升自己的胜率。它同时还采用了分布式计算平台，并采用多种策略，如蒙特卡洛树搜索、神经网络蒙特卡洛树搜索等，有效地优化了训练过程中的超参数选择。

# 2.背景介绍
AlphaStar是一个围棋AI项目，它的名称取自希腊神话中的阿拉贡。它是由多位围棋领域专家设计的，包括李平、李斯、李延炼等人。它的首席科学家是李子建。围棋AI项目有很多，包括俄罗斯方块、AlphaZero、Go playing bots等。但是，为什么会出现AlphaStar项目呢？

由于AlphaGo Zero的出现，围棋界有一批英雄扬威，比如李世石、柯洁、黄忠，他们统治围棋界三百年，屹立不倒。而且由于电脑算力的增长，围棋AI的研究和发展也越来越快。在2019年的开幕式上，AlphaGo于同年宣布获胜，这标志着围棋界迎来了一个新纪元的开始。

而到了2021年的春节，围棋界迎来了第一次疫情爆发，给围棋界的发展造成了巨大的冲击。随后，随着人们对疫情防控的关心加剧，围棋界也作出了反应，除了热度外，围棋AI的研究也逐渐开始放缓。直到最近，美国国家围棋联盟（NCAAB）在宣布禁止比赛，引发围棋界的一阵骚动。

在这样的背景下，AlphaGo Zero的成功彻底颠覆了围棋界三百年的梦想，而AlphaStar项目正好符合这股潮流。这也是作者的观察，他认为围棋AI的研究对社会的发展，无论对个人还是国家都是具有巨大的意义的。因此，AlphaStar的目标就是解决围棋界的问题，并且通过自我学习的方式提升围棋AI的能力，最终达到领先与独霸的境界。

# 3.基本概念术语说明
## （1）围棋规则
围棋的规则主要分为国际象棋规则和中国象棋规则。国际象棋规则中，又细分为九个格子排列的棋盘和黑白两方，双方轮流行动，落子位置必须形成五连线或四连线（横、竖、斜）或者活四（横、竖、斜）。第一子通过吃掉其他子就赢。棋盘大小是19x19，两方各执两个棋子。中国象棋规则与国际象棋规则类似，只是对角线连线的个数增加到六个。

围棋一般在双人对战模式下进行，每个玩家可以轮流控制两条不同的棋子，每一步只能在空白位置落子。游戏结束条件是：一方主动吃掉对方的所有棋子，且不能再进行合法的落子，这一方获胜；或双方都没有合法的落子，双方均输。

围棋的目标是用五子棋（通杀棋）或活四，吃掉更多的棋子并赢得比赛。围棋中有很多规则上的规定，如合法的走子范围、王路可行性、石头落子力度、禁手限制、边界线等等。围棋也有游戏规则的变体，如十步杀、五目相宜等。

## （2）蒙特卡洛树搜索（MCTS）
蒙特卡洛树搜索（Monte Carlo Tree Search，MCTS），是一种在决策问题上有效的搜索方法。MCTS通过模拟随机试验来评估各种可能的行动，然后根据这些评估结果构造出一棵树状结构——蒙特卡洛树。树的每一个叶节点表示一个可能的状态，树的根节点对应于当前游戏状态，而中间的中间节点表示不同于根节点的状态。每一次树搜索的开始，都会从根节点开始，按照某种启发式策略生成一条路径。随着搜索的进行，MCTS一直沿着这条路径选择子节点，直到到达叶节点，之后对该叶节点进行评估。最后，MCTS返回到父节点，使用价值网络（value network）或者策略网络（policy network）来决定在这条路径上的下一步是往哪走。

蒙特卡洛树搜索有以下几个优点：
1. 采样效率高：只需考虑已探索过的状态，不会陷入局部最优。
2. 收敛速度快：只要有足够的时间和资源，MCTS可以找到最优解，即使在大型问题上。
3. 鲁棒性强：对局面有一定的抗干扰能力，能够处理非完全信息下的问题。
4. 可扩展性强：不需要事先知道所有的状态，可以适用于许多问题。

## （3）神经网络
神经网络是由多个隐层（hidden layer）组成的多层次统计模型。输入数据会先经过一系列的处理过程（如卷积、池化等），然后传播到输出层，输出层根据输入数据的复杂程度，输出相应的值。

AlphaStar使用的神经网络有以下几个特点：

1. 基于蒙特卡洛树搜索（MCTS）的自我学习：AlphaStar使用MCTS来自我学习。首先，它训练两个神经网络——策略网络（policy network）和价值网络（value network），它们分别用来预测落子的概率和落子后对局面的估值。随着自我博弈的进行，模型会不断调整神经网络的参数，让它们更好的预测下一步的动作和对局面的估值。然后，它将策略网络的输出转换成用于博弈的动作概率分布。最后，它通过选择具有最大概率的动作进行落子。

2. 分布式计算平台：AlphaStar项目使用分布式计算平台，它可以训练多个AI模型，充分利用云服务器的计算能力。

3. 模块化设计：AlphaStar使用模块化的设计，允许用户根据需要增加、修改或替换神经网络的各个模块。例如，可以用别的神经网络来预测对手的策略和对局面的估值，或使用其他的策略函数来实现AlphaGo Zero的目标。

4. 强化学习：AlphaStar利用强化学习来训练模型，可以训练有策略的模型，提升训练效率。AlphaStar同时使用状态空间的表示方式，可以更好地捕捉局面的全局特征，而不是局部的局部特征。

# 4.核心算法原理和具体操作步骤以及数学公式讲解

## （1）蒙特卡洛树搜索
蒙特卡洛树搜索是AlphaStar项目的核心算法。它将蒙特卡洛模拟算法和树形搜索算法结合起来，构建一颗蒙特卡洛树，并用它来指导AI模型的搜索行为。蒙特卡洛树搜索的具体步骤如下：

### 1) 初始化蒙特卡洛树
初始化蒙特卡洛树，即创建一个跟节点，其包含所有状态节点，初始状态节点的价值是0。

### 2) 策略网络
每一步，模型都会预测一组动作对应的概率分布，并将其映射到蒙特卡洛树中。这里用到的是策略网络，它是一个神经网络，输入是当前状态的表示（如棋盘配置），输出是动作概率分布。

### 3) 树搜索
蒙特卡洛树搜索的关键是采用树搜索的方式来探索未知的状态空间。在每一步，AI都会按照策略网络生成一系列可能的动作，并将这些动作对应的子节点添加到树中。然后，它选择一个动作，沿着这条路径探索蒙特卡洛树。如果在搜索过程中发现有价值较高的叶节点（即证据），它就扩展这些叶节点，继续探索这些节点的后继节点。如果在搜索过程中遇到一个价值较低的叶节点，它就回退到父节点，重新探索新的子节点。树搜索的终止条件是到达游戏结束状态。

### 4) 价值网络
每当蒙特卡洛树搜索到达一个新的状态，它就会使用价值网络来估计该状态的价值。对于目标状态，它的估值会较高；而对于远离目标状态的子节点，它的估值会较低。

### 5) 更新树搜索的策略
在树搜索的每一步，AI都会更新它的估值网络的参数，以便让它的搜索更有针对性。当模型训练好后，AI就可以应用它来选择下一步的动作。

## （2）神经网络
AlphaStar项目使用了两种类型的神经网络，即策略网络（policy network）和价值网络（value network）。策略网络接收棋盘状态的输入，输出动作概率分布。价值网络接受棋盘状态的输入，输出该状态的估值。

### （2.1）策略网络
AlphaStar的策略网络是一个深度残差网络（ResNet）。它包含一个卷积层、两个残差块，以及一个全连接层。

卷积层用来提取局部信息，通过对局部关系进行抽象。残差块包含三个卷积层，并采用跳跃连接。跳跃连接的目的是解决梯度消失和梯度爆炸的问题。

全连接层用来输出动作概率分布。它使用softmax激活函数，将输出映射到0~1之间的实数上，且总和等于1。

### （2.2）价值网络
AlphaStar的价值网络是一个卷积神经网络。它包含三个卷积层和一个全连接层。卷积层用来提取局部信息，全连接层用来输出估值。

### （2.3）网络架构
AlphaStar的网络架构如下图所示：


左侧是AlphaGo Zero的网络架构。右侧是AlphaStar的网络架构。

## （3）分布式计算平台
AlphaStar项目使用分布式计算平台来训练多个模型，提高训练速度。平台可以部署在云服务器上，并通过分布式训练来降低单机的训练耗时。分布式训练可以有效地利用云服务器的计算能力，并提升训练速度。

## （4）超参数选择
AlphaStar项目使用超参数调优来优化模型的性能。超参数调优的目标是寻找能够提升模型训练效果的最佳设置。一般来说，超参数调优方法有随机搜索法、贝叶斯优化法、遗传算法等。

## （5）蒙特卡洛树搜索算法

蒙特卡洛树搜索算法由以下几步构成：

1. 随机选择一个结点作为根节点。
2. 在该结点所有子节点中随机选择一个子节点，作为下一个待扩展的结点。
3. 如果该结点没有子结点，则扩展该结点，计算其价值并赋值给它的价值函数，然后返回到第2步。
4. 如果该结点有一个子结点，则选中该子结点，返回到第2步。
5. 如果该结点有两个以上子结点，则对这两个子结点进行独立的扩展，并选出其中一个作为下一个待扩展的结点。
6. 如果这个结点的价值大于等于第一个选出的子结点的价值，则重复第3-5步。否则，回溯到父结点，并重复第3步。
7. 当到达搜索树的最底端，或者搜索到达最大搜索次数时，停止搜索。
8. 返回搜索树的根节点，即当前状态的概率。

蒙特卡洛树搜索算法的每一步的运算比较复杂，为了方便理解，我们来看一下简单的例子。假设有一个游戏场景，现在有两种类型的角色，它们可以执行一些操作。游戏结束条件是某个角色达到某个数量，那么游戏结束。现在假设我们有两个角色，一个对战另外一个，游戏进行到第三步时，它们的对手是策略函数f(s)，其中s表示当前的状态，f(s)表示根据状态s产生的动作概率分布。游戏开始时，我们假设各角色各有一个棋子，现在假设角色1的棋子位置为a，角色2的棋子位置为b。现在我们希望搜索完当前的局面，得到最有可能的下一步操作。

首先，我们初始化蒙特卡洛树，选择根节点，并令a为待扩展的结点。然后，我们根据当前状态，预测该状态下每个动作的概率。对于每个动作，我们选择下一步的结点，并重复第3-5步，直到到达搜索树的最底端或者搜索到达最大搜索次数。搜索到达最大搜索次数后，选择有较高奖励的那个动作作为当前动作，并作为搜索结束。

# 5.具体代码实例和解释说明

## （1）训练流程

AlphaStar项目的训练流程大致可以分为以下几步：

1. 获取训练数据：AlphaStar需要海量的数据才能训练成功。训练数据包括了游戏记录、训练参数、神经网络参数等。

2. 数据预处理：训练数据需要进行预处理。首先，数据集中的游戏记录需要转化成AlphaGo Zero的输入形式，即状态表示。然后，数据集中的训练参数需要进行归一化，保证数值的稳定。

3. 搭建网络：搭建神经网络。AlphaStar使用ResNet和卷积神经网络作为网络模型。

4. 训练网络：训练网络，监控模型训练过程，提升模型性能。

5. 测试网络：测试网络，验证模型训练是否正常。

6. 提供服务：提供训练好的模型供用户使用，或者直接应用训练好的模型对局局面进行分析。

## （2）代码实现

AlphaStar项目的代码实现主要是Python语言，使用PyTorch框架。下面是AlphaStar项目的关键代码。

### （2.1）神经网络

AlphaStar的神经网络是基于ResNet的。ResNet的主要改进点有：

1. 残差结构：ResNet通过引入残差结构来解决梯度消失和梯度爆炸的问题。残差结构的基本思想是让网络能够学习到输入和输出之间的小变化，并通过学习得到中间变量，避免了网络的大幅度损失。

2. 跳跃连接：ResNet中的跳跃连接是一种重要的组件，能够帮助模型学习到复杂的非线性变换。

3. 下采样：ResNet采用下采样的方式来减少网络大小，并降低计算复杂度。

下面是AlphaStar项目中PolicyNetwork类的实现。

```python
class PolicyNetwork(nn.Module):
    def __init__(self, board_size=19, input_channels=2, output_dim=19*19+1):
        super().__init__()

        self.board_size = board_size
        self.input_channels = input_channels
        self.output_dim = output_dim

        layers = [Conv2d(in_channels=input_channels, out_channels=256, kernel_size=(3,3), stride=1, padding=1)]

        for i in range(4):
            # add Residual block with two convolutional and one batchnorm layers each.
            resnet_block = nn.Sequential(
                Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding=1),
                BatchNorm2d(num_features=256),
                nn.ReLU(),
                Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding=1),
                BatchNorm2d(num_features=256))

            if i == 0:
                # first block doesn't need to downsample
                pass
            elif i == 1:
                # second block needs to downsample by factor 2 using maxpool
                resnet_block.add_module('downsample', nn.MaxPool2d((2,2)))
            else:
                # third, fourth blocks need to downsample by factor 2 using conv + stride 2
                resnet_block.add_module('downsample', Conv2d(in_channels=256, out_channels=256, kernel_size=(1,1), stride=2))
            
            layers.append(resnet_block)
        
        # last fully connected layer outputs policy logits (i.e., softmaxed action probabilities)
        layers.extend([
            Flatten(),
            Linear(in_features=256*self.board_size//2**len(layers[-3:])*self.board_size//2**(len(layers)-4), out_features=256),
            nn.ReLU(),
            Linear(in_features=256, out_features=self.output_dim)])

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return F.log_softmax(self.model(x), dim=-1)
    
    @property
    def device(self):
        """Gets model's device."""
        return next(self.parameters()).device
```

PolicyNetwork的构造函数定义了网络的结构，包括卷积层、残差块、全连接层。在forward函数中，网络接受棋盘状态作为输入，输出动作概率分布。log_softmax函数负责将网络的输出转换成对数似然值，因此输出的是对数概率。

### （2.2）蒙特卡洛树搜索

蒙特卡洛树搜索算法实现是在tree.py文件中。AlphaStar项目的树搜索算法有两种：前向搜索和逆向搜索。前向搜索由MCTS算法实现，逆向搜索由TD-lambda算法实现。MCTS算法根据树搜索的方式进行蒙特卡洛模拟，模拟出各个可能的动作对应的状态值。TD-lambda算法通过遍历历史数据的状态序列，逐步修正树搜索算法的偏差。

AlphaStar的蒙特卡洛树搜索算法实现非常简单。下面是tree.py文件中的Node类：

```python
class Node():
    def __init__(self, state, parent=None, prior=1.0):
        self._state = state      # current game state
        self._parent = parent    # pointer to node's parent
        self._children = {}      # pointers to node's children
        self._w = 0              # total visit count from this node to leaf nodes
        self._q = 0              # mean value of leaf nodes from this node
        self._n = 0              # number of visits on this node
        self._p = prior          # initial prior probability of choosing this node
        
    def select(self):
        '''Selects a child node according to UCT rule'''
        child = None
        best_uct = float('-inf')
        
        # loop through all possible actions at this node
        for a in self._children:
            n = self._children[a]._n   # number of visits on child node
            q = self._children[a]._q   # mean value of child node
            
            # calculate upper confidence bound (UCT) score
            exploitation = q
            exploration = math.sqrt(math.log(self._n)/float(n))
            ucb = exploitation + self._p * exploration
            
            if ucb > best_uct:
                best_uct = ucb
                child = self._children[a]
                
        assert child is not None, 'Error: unable to find valid move'
        
        return child
    
    def expand(self, actions, probs):
        '''Expands tree by adding new child nodes.'''
        assert len(actions) == len(probs), "Error: length mismatch between actions and their corresponding probabilities"
        for a, p in zip(actions, probs):
            s = make_move(self._state, a, self.player) # generate child state given action and player color
            
            # create child node object and add it as a subnode to the tree
            c = Node(s, parent=self, prior=p)
            self._children[a] = c
            
    def update(self, v):
        '''Updates stats after subtree traversal has finished'''
        while self!= None:
            self._n += 1
            self._w += 1
            self._q += v
            
            # backpropagate updated stats to root node
            self = self._parent
    
    def state(self):
        '''Returns current state'''
        return self._state
```

Node类存储了蒙特卡洛树的一个节点的相关信息。select函数根据UCT规则从当前节点选中一个子节点；expand函数扩展当前节点，添加新的子节点；update函数更新当前节点的统计信息。Node类还提供了几个其他的方法，如children()、is_leaf()、active_children()等。

AlphaStar的蒙特卡洛树搜索算法实现位于MCTS()函数中，具体的算法步骤如下：

1. 从根节点开始，依据先验分布进行搜索。
2. 每一步选择当前节点下的一个子节点。
3. 生成子节点对应的所有状态。
4. 对每个状态进行模拟，返回其平均奖励值。
5. 根据各个状态的奖励值和先验分布，重新调整每个子节点的权重和概率。
6. 将调整后的子节点加入到树中。
7. 重复2-6步，直到搜索到达叶节点。

```python
def MCTS(root, num_simulations, cpuct):
    '''Runs MCTS simulations starting from specified root node'''
    
    # set player color based on root node's current player position
    player = get_player(root.state())
    
    # run simulation loops until we have completed enough simulations
    for _ in range(num_simulations):
        node = root
        
        # start from root node and traverse downwards through the tree
        while True:
            if node.is_leaf():         # if current node is a leaf, perform rollout to determine its value
                break
                    
            # otherwise, choose among active children (those who haven't been visited enough yet)
            actives = node.active_children()
            if len(actives) == 0:     # if there are no eligible children, back up to previous decision point
                node = node._parent
                continue
            
            # select an appropriate child node based on UCB formula
            child = np.random.choice(list(actives))
            
            # descend down the search tree towards the selected child node
            node = child
        
        # once we reach a leaf node, simulate its outcome from scratch using rollout function
        leaf_states = []             # record visited states during rollout procedure
        leaf_actions = []            # record taken actions during rollout procedure
        prev_state = node.state()
        done = False                 # initialize flag indicating whether episode terminates
        
        # roll out episode starting from current state and following the optimal path determined by neural network
        while not done:
            legal_actions = legal_moves(prev_state)
            legal_probs = predict_action_probabilities(prev_state)[legal_actions]
            action = np.random.choice(list(zip(legal_actions, legal_probs)), p=[p for _, p in sorted(zip(legal_actions, legal_probs))])
            prev_state, reward, done = step(prev_state, player, action)
            
            # record visited state and action for later processing
            leaf_states.append(prev_state)
            leaf_actions.append(action)
        
        # compute average reward obtained from performing all recorded actions from this leaf node to its parent node
        values = reward_func(np.array(leaf_states))        # evaluate rewards along the simulated path
        avg_reward = sum(values) / len(values)
        
        # walk back up the tree to propagate updates to relevant statistics
        node = root
        while node is not None:
            node.update(avg_reward)                    # update node's accumulated reward and visit count
            actives = list(filter(lambda x: hash(str(x.state())) not in seen, node.active_children())) # filter unseen children
            
            # remove any inactive children that might have become inactive due to exhausted visits or terminal nodes
            while len(actives) < len(node._children):
                del node._children[tuple(sorted(set(node._children) - set(actives))[0])]
            
            # reset history of previously visited states when visit count drops below threshold 
            if node._n >= EXPLORATION_PARAM:
                seen.clear()
                seen.add(hash(str(node.state())))
                
            node = node._parent
```

MCTS()函数接受两个参数，第一个是根节点，第二个是模拟次数。每次调用函数时，树搜索算法都会选择一条路径进行模拟。模拟结束后，算法会通过反向传播更新树搜索算法的统计信息。

## （3）自我博弈

AlphaStar项目中的自我博弈部分实现在train.py文件中。训练算法的主要步骤如下：

1. 创建训练数据集。加载经历过的游戏记录，构造状态表示；加载经验池中保存的训练参数，对其进行归一化。

2. 创建神经网络模型。创建PolicyNetwork和ValueNetwork对象，并将它们加载到GPU上。

3. 设置优化器。设置学习率和优化器，如Adam。

4. 使用MCTS进行自我博弈。从根节点开始，使用蒙特卡洛树搜索算法来选择动作，更新神经网络的参数。

5. 训练网络。对每一个样本，使用蒙特卡洛树搜索算法，从根节点开始进行自我博弈。训练过程中，每一步都使用蒙特卡洛树搜索算法选择动作，并在每一步更新神经网络的参数。

6. 使用PPO进行策略梯度的优化。使用Proximal Policy Optimization (PPO)算法来优化策略网络的参数。

7. 更新经验池。每隔一段时间，保存经历过的游戏记录和训练参数。

8. 使用最新训练参数，进行测试。在每次测试的时候，使用蒙特卡洛树搜索算法从根节点开始进行自我博弈，评估模型的表现。

下面是train.py文件的代码实现。

```python
# Step 1: Create training dataset
dataset = ExperienceDataset("data", use_weights=True)

# Step 2: Construct Neural Network Model
policy_net = PolicyNetwork().to(DEVICE)
value_net = ValueNetwork().to(DEVICE)
target_value_net = ValueNetwork().to(DEVICE)

if args.load_checkpoint:
    checkpoint = torch.load(args.load_checkpoint)
    policy_net.load_state_dict(checkpoint['policy'])
    target_value_net.load_state_dict(checkpoint['value'])
    
# Step 3: Set Optimizer
optimizer = optim.Adam([{"params": policy_net.parameters()}, {"params": value_net.parameters()}], lr=LEARNING_RATE)

# Step 4: Self-Play
for epoch in range(EPOCHS):
    loader = DataLoader(dataset, BATCH_SIZE, shuffle=True)
    
    for sample in tqdm(loader):
        boards, targets = prepare_batch(sample)
        
        with torch.no_grad():
            # Sample opponent's moves
            moves = available_moves(boards).reshape(-1, 19*19+1)
            opp_probs = predict_opponent_action_probabilities(boards).reshape(-1, 19*19+1)
            opp_probs *= mask_invalid_moves(boards, invert=False).reshape(-1, 19*19+1)
            probs = (moves * opp_probs) / (moves * opp_probs).sum(axis=1).reshape((-1, 1))
            
            # Update Root Node
            mcts = MCTS(root, NUM_SIMULATIONS, CPUCT)
            actions, priors = mcts.run(boards, drop_out_value=args.drop_out_value)
            root.expand(actions, priors)
        
        optimizer.zero_grad()
        
        # Compute Losses and Backward Pass
        boards = boards.to(DEVICE)
        actions = tensorize_actions(actions).long().to(DEVICE)
        priors = tensorize_priors(priors).to(DEVICE)
        
        log_probs = policy_net(boards).gather(1, actions)
        value = value_net(boards)[:, :, 0].gather(1, actions[:, :, 0]).mean()
        target_value = target_value_net(targets)[:, :, 0].detach().mean()
        
        advantage = normalize(rewards - target_value)
        loss_policy = -(log_probs * advantage.detach()).mean()
        loss_value = (F.smooth_l1_loss(value, advantage)).mean()
        loss = args.loss_coeff * loss_policy + LOSS_VALUE_COEFF * loss_value
        
        loss.backward()
        optimizer.step()
```