
作者：禅与计算机程序设计艺术                    

# 1.简介
  


随着人工智能技术的迅速发展和传播，人们对下棋、围棋、象棋等纸牌类游戏的高精度控制需求越来越强烈。而这些要求，更多地依赖于对动作、状态等信息的充分理解，进而形成智能决策系统。机器人能够像人类一样认知和运用信息，并做出决策，也成为模仿学习的一种方式之一。

机器人学习的方式有很多种，其中最著名的就是基于值函数的强化学习方法。强化学习（Reinforcement Learning，RL）是一门关于如何使机器从环境中学习解决任务的方法论。其核心是一个智能体（agent），它通过与环境交互，接收奖励或惩罚，从而学会按照某种策略优化获得最大化的长期利益。由于在许多游戏领域都可以看到类似的学习过程，因此人们一直在探索如何利用强化学习方法来训练机器人来控制游戏。

本文将会着重分析一下，目前业界最主流的两款用于训练网络进行游戏 AI 决策的模型——AlphaGo Zero 和 AlphaZero。这两款模型都是基于强化学习方法训练的，而且采用了深度神经网络作为核心的网络结构，具有极高的学习能力。在本文的讨论中，我们只会从直觉上理解这些模型的工作原理，以及它们各自的特点与优缺点。

# 2.背景介绍

## 2.1 AlphaGo Zero

AlphaGo Zero 是 Google 团队于 2017 年提出的，一款结合了深度学习、蒙特卡洛树搜索（MCTS）和蒙特卡洛神经网络（NNet）的先进模型。它的玩法基于人类顶尖棋手围棋所下的棋谱数据，包括各种棋子位置、落子时的特征（如风云变化、气温变化等）等。该模型使用大量的游戏数据，包括完整的游戏棋盘信息、白、黑双方的棋子分布、落子顺序、对手的落子等，训练一个深度神经网络（NNet）模型，根据对手下一步的动作预测当前局面最有利的走法。之后，通过蒙特卡洛树搜索（MCTS）算法，模拟对手下每步的选择，最终得到模型所需的指导价值函数。


图 1: AlphaGo Zero 示意图

AlphaGo Zero 使用了一个卷积神经网络（CNN）处理输入图像，输出一个全局状态表示和七个不同动作对应的概率。全局状态表示由对称性、合法性和局势力量等多个特征组成，用作判断当前局面的胜负优劣。为了防止过拟合，AlphaGo Zero 在训练过程中加入了正则项约束，限制模型参数的范数不能太大。

蒙特卡洛树搜索（MCTS）算法是对搜索问题的一个有效且通用的框架，用来求解复杂问题的最佳路径。它的基本思路是将每个节点看作是一个“沙盒”，其中随机选择若干次子节点，然后根据子节点的访问次数计算其“平均”收益，最后返回达到目标值的节点作为最佳节点。MCTS 对下棋有独到的特性，可以考虑到对手的行动，并且采用蒙特卡洛规则保证收敛性和稳定性。

AlphaGo Zero 的优势主要在于：

1. **效率高**：AlphaGo Zero 用到了深度神经网络和 MCTS 技术，在训练过程中充分利用 GPU 的性能加快运算速度。同时，蒙特卡洛树搜索算法通过模拟对手下棋的方式找到了比较好的方案，提升了搜索效率。AlphaGo Zero 在围棋领域取得了当时记录，是国际顶级的AI之一。

2. **领域广**：AlphaGo Zero 可以直接用于游戏中的复杂棋盘，无需人工设计特征工程，而其他基于模型学习的方法都需要专业领域知识才能解决。

3. **效果好**：AlphaGo Zero 的表现远胜其他机器学习方法，在围棋、五子棋、战棋等经典游戏上取得了不俗的成绩。

但是，AlphaGo Zero 有一些局限性：

1. **联想搜索**：AlphaGo Zero 只使用局部信息进行决策，而没有全局信息，无法完整认识对手的战略和可能的走法，导致无法应付一些复杂的局面。

2. **专注于单子棋局**：AlphaGo Zero 仅适用于二维平面上的九宫格棋盘，对于更高维度或更复杂的棋盘无法充分发挥作用。

3. **小范围落子**：AlphaGo Zero 受限于小范围的落子，对于对手较为复杂的下一步行为的影响有限。

## 2.2 AlphaZero

AlphaZero 是微软亚洲研究院和 Deepmind 提出的新一代 AI 模型，它相比于 AlphaGo Zero 更加突破性，在多个棋类游戏上击败了人类职业棋手。


图 2: AlphaZero 示意图

AlphaZero 使用 AlphaGo Zero 中的深度神经网络改造而来，但完全重写了蒙特卡洛树搜索（MCTS）算法。在改写之前，AlphaGo Zero 使用的蒙特卡洛树搜索算法，是在一个围棋局面上执行对弈，假设对手是一位完全随机的“智能体”。当游戏结束后，评估一个棋子布局的好坏依然需要耗费大量时间，导致 MCTS 算法在计算指导价值函数时的效率低下。因此，为了提高 MCTS 算法的效率，AlphaZero 采用了一个完全不同的蒙特卡洛树搜索算法。

AlphaZero 不再使用传统的蒙特卡洛规则，而是采取了一种“公平竞争”的思路，即每个子节点引入了一定的探索率，在搜索过程中，将各节点的访问次数用二者的乘积来衡量，这样可让差距大的节点有机会被选中，探索性增强了。同时，通过实践证明，这种探索性机制能够带来更好的结果，提升了对手的下场。

AlphaZero 在不同游戏和难度层面上的表现都超过了 AlphaGo Zero。比如，它在五子棋上勇夺了当时最优雅的 AI AlphaGo，还在象棋领域赢得了本领。AlphaZero 在处理棋类游戏上独具慧眼，取得了非凡成就。

不过，AlphaZero 也存在一些局限性。首先，它是一个更加复杂的模型，涉及更多的变量和超参数，需要更多的训练数据和算力支持。另外，它仍然没有完全消除联想搜索的问题，由于使用了深度神经网络，只能捕捉局部信息。为了克服这些局限性，Google Brain 团队提出了一种新的模型—— AlphaStar，在多个游戏和强化学习环境上训练得到的模型，称为 AlphaStar。

# 3.核心算法原理和具体操作步骤以及数学公式讲解

## 3.1 AlphaGo Zero 算法

### （1）模型网络结构

AlphaGo Zero 的模型网络结构非常简单，只有四层全连接层。第一层是三层感知器（Fully connected layer with three hidden layers）；第二层是六个神经元（Six neurons for the policy head and value head）。第三层和第四层分别是针对策略头和价值头的输出，分别有 $6 \times 7 \times 64$ 和 $(1 + 6) \times 64$ 个神经元。输出向量大小为 $(\text{board_size} \times \text{board_size}) \times (6 \times 64)$ 和 $\text{board_size} \times (\text{board_size} \times \text{board_size})\times (64)$ 。


图 3: AlphaGo Zero 模型网络结构示意图

### （2）蒙特卡洛树搜索（MCTS）算法

蒙特卡洛树搜索（Monte Carlo Tree Search，MCTS）是一种搜索算法，用来找出游戏中最优的动作序列。它借助蒙特卡罗方法（Monte Carlo method）模拟对战过程，即每次模拟一个随机策略，并根据结果更新搜索树中的结点。蒙特卡洛树搜索算法包含以下三个步骤：

1. 初始化搜索树：在根结点处创建一个虚拟结点，表示整个搜索空间。

2. 搜索：从根结点开始，遍历整棵树，依据叶子结点的数量来决定是否扩展子结点，并在扩展后的结点上添加随机噪声。

3. 反向传播：模拟完成后，反向传播结果给各个结点，更新自己的胜率（Win rate），即随机策略下赢得对手的概率。

蒙特卡洛树搜索算法的具体流程如下：

1. 在当前结点处创建 $n$ 个子结点，依照蒙特卡洛公式，用以模拟 $n$ 次以随机策略下赢得对手的概率，称为“平均策略价值（Average action value）”。

2. 根据子结点对应的“平均策略价值”，选择其中具有最高值、概率最高的子结点作为父结点。

3. 重复以上步骤，直到达到搜索终止条件（如达到最大搜索深度或已找到满意解）。

4. 在每一步搜索结束后，以此为基础，结合历史访问情况，对结点进行“学习”，生成“价值”和“置信度”。

蒙特卡洛树搜索算法能够有效求解复杂问题，其理论基础是“博弈论”，是计算机科学领域的一大分支。它认为，智能体与环境之间存在着一套互动过程，智能体在其中根据自身的策略和遵循游戏规则来选择行动，同时环境也在不断反馈给智能体有关信息。智能体只能在这些信息的影响下，按照某个策略做出最优的决策。蒙特卡洛树搜索算法能够以有限的时间内，模拟出这一过程，找出最优的策略，即“最佳动作序列”。

### （3）训练过程

AlphaGo Zero 训练过程包括两个阶段：

1. 预训练：在训练前几千次迭代轮次，仅仅用强化学习方式对模型进行初始化，相当于预训练阶段，可以减少模型训练时对当前模型的依赖。

2. 微调阶段：在训练后几千次迭代轮次，使用蒙特卡洛树搜索算法进行训练。初始阶段，模型仅仅被训练成一个完全随机的 AI，随着蒙特卡洛搜索算法的不断迭代，模型逐渐被训练成更有策略性的 AI。

蒙特卡洛树搜索算法的训练步骤如下：

1. 从 $m$ 个初始位置中随机抽取 $b$ 个局面，作为输入图像。
2. 将 $b$ 个局面输入到 CNN 中，获取 $b$ 个局面对应的输出向量，即表示各局面全局状态的特征向量。
3. 通过蒙特卡洛搜索算法，模拟 $c$ 次对局，在每个对局中，以“平均策略价值”和“价值”来更新结点的访问次数、获胜次数和访问顺序。
4. 更新模型参数，使模型不断改善自己的策略，并最终获得更有策略性的 AI。

蒙特卡洛搜索算法的参数设置如下：

- $m$ 为模拟次数，默认为 $50$ ，代表从 $m$ 个初始位置中随机抽取样本数目。
- $b$ 为经验池大小，默认为 $256$ ，代表每个训练轮次输入的样本数。
- $c$ 为模拟回合数，默认为 $500$ ，代表蒙特卡洛树搜索算法运行次数。
- 每次蒙特卡洛搜索过程，都会记录当前状态下的访问次数、获胜次数、访问顺序等信息。

### （4）超参数设置

超参数设置如下：

- $\gamma$ 表示折扣因子，用来衰减最终的指导价值，默认值为 $0.95$ 。
- $lmbda$ 表示探索参数，用来控制探索水平，默认值为 $0.75$ 。
- $lr$ 表示学习率，用来控制模型参数的更新幅度，默认值为 $0.01$ 。
- $n_simulations$ 表示蒙特卡罗模拟次数，默认值为 $50$ 。
- $T_target$ 表示探索终止阈值，用来控制探索次数，默认值为 $10^6$ 。

### （5）数据集

数据集采用的是 Go、Chess AlphaZero 和 Shogi AlphaZero 中使用的合作伙伴合作，并进行了重采样和扩充，总共包含了近 100 万次游戏数据的样本。

## 3.2 AlphaZero 算法

### （1）模型网络结构

AlphaZero 的模型网络结构与 AlphaGo Zero 大致相同，但有些地方也有所不同。首先，AlphaZero 的模型网络结构更加复杂，包括十一层全连接层，模型规模更大。第二，除了采用全连接层外，AlphaZero 还添加了残差网络模块（Residual block）。


图 4: AlphaZero 模型网络结构示意图

### （2）蒙特卡洛树搜索（MCTS）算法

AlphaZero 使用的蒙特卡洛树搜索算法和 AlphaGo Zero 非常相似，但也有区别。首先，AlphaZero 采用了完全不同的蒙特卡罗树搜索算法。其次，AlphaZero 的蒙特卡洛树搜索算法和 AlphaGo Zero 不同，更加关注探索性，在蒙特卡罗树搜索的过程中引入了探索策略，增加了对自己模型参数的探索性。

AlphaZero 蒙特卡洛树搜索的具体算法如下：

1. 在当前结点处创建 $n$ 个子结点，依照蒙特卡洛公式，用以模拟 $n$ 次以最大策略下赢得对手的概率，称为“平均策略价值（Average action value）”。
2. 根据子结点对应的“平均策略价值”，选择其中具有最高值、概率最高的子结点作为父结点。
3. 重复以上步骤，直到达到搜索终止条件（如达到最大搜索深度或已找到满意解）。
4. 在每一步搜索结束后，以此为基础，结合历史访问情况，对结点进行“学习”，生成“价值”和“置信度”。

蒙特卡洛树搜索算法的不同之处如下：

1. AlphaZero 使用完全不同的数据结构——神经网络蒙特卡罗树（NNet Monte Carlo tree search，NN-MCTS）。它使用神经网络来代表状态的特征，而不是采用规则。
2. AlphaZero 使用置信度（Confidence）来评估每一次模拟的准确度，并在每一步选择中反映到结点上。
3. AlphaZero 使用神经网络来探索模型参数，并对模型进行训练，以实现更加自主的学习过程。

### （3）训练过程

AlphaZero 的训练过程也是两个阶段：

1. 预训练：和 AlphaGo Zero 一样，在训练前几千次迭代轮次，仅仅用强化学习方式对模型进行初始化，减少模型训练时对当前模型的依赖。

2. 微调阶段：在训练后几千次迭代轮次，使用蒙特卡洛树搜索算法进行训练。由于 NNet-MCTS 使用神经网络来模拟，不需要随机策略，因此不需要采集更多的数据。这也是 AlphaZero 相比于 AlphaGo Zero 更加强大的原因之一。

蒙特卡洛树搜索算法的训练步骤如下：

1. 从 $m$ 个初始位置中随机抽取 $b$ 个局面，作为输入图像。
2. 将 $b$ 个局面输入到 NN-MCTS 中，获取 $b$ 个局面对应的输出向量，即表示各局面全局状态的特征向量。
3. 通过蒙特卡洛搜索算法，模拟 $c$ 次对局，在每个对局中，以“平均策略价值”和“价值”来更新结点的访问次数、获胜次数和访问顺序。
4. 更新模型参数，使模型不断改善自己的策略，并最终获得更有策略性的 AI。

蒙特卡洛搜索算法的参数设置如下：

- $m$ 为模拟次数，默认为 $50$ ，代表从 $m$ 个初始位置中随机抽取样本数目。
- $b$ 为经验池大小，默认为 $256$ ，代表每个训练轮次输入的样本数。
- $c$ 为模拟回合数，默认为 $500$ ，代表蒙特卡洛树搜索算法运行次数。
- 每次蒙特卡罗树搜索过程，都会记录当前状态下的访问次数、获胜次数、访问顺序等信息。

### （4）超参数设置

超参数设置如下：

- $\gamma$ 表示折扣因子，用来衰减最终的指导价值，默认值为 $0.95$ 。
- $lmbda$ 表示探索参数，用来控制探索水平，默认值为 $0.75$ 。
- $lr$ 表示学习率，用来控制模型参数的更新幅度，默认值为 $0.01$ 。
- $T_target$ 表示探索终止阈值，用来控制探索次数，默认值为 $10^6$ 。
- $hidden\_layer\_size$ 表示隐藏层神经元个数，默认为 $256$ 。

### （5）数据集

数据集采用的是 Go、Chess AlphaZero 和 Shogi AlphaZero 中使用的合作伙伴合作，并进行了重采样和扩充，总共包含了近 100 万次游戏数据的样本。

# 4.具体代码实例和解释说明

AlphaGo Zero 和 AlphaZero 开源项目链接如下：


## 4.1 AlphaGo Zero 示例代码

AlphaGo Zero 中有一个实现训练和评估的示例代码，我们可以使用这个示例代码对 AlphaGo Zero 进行测试和训练。

```python
import numpy as np
from alphazero.Game import Game
from alphazero.NNetWrapper import NNetWrapper
from alphazero.GenericPlayers import RawMCTSPlayer
from alphazero.learn import learn

game = Game(6) # Set up game instance
nn = NNetWrapper(game) # Set up neural network instance
args = dotdict({'numMCTSSims': 50, 'cpuct':1.0}) # Set up training parameters
trainExamplesHistory = [] # Set up array to store examples

for i in range(1):
    print('Starting iteration', i+1)

    nn.load_checkpoint('./temp/', 'best.pth.tar') # Load saved model

    mctsPlayer = RawMCTSPlayer(game, args, nn) # Create MCTS player based on neural net
    winner = None
    
    while True:
        canonicalBoard = game.getCanonicalForm(game.getInitBoard(), game.curPlayer) # Get initial board representation

        if display:
            render_game(canonicalBoard, end='\r')
        
        pi = mctsPlayer.getActionProb(canonicalBoard, temp=0) # Get predicted move probabilities from neural net
        
        sym = game.getSymmetries(canonicalBoard, pi) # Generate all possible symmetries

        for b, p in sym:
            trainExamplesHistory.append([b, pi])
            
        action = np.random.choice(len(pi), p=pi) # Choose most likely move according to predicted probability distribution
        valids = game.getValidMoves(canonicalBoard, 1) # Get list of legal moves
        
        if valids[action] == 0: # Handle invalid move case
            print(valids)
            assert valids[action] > 0
        
        board, _ = game.getNextState(canonicalBoard, 1, action) # Apply chosen move to current state

        if game.getGameEnded(board, 1)[0]: # Check whether the game has ended
            break
            
    if len(trainExamplesHistory) > batchSize:
        # Sample training examples from history
        samples = random.sample(trainExamplesHistory, batchSize)
        X, y = [], []
        for sample in samples:
            X.append(np.array(sample[0]))
            y.append(sample[1])
        X = np.array(X).astype(float)
        y = np.array(y).astype(float)
    
        loss = nn.train(X, y) # Train neural network with sampled data
        
    save_model() # Save trained model after each iteration
    
```

## 4.2 AlphaZero 示例代码

AlphaZero 中有一个实现训练和评估的示例代码，我们可以使用这个示例代码对 AlphaZero 进行测试和训练。

```python
import torch
import numpy as np
from AlphaZero.processing.state_converter import StateConverter
from AlphaZero.processing.parallel_processor import ParallelProcessor
from AlphaZero.playout.simple_player import SimplePlayoutPolicy
from AlphaZero.trainer.supervised_policy_trainer import SupervisedPolicyTrainer
from AlphaZero.ai.rl_nn import RLNetwork

if __name__ == '__main__':
    device = torch.device("cuda:0")
    
    trainer = SupervisedPolicyTrainer(params={
        "n_states": StateConverter().output_shape()[0], 
        "n_actions": 7 * 6,
        "n_filters": 256,
        "filter_sizes": [5]*3,
        "hidden_units": 256,
        "batch_size": 1024,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4}, device=device)
    
    processor = ParallelProcessor()
    nnet = RLNetwork(device=device)
    
    # Train the supervised learning agent
    processor.run_supervised(nnet,
                             supervisor=trainer,
                             epochs=1000,
                             init_games=10,
                             update_epochs=100,
                             log_interval=10,
                             verbose=True)
    
    # Evaluate the learned agent
    num_games = 100
    res = evaluator.evaluate(SimplePlayoutPolicy(), processor.create_player(nnet))
    
    print("Final Result:",
          f"{res['won']} / {num_games} games won ({round((res['won'] / num_games)*100)}%)")

```

# 5.未来发展趋势与挑战

## 5.1 AlphaGo Zero 与 AlphaZero 之间的巨大区别

虽然两者都是基于强化学习方法训练的 AI 模型，但是两者却存在着巨大的区别。AlphaGo Zero 以聪明的手段对局、在多个游戏中击败了人类世界冠军，但它并没有被广泛采用，主要原因是它基于完全随机的蒙特卡洛树搜索算法，无法充分发挥自适应的探索机制。另一方面，AlphaZero 已经超过了 AlphaGo Zero ，并且获得了更优秀的性能，但它仍然有着巨大的潜力，因为它应用了专门用于强化学习训练的网络结构。

AlphaGo Zero 和 AlphaZero 在训练过程中都使用了蒙特卡洛树搜索算法，在探索过程中也都采用了 MCTS 方法。这两种算法都构建在概率论上，以期待在未来采取更好的决策。不过，这两种算法之间的区别也很重要，它们代表着两种不同性质的 AI 模型。

AlphaGo Zero 是一个全新的、高效的 AI 模型，它对局和博弈的理念都源自 AlphaGo 程序，是有史以来第一个采用蒙特卡洛树搜索算法的围棋模型。它的优势在于，它使用了强化学习和蒙特卡洛树搜索的最新思想，能够得到更好的结果，而且只需要极少的数据。不过，它也存在着一些局限性，比如它不能完整认识对手的进攻和防御策略，也不适合高维度棋盘、不适合高阶博弈。

AlphaZero 是一个应用深度学习技术的 AI 模型，它的创新之处在于，它构建了完全不同的数据结构——神经网络蒙特卡罗树（Neural Network Monte Carlo Tree，NNet-MCTS），它使用神经网络来代替规则，将策略抽象成状态空间和动作空间之间的映射。它可以更容易地捕捉局部信息，能够在模拟过程中增加自适应的探索能力。AlphaZero 引入 NNet-MCTS 后，就可以从任意状态下，生成一系列合法的行为并选择最佳行为。NNet-MCTS 所具备的自适应探索能力，使得它可以更好地掌握未来的局面，从而达到比 AlphaGo Zero 更好的效果。

## 5.2 深度学习与强化学习的结合

在深度学习与强化学习的结合方面，AlphaZero 拥有独特的优势。AlphaZero 用神经网络作为胜任的黑马，将蒙特卡罗树搜索与强化学习相结合，最终以比 AlphaGo Zero 更优异的性能赢得围棋比赛。这是因为神经网络可以在不依靠传统规则的情况下进行自我学习，并且可以自由地探索环境，因此可以处理高阶博弈，从而赢得比其他强化学习方法更强的胜利。

当然，AlphaZero 也存在着局限性。首先，它的学习能力依赖于大量的训练数据，而且要花费数年甚至更久的时间才会有成果。其次，它的搜索能力也受限于蒙特卡洛树搜索，而且不适合对高维度棋盘进行搜索。最后，AlphaZero 的学习过程还是受限于强化学习的片刻训练，因此它的表现不会随着数据的增多而变得更好。

# 6.附录：常见问题与解答

**问：**什么是蒙特卡罗树搜索？为什么需要蒙特卡罗树搜索？

**答：**蒙特卡罗树搜索（Monte Carlo Tree Search，MCTS）是一种搜索算法，用来找出游戏中最优的动作序列。其基本思想是，在每次搜索之前，对所有可能的路径进行评估，然后通过模拟随机策略来评估真实的行动路径，并使用统计学的方法来估计胜率。与启发式搜索不同，MCTS 不是从根本上考虑所有可能的搜索树，而是采用随机策略来进行模拟。这个方法通过多次模拟，不断构造出不同模拟路径下的动作评估值，从而寻找最优的路径。

MCTS 可用于各类游戏，其中有一些典型的应用场景，包括机器人对战、电脑博弈、股票交易等。它不仅可以帮助蒙特卡罗方法在多个游戏上训练出具有策略性的 AI，而且也可以帮助开发人员在许多领域中进行快速迭代和尝试。与蒙特卡洛方法不同，MCTS 能够有效处理较为复杂的问题，并且对搜索进行更少的迭代次数，因此可以节省计算资源。

**问：**AlphaGo Zero 和 AlphaZero 的区别有哪些？

**答：**AlphaGo Zero 和 AlphaZero 是基于强化学习方法训练的两款 AI 模型，它们都采用蒙特卡洛树搜索算法来进行训练和评估。它们各自的特点如下：

- AlphaGo Zero：是在 AlphaGo 程序的基础上建立的，是第一个采用蒙特卡洛树搜索算法的围棋模型，具有高度的效率和准确性。它拥有对局和博弈的理念，并采用了强化学习和蒙特卡洛树搜索的最新思想。但是，它的表现仍然有限，它不能完整认识对手的战略和可能的走法，也不适合高维度棋盘、不适合高阶博弈。AlphaGo Zero 在最近几年获得了不俗的成绩，并将围棋引擎推向了新的高度。

- AlphaZero：是一款应用深度学习技术的 AI 模型，它构建了完全不同的数据结构——神经网络蒙特卡罗树（NNet-MCTS），它使用神经网络来代替规则，将策略抽象成状态空间和动作空间之间的映射。它可以更容易地捕捉局部信息，能够在模拟过程中增加自适应的探索能力。AlphaZero 引入 NNet-MCTS 后，就可以从任意状态下，生成一系列合法的行为并选择最佳行为。NNet-MCTS 所具备的自适应探索能力，使得它可以更好地掌握未来的局面，从而达到比 AlphaGo Zero 更好的效果。

AlphaGo Zero 和 AlphaZero 的区别还有很多，比如，它们的训练时间、搜索方式、价值函数等等。这些差异将对围棋、斗地主等众多的游戏产生深远的影响。