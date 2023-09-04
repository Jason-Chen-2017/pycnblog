
作者：禅与计算机程序设计艺术                    

# 1.简介
  

> 本文旨在总结与分享一下我在AI领域的一些经验。本文的主要内容包括机器学习、深度学习、强化学习以及数据分析的基础知识。希望能够对读者提供一些启发或借鉴。


# 2. 概览
- **机器学习**

  - 定义
  - 分类
  	- 监督学习
    - 无监督学习（聚类）
    - 半监督学习（标注少的数据集）
  	- 强化学习（游戏）
  - 算法
  	- 线性回归
  	- 逻辑回归
  	- KNN
  	- Naive Bayes
  	- SVM
  	- 决策树
  	- GBDT/XGBoost
  	- Deep Learning (DNN)
    - Convolutional Neural Networks(CNN)
    - Recurrent Neural Networks(RNN)
    - GAN（Generative Adversarial Networks）
  - 特征工程
  	- 数据预处理
  	- 文本数据
  	- 图像数据
  - 模型评估
  	- 交叉验证
  	- 超参数优化
  - 模型部署
  	- RESTful API
  	- Microservices Architecture
  	- Serverless Computing
  	- Containerization and Orchestration Tools
  

- **深度学习**
  
  - CNN（卷积神经网络）
  	- LeNet
  	- AlexNet
  	- VGG
  	- ResNet
  	- DenseNet
  - RNN（循环神经网络）
  	- LSTM
  	- GRU
  - GAN（生成式对抗网络）
  	- 生成模型（判别模型）
  	- 损失函数
  	- 生成器网络
  	- 判别器网络
  	- 对抗训练
- **强化学习**
  
  - Markov Decision Process
  	- MDP建模
  	- Value Iteration
  	- Q-Learning
  - Monte Carlo Tree Search(MCTS)
  	- 动作选择
  - AlphaGo
  	- 深度学习
  	- 模拟退火
  	- 蒙特卡洛树搜索（MCTS）
  	- 自博弈
  	- Go 19x19 棋盘游戏
  - Open AI Gym
  	- CartPole-v1
  	- Pong-v0
  - Unsupervised learning
  	- k-means聚类算法
  	- DBSCAN聚类算法
  	- 层次聚类算法

- **数据分析**

   - Pandas
   - Matplotlib
   - Seaborn
   - Scikit-learn
   - Statsmodels
   - TensorFlow / Keras
   - Pytorch
   
# 3. 一些基础知识点
## （1）分类模型
  
### （1.1）监督学习  
　　监督学习，也称为教学学习、有监督学习或者标注学习，是通过给定的输入、输出及目标，训练一个模型，使模型可以对新数据进行正确的预测或分类。常见的监督学习算法包括朴素贝叶斯、决策树、随机森林、支持向量机等。  
  
### （1.2）无监督学习（聚类）  

　　无监督学习，也叫非监督学习，是指对数据没有人为的标签。聚类算法就是一种无监督学习的方法。常见的聚类算法包括K-means、DBSCAN、层次聚类等。  
  
### （1.3）半监督学习（标注少的数据集）  
  
　　半监督学习，即部分拥有标注数据的样本集合中既含有少量无标注数据。解决这一问题的常用方法有将数据分成两组，其中一组作为有标注数据，另一组作为无标注数据，然后利用有标注数据训练模型，再利用无标注数据测试模型。还有一种方法是先用有标注数据训练模型，然后将未标记数据划分到每个类别的概率分布图上，根据概率分布图上形状将未标记数据分配到最可能属于哪个类别。  
  
### （1.4）强化学习（游戏）  
  
 　　强化学习，也称为博弈学习、有奖励的学习，是基于马尔可夫决策过程的概念。该方法从环境中接收到初始状态，并在每一步都由策略做出决定，从而实现最大化累计奖赏。它适用于许多与人工智能相关的问题，如约束满足问题、股票市场、机器人控制、智能电网、游戏、智能驾驶等。深度强化学习（Deep Reinforcement Learning，DRL），是强化学习中的一种应用。DRL通过构建深度神经网络来学习环境的动态特性，从而达到自动化学习的目的。常用的DRL算法有DQN、A3C、PPO等。