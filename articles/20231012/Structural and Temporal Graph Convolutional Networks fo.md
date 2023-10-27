
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


Traffic forecasting is an essential component of urban traffic management system that provides valuable information to transportation planners and decision-makers on the congestion status and flow patterns in real time. Among various approaches, graph convolutional networks (GCNs) have shown promising performance with good accuracy, scalability and robustness. Despite their recent success, there are several limitations such as no global view or long term memory to capture the spatio-temporal dependencies across different road segments. In this paper, we propose a novel framework called ST-GCNN for traffic forecasting based on structural and temporal GCNs (ST-GCNN). The framework integrates both spatial and temporal features by using an effective combination of structural information obtained from graphs and temporal information obtained from sequential data. The key idea behind our approach is to first define two layers of ST-GCNN separately for capturing local dependencies and globally connected dependencies across multiple adjacent segments. We then combine these separate layers together into one unified model for predicting future traffic flows over multiple horizons. Moreover, we introduce attention mechanism to focus more important nodes during training to improve the learning efficiency. The experimental results show that our proposed method outperforms existing state-of-the-art methods significantly for short-term prediction tasks. Further analysis also reveals that our method achieves better performance under high-speed rush hour conditions where the stochastic dynamics change rapidly due to vehicular activities.
# 2.核心概念与联系
## 2.1 结构图卷积网络（Structure Graph Convolutional Network）
结构图卷积网络是一种基于图卷积神经网络(Graph Convolutional Neural Network, GCN)的网络模型。它提出了将空间关系和时间关系结合的方法进行交互学习，通过学习全局信息从而提升预测准确性。GCN在节点特征表示、图分类任务等方面都有着广泛的应用。结构图卷积网络将所有节点之间的空间依赖关系融入到卷积核中，通过对邻域节点进行池化得到空间特征图，进一步增强空间相似性；同时还包括时间相关的信息，通过引入时间卷积层可以捕获节点随时间变化的特性。结构图卷积网络有如下几个主要优点：

1. 多模态学习：节点特征从不同的视角融合，充分考虑节点之间的空间依赖关系及时间序列关系。

2. 时空记忆建模能力强：能够建模不同时间段内的时空依赖关系，实现长期记忆和全局记忆功能。

3. 适应多尺度数据：由于其具有空间与时间的双向交互性，所以适用于不同类型的数据，如静态图、动态图、时序信号、文本数据等。

4. 自适应采样方法：结构图卷积网络采用自适应采样方法，即不仅考虑节点位置信息，而且利用邻居节点之间的邻域信息对采样点进行进一步约束，最终获得更精准的结果。

5. 端到端训练：结构图卷积网络可以端到端地进行训练，不需要像传统的GCN那样先生成一个子图再用其做节点分类，可以直接利用原始的图数据进行训练。

## 2.2 时序图卷积网络（Temporal Graph Convolutional Network）
时序图卷积网络（Temporal Graph Convolutional Network, T-GCN）是在结构图卷积网络上的扩展，它同时考虑节点间的时间依赖关系。T-GCN借鉴时序卷积神经网络的思想，将时间上的相关性引入到GCN的卷积核中，从而提取时间上的特征，形成时序特征图。时序图卷积网络有如下几个主要优点：

1. 考虑时间依赖关系：T-GCN可以考虑到节点的时序关系，从而利用时间上的特征，增加预测的准确率。

2. 解决长期记忆问题：因为T-GCN考虑到了节点间的时间依赖关系，因此可以提供全局信息，长期记忆能力比结构图卷积网络好很多。

3. 提供时序特征：T-GCN可以提供每个节点在不同时间上的特征，提高预测的准确率。

4. 更优秀的预测能力：T-GCN可以很好地学习到节点间的空间和时间依赖关系，因此可以提供更好的预测能力。

5. 可扩展性强：T-GCN的扩展性强，可以在不同类型的图上进行训练，例如静态图、动态图、时序信号、文本数据等。

## 2.3 整体框架
结构图卷积网络和时序图卷积网络各自独立地抽取空间特征和时间特征，并通过融合两种特征，构建一个统一的预测模型，达到整体提升预测准确率的目的。整体的框架如图所示：