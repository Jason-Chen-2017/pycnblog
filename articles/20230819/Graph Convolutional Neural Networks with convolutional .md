
作者：禅与计算机程序设计艺术                    

# 1.简介
  

在图神经网络（Graph Neural Network）中，传播消息的关键一步就是如何处理图结构中的节点特征。传统上，节点的表示学习通常采用图卷积神经网络(GCN)方法[1]或自注意力机制(Self-Attention mechanism) [2]。两者都可以看作对节点特征进行整合的过程，但都缺乏足够的理论支持和实践经验。因此本文借鉴GCN的思想，将其作为正则化器应用到卷积神经网络CNN上。这种做法将CNN视为一种图形分类模型，并扩展CNN的适应性，提升其分类能力。因此通过卷积核权重矩阵，CNN可以模拟图卷积神经网络，并在训练过程中不断更新参数，以达到最佳效果。

本文主要基于以下观点：
1. CNNs are powerful feature extractors and graph based models can be seen as a special case of CNNs in which the input is represented by graphs rather than images or text.
2. We can apply GCN as a regularizer to any CNN architecture by simply modifying its convolution kernel weights during training. 
3. The learned convolutional features from GCN can serve as a rich input for other downstream tasks such as classification, regression, and segmentation. 

为了能够利用到图卷积神经网络中的一些优秀特性，本文选择了一种特定的图卷积网络设计方案——DeeperGCN [3],即多个单层的GCN组成了一个深层的GCN模块。其中每个GCN模块都由两个图卷积层组成，前一个图卷积层编码输入图的信息，后一个图卷积层解码抽象的特征表示。这样的设计带来两个好处：第一，更充分地利用了多层的GCN网络；第二，可以在训练过程中动态调整GCN的深度，防止过拟合。

同时，本文还需要解决一下三个核心问题：
1. 如何把CNN当做图分类模型？
2. GCN作为正则化器是否有效？
3. GCN学习到的抽象表示如何用在其他下游任务上？

# 2.背景介绍
## 2.1 图神经网络简介

图神经网络(Graph Neural Network, GNN)，是近几年才被广泛关注的一个技术领域。它所涉及的数学模型和技术，也正在逐渐成为许多学科领域的基础，如图分类、文本处理、计算机视觉等。GNN从最原始的图分析角度出发，将图划分成若干个子图或区域，然后在这些区域上定义节点间的关系。然后再考虑节点的聚集和传递，得到整个图上的全局表示。这些抽象操作分别由各类神经网络模型实现，比如图卷积神经网络(Graph Convolutional Neural Network, GCN)。下面是GCN的基本框架示意图:


GCN中的卷积核权重矩阵W对每一个节点x进行更新：wij = sum_neigh(xj * Xi), neigh表示与节点x相连接的所有邻居节点。该公式可以清晰地表达出GCN对邻接信息的聚合，并生成中心节点的嵌入向量。

根据输入的图的复杂程度，不同类型图有不同的GNN模型。目前主要分为三种：
- 无监督的图表示学习(Unsupervised Graph Representation Learning): 用GNN学习节点之间的相似性，主要用于节点分类、链接预测等任务。
- 有监督的图分类模型(Semi-Supervised Graph Classification): 在有标签的数据集上，同时使用节点标签和边信息，训练GNN来推理整个图的标签分布。
- 强化学习环境下的图任务(Reinforcement Learning Environment for Graph Tasks): 使用GNN模型来训练强化学习(RL) agent，在复杂的图环境中规划行为策略。

## 2.2 CNN简介

卷积神经网络(Convolutional Neural Network, CNN)是机器学习领域中最著名的图像分类模型之一。其在图像分类领域中获得了较好的效果。CNN由卷积层、池化层和全连接层构成，输入是一个高维的图像特征，输出是一个高维的分类结果。卷积层通过滑动窗口的形式对输入图像进行卷积，提取局部特征。池化层则对局部特征进行降采样，进一步提取全局特征。全连接层则将卷积和池化后的特征进行融合，得到分类结果。


<center>图1：CNN架构</center><|im_sep|>