
作者：禅与计算机程序设计艺术                    

# 1.简介
  

本系列博客将从A Primer of Neural Networks and Deep Learning(神经网络与深度学习导论)出发，系统性地回顾、总结并理解机器学习、深度学习的基础知识、技术、理论，力求做到知其然、知其所以然。本篇博客将介绍Attention Is All You Need论文中的第五部分（《Self-Attention with Relative Position Representations》）。在神经网络中，每一个节点都可以看做是一个神经元单元，它能够接受输入的数据、处理数据，并且给出输出结果。Attention mechanism是一种重要的深度学习技术，通过注意力机制，不同的输入信息被赋予不同的权重，从而实现不同子任务间的信息交互。Self-attention是一种特殊的Attention机制，其中查询和键之间的相对位置信息也会被学习到，因此，Self-attention具有更好的表达能力。本篇博客将主要介绍Self-attention及其局限性。
# 2.Attention Mechanism
Attention mechanism表示一种表征学习的过程。当输入数据过多时，如何选择重要的信息、忽略不重要的信息是一个需要解决的问题。Attention mechanism就是通过学习输入数据的注意力分布，根据注意力分布对输入数据进行重新加权处理，从而得到有用的输出。由于不同的输入项可能有不同的重要性，因此，Attention mechanism也是一种层次化学习的过程。Attention mechanism分为两种类型：全局Attention mechanism 和 局部Attention mechanism 。前者将整体输入考虑进去，后者只考虑局部区域的输入。如图所示，全局Attention mechanism 是基于整个输入序列计算注意力分布；局部Attention mechanism 只关注某一片段或一个区域内的输入。
图1:Attention Mechanism

# 3.Self-Attention Mechanism
Self-attention mechanism是一种特殊的Attention mechanism，其中查询和键之间的相对位置信息也会被学习到。Self-attention相对于传统的Attention mechanism有很多优点，比如可以增强模型的可塑性、对长距离依赖关系建模能力好、在不增加参数的情况下提高效率等。但是，由于Self-attention引入了位置信息，因此只能处理顺序数据。具体来说，Self-attention由两个步骤组成：Attention计算步骤和输出计算步骤。首先，利用queries和keys计算注意力矩阵，从而生成注意力权重；然后，用注意力权重与values计算输出向量。值得注意的是，Self-attention能直接利用局部的相对位置信息。具体而言，查询和键之间的相对位置信息会被学习到，并且可以直接利用。如下图所示，一个Self-attention模块在时间维度上循环计算。
图2:Self-attention Module

# 4.Relative Position Representation
为了使Self-attention可以利用局部的相对位置信息，作者们设计了一个新的相对位置编码方法。相对位置编码方法不是简单地将绝对位置信息作为向量拼接到每个向量上，而是将两个向量之间的相对位置信息作为向量拼接到每个向量上。具体来说，假设一个向量的长度为d，那么其相对位置信息就可以用另一个小于等于d的向量表示出来。例如，如果q=v，则r=-q，如果q<v，则r=(q-v)，如果q>v，则r=(q+d)-v，也就是说，在其他位置上的元素会对当前位置产生作用。将这些相对位置信息表示为d维的向量，可以让Self-attention能够很好地利用局部的相对位置信息。

与传统的位置编码方法不同，相对位置编码方法不会让模型陷入困境，因为它可以灵活地捕获不同距离之间的依赖关系。同时，相对位置编码方法也避免了绝对位置编码带来的问题，如相邻元素之间相关性过大、模型过于复杂。相对位置编码方法虽然不能替代绝对位置编码，但它可以很好地补充其缺陷。如图3所示，相对位置编码方法可以帮助Self-attention模块更好地捕获全局和局部的依赖关系。
图3:Positional Encoding

# 5.Conclusion
Self-attention mechanism是一种最新的深度学习技术，通过学习输入序列的相对位置信息，从而实现不同子任务间的信息交流。相比于传统的Attention mechanism，Self-attention可以在不增加参数的情况下提升效率、可以捕获更丰富的依赖关系、可以适应序列数据的顺序特征。但是，Self-attention无法直接利用局部的相对位置信息，这就限制了它的表达能力。随着技术的发展，局部的相对位置信息已经成为一种越来越重要的特征。因此，Self-attention是一类有待发展的新型Attention mechanism。