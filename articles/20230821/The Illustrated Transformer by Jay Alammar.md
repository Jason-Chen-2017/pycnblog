
作者：禅与计算机程序设计艺术                    

# 1.简介
  

“The Illustrated Transformer” 是由 Jay Alammar 发表于 Medium 的文章，旨在用简单的图片和图文形式，准确又易懂地阐述 Transformers 的概念、结构、工作原理及其局限性。文章分为七章节，包括前言、导读、概览、编码器-解码器架构、位置编码、自注意力机制、混合注意力机制、条件语言模型和GPT-2模型。每章节末尾均有作者对该章节内容的评价，作者详细介绍了每一块知识点背后的理论和数学基础，还提供了相应的代码实现和实验结果。本文将尝试通过对 Transformers 相关知识的展示，帮助读者从不同的视角理解和运用 Transformers 在 NLP 和 CV 中的应用。
## 2.导读（Introduction）
Transformers 是近年来最火爆的深度学习技术之一，它已经成为 NLP 和 CV 领域中的新宠。那么，什么是 Transformers?为什么它如此受欢迎？它的主要特征是什么？其局限性又在哪里呢？这些都是我们需要了解和明白的。正因为如此，这篇文章就是为了解决这个问题而撰写的。作者从大量篇幅的研究与实践中总结出了一个完整的关于 Transformers 的知识体系。
## 3.概览（Overview）
### 3.1 NLP 中的 Transformers
先说一下 NLP 中的 Transformers。NLP 任务都离不开词嵌入(Word Embedding)和循环神经网络(Recurrent Neural Network, RNN)。RNN 的特点是会记录之前的信息，但是对于长文本来说，其训练困难，这时 Transformers 应运而生。NLP 中的 Transformers 可以被看作是 RNN 的升级版。它们的结构与 RNN 有些类似，但也有一些不同。这里作者给出一个例子。假设有一个句子 “I love you”，传统的词嵌入方法可能会得到如下向量表示：
```
[0.9  0.   0.7  0.3 -0.2...] # 表示 "I"
[0.8  0.4  0.3 -0.5  0.9...] # 表示 "love"
[0.    0.2 -0.4 -0.8 -0.6...] # 表示 "you"
```
RNN 的结构一般是这样的：
```
Embedding -> LSTM/GRU Layer -> Hidden State -> Output Layer
```
而对于 Transformers 来说，它的结构如下所示：

整个模型由多个 Encoder 和 Decoder 组成，每个 Encoder 负责产生隐藏状态，而每个 Decoder 根据隐藏状态生成输出序列。Encoder 和 Decoder 之间的信息传递是通过多头注意力机制进行的，并且这些注意力机制可以并行计算。因此，单个模型能够处理长度差异很大的输入序列，这使得它非常适合于处理长文档或者数据集。同时，由于采用这种结构，因此模型的参数量比 RNN 小很多。


### 3.2 CV 中的 Transformers
CV 中的 Transformers 与 NLP 中类似。除了输入的维度有所变化外，CV 中的 Transformers 的结构也与 NLP 中的基本相同。这里给出一个示例：

首先，介绍一下 Self-Attention 模块。Self-Attention 模块接收一个查询张量 $Q$ ，一个键张量 $K$ ，一个值张量 $V$ 作为输入。其中，$Q$, $K$ 和 $V$ 都是相同维度的张量，例如在图像分类任务中， $Q$, $K$ 和 $V$ 分别为特征图，可以表示为 $N \times C \times W \times H$ 。假设 $W$ 和 $H$ 为相等的值。那么 Self-Attention 的输出张量为 $\text{Attention}(Q, K, V)$ 。可以看到，该模块对输入张量做一次线性变换后，把它拆分成 Q， K 和 V 三者分别处理。然后，计算 Q 和 K 矩阵的内积，即得出权重矩阵 $A$ ，再乘上 V 矩阵，即可得出输出张量。

接下来介绍一下 Transformer 模型。Transformer 是一个编码器－解码器模型，可以处理序列到序列的问题，它的结构如下图所示：


Encoder 模块接受输入序列，将其拆分成多个子序列，然后利用 Self-Attention 对每个子序列进行特征提取。最后，通过多个层次的堆叠，将各个子序列的隐藏态映射到同一空间中。Decoder 模块则按照与 Encoder 相同的方式进行特征提取。但这里有一个重要的区别，那就是 Decoder 需要连续生成输出序列，而不是像 RNN 只能输出一个值。为了实现这一点，Decoder 使用 Masking 操作屏蔽掉未来的信息，这样才能让模型只关注当前的信息。

下面，我们就进入文章的第二部分——基本概念、术语说明。