
作者：禅与计算机程序设计艺术                    

# 1.背景介绍




近几年来，神经网络已经在NLP领域中扮演着越来越重要的角色，主要原因就是其能够很好地解决序列数据的建模、分类和生成等任务。特别是在深度学习兴起之前，RNN这种比较简单的模型已经能够取得不错的结果。但是随着深度学习技术的迅速发展，出现了诸如Transformer、BERT等基于Attention的模型，它们不仅可以对序列数据进行建模，而且可以有效解决长时依赖的问题。因此，基于Attention机制的模型逐渐成为当下NLP领域中最热门的模型之一。



本文作者Vaswani、Shazeer等人在2017年NIPS发表了Transformer模型[1]，它主要受到“Self-Attention”模块的启发，将注意力机制应用于机器翻译、文本摘要、问答系统等任务中。通过引入Attention机制，Transformer能够在一个序列上关注不同位置的信息并产生全局有效的表示，从而克服了RNN、CNN等传统模型在处理序列信息时的一些缺陷。Transformer的结构十分简单，但却具有极大的计算效率。本文主要是围绕这个模型进行阐述。

2.核心概念与联系


为了理解Transformer，首先需要了解一下相关的一些概念。



## Transformer概览


Transformer是一个由Self-Attention层和多头注意力机制组成的编码器－解码器(Encoder-Decoder)模型。它的特点是端到端的训练，并且在计算复杂度方面远远超过其他模型。如下图所示，Transformer包括两个相同的结构相同大小的子层，即encoder和decoder。Encoder把输入序列$X=\left\{x_{1}, x_{2}, \cdots, x_{t}\right\}$变换成一种固定长度的表示$z=\left\{z_{1}, z_{2}, \cdots, z_{t}\right\}$；Decoder则根据此表示生成输出序列$Y=\left\{y_{1}, y_{2}, \cdots, y_{n}\right\}$。







在实践中，通常使用Encoder来处理输入序列中的全局特征，例如符号表示或词嵌入；而Decoder则利用Encoder提供的上下文信息来完成当前目标的生成，例如语言模型或序列到序列的翻译任务。

## Self-Attention


在每个子层中，Transformer使用Self-Attention层来提取输入序列中全局特征。Self-Attention层采用了两个相同的子层，分别被称作Multi-Head Attention层和Positionwise Feedforward层。如下图所示，其中红色框内代表自注意力层。







### Multi-head attention layer


Multi-head attention层由多个并行的自注意力子层组成，每一个子层都使用不同的线性变换计算输入序列与其自己的部分之间的相似性，然后将这些相似性汇总得到最终的表示。这样做可以使得模型能够捕获不同位置之间的关联关系，从而更好地处理长时依赖问题。

具体来说，每个子层首先使用$Q,K,V$矩阵将输入序列$X$变换成$Q=(q_{\cdot 1}, q_{\cdot 2}, \cdots, q_{\cdot d})^{T}$、$K=(k_{\cdot 1}, k_{\cdot 2}, \cdots, k_{\cdot d})^{T}$、$V=(v_{\cdot 1}, v_{\cdot 2}, \cdots, v_{\cdot d})^{T}$三个向量。然后通过求内积得到输入序列各元素对另一部分元素的关联性，并用softmax归一化得到权重分布，进而得到最终的表示$Z=Softmax(\text{Score}(Q, K))V$。其中$\text{Score}(Q, K)=QK^T/\sqrt{d_k}$为匹配函数，$d_k$为表示维度。



### Position-wise feed-forward network


第二个子层是Position-wise Feed-Forward Network（FFN），它由两层神经网络组成，分别应用于输入的序列$Z$和输出的序列$Y$，前者用于计算中间表示，后者用于生成输出序列。具体来说，FFN层的计算公式如下：




$$
FFN(x)=\max (0, xW_1+b_1) W_2+b_2
$$




其中$x$为输入向量，$W_1, b_1, W_2, b_2$为参数。



### Embedding and softmax


在进入下一章节前，还有一个重要的步骤是对输入序列进行Embedding。Embedding是将原始输入映射到高维空间的过程，目的是使得语义信息更容易被模型学习到。由于词嵌入矩阵会非常大，因此一般只在开始阶段一次性学习，之后只使用对应的词向量。



最后一步是使用softmax归一化函数将输出序列的预测分布改造成实际的标签分布。



## Encoder and Decoder


Transformer也使用两个相同的结构相同大小的子层——Encoder和Decoder——来处理输入序列和输出序列。Encoder和Decoder都由多层相同结构的子层组成，在这种情况下，每个子层由两个相同的子层组成，分别是Multi-Head Attention层和Position-wise Feedforward层。







但是与之前的Self-Attention层不同，在Encoder和Decoder的子层之间不共享参数。这意味着对于输入序列的第i个位置，Encoder和Decoder的每一层只能看到到序列的前j个位置（j>=i）。在实践中，Transformer一般选择几层为共享层。



同时，不同层在学习输入序列的方式也不同。Encoder通常在学习全局特征，而Decoder则在学习局部特征。因此，Decoder的第一层可以更关注编码器的输出，而Encoder的最后一层可以更关注解码器的输出。