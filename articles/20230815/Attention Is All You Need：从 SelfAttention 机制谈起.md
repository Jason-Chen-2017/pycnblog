
作者：禅与计算机程序设计艺术                    

# 1.简介
  

自注意力（self-attention）机制，是在 transformer 模型中被提出的一个关键技术。它最初由斯坦福大学、新加坡国立大学等机构的研究者们提出，目的是为了解决机器翻译和图像识别领域中的长期依赖问题，即模型中出现太多长尾词汇或者错误词导致的性能下降。基于注意力的神经网络模型可以在学习阶段自动进行特征组合，能够对输入序列中的不同位置之间的关系建模，并且可以有效的处理变长序列，因此在 NLP 和 CV 领域取得了很大的成功。

# 2.基本概念术语说明
## Transformer
Transformer 是一种基于 attention 的神经网络模型。它主要由 encoder 和 decoder 组成。encoder 和 decoder 中都包含若干相同层数的堆叠的子层（Sublayers），每个子层又包括两个操作：multi-head self-attention 运算和 position-wise feed forward 运算。其中，attention 运算是一个关键的运算，其作用就是关注输入序列中某一点与其他点之间存在的关联性，并根据关联性对各个输入点进行加权求和得到输出。而 multi-head self-attention 则是对 attention 进行多头分离的操作。对于每一次 attention 操作，只需要计算一部分输入信息即可，这样就可以将模型参数量减少至一定的程度。

## Scaled Dot-Product Attention
Scaled Dot-Product Attention 是 attention 机制中的一种方法。给定 Q、K、V 三个矩阵（矩阵维度分别为 head 个数 * len(Q) * dim_q，len(Q) 为序列长度，dim_q 为 Q 的特征维度；K 和 V 维度均为 head 个数 * len(K) * dim_k/v，dim_k/v 为 K 或 V 的特征维度），计算 attention 时使用如下公式：

$$
\text{Attention}(Q,K,V)=softmax(\frac{(QK^T}{\sqrt{d_k}})V)\\
d_k=\text{dim}\text{(K)}
$$

其中，$softmax$ 函数用来归一化 attention 概率分布。$\frac{\sqrt{}}{d_k}$ 表示缩放因子，用于调整 softmax 函数对矩阵元素大小的影响。一般来说，$\text{dim}\text{(K)}\approx \text{dim}\text{(V)}$ ，因此缩放因子可以设为 1。

## Multi-Head Attention
Multi-Head Attention （MHA）是在同一个时刻对不同子空间的 feature map 进行 attention 计算，以获取更丰富的信息。具体地，MHA 将输入向量拆分为多个子向量，再将这些子向量分别做 attention，最后将结果拼接起来作为输出。如下图所示：


如上图所示，输入向量首先被分割成几个子向量，然后分别对这些子向量做 attention，最后将 attention 的结果拼接起来作为输出。这里的 head 可以理解为 subspaces 。每个 head 根据不同的特征进行 attention ，因此不同 heads 会捕获到不同的数据模式。因此 MHA 在一定程度上能够增强模型的表达能力。

## Positional Encoding
Positional Encoding 是一种常用的方式，可以让模型对于位置相关的信息有所掌握。其原理就是引入一些位置编码，使得词嵌入表示能够反映它们在句子中所处的位置信息。位置编码可以看作是位置向量，它与词嵌入相乘后会产生一个新的位置编码向量，表示这个词的位置信息。

最简单的方法是直接把位置信息编码成向量，如 "Hello" 这个词对应的位置向量可能为 [1,2] ，"world!" 这个词对应的位置向vedctor可能为 [2,3] 。这种方式的缺点是缺乏全局的考虑，只能局部编码位置信息。另一种方式是使用一串具有不同频率的 sinusoid 函数来编码位置信息，如下图所示：


假设有 $N$ 个位置向量，那么第 $n$ 个位置向量的形式可以用以下的公式表示：

$$
PE_{(pos,2i)} = sin(pos / (10000^{\frac{2i}{d_model}}))\\
PE_{(pos,2i+1)} = cos(pos / (10000^{\frac{2i}{d_model}}))
$$

其中，$PE_{(pos,2i)}$ 和 $PE_{(pos,2i+1)}$ 分别代表第 n 个位置向量在偶数维和奇数维上的分量，$d_model$ 表示模型的维度，$pos$ 是当前位置的序号，一般设为一个整数。那么通过以上公式计算出的位置向量的全体可以看作是一个正弦或余弦函数的周期性。其特点是全局编码位置信息，并且可以较好地保持词向量的稀疏性。