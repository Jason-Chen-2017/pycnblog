
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Transformer 模型已经成为构建大规模语言模型的标准方法之一。最近的研究表明， Transformer 在内存消耗、训练速度等方面存在着巨大的性能缺陷。为了提升 Transformer 的性能，作者们在设计新模型时，开始重新考虑如何减少模型的显存消耗，同时保持模型的准确率。本文将从两方面出发，分别是：
- Reformulate the original attention mechanism as a lightweight memory bank
- Propose a new positional encoding that leverages both content and location information in language modeling tasks

基于以上两个改进点，作者提出了一个名为 GPT-3 的模型，该模型具有比原始的 Transformer 更高效的训练和推理时间，并且可以更好地处理长文档、对话等领域任务。下面的内容将会描述这两个改进点的实现方式。
# 2.Reformulating Attention Mechanism
## 2.1 Memory Bank
Transformer 中的多头自注意力机制（multi-head self-attention）是一个高度复杂且占用内存较大的模块。由于 self-attention 需要计算输入向量之间的相似性，因此当模型输入变长时，需要增大模型的显存消耗。为了解决这个问题，作者提出了一种轻量级的记忆库（memory bank）。该记忆库包含多个随机初始化的小矩阵，每个矩阵都可以看作是一个知识库或缓存区，存储着一个注意力头的权重向量。通过矩阵乘法，可以快速计算输入向量与缓存中的各个向量之间的相似性，从而避免了直接计算输入向量之间的相似性导致的大量计算开销。此外，作者还引入了残差连接（residual connection），使得 Transformer 中间层的输出可以直接加到记忆库中进行下一次注意力计算。
## 2.2 Positional Encoding with Relative Distance
Transformer 在计算编码器输出时，一般采用绝对位置编码，即将位置信息作为单独的向量嵌入到输入序列中，因此对于较长的序列，其维度会很大。为了缓解这个问题，作者在位置编码中引入相对距离信息。相对距离信息主要用来指导词或短语在句子中的相对位置关系，相对距离可以由下式计算得到：

r_{ij}=\sqrt{\sum_{k=1}^K(q_i^k-q_j^k)^2}

其中，$q_i^k$ 和 $q_j^k$ 分别表示第 i 个位置的 k 个查询向量，K 表示查询向量个数。在作者的实验中，作者设置 K 为 64。这样，相对距离矩阵 r 可以被嵌入到位置编码中。相对距离信息有助于捕捉输入序列中相对位置上的关系，从而让模型能够学到更丰富的上下文信息。最终，位置编码的计算如下图所示：

PE(pos,2i)=sin(\frac{pos}{10000^{\frac{2i}{dim}}}) \quad PE(pos,2i+1)=cos(\frac{pos}{10000^{\frac{2i}{dim}}})

其中 pos 是输入序列的位置，dim 是位置编码的维度。