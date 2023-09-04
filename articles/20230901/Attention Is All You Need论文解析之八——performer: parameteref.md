
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## Performer: Parameter-Efficient Attention in Linear Time and Space
从标题可知，Performer是一种parameter-efficient的attention模型，作者声称该模型的时间复杂度为$O(L\sqrt{D})$、空间复杂度为$O(LD)$，是目前最快速度的attention模型。本文主要讲述了Performer模型的设计思想、基于Transformer的实现细节、效果、未来的发展方向以及性能对比等。

Performer是一个用于提升计算效率的参数量级模型，不同于目前存在的self-attention结构（线性时间复杂度$O(\text{seq_len}^2 \times d_{\text{model}})），Performer利用卷积神经网络中的神经元来做attention运算。相较于一般的CNN或者RNN进行特征抽取，Performer能够在不增加参数量级的情况下提升计算效率。Performer的基本思路就是先使用卷积神经网络生成一个基于位置的特征向量集合$K$, $Q$, 和$V$,然后通过点积运算得到注意力权重矩阵$\alpha$. 


这样一来，Performer只需要一次卷积运算就可以完成所有的attention计算，并且可以一次性得出所有维度上的attention权重。由于卷积运算的时间复杂度是$O(K\log_2 K)$,因此对于短序列来说，Performer的计算效率还是很高的。另外，使用了卷积运算，因此其参数量级和计算量级都远小于Transformer模型。

与其他模型比较，Performer对参数量级的要求相对更低，所以更适合于嵌入式设备或一些计算密集型任务中。除此之外，Performer同样兼顾准确性与效率，在很多领域都取得了优秀的成绩。

# 2.背景介绍
Attention机制在自然语言处理（NLP）、视觉任务中扮演着至关重要的角色，能够显著提升模型的性能。传统的注意力模型如Transformer、BERT等采用self-attention作为基础结构，能够捕捉全局依赖关系并有效地进行编码。但是，self-attention也存在着计算复杂度的瓶颈，因此出现了许多改进版的注意力模型，如long short-term memory (LSTM)、gated recurrent unit (GRU)、hierarchical attention networks (HAN)。这些模型虽然提升了计算效率，但依旧存在着较大的参数量级，在很多场景下都不能满足需求。

2020年初，微软亚洲研究院团队搭建了一个名为GPT-3的强大的NLP模型，它是一个基于transformer的Sequence to sequence (Seq2Seq) 预训练模型。其中，transformer模块构建了最为基础的encoder-decoder框架，而decoder部分则是借鉴GPT-2的变压器模型。GPT-3在无监督学习、评估、推断、总结、语法分析等多个NLP任务上都表现卓越，而且它的参数数量仅为175亿个。随着硬件性能的增长，以及模型的升级换代，越来越多的人开始关注计算资源的优化。

2021年初，阿里巴巴团队发表了一篇名为“Attention is all you need”的论文，针对自然语言处理领域的最新研究成果，提出了一种新的注意力机制，即performer。通过多层卷积神经网络的应用，Performer能够在不增加参数量级的情况下提升计算效率。实验结果表明，在英文文本、图像等具有全局依赖关系的数据集上，Performer的准确率可以达到甚至超过目前state-of-the-art的各种模型。另外，Performer在transformer模块的输出特征维度仍保持了较高的一致性，因此不受影响，与其他注意力模型兼容，实现了parameter-efficient。