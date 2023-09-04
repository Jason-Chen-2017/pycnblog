
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 一、模型概述

Reformer是一个Transformer变体，是为了解决Transformer中的效率问题而提出的。它引入了两种新方法：Reversible network和LSH attention。

## 二、主要优点

1. Reversible network: 概念类似于LSTM，通过反向传播（backpropagation）的方式更新参数，可以在不依赖中间状态的情况下进行梯度计算，使得训练更加快速；

2. LSH attention: 使用局部性sensitive hashing (LSH)的方法来计算注意力矩阵，相较于一般的attention机制，LSH可以降低内存占用和计算量，减少计算时间，提高效率。



## 三、主要结构图


1. Encoder Layer:

    a. Self-attention layer:

      - Multi-head attention mechanism is used in this part of the model for reducing redundancy and capturing local dependencies between different positions in the sequence. The queries, keys and values are obtained from the input sequences using the same dense layers with learnable projections. A scaled dot product attention function is used to compute the attention weights based on these three vectors and performs softmax normalization over them.

      - Residual connection is applied here to add the self-attention output to the original inputs before being passed through activation functions such as GELU or Swish. This helps to avoid vanishing gradients when backpropagating errors.
      
    b. Feed forward layer: 

      - It uses two linear layers followed by swish activation function to perform feature transformation on the concatenated vector.

    c. Residual connection is also applied here to combine the outputs from both feedforward and self-attention layers.
    
2. Decoder Layer:

   Follows similar structure as that of encoder except some minor differences include masked multi-head attention layer which allows the decoder only attend to relevant positions at each step during decoding process.
   
   Note: The key differences between reformer and transformers generally involve introducing new ideas and techniques to improve their efficiency, while maintaining the core components such as attention mechanisms and feedforward networks.
  
## 四、主要组件介绍

### Reversible network

1. 概念及作用：在神经网络中，反向传播被用来求导，但如果参数更新需要依赖之前的中间状态值的话，效率就会受到影响。因此，LSTM等RNN结构采用反向传播的方式来更新参数，这种方式在一定程度上能够减轻梯度爆炸的问题。但是这种方式又导致了存储空间和计算量的增加，特别是在长序列的情况。Reversible network则通过引入新的结构，比如门控网络，可以有效地避免此类问题。

2. 结构示意图：


   1. 将神经网络中任意一个参数分割成两个子部分：A和B。

   2. 在正向过程中，将输入A做一些运算得到输出C。然后将输入B与这个输出C连接起来作为新的输入B，再重复相同的运算过程。输出C与原始输入A连接起来作为新的输出C，即完成了一次正向运算。

   3. 在反向过程中，将输出C的梯度传回输入A，然后在这里进行一些运算得到输入B的梯度。首先，利用链式法则反算出C=A*D*E+B*F。然后利用链式法则再次计算出原始输入A的梯度D'=1*C，原始输入B的梯度F'=1*(C-D*E)。最后，把D'和F'加在一起作为最终的梯度传回给输入A和输入B。

3. 为什么要这样设计？

   如上所述，LSTM等RNN结构采用反向传播的方式来更新参数，但存在着存储空间和计算量的问题。因此，提出了Reversible network的想法。

   另一方面，Reversible network也具有其他很多优点。例如，在训练GAN网络时，利用Reversible network可以同时生成图像数据和标签数据，并使用普通的训练方式来进行训练，而不是像传统的GAN那样，生成的图像数据和标签数据要单独作为输入进行训练。而且，Reversible network还可以用于强化学习领域，因为它可以像正常的网络一样处理输入数据和梯度信息，不需要额外的采样过程。