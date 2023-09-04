
作者：禅与计算机程序设计艺术                    

# 1.简介
  

自从Transformer出现在NLP领域后，它被广泛的应用于各个领域中，特别是在机器翻译、文本摘要、机器问答等任务上取得了惊艳的成果。由于其编码器-解码器（Encoder-Decoder）架构，在序列生成任务中比RNN（Recurrent Neural Networks）更加灵活且高效，所以受到许多研究者的关注。但是，传统RNN对长序列具有不利的性能，因为每一步都需要计算完整的历史信息，而此时RNN并不能充分利用上下文信息。基于这一点，作者提出了一种新的自回归（Auto Regressive）的Transformer模型——Fast Autoregressive Transformer（FAT）。这种模型能够学习到长程依赖关系，并且能有效地处理长序列。FAST相对于传统RNN，可以降低梯度消失或爆炸的问题，能够更好地捕捉局部和全局的依赖关系。而且，FAST模型可以直接通过Self-Attention机制实现序列的多层次建模，因此不需要堆叠RNN层。本文主要阐述了FAST Transformer的原理和工作流程。


# 2.基本概念术语说明
## 2.1 Transformer概览
论文标题中的“Transformers”指的是Transformer模型，由两位华裔学者Vaswani和Wszawa共同提出，是目前最成功的自注意力模型。Transformer是一种基于位置编码(positional encoding)的自编码器（autoencoder），其内部采用了一套多头自注意力机制(multi-head self-attention mechanism)。在自回归语言模型任务中，输入序列通过Encoder把它转换为固定长度的向量表示，再通过Decoder生成输出序列。多头自注意力机制能够在不同的子空间（subspaces）之间同时做注意力学习，增加模型的表达能力。

## 2.2 FAST Transformer概览
FAST Transformer是一种对Transformer模型进行改进和扩展的模型。在传统的Transformer中，每个位置只能由前面的几个位置所影响，这样在处理长序列时，每一步都需要计算完整的历史信息。但这种局部依赖关系可能会导致梯度爆炸或者消失，并且可能对模型产生不稳定性。为了解决这一问题，作者在Multi-Head Self-Attention模块中引入位置相关性(positional correlation)，即利用相邻的位置之间的联系，来增强模型的表现力。另外，作者设计了一个基于位置编码的结构，能够提高模型的学习效率。基于这些改进，作者提出了一种新的自回归Transformer——FAST Transformer。

## 2.3 相关术语
### 2.3.1 Position Encoding
Position Encoding是一种通过训练得到的矢量，用来将序列位置信息编码到输入特征中。这项技术被广泛用于各种自然语言处理任务，包括语言模型、序列到序列模型、图像描述、音频、视频生成等。当序列长度较短时，通常会采用 learned positional embedding (LePE) 或 fixed positional embedding (FPE)的方式，即用一个 learned matrix 或 嵌入矩阵 来生成编码向量。在前两种方式下，编码向量的元素数量等于序列的长度。除此之外，还有相对位置编码(Relative Position Embedding, REP)或绝对位置编码(Absolute Position Embedding, APE)的变体。REP通过学习不同的矩阵来编码不同距离下的位置差异，在序列较短的时候仍然可以使用固定位置编码。

### 2.3.2 Scaled Dot-Product Attention
Scaled Dot-Product Attention 是Transformer中的核心组件之一，用于计算注意力权重。它由两个步骤组成：首先，先求出query和key矩阵的点积，然后缩放这个乘积的结果。缩放系数取决于模型参数d_k和维度大小d_model，如下所示：

$$softmax(\frac{Q K^T}{\sqrt{d_k}})$$ 

其中$Q \in R^{n\times d_k}$, $K \in R^{m\times d_k}$, n和m分别是查询序列和键序列的长度，d_k和d_model是模型参数。上述公式就是Scaled Dot-Product Attention的计算过程。注意，该模块只涉及输入数据的线性组合，而无需关注时间序排列。

### 2.3.3 Multi-Head Attention
Multi-Head Attention是一个在Self-Attention层次上的改进方法。一般来说，Self-Attention机制可以看作是同时考虑到当前位置和之前所有位置的情况，因而称为Self-Attention。然而，当序列很长时，这种全局性的依赖关系就不一定合适了。因此，作者提出了一种改进的模块——Multi-Head Attention。在Self-Attention中，仅有一个头指针向量与整个序列相关；而在Multi-Head Attention中，每个头指针向量都有自己的依赖关系。这样可以充分利用全局和局部信息。

### 2.3.4 Positional MLP
Positional MLP是一种通过位置编码学习到的非线性映射，用来增强模型的表达能力。它是一种残差网络，它的输出是当前位置的编码向量和过去固定窗口内的位置编码向量做点积后的结果。Positional MLP的窗口大小也是模型参数的一部分。

### 2.3.5 GPT
GPT (Generative Pre-Training)是一种预训练语言模型，用于训练大规模的文本生成模型。在这种模型中，输入序列是一串单词，输出序列也是一串单词。在训练阶段，GPT通过不断迭代损失函数，来拟合输入序列和输出序列之间的映射关系。

### 2.3.6 Long-Short Term Memory (LSTM)
LSTM是一种用来捕获和记忆序列信息的神经网络单元。它通过门机制控制信息的流动，避免了RNN中的梯度弥散现象。其内部由三个门组成：遗忘门、输入门和输出门。在训练阶段，LSTM会学习到捕获和存储上下文信息的模式。

# 3.核心算法原理和具体操作步骤以及数学公式讲解
## 3.1 FAST Transformer模型结构
如图所示，FAST Transformer是一种自回归的Transformer模型，由Encoder、Decoder和两个多头自注意力机制构成。Encoder的输入是输入序列x，通过一系列的层(layer)来编码输入序列的信息，得到固定长度的向量z。Decoder的输入是目标序列y，根据Encoder的输出z和y，通过自回归模块逐步生成输出序列。

### 3.1.1 Encoder模块
Encoder由多个层(layer)组成，每层包含以下操作：

1. Additive Positional Encoding：添加位置编码，在Encoder的第一层，位置编码是输入向量x和固定长度向量e进行相加得到，e的生成和训练是重要的，它可以帮助模型捕捉位置信息。如图1左侧所示。

2. Dropout：随机扰乱网络，防止过拟合。

3. Layer Normalization：归一化，用于加快训练速度，同时减少梯度消失或爆炸。

4. Self-Attention：多头自注意力机制。在第l层，q、k、v分别是Encoder的输出z、z和z，然后计算注意力权重，即权重值alpha。如公式如下：

   $$\text{Attention}(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})\cdot V$$
   
   在本文中，d_k=d_v=d_model=h=d_ff=512，也就是说，使用一个512维的向量表示一个样本，头数h=8。因此，将原始输入x划分为h块，每块里面有d_model/h=64维，因此，总共有8个头指针向量Wq，Wk，Wv。使用Softmax函数计算注意力权重，并与值向量进行点积。最后，加上一个残差连接并使用ReLU激活函数，获得新的隐藏状态h。最后，重复以上步骤，生成多个头指针向量。

   
   $$ MultiHead(Q,K,V)=Concat(\text{head}_1,\dots,\text{head}_h)W^O\\ \text{where } \text{head}_i=\text{Attention}(QW^Q_i+KW^K_i+VW^V_i)\\ W^Q_i\in R^{(1\times h}\times d_{\text{model}/h)}, W^K_i\in R^{(1\times h}\times d_{\text{model}/h)}, W^V_i\in R^{(1\times h}\times d_{\text{model}/h}) \\ W^O\in R^{(1\times hd_\text{model}}\times d_\text{ff}), d_\text{ff}=2048$$
   


5. Positionwise Feedforward Network：多层感知机，由两个全连接层组成，第一个全连接层的输入是h，第二个全连接层的输入是FFN(h)+h，FFN作用是对h进行特征提取，以便于后续处理。

### 3.1.2 Decoder模块
Decoder由三个多头自注意力机制构成，包括Q、K、V，对源序列的隐含表示(hidden state of the encoder)进行多头自注意力运算，并将其作为Query，用于注意力权重计算。Q是decoder的输入，KV则来自于encoder的输出。与encoder一样，在每个层里，q、k、v分别是decoder的输出、decoder的输出和decoder的输出，注意力权重alpha也一样。

### 3.1.3 通用的Attention模块
注意力模块共有三种形式，分别是Additive Attention，Dot-Product Attention和Scaled Dot-Product Attention。这三种形式的区别在于计算注意力权重的方式。

1. Additive Attention：首先计算query和所有key值的加权和，再与所有value值相乘。这种形式的注意力运算适用于inputs具有相同维度的情况，否则需要对inputs进行线性变化。

2. Dot-Product Attention：计算query与所有key的点积，再除以根号dk，然后与所有value进行点积。这种形式的注意力运算对计算资源要求较高，因此，在实际工程应用中一般采用其他两种注意力形式。

3. Scaled Dot-Product Attention：与Dot-Product Attention类似，计算query与所有key的点积，并乘以根号dk。但它还有一个额外的缩放因子，其目的是为了解决Dot-Product Attention存在的梯度消失问题。当注意力权重接近于0或1时，缩放因子使得它们的值更稳定，能有效缓解梯度消失或爆炸的问题。

### 3.1.4 Padding Masking和Lookahead Masking
Padding Masking和Lookahead Masking都是为了阻止模型过早地依赖未来的信息，从而起到预测的延迟效果。Padding Masking和Lookahead Masking均可用于给输入序列增加mask，在计算注意力权重时，将对应的mask置0。

1. Padding Masking：Padding Masking是为了掩盖掉输入序列的padding信息，保证注意力模型能够正确地关注有效信息。当模型生成padding token时，如果没有padding mask，就会按照padding的存在来计算注意力权重，从而影响最终的输出。因此，Padding Masking的目的就是让模型在生成padding token时不计算注意力权重。

2. Lookahead Masking：Lookahead Masking的目的是为了防止模型过早地看到future的信息，从而导致信息的丢失。假设模型当前正在生成第t个token，那么模型不会看到第t+1到t+n个token，而只是看到当前token之后的序列。因此，Lookahead Masking是为了阻止模型看到future的信息，从而预测错误。