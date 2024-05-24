# Transformer解码器:自回归与掩码的奥秘

## 1.背景介绍

### 1.1 序列到序列模型的演进

在自然语言处理(NLP)和机器学习领域,序列到序列(Sequence-to-Sequence,Seq2Seq)模型是一类广泛使用的模型架构,它可以将一个序列(如句子)映射为另一个序列。典型的应用包括机器翻译、文本摘要、对话系统等。

早期的Seq2Seq模型主要基于循环神经网络(Recurrent Neural Network,RNN),如长短期记忆网络(Long Short-Term Memory,LSTM)和门控循环单元(Gated Recurrent Unit,GRU)。尽管RNN擅长处理序列数据,但由于其递归特性,难以有效利用现代硬件(如GPU)进行并行计算,从而在处理长序列时存在效率低下的问题。

2017年,Transformer模型被提出,它完全放弃了RNN的结构,利用注意力(Attention)机制直接对输入序列进行建模。Transformer模型不仅在机器翻译任务上取得了极大的突破,而且在其他NLP任务中也展现出了强大的能力,成为了NLP领域的主流模型之一。

### 1.2 Transformer解码器的作用

Transformer是一种编码器-解码器(Encoder-Decoder)架构,用于将源序列编码为中间表示,再由解码器从该表示生成目标序列。在机器翻译等任务中,解码器的作用是根据编码器输出的上下文向量,自回归地生成目标序列。

Transformer解码器的设计关键在于,如何有效地利用编码器输出的上下文信息,并保证生成序列的连贯性。为此,它采用了两种核心技术:

1. **自回归(Auto-Regressive)机制**:每个时间步的输出仅依赖于输入序列和之前的输出,确保生成序列的连贯性。
2. **掩码(Masked)多头注意力**:将未来时间步的信息从注意力计算中排除,避免不合理地借助未来信息。

本文将重点剖析Transformer解码器的自回归和掩码机制,阐述其原理和实现细节,以加深读者对该模型的理解。

## 2.核心概念与联系 

### 2.1 自回归(Auto-Regressive)

自回归是指当前时间步的输出仅依赖于输入序列和之前时间步的输出,而与未来时间步无关。形式化地,对于目标序列 $\boldsymbol{y} = (y_1, y_2, \ldots, y_T)$,其生成概率可表示为:

$$
P(\boldsymbol{y}|\boldsymbol{x}) = \prod_{t=1}^{T} P(y_t|y_{<t}, \boldsymbol{x})
$$

其中 $\boldsymbol{x}$ 是源序列, $y_{<t}$ 表示时间步 $t$ 之前的目标序列。自回归模型生成序列的过程是逐位解码,即在每个时间步只需要生成一个标记(token)。这种做法保证了生成序列的连贯性和合理性。

在Transformer解码器中,自回归是通过掩码多头注意力机制实现的。具体来说,在计算当前时间步注意力时,将来自未来时间步的信息予以掩码,使当前时间步的输出仅依赖于输入序列和之前的输出。

### 2.2 掩码多头注意力(Masked Multi-Head Attention)

注意力机制是Transformer的核心,它能够捕捉输入序列中不同位置的信息,并对它们赋予不同的权重。在解码器中,采用了掩码多头注意力,以实现自回归特性。

掩码多头注意力包含三个输入:查询(Query)、键(Key)和值(Value)。对于时间步 $t$,查询 $\boldsymbol{q}_t$ 是当前解码器状态,而键 $\boldsymbol{K}$ 和值 $\boldsymbol{V}$ 来自于编码器输出和之前的解码器输出。注意力分数由查询与键的点积得到:

$$
\text{Attention}(\boldsymbol{q}_t, \boldsymbol{K}, \boldsymbol{V}) = \text{softmax}\left(\frac{\boldsymbol{q}_t \boldsymbol{K}^\top}{\sqrt{d_k}}\right) \boldsymbol{V}
$$

其中 $d_k$ 是缩放因子,用于防止点积过大导致梯度消失。为了实现掩码,在计算注意力分数时,将来自未来时间步的键对应的分数设置为负无穷,从而在 softmax 后对应的权重为 0。

掩码多头注意力将注意力计算过程分成多个头(Head),每个头对查询、键和值进行不同的线性投影,然后将所有头的注意力输出拼接起来,再进行线性变换以生成最终的注意力输出。多头注意力可以关注输入序列中不同的位置信息,增强了模型的表示能力。

## 3.核心算法原理具体操作步骤

掩码多头自注意力是Transformer解码器的核心机制,下面将详细介绍其具体计算步骤。为了简化说明,我们先忽略多头的部分,只考虑单头情况。

假设解码器的输入是一个形状为 $(T, d_\text{model})$ 的张量 $X$,其中 $T$ 是序列长度, $d_\text{model}$ 是特征维度。我们希望计算时间步 $t$ 的自注意力输出 $y_t$。

1. **线性投影**:首先将输入 $X$ 分别投影到查询 $Q$、键 $K$ 和值 $V$ 上:

    $$
    \begin{aligned}
    Q &= X W_Q \\
    K &= X W_K \\
    V &= X W_V
    \end{align}
    $$

    其中 $W_Q, W_K, W_V \in \mathbb{R}^{d_\text{model} \times d_k}$ 是可学习的投影矩阵。

2. **计算注意力分数**:对于时间步 $t$,我们计算其查询向量 $\boldsymbol{q}_t$ 与所有键向量的点积,得到未缩放的注意力分数:

    $$
    e_{tj} = \boldsymbol{q}_t \boldsymbol{k}_j^\top
    $$

    其中 $\boldsymbol{k}_j$ 是时间步 $j$ 的键向量。为了实现掩码,我们将来自未来时间步 $(j > t)$ 的注意力分数设置为负无穷:

    $$
    e_{tj}' = \begin{cases}
    e_{tj}, & \text{if}\ j \le t \\
    -\infty, & \text{if}\ j > t
    \end{cases}
    $$

3. **计算注意力权重**:将注意力分数缩放后通过 softmax 函数得到注意力权重:

    $$
    \alpha_{tj} = \text{softmax}_j\left(\frac{e_{tj}'}{\sqrt{d_k}}\right) = \frac{\exp\left(e_{tj}' / \sqrt{d_k}\right)}{\sum_{l=1}^T \exp\left(e_{tl}' / \sqrt{d_k}\right)}
    $$

    其中 $\sqrt{d_k}$ 是缩放因子,用于防止点积过大导致梯度消失。由于掩码操作,来自未来时间步的注意力权重将为 0。

4. **计算加权和**:将注意力权重与值向量加权求和,得到时间步 $t$ 的注意力输出:

    $$
    \boldsymbol{y}_t = \sum_{j=1}^T \alpha_{tj} \boldsymbol{v}_j
    $$

对于多头注意力,上述过程在每个头上独立进行,最后将所有头的输出拼接,并进行线性变换以生成最终的注意力输出。

通过自回归掩码机制,Transformer解码器确保了在生成序列时,每个时间步的输出只依赖于输入序列和之前的输出,避免了不合理地利用未来信息,保证了生成序列的连贯性和合理性。

## 4.数学模型和公式详细讲解举例说明

在上一节中,我们已经介绍了掩码多头自注意力的计算过程。现在让我们通过一个具体例子,进一步解释相关数学模型和公式。

假设我们要生成一个长度为 4 的序列,解码器的输入张量 $X$ 的形状为 $(4, 512)$,注意力头数为 8,每个头的特征维度 $d_k = d_v = 64$。首先通过线性投影将 $X$ 投影到查询 $Q$、键 $K$ 和值 $V$ 上:

$$
\begin{aligned}
Q &= X W_Q^{(h)} &&\in \mathbb{R}^{4 \times 64} \\
K &= X W_K^{(h)} &&\in \mathbb{R}^{4 \times 64} \\
V &= X W_V^{(h)} &&\in \mathbb{R}^{4 \times 64}
\end{align}
$$

对于第 2 个时间步,我们计算其查询向量 $\boldsymbol{q}_2$ 与所有键向量的点积,得到未缩放的注意力分数:

$$
\begin{bmatrix}
e_{21} & e_{22} & e_{23} & e_{24}
\end{bmatrix} = \boldsymbol{q}_2 \begin{bmatrix}
\boldsymbol{k}_1^\top & \boldsymbol{k}_2^\top & \boldsymbol{k}_3^\top & \boldsymbol{k}_4^\top
\end{bmatrix}
$$

由于我们需要实现自回归,因此将来自未来时间步 $(j > 2)$ 的注意力分数设置为负无穷:

$$
\begin{bmatrix}
e_{21}' & e_{22}' & e_{23}' & e_{24}'
\end{bmatrix} = \begin{bmatrix}
e_{21} & e_{22} & -\infty & -\infty
\end{bmatrix}
$$

然后将注意力分数缩放并通过 softmax 函数得到注意力权重:

$$
\begin{bmatrix}
\alpha_{21} & \alpha_{22} & \alpha_{23} & \alpha_{24}
\end{bmatrix} = \text{softmax}\left(\frac{1}{\sqrt{64}} \begin{bmatrix}
e_{21}' & e_{22}' & e_{23}' & e_{24}'
\end{bmatrix}\right)
$$

由于 $e_{23}'$ 和 $e_{24}'$ 为负无穷,因此 $\alpha_{23} = \alpha_{24} = 0$,即时间步 2 的输出不会考虑来自未来时间步的信息。

最后,我们将注意力权重与值向量加权求和,得到时间步 2 的注意力输出:

$$
\boldsymbol{y}_2 = \alpha_{21} \boldsymbol{v}_1 + \alpha_{22} \boldsymbol{v}_2
$$

对于其他时间步,计算过程类似。在多头情况下,上述过程在每个头上独立进行,最终将所有头的输出拼接并进行线性变换。

通过这个例子,我们可以更好地理解掩码多头自注意力的数学模型和公式。自回归掩码机制确保了在生成序列时,每个时间步的输出只依赖于输入序列和之前的输出,避免了不合理地利用未来信息,保证了生成序列的连贯性和合理性。

## 4.项目实践:代码实例和详细解释说明

为了更好地理解Transformer解码器的自回归和掩码机制,我们将通过PyTorch代码示例来实现掩码多头自注意力层。

```python
import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # 线性投影
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 分头
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_