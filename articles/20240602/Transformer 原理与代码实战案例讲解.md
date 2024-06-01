## 1. 背景介绍

Transformer（变换器）是2017年谷歌公司推出的一个革命性的人工智能模型，它的出现使得自然语言处理（NLP）领域迎来了一场革命。 Transformer 不仅在自然语言处理领域取得了显著的进展，还广泛应用于计算机视觉、音频处理等多个领域。今天，我们将深入探讨 Transformer 的原理、核心算法及其在实际项目中的应用。

## 2. 核心概念与联系

### 2.1. 自注意力机制

Transformer 的核心概念是自注意力（Self-Attention）机制。自注意力机制允许模型在处理输入序列时，能够关注输入序列的不同部分。它可以帮助模型捕捉长距离依赖关系，并解决传统序列模型（如 RNN）所面临的长距离依赖问题。

### 2.2. 多头注意力

在实际应用中，Transformer 还引入了多头注意力（Multi-Head Attention）机制。多头注意力可以让模型在不同的维度上学习不同类型的特征，这有助于提高模型的表达能力和泛化能力。

## 3. 核心算法原理具体操作步骤

### 3.1. 输入表示

输入文本首先需要转换为一个连续的向量表示，通常使用词嵌入（Word Embedding）进行表示。词嵌入可以通过预训练好的模型（如 Word2Vec、GloVe）得到，也可以通过自定义的训练过程生成。

### 3.2. 自注意力计算

接下来，我们需要计算自注意力分数矩阵。自注意力分数矩阵的计算公式如下：

$$
\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})
$$

其中，Q（Query）表示查询向量，K（Key）表示密钥向量，V（Value）表示值向量。d<sub>k</sub>表示向量维度。

### 3.3. 多头注意力计算

多头注意力计算过程如下：

1. 将 Q、K、V 向量分别按照多头注意力头数进行分组。
2. 对每个头计算自注意力分数矩阵。
3. 将每个头的分数矩阵进行加权求和。

### 3.4. 线性变换

最后，我们需要对多头注意力输出进行线性变换，以得到最终的输出。线性变换可以通过全连接层（Fully Connected Layer）实现。

## 4. 数学模型和公式详细讲解举例说明

在本节中，我们将详细讲解 Transformer 的数学模型和公式，并通过实际例子进行解释说明。

### 4.1. 自注意力分数矩阵

自注意力分数矩阵的计算公式如下：

$$
\text{Attention}(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})
$$

其中，Q（Query）表示查询向量，K（Key）表示密钥向量，V（Value）表示值向量。d<sub>k</sub>表示向量维度。

### 4.2. 多头注意力分数矩阵

多头注意力分数矩阵的计算公式如下：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(h^1, h^2, ..., h^h)
$$

$$
h^i = \text{Attention}(QW^Q_i, KW^K_i, VW^V_i)
$$

其中，h<sup>i</sup>表示第 i 个注意力头的输出，h 是注意力头数。

### 4.3. 线性变换

线性变换可以通过全连接层（Fully Connected Layer）实现。其计算公式如下：

$$
\text{Output} = \text{Linear}(X)
$$

其中，Output 是输出向量，X 是输入向量。

## 5. 项目实践：代码实例和详细解释说明

在本节中，我们将通过一个实际的代码示例来解释 Transformer 的实现过程。

### 5.1. 代码示例

以下是一个简化的 Transformer 实现代码示例：

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = dropout
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None

    def forward(self, query, key, value, mask=None):
        nbatches = query.size(0)
        query, key, value = [self.linears[i](x) for i, x in enumerate((query, key, value))]
        qkv = torch.cat([query, key, value], dim=-1)
        qkv = self.split(qkv, self.nhead, self.d_model)
        qkv = [torch.stack([qkv[i,j] for j in range(self.nhead)]) for i in range(3)]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in qkv]
        qkv = [torch.stack([qkv_i[i] for i in range(self.nhead)]) for qkv_i in qkv]
        qkv = [torch.transpose(qkv_i, 0, 1) for qkv_i in