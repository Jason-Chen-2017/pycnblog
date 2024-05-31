# Transformer 原理与代码实例讲解

## 1. 背景介绍
### 1.1 Transformer 的诞生
2017年,Google 机器翻译团队在论文《Attention is All You Need》中提出了 Transformer 模型,开创了 NLP 领域的新纪元。Transformer 抛弃了传统的 RNN 和 CNN 结构,完全依靠注意力机制(Attention)来学习文本中的上下文关系,在机器翻译任务上取得了 SOTA 的效果。

### 1.2 Transformer 的影响力 
Transformer 不仅在机器翻译领域独领风骚,其思想很快被广泛应用到 NLP 的各个任务中,如文本分类、命名实体识别、问答系统、文本摘要等。各种 Transformer 变体如 BERT、GPT、XLNet 层出不穷,不断刷新各项任务的最高性能。Transformer 已经成为了当前 NLP 领域的标配模型。

### 1.3 Transformer 的优势
- 并行计算能力强:抛弃了 RNN 的串行结构,采用了全注意力机制,计算过程可以高度并行化,训练速度大幅提升。
- 长距离依赖学习能力强:通过 Self-Attention 机制,每个单词都能和句子中的任意单词建立直接联系,更好地捕捉长距离依赖。 
- 结构灵活:Transformer 的编码器和解码器都由相同的层结构堆叠而成,非常适合迁移学习和预训练。

## 2. 核心概念与联系
### 2.1 Self-Attention
Self-Attention 是 Transformer 的核心,用于学习文本内部的依赖关系。对于每个单词,通过 query 向量去查询其他单词的 key 向量,根据匹配程度得到权重系数,再对 value 向量加权求和,得到该单词的上下文编码。公式如下:

$$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

其中 $Q$,$K$,$V$ 分别是 query,key,value 向量矩阵,$d_k$ 为向量维度。

### 2.2 Multi-Head Attention
Multi-Head Attention 将 Self-Attention 进行多次线性变换,然后并行执行多个 scaled dot-product attention,以学习不同子空间的文本表示。最后把各个 head 的结果拼接起来:

$$MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O \\
head_i=Attention(QW_i^Q,KW_i^K,VW_i^V)$$

其中$W^O$,$W_i^Q$,$W_i^K$,$W_i^V$是可学习的参数矩阵。

### 2.3 Positional Encoding
由于 Transformer 没有 RNN 的循环结构来捕捉序列信息,因此需要在输入 embedding 中加入位置编码,来表示单词在句子中的位置。位置编码通过三角函数实现:

$$PE_{(pos,2i)} = sin(pos/10000^{2i/d_{model}}) \\
PE_{(pos,2i+1)} = cos(pos/10000^{2i/d_{model}})$$

其中 $pos$ 表示位置,$i$ 表示维度,$d_{model}$ 为 embedding 维度。

### 2.4 Transformer 整体结构
![Transformer结构图](https://pic3.zhimg.com/80/v2-dce7d6f4ae3c8da923f6b0a62bc9c893_720w.jpg)

如上图所示,Transformer 由 6 层编码器和解码器堆叠而成。每一层包含两个子层:
- 第一个子层是 Multi-Head Self-Attention
- 第二个子层是全连接前馈网络 

两个子层之间通过残差连接和 Layer Normalization 连接。

编码器负责对输入序列进行特征提取,解码器负责根据编码器的输出和之前的预测结果,生成下一个单词的概率分布。

## 3. 核心算法原理与操作步骤
### 3.1 编码器(Encoder)
1. 输入序列 $X=(x_1,x_2,...,x_n)$ 通过 word embedding 和 positional encoding 相加,得到输入表示 $H^0=(h_1^0,h_2^0,...,h_n^0)$。

2. 对于第 $l$ 层 Encoder,首先对 $H^{l-1}$ 进行 Multi-Head Self-Attention:
$$MidH^l = MultiHead(H^{l-1},H^{l-1},H^{l-1}) \\
 H^l = LayerNorm(H^{l-1} + MidH^l)$$

3. 然后经过全连接前馈网络 FFN 得到最终输出:
$$OutH^l = FFN(H^l) \\
 H^l = LayerNorm(H^l+OutH^l)$$

4. 重复步骤 2~3,堆叠 $N$ 层 Encoder,得到最终输出 $H^N$。

### 3.2 解码器(Decoder)
1. 目标序列 $Y=(y_1,y_2,...,y_m)$ 通过 word embedding 和 positional encoding 相加,得到输入表示 $S^0=(s_1^0,s_2^0,...,s_m^0)$。

2. 对于第 $l$ 层 Decoder,首先对 $S^{l-1}$ 进行 Masked Multi-Head Self-Attention,得到 $MidS^l$。Mask 操作是为了避免看到未来的信息。

3. 然后让 $MidS^l$ 与 Encoder 的输出 $H^N$ 进行 Multi-Head Attention:
$$OutS^l = MultiHead(MidS^l,H^N,H^N)$$

4. 再经过全连接层,残差连接和 Layer Normalization,得到 $S^l$。

5. 重复步骤 2~4,堆叠 $M$ 层 Decoder,得到最终输出 $S^M$。

6. 将 $S^M$ 经过线性层和 softmax 层,得到下一个单词的概率分布 $P(y_{t+1}|y_{\leq t},X)$。

## 4. 数学模型和公式详解
### 4.1 Scaled Dot-Product Attention
$$Attention(Q,K,V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$$

- $Q$,$K$,$V$ 分别是 query,key,value 矩阵,大小为 $(m,d_k)$,$(n,d_k)$,$(n,d_v)$。
- $QK^T$ 计算 query 和 key 的匹配程度,得到 $(m,n)$ 的矩阵。
- $\sqrt{d_k}$ 起到调节作用,使内积不至于太大。
- softmax 对每一行进行归一化,使其成为概率分布。
- 最后右乘 $V$ 矩阵,得到加权求和的 attention 值,大小为 $(m,d_v)$。

直观理解就是,通过 query 去查询 key,根据匹配程度对 value 进行加权求和。

### 4.2 Multi-Head Attention
$$MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O \\
head_i=Attention(QW_i^Q,KW_i^K,VW_i^V)$$

- 先将 $Q$,$K$,$V$ 通过线性变换投影到 $h$ 个不同的子空间,变成维度为 $d_k/h$,$d_k/h$,$d_v/h$ 的 $Q_i$,$K_i$,$V_i$。
- 然后对每组 $Q_i$,$K_i$,$V_i$ 并行执行 scaled dot-product attention,得到 $d_v/h$ 维的 $head_i$。
- 最后把所有 $head_i$ 拼接起来,经过线性变换,得到最终的 Multi-Head Attention 输出,维度为 $d_v$。

直观理解是,Multi-Head Attention 允许模型在不同的子空间里学习到不同的语义信息,提高了模型的表达能力。

### 4.3 残差连接与 Layer Normalization
$$Output = LayerNorm(Input + Sublayer(Input))$$

- Sublayer 可以是 Multi-Head Attention 或 FFN。
- 残差连接有助于缓解梯度消失,加速收敛。
- Layer Normalization 可以稳定训练,加速收敛。

## 5. 代码实例和详解
下面是一个用 PyTorch 实现的简化版 Transformer 代码,用于学习文本分类任务。

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model) 
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        
        q = self.q_linear(q).view(bs, -1, self.num_heads, self.head_dim)
        k = self.k_linear(k).view(bs, -1, self.num_heads, self.head_dim)
        v = self.v_linear(v).view(bs, -1, self.num_heads, self.head_dim)
        
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim**0.5
        if mask is not None:
            scores = scores.masked_fill(mask==0, -1e9)
        scores = torch.softmax(scores, dim=-1)
        
        output = torch.matmul(scores, v)
        output = output.transpose(1,2).contiguous().view(bs, -1, self.d_model)
        output = self.out(output)
        return output

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, src, src_mask=None):
        src2 = self.attention(src, src, src, src_mask)
        src = src + self.dropout(src2) 
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src
        
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dim_feedforward, dropout):
        super().__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout) 
                                     for _ in range(num_layers)])
    
    def forward(self, src, src_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask)
        return output
        
class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, dim_feedforward, dropout, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, dim_feedforward, dropout)
        self.fc = nn.Linear(d_model, num_classes)
        
    def forward(self, src, src_mask=None):
        src = self.embedding(src)
        src = self.pos_encoding(src)
        output = self.encoder(src, src_mask)
        output = output.mean(dim=1)
        output = self.fc(output)
        return output
```

代码讲解:
- MultiHeadAttention 实现了多头注意力机制,通过线性变换将 Q,K,V 映射到多个子空间,并行计算 attention,最后拼接输出。
- TransformerEncoderLayer 实现了 Transformer 的编码器层,包含 Multi-Head Attention 和 FFN 两个子层,以及 Add&Norm 操作。
- TransformerEncoder 通过堆叠 N 个 TransformerEncoderLayer 得到完整的编码器。
- Transformer 实现了用于文本分类的完整模型,Embedding 和 Positional Encoding 后接 TransformerEncoder,最后通过均值池化和全连接得到分类输出。

## 6. 实际应用场景
Transformer 及其变体可以应用于以下场景:
- 机器翻译:Transformer 最初就是为此而提出,用于将一种语言翻译成另一种语言。
- 文本分类:通过 Transformer