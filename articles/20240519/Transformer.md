# Transformer

作者：禅与计算机程序设计艺术

## 1. 背景介绍

### 1.1 序列到序列模型的局限性

传统的序列到序列(Seq2Seq)模型如RNN、LSTM等在处理长序列时存在梯度消失和梯度爆炸的问题,且难以并行化加速训练。此外,RNN系列模型只能捕捉有限的长距离依赖关系。

### 1.2 Transformer的诞生

Google于2017年发表了题为《Attention is All You Need》的论文,提出了Transformer模型。Transformer抛弃了传统的RNN、CNN等结构,完全依赖注意力机制(Attention)来学习序列之间的依赖关系,大幅提高了模型的并行能力和长距离特征捕捉能力,在机器翻译任务上取得了state-of-the-art的效果。

### 1.3 Transformer的影响力

Transformer作为一种通用的序列建模框架,不仅在机器翻译领域大放异彩,在自然语言处理的其他任务如文本分类、问答系统、语言模型等都取得了非常好的效果。Transformer的思想也被广泛应用到计算机视觉、语音识别、图网络等其他领域。可以说,Transformer引领了深度学习的新时代。

## 2. 核心概念与联系

### 2.1 Encoder-Decoder框架

- Encoder:将输入序列编码为隐向量
- Decoder:根据Encoder的输出和之前的输出,解码生成目标序列
- Encoder和Decoder通过Attention建立联系

### 2.2 Self-Attention

- 序列内部元素之间两两计算注意力权重,捕捉序列内部的依赖关系
- 相比RNN,Self-Attention能够一步到位捕捉任意距离的依赖
- 计算过程可以高度并行化

### 2.3 Multi-Head Attention

- 将输入进行多次线性变换,并行计算多个注意力权重
- 类似CNN中的多通道,从不同的子空间捕捉特征
- 最后将多头的结果拼接在一起

### 2.4 Positional Encoding

- 由于Transformer不包含RNN等顺序操作,需要显式引入位置信息
- 用固定的正余弦函数生成位置向量,与词向量相加
- 使得模型能够利用序列的顺序信息

### 2.5 Layer Normalization与残差连接  

- Layer Norm在每一层的输出上做归一化,加速收敛
- 残差连接使信息能够直接传递,缓解梯度消失

## 3. 核心算法原理与具体操作步骤

### 3.1 Encoder

#### 3.1.1 输入嵌入与位置编码

1. 将输入token转为固定维度的嵌入向量 $X \in \mathbb{R}^{n \times d}$
2. 生成对应长度n的位置编码向量 $P \in \mathbb{R}^{n \times d}$  
3. 将位置编码与嵌入向量相加 $X+P$

#### 3.1.2 Self-Attention

1. 通过线性变换得到Q、K、V矩阵

$$
\begin{aligned}
Q &= (X+P)W^Q \\
K &= (X+P)W^K \\
V &= (X+P)W^V
\end{aligned}
$$

2. 计算注意力权重矩阵

$$ A = \text{softmax}(\frac{QK^T}{\sqrt{d}}) $$

3. 加权求和得到输出

$$ \text{Attention}(Q,K,V) = AV $$

#### 3.1.3 Multi-Head Attention

1. 将Q、K、V通过h次不同的线性变换,得到h组Q、K、V
2. 各自分别过Self-Attention,得到h个输出
3. 将h个输出拼接后再过一个线性层

$$
\begin{aligned}
\text{MultiHead}(Q,K,V) &= \text{Concat}(\text{head}_1,...,\text{head}_h)W^O \\
\text{head}_i &= \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
\end{aligned}
$$

#### 3.1.4 Layer Normalization与前馈网络

1. Multi-Head Attention的输出先做Layer Norm再残差连接

$$ X_1 = \text{LayerNorm}(X + \text{MultiHead}(X)) $$

2. 过两层前馈网络,再Layer Norm和残差连接

$$
\begin{aligned}
\text{FFN}(X_1) &= \text{ReLU}(X_1W_1 + b_1)W_2 + b_2 \\  
X_2 &= \text{LayerNorm}(X_1 + \text{FFN}(X_1))
\end{aligned}
$$

### 3.2 Decoder

Decoder的结构与Encoder类似,也是由Self-Attention、Layer Normalization、前馈网络等组成。不同点在于:

1. Decoder的Self-Attention中,每个位置只能看到该位置及其之前的所有位置,后面的位置被Mask掉。这是为了避免在生成时看到未来的信息。

2. 在Decoder的Self-Attention之后,还有一个Encoder-Decoder Attention层。它以Decoder的隐状态为Q,Encoder的输出为K和V,用于在生成每个token时attend to输入序列的相关信息。

### 3.3 输出层

Decoder最后的隐状态经过线性变换和softmax,得到每个位置的输出概率分布。

$$ P(y_t|y_{<t},X) = \text{softmax}(HW^{vocab}) $$

其中H为Decoder最后一层的输出,$W^{vocab}$为输出词表的嵌入矩阵。

## 4. 数学模型和公式详细讲解举例说明

### 4.1 Scaled Dot-Product Attention

Transformer的核心是Scaled Dot-Product Attention:

$$
\text{Attention}(Q,K,V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
$$

其中Q、K、V分别表示query、key、value,维度分别为$d_q$、$d_k$、$d_v$。常见的做法是令$d_q=d_k=d_v=d$。

这个公式可以这样理解:

1. 将每个query与所有的key做点积,得到它们的相似度
2. 除以$\sqrt{d_k}$进行缩放,以避免点积结果过大
3. 对相似度做softmax,得到归一化的注意力权重
4. 将注意力权重与value加权求和,得到最终的注意力结果

举例说明:

假设有一个长度为3的输入序列"I love dogs",要计算第2个词"love"的Self-Attention。

1. 将输入嵌入为Q、K、V矩阵,每一行对应一个词。假设嵌入维度为4。

$$
Q=K=V=
\begin{bmatrix}
0.1 & 0.2 & 0.3 & 0.4 \\
0.5 & 0.6 & 0.7 & 0.8 \\
0.9 & 1.0 & 1.1 & 1.2
\end{bmatrix}
$$

2. 取第2行作为query,与K的转置做点积,并除以$\sqrt{d_k}=2$

$$
\frac{QK^T}{\sqrt{4}}=
\begin{bmatrix}
1.30 & 3.25 & 4.60  
\end{bmatrix}
$$

3. 做softmax得到注意力权重

$$
\text{softmax}(\frac{QK^T}{\sqrt{d_k}}) = 
\begin{bmatrix}
0.02 & 0.16 & 0.82
\end{bmatrix}
$$

4. 将注意力权重与V相乘,得到加权和

$$
\text{Attention}(Q,K,V) = 
\begin{bmatrix}
0.02 & 0.16 & 0.82  
\end{bmatrix}
\begin{bmatrix}
0.1 & 0.2 & 0.3 & 0.4 \\
0.5 & 0.6 & 0.7 & 0.8 \\
0.9 & 1.0 & 1.1 & 1.2
\end{bmatrix}
=
\begin{bmatrix}
0.80 & 0.91 & 1.01 & 1.12
\end{bmatrix}
$$

可以看到,"love"这个词主要attend to了第三个词"dogs",说明Attention捕捉到了它们之间的依赖关系。

### 4.2 Multi-Head Attention

Multi-Head Attention相当于同时执行h次不同的Self-Attention,然后将结果拼接。其公式为:

$$
\begin{aligned}
\text{MultiHead}(Q,K,V) &= \text{Concat}(\text{head}_1,...,\text{head}_h)W^O \\
\text{head}_i &= \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
\end{aligned}
$$

其中$W_i^Q \in \mathbb{R}^{d \times d_q}, W_i^K \in \mathbb{R}^{d \times d_k}, W_i^V \in \mathbb{R}^{d \times d_v}, W^O \in \mathbb{R}^{hd_v \times d}$。

举例说明:

假设我们有8个头(h=8),每个头的维度为64(d_q=d_k=d_v=64)。输入X的维度为512。

1. 将X分别乘以8组不同的$W_i^Q, W_i^K, W_i^V$矩阵,每组得到一个64维的Q、K、V。

2. 各自通过scaled dot-product attention,得到8个64维的输出$\text{head}_i$。

3. 将8个$\text{head}_i$拼接成一个8*64=512维的向量。

4. 乘以$W^O$矩阵,得到最终的512维输出。

Multi-Head Attention通过引入多个独立的attention函数,增强了模型的表达能力,使其能够关注输入的不同方面。

### 4.3 Positional Encoding

Transformer中的Positional Encoding使用固定的正余弦函数:

$$
\begin{aligned}
PE_{(pos,2i)} &= \sin(pos/10000^{2i/d}) \\
PE_{(pos,2i+1)} &= \cos(pos/10000^{2i/d})
\end{aligned}
$$

其中pos为位置索引,i为维度索引,d为嵌入维度。

举例说明:

假设嵌入维度为4,序列长度为3。则Positional Encoding为:

$$
\begin{aligned}
PE_{(0,0)} &= \sin(0/10000^{0/4}) = 0 \\
PE_{(0,1)} &= \cos(0/10000^{0/4}) = 1 \\  
PE_{(0,2)} &= \sin(0/10000^{2/4}) = 0 \\
PE_{(0,3)} &= \cos(0/10000^{2/4}) = 1 \\
PE_{(1,0)} &= \sin(1/10000^{0/4}) \approx 0.0001 \\
PE_{(1,1)} &= \cos(1/10000^{0/4}) \approx 1 \\
PE_{(1,2)} &= \sin(1/10000^{2/4}) \approx 0.0001 \\
PE_{(1,3)} &= \cos(1/10000^{2/4}) \approx 1 \\
PE_{(2,0)} &= \sin(2/10000^{0/4}) \approx 0.0002 \\ 
PE_{(2,1)} &= \cos(2/10000^{0/4}) \approx 1 \\
PE_{(2,2)} &= \sin(2/10000^{2/4}) \approx 0.0002 \\
PE_{(2,3)} &= \cos(2/10000^{2/4}) \approx 1 \\
\end{aligned}
$$

最终的Positional Encoding矩阵为:

$$
PE=
\begin{bmatrix}
0 & 1 & 0 & 1 \\
0.0001 & 1 & 0.0001 & 1 \\  
0.0002 & 1 & 0.0002 & 1
\end{bmatrix}
$$

可以看到,Positional Encoding矩阵的每一行都是一个正弦/余弦函数,频率随着位置递增而变化。将其与词嵌入相加,就给每个词引入了位置信息。

## 5. 项目实践:代码实例和详细解释说明

下面我们用PyTorch实现一个简单的Transformer模型,并应用于文本分类任务。

### 5.1 定义模型结构

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0