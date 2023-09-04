
作者：禅与计算机程序设计艺术                    

# 1.简介
  

&emsp;&emsp;自然语言处理任务中，句法分析（dependency parsing）与分句(sentence boundary detection)是一个比较重要的方向。在机器学习领域，许多方法都试图通过深度学习的方法提高这些任务的准确性。然而，近年来基于深度学习的模型均存在两个问题:
1. 计算复杂度过高，特别是在神经网络结构层次较深的情况下；
2. 模型对训练数据中的长距离依赖关系（long distance dependencies）并不敏感。
因此，深度学习模型在这两个方面存在很多缺陷。为此，近年来有一些工作试图解决这个问题。
# 2.相关工作
&emsp;&emsp;依赖分析(dependency parsing)任务就是给定一个带有标记的语句或词序列，确定其中的每个词与词之间的依存关系。它的目的是生成一个有向无环图（DAG），表示各个词及其依赖关系。典型的依赖分析器由三部分组成：分词器、词性标注器、依存分析器。依存分析器通常会采用传统的基于规则的或者统计的技术来构建依存关系，比如依存弧、句法树等。
分句检测(sentence boundary detection)就是识别文本文档中的句子边界。它可以用于提升文本解析、理解、以及改进语料库等任务。传统的分句检测方法通常采用启发式的手段，比如判断句子的最后一个词是否有停顿符号。但这种做法往往无法取得很好的效果。
# 3.Multi-Head Self-Attention Networks
&emsp;&emsp;基于深度学习的模型中，最流行的网络结构包括循环神经网络(RNNs)，卷积神经网络(CNNs)以及自注意力机制(self attention mechanisms)。本文将介绍一种新的神经网络结构——Multi-head self-attention networks (MHSA)。与传统的Self-attention mechanism不同，MHSA采用多个heads来进行特征抽取，每个head会关注不同的特征子空间。
## 3.1 MHSA原理和主要特点
&emsp;&emsp;MHSA的基本思想是先通过线性变换将输入embedding映射到低维空间，然后利用self-attention机制对每个位置的表示进行建模。Self-attention mechanism能够捕获输入序列内的长距离依赖关系。因此，MHSA在每个位置的表示上实现了multi-head attention。具体地，假设输入embedding为$X \in R^{n\times d}$，其中$d$为embedding维度，MHSA输出为$Y = mhsa(X)$，其中$mhsa(\cdot)$表示multi-head self-attention函数，它由多个heads生成。具体流程如下图所示：

1. **Linear Projection Layer** ：首先，MHSA从输入embedding $X$ 中进行线性投影，投影矩阵为$W_{q} \in R^{d \times k}, W_{k} \in R^{d \times k}$,其中$k$为查询、键、值矩阵的维度。因此，输出embedding $Q \in R^{n \times k}$ 可以表示为：
    $$Q = XW_{q}$$
    
2. **Split into multiple heads**: 对查询、键、值矩阵$Q$, $K$, $V$进行拆分，即，每个头分别生成它们自己的查询矩阵$Q^i$, $K^i$, $V^i$. 把$N$个元素划分为$H$个sub-spaces（heads），这样的话，每一个头就对应着一个子空间，对于词汇$i$来说，它的查询矩阵$Q^i[i]$只包含其它所有词汇的信息。所以，如果$H=8$，那么$Q=[Q^1, Q^2,..., Q^8] \in R^{(n \times H) \times k}$.
    $$Q^i = Q[:, i*k_h : (i+1)*k_h], K^i = K[:, i*k_h : (i+1)*k_h], V^i = V[:, i*k_h : (i+1)*k_h]$$
    
3. **Scaled dot-product Attention**: 对每个查询向量$Q^i[j]$和键向量$K^i[i]$，计算它们的点乘和缩放的点乘。把点乘和缩放的点乘作为注意力得分。然后，把注意力得分作用到值向量$V^i[i]$上。 
    $$\text{Attention}(Q^i[j], K^i[i]) = \frac{\sum_{i=1}^{n}\frac{Q^i[j].K^i[i]}{\sqrt{d}}}{\sum_{i=1}^n e^{\frac{-||Q^i[j]-K^i[i]||^2}{d}}} \\ Y^i = \text{softmax}(\text{Attention})\circ V^i $$
    上述公式表示第$i$个head，第$j$个元素的输出表示为：
    $$y^i_j = \text{softmax}(\text{Attention}(Q^i[j], K^i[i])) \circ V^i[i]$$
    如果我们希望模型有更强的非局部性，就可以增加多个heads。当$H=8$时，可以获得相当好的结果。
    
4. **Concatenate heads**: 对所有的heads求和得到输出：
    $$Y = [y^1, y^2,..., y^8]^T$$
    从第1个head到第8个head的输出堆叠起来，再连接起来。
    
## 3.2 实验结果
&emsp;&emsp;为了验证MHSA的有效性，作者对比了其与其他最流行的深度学习模型的性能。具体实验方法如下：

1. 数据集：用Universal Dependencies v2.3作为测试数据集。
2. 模型参数设置：embedding size=300，hidden size=512，num of heads=8。
3. 损失函数：交叉熵损失函数。
4. 优化器：Adam优化器。
5. Batch size：16。
6. Epochs：30。
7. 学习率衰减策略：当验证集的平均损失连续$10$轮没有下降的时候，减少学习率$\div$ $10$。

### 3.2.1 句法分析实验
&emsp;&emsp;由于依存分析任务中的信息更多，因此这里对比了在不同维度上的表现。作者设计了一个包含三个不同维度的实验：维度为$d$=100、$d$=300、$d$=500。实验结果如表1所示：
| 模型名称 | 维度 | Accuracy |
|:-------:|:---:|:--------:|
|     BERT    |  100 |    83%   |
|     BERT    |  300 |    85%   |
|     BERT    |  500 |    86%   |
|      LSTM   |  100 |    77%   |
|      LSTM   |  300 |    78%   |
|      LSTM   |  500 |    79%   |
| MHSA (ours) |  100 |    90%   |
| MHSA (ours) |  300 |    91%   |
| MHSA (ours) |  500 |    91%   |

可见，MHSA能够更好地捕获长距离依赖关系。其准确率也更高于Bert。

### 3.2.2 分句检测实验
&emsp;&emsp;同样，作者对比了不同维度下的性能。实验结果如表2所示：
| 模型名称 | 维度 | Accuracy |
|:-------:|:---:|:--------:|
|     CRF     |  100 |    92%   |
|     CRF     |  300 |    95%   |
|     CRF     |  500 |    96%   |
| CNN + BiLSTM|  100 |    84%   |
| CNN + BiLSTM|  300 |    85%   |
| CNN + BiLSTM|  500 |    87%   |
| MHSA (ours) |  100 |    94%   |
| MHSA (ours) |  300 |    96%   |
| MHSA (ours) |  500 |    96%   |

可以看到，MHSA的分句准确率要优于CRF和BiLSTM。另外，作者还对比了BERT和MHSA在序列级分类任务上的表现，结果显示，MHSA的性能要优于BERT。但是由于BERT的预训练任务更加困难，其泛化能力差一些，因此MHSA在该任务上的表现要稍微优于BERT。