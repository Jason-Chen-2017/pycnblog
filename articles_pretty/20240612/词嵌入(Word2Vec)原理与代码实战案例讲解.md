# 词嵌入(Word2Vec)原理与代码实战案例讲解

## 1. 背景介绍
### 1.1 词嵌入的概念
词嵌入(Word Embedding)是自然语言处理(NLP)领域中一种重要的文本表示方法。它将词汇表中的每个单词映射到一个固定维度的实数向量空间中,使得语义相近的词在向量空间中的距离也比较近。通过这种方式,词嵌入可以有效地捕捉单词之间的语义关系,为下游的NLP任务提供高质量的输入特征。

### 1.2 词嵌入的发展历程
早期的词表示方法主要是 One-hot Encoding,即每个单词用一个独热向量表示。这种方法虽然简单直观,但无法刻画单词间的语义关系,且容易产生维度灾难。为了克服这些问题,研究者们先后提出了分布式语义表示模型如 NNLM、RNNLM 等。2013年,Google 的 Mikolov 团队提出了 Word2Vec 模型,使得词嵌入的训练更加高效,并被广泛应用于各种 NLP 任务中。

### 1.3 Word2Vec 的优势
相比于传统的词表示方法,Word2Vec 具有以下优势:

1. 语义表示能力强:通过对大规模语料的训练,Word2Vec 学习到的词向量可以很好地反映单词的语义信息。
2. 维度相对较低:Word2Vec 生成的词向量通常在几十到几百维,远低于词汇表大小,减轻了维度灾难问题。  
3. 可以进行词的语义运算:词向量支持加减等数学运算,比如 vec("king") - vec("man") + vec("woman") ≈ vec("queen")。
4. 训练效率高:Word2Vec 采用了高效的训练策略,在保证性能的同时大幅提升了训练速度。

## 2. 核心概念与联系
### 2.1 CBOW 与 Skip-Gram 模型
Word2Vec 包含两个经典模型:CBOW(Continuous Bag-of-Words)和 Skip-Gram。它们的核心思想是利用一个单词的上下文信息来预测该单词本身(CBOW),或根据一个单词来预测它的上下文(Skip-Gram)。两个模型的网络结构类似,主要区别在于输入和输出的形式。

### 2.2 词向量的两个矩阵
Word2Vec 模型包含两个权重矩阵:输入词矩阵 W 和输出词矩阵 W'。在 CBOW 中,W 用于将多个上下文词向量叠加平均为隐藏层向量,W' 用于根据隐藏层向量预测中心词。在 Skip-Gram 中,W 用于将中心词映射为隐藏层向量,W' 用于根据隐藏层向量预测上下文词。训练完成后,输入词矩阵 W 就是我们需要的词向量。

### 2.3 层序与功能
Word2Vec 的网络结构一般包含三层:输入层、投影层(隐藏层)和输出层。

- 输入层:用 One-hot 向量表示每个样本的上下文词(CBOW)或中心词(Skip-Gram)。
- 投影层:CBOW 将多个上下文词向量求平均并传递到下一层;Skip-Gram 将中心词向量传递到下一层。
- 输出层:使用 Softmax 函数计算中心词(CBOW)或上下文词(Skip-Gram)的概率分布。

网络的目标是最大化样本的似然概率,即最小化输出分布与真实分布的交叉熵损失。

### 2.4 负采样(Negative Sampling) 
为了提高训练效率,Word2Vec 通常使用负采样策略替代原始的 Softmax 函数。负采样将多分类问题转化为二分类问题,即判断一个词是否为中心词的上下文。对于每个正样本(真实的上下文词),随机采样 k 个负样本(不是上下文的词)进行二分类。这种方法不仅大幅降低了计算复杂度,而且对高频词进行了下采样,缓解了词频分布不均衡的问题。

## 3. 核心算法原理具体操作步骤
下面以 CBOW 模型为例,详细介绍 Word2Vec 的训练过程。

输入:语料库 $\mathcal{D}=\{w_1,w_2,...,w_T\}$,上下文窗口大小 $c$,词向量维度 $d$,负采样数 $k$,学习率 $\eta$。

输出:词向量矩阵 $W\in \mathbb{R}^{|V|\times d}$。

1) 随机初始化输入词矩阵 $W$ 和输出词矩阵 $W'$。

2) 对于语料库中的每个位置 $t=1,2,...,T$:

a. 获取中心词 $w_t$ 及其上下文 $\mathcal{C}_t=\{w_{t-c},...,w_{t-1},w_{t+1},...,w_{t+c}\}$。

b. 将上下文词 $\mathcal{C}_t$ 的词向量求平均得到隐藏层向量:
$$
\mathbf{h} =\frac{1}{2c} \sum_{w_i \in \mathcal{C}_t} \mathbf{v}_{w_i}
$$

c. 对于中心词 $w_t$,采样 $k$ 个负样本 $\{w_j\}_{j=1}^k$。

d. 计算正样本 $(w_t,1)$ 和负样本 $\{(w_j,0)\}_{j=1}^k$ 的似然概率:
$$
\mathcal{L}=\log \sigma(\mathbf{u}_{w_t}^\top \mathbf{h}) +\sum_{j=1}^k \log \sigma(-\mathbf{u}_{w_j}^\top \mathbf{h})
$$

e. 计算损失函数关于各参数的梯度:
$$
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial \mathbf{u}_{w_t}} &= (1-\sigma(\mathbf{u}_{w_t}^\top \mathbf{h}))\mathbf{h}\\
\frac{\partial \mathcal{L}}{\partial \mathbf{u}_{w_j}} &= -\left(1-\sigma(-\mathbf{u}_{w_j}^\top \mathbf{h})\right)\mathbf{h}\\
\frac{\partial \mathcal{L}}{\partial \mathbf{h}} &= (1-\sigma(\mathbf{u}_{w_t}^\top \mathbf{h}))\mathbf{u}_{w_t}-\sum_{j=1}^k\left(1-\sigma(-\mathbf{u}_{w_j}^\top \mathbf{h})\right)\mathbf{u}_{w_j}\\
\frac{\partial \mathcal{L}}{\partial \mathbf{v}_{w_i}} &= \frac{1}{2c}\frac{\partial \mathcal{L}}{\partial \mathbf{h}}, \forall w_i \in \mathcal{C}_t
\end{aligned}
$$

f. 根据梯度更新参数:
$$
\begin{aligned}
\mathbf{u}_{w_t} &\leftarrow \mathbf{u}_{w_t} + \eta \frac{\partial \mathcal{L}}{\partial \mathbf{u}_{w_t}}\\
\mathbf{u}_{w_j} &\leftarrow \mathbf{u}_{w_j} + \eta \frac{\partial \mathcal{L}}{\partial \mathbf{u}_{w_j}}, j=1,...,k\\
\mathbf{v}_{w_i} &\leftarrow \mathbf{v}_{w_i} + \eta \frac{\partial \mathcal{L}}{\partial \mathbf{v}_{w_i}}, \forall w_i \in \mathcal{C}_t
\end{aligned}
$$

3) 返回输入词矩阵 $W$ 作为最终的词向量。

Skip-Gram 模型的训练过程与 CBOW 类似,主要区别在于隐藏层向量来自中心词而非上下文词。此外,Skip-Gram 需要对每个上下文词单独计算损失和梯度。

## 4. 数学模型和公式详细讲解举例说明
### 4.1 词向量的数学表示
给定大小为 $|V|$ 的词汇表,词向量将每个单词映射为一个 $d$ 维实数向量。因此,整个词汇表可以用一个矩阵 $W\in \mathbb{R}^{|V|\times d}$ 表示,其中第 $i$ 行 $\mathbf{w}_i$ 就是第 $i$ 个单词的词向量。

例如,假设词汇表为 $\{$"cat","dog","animal","apple"$\}$,词向量维度为3,则词向量矩阵可能为:
$$
W=
\begin{bmatrix}
0.1 & -0.2 & 0.3 \\
0.2 & -0.1 & 0.4 \\
0.3 & 0.1 & 0.5 \\
-0.1 & 0.2 & -0.3
\end{bmatrix}
$$

其中,"cat"的词向量为 $[0.1,-0.2,0.3]$,"dog"的词向量为 $[0.2,-0.1,0.4]$ 等。

### 4.2 CBOW 的数学描述
给定一个长度为 $T$ 的语料库 $\mathcal{D}=\{w_1,w_2,...,w_T\}$,CBOW 的目标是最大化如下似然概率:
$$
\mathcal{L}(\mathcal{D})=\prod_{t=1}^T P(w_t|\mathcal{C}_t)
$$

其中 $\mathcal{C}_t$ 表示单词 $w_t$ 的上下文。为了计算条件概率 $P(w_t|\mathcal{C}_t)$,CBOW 首先将上下文词向量求平均得到隐藏层向量:
$$
\mathbf{h}=\frac{1}{2c}\sum_{w_i \in \mathcal{C}_t} \mathbf{v}_{w_i}
$$

然后通过 Softmax 函数计算条件概率:
$$
P(w_t|\mathcal{C}_t)=\frac{\exp(\mathbf{u}_{w_t}^\top \mathbf{h})}{\sum_{w\in V}\exp(\mathbf{u}_w^\top \mathbf{h})}
$$

其中 $\mathbf{u}_w$ 是词汇表中每个单词对应的输出向量。

例如,假设词汇表为 $\{$"cat","dog","animal","apple"$\}$,上下文窗口大小为2,语料库为"animal dog cat apple"。对于中心词"dog",它的上下文为 $\mathcal{C}_{\text{dog}}=\{$"animal","cat"$\}$,隐藏层向量为:
$$
\mathbf{h}=\frac{1}{2}(\mathbf{v}_{\text{animal}}+\mathbf{v}_{\text{cat}})
$$

条件概率 $P($"dog"$|\mathcal{C}_{\text{dog}})$ 为:
$$
P(\text{"dog"}|\mathcal{C}_{\text{dog}})=\frac{\exp(\mathbf{u}_{\text{dog}}^\top \mathbf{h})}{\sum_{w\in V}\exp(\mathbf{u}_w^\top \mathbf{h})}
$$

### 4.3 负采样的数学描述
负采样将多分类问题转化为二分类问题。对于一个正样本 $(w,\mathcal{C})$,负采样根据噪声分布 $P_n(w)$ 采样 $k$ 个负样本 $\{(w_j,\mathcal{C})\}_{j=1}^k$。噪声分布通常选择 $P_n(w) \propto U(w)^{3/4}$,其中 $U(w)$ 是词频。

负采样的目标是最大化如下似然概率:
$$
\mathcal{L}=\log \sigma(\mathbf{u}_w^\top \mathbf{h}) + \sum_{j=1}^k \log \sigma(-\mathbf{u}_{w_j}^\top \mathbf{h})
$$

其中 $\sigma(x)=\frac{1}{1+e^{-x}}$ 是 Sigmoid 函数。直观地看,负采样试图将正样本的得分 $\mathbf{u}_w^\top \mathbf{h}$ 最大化,同时将负样本的得分 $\mathbf{u}_{w_j}^\top \mathbf{h}$ 最小化。

## 5. 项目实践:代码实例和详细解释说明
下面使用 Python 和 NumPy 库从零实现一个简单的 CBOW 模型。

```python
import numpy as np

class CBOW:
    def __init__(self, vocab_size, embed_size, context_size):
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.context_size = context_size
        self.input_weights = np.random.uniform(-1, 1, (vocab