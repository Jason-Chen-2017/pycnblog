
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## Performer: Parameter-Efficient Attention in Linear Time and Space
Performer（中文名：Perfomer）是一种注意力机制。它的提出是为了解决Transformer模型中参数量过多的问题。Performer通过减少softmax运算，将注意力分布的参数数量降低到与输入相同的数量，并在计算时进行分组聚合。这样可以在不增加参数数量和计算量的情况下，达到比标准Transformer更好的效果。而且，由于attention是所有Transformer层的基础模块，因此可以应用于各种深度学习模型中，如GPT、BERT等。
本文主要介绍Performer中的重要知识点：

1. Self-Attention and its Complexity
2. Performance Bottleneck of Softmax-based Attention Mechanisms
3. Local Attention and Efficient Permutation Invariance via Random Projections
4. Locality Matters: Sampling for Performer and its Effect on Model Size and Speed
5. Favoring High-Dimensional Input Dimensions with Factorized Transformations
6. Partitioned Feedforward Networks as an Alternative to ReLU Activation Functions
7. Contrastive Predictive Coding: An Architecture for Memory-Augmented Neural Networks
8. The Surprising Power of Cross-Attention and performer:parameter efficient attention in linear time and space

为了将以上内容清晰地表达出来，本文分9段分别介绍每个知识点的内容。每段的标题会根据每个知识点的标题加以简要阐述。
# 一、Self-Attention and Its Complexity
Transformer模型中的self-attention是一个非常重要的模块。其能够实现两个输入之间的直接信息交互。但是，当词嵌入维度越高的时候，其所占用的内存也就越多。而且，softmax计算对于每一个位置的特征向量，都需要进行一次计算。因此，当维度或序列长度增长时，计算开销就会随之增加，而模型的性能也会变得越来越差。因此，为了解决上述问题，Performer采用了两种方式：

1. Partitioning：将输入空间分割成多个区块，其中每一个区块被分配给不同的头部。这样，每个头部只关注自身区块内的信息。
2. Reducing the softmax operation: 使用局部性质，只对相关区块上的权重进行归一化，而不是全局所有的权重。

下图展示的是经典的self-attention矩阵。其中每个方格代表一个词或句子中的一个位置，而每个条纹代表一个头部。每个头部关注当前位置周围的词的注意力。但是，经典的self-attention存在两个问题：

1. Computing complexity is exponential in sequence length
2. Memory consumption increases quadratically with input dimension

因此，Performer首先研究如何解决第一个问题。下面是解决方案：

1. Self-Attention Block Decomposition: 将每个位置的计算拆分成多个小方格，每一个小方格对应一个头部。这样，就可以有效地减少softmax计算。
2. Switch Transformer: 通过使用switch矩阵，能够在计算时同时考虑多种类型的注意力。


上图展示了一个经典的self-attention矩阵。每个方格代表一个词或句子中的一个位置，而每个条纹代表一个头部。每个头部只关注当前位置周围的词的注意力。由于词嵌入维度越高，整个注意力矩阵的大小也会随之增加。但实际上，很多位置之间的注意力都是相似的，因此可以通过分块的方式进行计算，从而避免重复计算。例如，上图左边的四个头部只关注词向量w[i-1]、w[i]、w[i+1]和w[i+2]；而右边的两个头部只关注词向量w[j-1]和w[j+1]。这样就可以降低计算复杂度，并且还能较好地减少内存消耗。

除了解决计算复杂度外，另一个需要解决的问题就是内存消耗。因为每个词或句子都对应一个向量，因此，如果词嵌入维度很高的话，需要消耗大量的内存。为了解决这个问题，Performer提出了以下两个方法：

1. Low-rank projection matrices: 投影矩阵通过限制投影向量的数量，使得矩阵的秩较低，从而降低存储空间。
2. Long-range interactions: 在进行注意力计算时，仅利用局部范围内的词向量进行通信。

因此，Performer的计算复杂度与模型尺寸之间的tradeoff便出现了。Performer通过分块的方式降低了复杂度，但同时又保持了模型的内存消耗与词嵌入维度之间的tradeoff。此外，Performer也试图通过随机投影矩阵，改进模型的性能。Random Projections的方法是：

假设输入X是一个m×n矩阵，其元素满足均匀分布。那么，如果随机选择r列向量作为投影向量，就得到一个新的矩阵W∈Rn×m，且有U=XVW的计算，其中V是随机选择的n×r矩阵。因此，新矩阵的秩为r，且计算过程需要额外的O(mnklogn)时间，其中k是随机生成的k列向量的数量。换句话说，当投影向量数量较少时，可能会获得较好的性能。而当投影向量数量较多时，则会降低计算时间。 

因此，Locality Matters: Sampling for Performer and its Effect on Model Size and Speed
第二个方法Local Attention与Long Range Interactions有关。首先，Local Attention指的是仅对局部区域内的词向量进行注意力计算。举例来说，如果句子中有一句话“I like apple”中的“apple”，那么其他的词向量都不需要参与计算。Performer通过采样策略，在训练和预测阶段，均衡地进行局部注意力计算。具体做法如下：

1. During training phase: 在训练时，选取一些样本，并在其后面加入特殊符号“<EOS>”。然后，依次对这些样本进行训练。
2. During inference phase: 当模型生成新文本时，不要为完整的文本生成完整的输出。而是在生成过程中，每生成一个词，就切换到与该词相关联的局部区块。

这种采样策略能够保证模型的鲁棒性。因此，模型的性能不会受到影响太多，但会改善模型的训练速度。此外，通过局部注意力计算，Performer模型的尺寸和计算时间可以得到优化。Locality Matters之后，Factorized Transformations方法、Partitioned Feedforward Networks、Contrastive Predictive Coding方法均属于面向深度学习领域的最新研究范畴。下面介绍一下具体方法。
# 三、Low Rank Projection Matrices
### Introduction
Low-rank projection matrices是一种非常有效的降低矩阵维度的方法。它可以减少存储空间和计算资源消耗，并提升模型的性能。然而，通常情况下，矩阵的秩（Rank）往往会比期望值小很多。为此，Performer提出了一个训练过程，用不同的随机投影矩阵来尝试不同的秩。如下图所示：


如上图所示，左边部分是原始的矩阵X。右边部分的每个圆圈代表一个不同的随机投影矩阵V。在训练过程中，随机投影矩阵V都会被更新，通过最小化Frobenius距离和秩约束来找到最优的投影矩阵。其中，rho是一个超参数，用于控制模型的能力。该过程的目标函数定义如下：


训练过程结束之后，根据设定的目标秩，会选出相应的投影矩阵，然后利用该投影矩阵对输入矩阵X进行投影，并得到新的矩阵Z。最后，模型的计算可以进行基于Z的矩阵乘法。因此，通过调整目标秩，可以在不增加参数的情况下，降低模型的计算复杂度和内存消耗。

在实验结果上表明，Performer的低秩投影矩阵能够显著降低模型的计算复杂度和内存消耗。实验数据表明，当目标秩设置为32时，Performer的性能比一般的Transformer模型具有更好的性能。因此，Performer的投影矩阵方法可以提供优化的性能。除此之外，该方法还有其它优点，如模型尺寸更小、训练更快、泛化能力更强、可迁移性更强等。

除了Low-rank projection matrices方法外，Performer还提出了两个其它方法：Factorized Transformations方法和Partitioned Feedforward Networks方法。接下来分别介绍。
# 四、Factorized Transformations Method
### Introduction
Factorized Transformations Method是Performer的一项创新工作。它是指将线性变换分解为若干因子，然后针对不同的因子进行不同的操作。如下图所示：


如上图所示，在图中，x1，x2，...，xk是输入向量。每一步都将x的维度压缩为两倍的维度。图中每一个蓝色圆圈代表一次线性变换。例如，第一次将向量分成四份，表示为W1, W2，W3，W4，然后将这四份向量相加得到y1。再次线性变换，将y1分成四份，得到y2。依此类推。同样，也可以对y进行类似操作。因此，通过对不同维度进行不同操作，Factorized Transformations Method能够降低模型的计算复杂度。

但是，当出现奇异值分解时，一般的线性变换无法进行。因此，Performer提出了两个改进方案：

1. Singular Value Clipping: 在奇异值分解时，限制最大的奇异值。
2. Strong Eigenvalue Regularization: 通过在奇异值分解之前施加强大的正则化，使得模型的性能更稳定。

Singular Value Clipping是指限制最大的奇异值。当某一行或某一列对应的奇异值为零时，会导致矩阵不可逆。因此，为了防止这种情况发生，Performer会在奇异值分解之前，对奇异值进行裁剪。这样，在进行奇异值分解时，只有特别大的奇异值才会对分解产生影响。如下图所示：


如上图所示，设定最大的奇异值为γ。对于奇异值σi≥γ的i，把它们置零。对角矩阵的最小奇异值等于γ。Performer认为，限制最大的奇异值可以保留更多的信息。而且，对角阵的限制并不是十分必要，因为对角阵在计算时才起作用。因此，Performer只在需要进行奇异值分解时，才对奇异值进行限制。

Strong Eigenvalue Regularization是指在奇异值分解之前，在损失函数中加入强大的正则化。举例来说，令λ是模型的参数，Φ(λ)是损失函数。Φ(λ)的设计是希望让模型学习到可分的特征，即希望λ的值尽可能地小。但是，由于λ可以取任意值，所以难以固定λ。因此，Performer提出在λ的梯度方向上施加一个强大的正则化，在损失函数的梯度方向上引入惩罚项。如下图所示：


如上图所示，Φ(λ)函数的变化会由损失函数θ(λ)决定。θ(λ)是一个负的概率密度函数，即λ越大，则θ(λ)越小。但是，θ(λ)不是凸函数。因此，如果θ(λ)不是凸函数，那么优化就无法收敛。但是，如果θ(λ)是一个凸函数，则说明λ的值是允许的。这样，当θ(λ)的梯度和负梯度相遇时，我们可以利用线搜索法找到合适的λ值。具体地，通过梯度的负曲率和导数的负半值来近似θ(λ)的负梯度。利用这一关系，可以得到精确的λ值。同时，在训练时，不断更新参数λ，直至其满足θ(λ)的条件。

总结一下，Factorized Transformations Method通过在奇异值分解前加入新的限制和正则化，可以减少模型的计算复杂度。新的限制使得模型能够保留更多的信息，而正则化使得模型能够拟合可分的数据。

后面的Partitioned Feedforward Networks方法和Contrastive Predictive Coding方法也是建立在Factorized Transformations Method方法上的。因此，了解了Factorized Transformations Method后，这两个方法的内容比较简单，没有太大的可解释性。因此，下面的内容只是介绍一下，并没有细致地阐述。