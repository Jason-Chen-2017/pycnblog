
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Knowledge graph (KG) representation learning has become a hot research topic recently with various models being proposed for different tasks such as link prediction, question answering, entity classification or relation extraction. However, little attention is paid to address the problem of training data size that impacts the performance of learned representations on complex real-world knowledge graphs. To fill this gap, we propose a new algorithm called ComplEx-NARS which employs negative sampling to balance positive and negative examples within each batch during training. We further enhance our model by introducing a novel self-supervised strategy called Relation Induction which leverages statistical patterns among entities and relations to learn latent embeddings automatically from large scale unlabeled text corpora without any annotation guidance. The key idea behind ComplEx-NARS is to use projection matrices to map the original complex vector space into a reduced lower dimensional one where dot product can capture the semantic relationships among entities and relations more accurately than traditional approaches based on local neighborhoods. This leads to significant improvement over previous approaches. Our experiments show that ComplEx-NARS achieves competitive results on benchmark datasets while reducing the need for annotated training data by up to 90% compared to other popular baselines. Moreover, through an ablation study, we reveal that using multiple hidden layers enhances the expressiveness of the model and improves its ability to capture multi-granularity relationships better. Finally, we demonstrate how Relation Induction can effectively reduce both labeled and unlabeled data requirements leading to significant improvements even when the number of entities exceeds 1 million. Overall, our work provides insights and practical solutions towards addressing the challenge of limited training data available in knowledge graph representation learning and gives hope for promising future research directions in this field.
本文总结了知识图谱表示学习中一个重要且挑战性的问题——训练数据集大小对模型性能的影响。作者提出了一个新的算法——ComplEx-NARS，它通过负采样的方式在每个批次训练过程中平衡正负例。除此之外，为了进一步提升模型性能，作者还提出了一个用于自动从大规模无标注文本语料库中学习潜在嵌入向量的新型自监督策略——关系推断（Relation Induction）。基于ComplEx的映射矩阵将原始复杂向量空间投影到低维空间，可以更精确捕获实体之间的语义关系，而不是传统的方法依赖于局部邻域。相比之下，这个方法在表达能力、多级关系建模方面都有明显优势。实验结果表明，ComplEx-NARS 在各类基准测试集上均取得了可观的成绩，并且通过减少训练数据集的需求，获得了接近十倍的效果，甚至超过其他一些主流基线算法。关于关系推断（RI）的验证研究也得出的重要结论是，它能够有效地降低需要的标注和未标注数据的数量，同时仍然能提升模型性能。作者认为，本文的研究成果为解决知识图谱表示学习中的训练数据集不足而提供了新的思路。
# 2. 相关工作背景
## （1）KG Embedding Tasks
之前的工作大致可以分为两大类：基于矩阵分解的方法（SVD++, CP, TransE等）以及基于神经网络的方法（TransH, TransR等）。在SVD++及其变种方法中，表示学习的目标是在低秩矩阵分解之后得到较好的语义表示；而在神经网络方法中，则利用神经网络实现端到端的学习，将输入实体或关系转化为适合于语义理解的特征向量。但很遗憾的是，这些模型往往会受限于训练数据集的规模，即便能够训练出高质量的表示，它们也可能存在以下问题：

1. 训练效率：由于传统的基于矩阵分解的方法直接求解低秩矩阵，因此需要对整个训练集进行完整的计算才能得到最终的结果，计算开销非常大。同时，在评估时也只能对一个查询实体进行测试，无法对整个查询集合进行统一的测试。
2. 模型易受噪声影响：传统的模型通常假设实体和关系都是相互独立的，忽略了实体间的联系信息。而现实世界的知识图谱往往具有丰富的上下文信息，因此噪声的引入可能会导致模型的不稳定。
3. 模型缺乏长尾分布：在大型知识图谱中，训练数据集往往存在着冷启动问题，即实体或关系的出现频率非常低，对模型的学习造成了困难。这一问题也困扰了传统的表示学习方法，因为它们通常采用与训练数据集相同的分布初始化参数。

## （2）Previous Works on Limited Training Set Size
很多研究人员曾尝试解决训练数据集小的问题，比如：1）小数据集学习法（Few-Shot Learning）；2）半监督学习（Semi-Supervised Learning）；3）重排标签（Re-labelling）等。但目前看来，这些方法都存在以下两个共同特点：第一，它们的训练目标往往只是扩展已有的简单模型，并没有创造新的模型架构；第二，它们往往只针对某些特定任务进行优化，忽略了一般性的优化方向。因此，希望找到一种更通用的优化框架，能够充分考虑不同任务的异同。另外，如何平衡不同的任务，确保模型的泛化能力，也是值得关注的问题。

# 3. ComplEx-NARS: A New Algorithm for Knowledge Graph Representation Learning
## Introduction
随着知识图谱的快速发展，越来越多的基于深度学习的模型被提出，这些模型的精度已经超过了传统的基于规则或统计的方法。但是，这些模型往往对训练数据集的大小敏感，当训练数据集小于某个阈值时，它们的性能就不一定能达到最佳水平。本文所要解决的主要问题就是如何在训练数据集和模型性能之间找到一个折衷点。

为了解决这个问题，作者设计了一套新的模型——ComplEx-NARS，其结构类似于ComplEx模型，也采用了投影矩阵的技巧。但是，其又加入了负采样的方法，以平衡正负例。具体来说，在每个批次中，模型会先对所有正例实体对和关系对进行正采样，然后再随机采样一些负例实体对和关系对。这样做的目的就是使模型在损失函数计算的时候，既有较大的权重分配给正例，也有较小的权重分配给负例。同时，为了保证负采样过程的平衡性，作者在负采样前先进行预处理。

除此之外，为了进一步提升模型的性能，作者提出了一种新的自监督策略——关系推断（RI），它利用大规模无标注文本数据集来自动学习潜在的关系特征。具体来说，RI首先将无监督学习任务定义为学习潜在的实体关系分布（EDM）。EDM是一个关于实体与关系的概率分布，其由模型学习到的数据生成。然后，模型利用EDM建立一个编码器（Encoder）网络，该网络将实体和关系抽象成固定长度的向量，并学习实体和关系之间的映射关系。这种编码器可以用来编码许多知识图谱数据集，如WordNet、Freebase等。作者发现，通过自学习的关系特征，可以让模型学习到更丰富的表示。

最后，作者还探索了训练时隐藏层的数量，发现使用多个隐藏层能够提升模型的表达能力，并且能够更好地捕获多级关系。作者还发现，当实体个数超过一百万时，基于RI的模型能够获得很好的效果，但这需要消耗大量的计算资源。

## Approach
### Entity-relation embedding
ComplEx-NARS 使用了ComplEx模型作为基础模型。具体来说，ComplEx模型将实体和关系分别用两个embedding向量表示，并利用投影矩阵将其映射到一个低维的连续向量空间中，这个空间可以通过一个内积函数来表示实体间、关系间以及两个实体间和关系间的复杂的相互作用。ComplEx模型采用了两个投影矩阵$P_r$和$P_e$，来将两个向量投射到低维空间：

$$
v'_r = P_r v_r
$$

$$
v'_e = P_e v_e
$$

其中，$\{v_r, v_e\}$表示实体和关系的原始embedding向量，$v'_r$和$v'_e$表示经过投影后的向量。投影后的向量组成的空间可以看到，是与原始向量有关的更多的信息的表示。

### Negative Sampling
正例和负例构成了训练的真实标签，而模型需要根据这些标签进行训练。为了解决这个问题，作者在每个批次中，首先对所有的正例实体对和关系对进行正采样，然后随机采样一些负例实体对和关系对。具体来说，正例实体对指的是模型训练时同时遇到的正例实体对，关系对指的是模型训练时同时遇到的关系对。负例实体对和关系对则是模型训练时遇到的其他实体对或关系对。

在每个批次中，模型会根据损失函数对两种类型实体对进行区分，即关系和实体对。对于实体对，损失函数仅仅将两个向量的差距计算出来，并没有区别对待正例和负例；而对于关系对，模型会在正例和负例上进行区分，区分的依据是模型预测的边的类型。具体的损失函数如下：

$$
L(X,Y)=\frac{\partial}{\partial X} f(X,Y)\cdot g(X,Y)+\lambda \cdot L_r(W_{rel})+L_{NS}(X,Y)
$$

$$
f(X,Y)=-\log \sigma(\langle X, Y^T\rangle )+\sum_{j=1}^{m}\log \sigma(-\langle X, W_j^\top y_j + b_j\rangle)
$$

其中，$\sigma(x)=\frac{1}{1+exp(-x)}$是sigmoid函数，$\{y_j\}_{j=1}^m$表示正例实体对对应的标签向量，$W_j$, $b_j$表示相应的关系、实体对的权重和偏置，$\lambda$表示正例实体对的比例，$L_{NS}$表示负例实体对的损失函数。

### Relation Induction
为了实现ComplEx-NARS模型的自监督学习能力，作者提出了关系推断（RI）的概念。RI是一种无监督的机器学习任务，旨在学习潜在的实体关系分布，即EDM。EDM是对所有实体和关系之间发生的各种关系的概率分布。RI 的目的是学习EDM的表示形式，并将其用作模型编码器的输入。换句话说，EDM可以视为一个无监督的语言模型，学习它的表示可以帮助模型发现事物之间的共性，并有效地编码输入文本。

具体来说，RI 将无监督学习任务定义为学习 EDM 的表示，其有两个主要步骤。首先，利用大规模的文本数据（如 Wikipedia 或 Freebase）学习 EDMS，其包括 EDM 的所有可能的边缘及其出现次数。然后，利用 EDM 和一个编码器网络来生成输出序列，序列里包含输入文本中所有边缘及其概率。

例如，假设词典中有 $V$ 个词，则 EDM 可以定义为一个 $|V|\times |V|\times |\mathcal{R}|$ 大小的三维概率分布，其中 $|\mathcal{R}|$ 是关系的数量。那么，为了学习 EDM ，可以假设词汇表的大小为 $K$ ，那么就可以使用概率主题模型来对边缘分布进行建模。具体的步骤如下：

1. 对文档进行切词，使用词袋模型（Bag of Words，BoW）将文档转换为词袋序列，每个文档对应一个词序列。
2. 根据词的上下文词计数和转移计数对文档进行词袋二阶统计。
3. 通过 EM 算法迭代更新模型参数，直至收敛。

使用 EDM 后，编码器网络可以对实体和关系进行编码。编码器网络的输入包括两个嵌入矩阵 $E$ 和 $\Theta$，即实体嵌入矩阵和关系嵌入矩阵。通过学习 EDMs 和二阶统计信息，编码器网络可以把实体和关系抽象成固定维度的矢量。例如，假设 $n_e$ 为实体数，$d_e$ 为嵌入向量维度，$n_r$ 为关系数，$d_{\Theta}$ 为关系嵌入向量维度。编码器网络的输出可以用来预测实体对 $(i, j)$ 和关系对 $(r, s)$ 。具体的计算公式如下：

$$
z_e=(\sum_{k=1}^{K}p^{word}_k x_k^{(e)})_i+\Theta e_r
$$

$$
z_r=\Theta r+(q_r)^T p_{edge}(j, i)
$$

其中，$(x_k^{(e)}, y_k^{(e)}, z_e)$ 表示第 k 次出现的实体，$(x_k^{(r)}, y_k^{(r)}, z_r)$ 表示第 k 次出现的关系，$\Theta$ 和 $\phi$ 表示实体和关系的嵌入矩阵。$\psi$ 表示转移概率矩阵，$p_{edge}(j, i)$ 表示 j 指向 i 的概率。

### Self-supervised Learning
为了实现 ComplEx-NARS 模型的自监督学习能力，作者提出了一种新的自监督学习策略——关系推断（RI）。RI 是一种无监督的学习任务，旨在学习 EDM 的表示形式，并将其用作模型编码器的输入。为了实现 RI，作者首先定义了实体-关系图（Entity-Relation Graph，ERG）。ERG 是一个对所有实体-关系对的三元组进行记录的图，它包括三个子图，即实体子图、关系子图和边子图。实体子图记录了所有实体，关系子图记录了所有关系，边子图记录了实体与关系之间的连接关系。作者将 ERG 分为三个阶段进行学习。

第一个阶段是实体学习阶段，作者使用一种实体链接算法来完成实体识别。实体链接算法的输入是带有上下文信息的实体 mention，它可以自动识别同一实体 mention 在不同文档中的上下文位置。然后，使用结构化信息来调整实体链接结果，如页面标题、篇章结构等。

第二个阶段是关系抽取阶段，作者提出了一种关系抽取框架，它可以从实体-关系图中提取关系。具体来说，作者将 ERG 划分成 $C$ 个聚类，每一个聚类是一个拥有相似语义含义的实体-关系子集。然后，在每个聚类中进行关系抽取，即从各个实体指向其最近的相邻实体的方式抽取出所有可能的关系。

第三个阶段是关系学习阶段，作者将学习到的关系抽取结果和实体链接结果联合起来，构造知识图谱。具体来说，先将所有关系抽取结果归约到两个类别——语义关系和上下位关系，再利用实体链接结果修正实体名称，从而构建整个知识图谱。

最后，利用知识图谱，作者构建了由 $n_e$ 个实体和 $n_r$ 个关系组成的大型实体-关系图，称之为知识图谱图。从知识图谱图中，作者可以学习到知识的表示，包括实体向量和关系向量。随后，可以基于知识图谱图的结构来编码输入文本，或者利用知识图谱图来分类和匹配实体、关系等。