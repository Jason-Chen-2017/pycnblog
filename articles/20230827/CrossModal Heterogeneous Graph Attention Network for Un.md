
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 概述
在多模态领域，大量研究试图构建统一的框架将不同模态的特征整合到一起进行高效的预测或分析。然而，传统的基于深度学习的多模态分析方法存在着不足之处，特别是在多种模态之间存在高度重叠、冗余时，模型可能难以从多个模态中学习有效的共同特征，并因此导致预测准确率低下。

针对这个问题，作者提出了一个名为CroGAT的新型跨模态异构图注意力网络(cross-modal heterogeneous graph attention network)，该网络能够同时利用两种不同的输入模态——位置信息(location information)和用户特征(user features)——来获取独有的信息。通过引入图神经网络模块(graph neural networks module), CroGAT能够捕获异构数据的全局依赖关系并提取有效的特征表示。除此之外，作者还设计了具有实验性质的新型异构图注意力机制来对数据中的模式进行推理。

本文将主要围绕以下几个方面展开：

1. Cross-Modal Heterogeneous Graph Attention Networks: 介绍CroGAT的设计理念及其组成模块；
2. Unified multi-modal understanding and prediction: 论证CroGAT能够显著提升多模态预测任务的效果；
3. Experiments: 在一个真实的交通移动预测数据集上进行实验验证；
4. Future Directions: 对未来的研究方向进行展望。

# 2.相关工作
多模态学习(Multi-Modal Learning)已成为许多计算机视觉、自然语言处理、社会计算等领域的热门话题。多模态数据通常包括图像、文本、音频等多种类型的数据，这些数据的融合可以增强机器学习模型的能力。传统的多模态学习方法主要包括基于单模态分类器(Single Modality Classifier)的方法、基于深度学习的联合特征嵌入(Joint Feature Embedding)方法、和多任务学习(Multi-Task Learning)方法。近年来，基于深度学习的多模态学习方法也得到越来越多的关注，如基于神经网络的双向多模态学习(Bidirectional Multimodal Learning)、时序多模态学习(Temporal Multimodal Learning)、跨模态理解(Cross-Modal Understanding)和预测(Prediction)。但是，当前的多模态学习方法仍存在一些局限性，比如：

1. 模型过于简单，无法捕获复杂的全局关系；
2. 假设输入数据的分布和相似度是固定的，不能适应新的场景；
3. 不适用于多步预测任务。

为了解决以上问题，本文提出一种新型的跨模态异构图注意力网络(cross-modal heterogeneous graph attention network)。这种网络能够同时利用两种不同的输入模态——位置信息(location information)和用户特征(user features)——来获取独有的信息。通过引入图神经网络模块(graph neural networks module), CroGAT能够捕获异构数据的全局依赖关系并提取有效的特征表示。除此之外，作者还设计了具有实验性质的新型异构图注意力机制来对数据中的模式进行推理。



# 3.算法描述
## Cross-Modal Heterogeneous Graph Attention Networks (CroGAT)
### 模型概览
CroGAT是一个基于图卷积神经网络(Graph Convolutional Neural Networks, GCN)的跨模态异构图注意力网络。该网络由两部分组成：位置编码模块(Location Encoding Module)和用户特征编码模块(User Features Encoding Module)。两个模块分别独立地编码位置信息和用户特征，然后将编码后的结果送入相同的图卷积神经网络层(Graph Convolutional Neural Networks Layer)中进行融合。为了学习全局的信息依赖关系，CroGAT采用了两种类型的图注意力机制：1）异构图注意力机制(Heterogeneous Graph Attention Mechanism): 利用位置编码模块生成的位置特征向量和用户特征编码模块生成的用户特征向量，从而实现位置相关的用户特征的重建；2）异构序列注意力机制(Heterogeneous Sequence Attention Mechanism): 将位置编码模块生成的位置序列向量和用户特征编码模块生成的用户特征序列向量进行匹配，从而捕获序列动态信息的变化，用于提升模型的鲁棒性。


图1: CroGAT模型结构示意图

### Location Encoding Module
位置编码模块由若干个位置感知单元(Location-Aware Unit, LAU)组成，每个LAU负责学习特定区域的位置特征。位置特征由三部分组成：1）位置嵌入向量(Position Embedding Vector): 每个位置由唯一的位置嵌入向量表示；2）位置上下文特征(Position Contextual Features): 通过将同一区域内其他位置的特征结合得到区域的上下文特征；3）位置偏差(Position Bias): 通过统计其他位置嵌入向量之间的距离来对每个位置添加位置偏差。位置上下文特征和位置偏差都可通过图卷积网络进行学习。


图2: 位置编码模块示意图

### User Features Encoding Module
用户特征编码模块由若干个用户特征感知单元(User-Feature Aware Unit, UFAU)组成，每个UFAU负责学习特定用户的特征。用户特征由两部分组成：1）用户嵌入向量(User Embedding Vector): 每个用户由唯一的用户嵌入向量表示；2）用户上下文特征(User Contextual Features): 通过将同一用户其他时间点的特征结合得到用户的上下文特征。用户上下文特征可通过图卷积网络进行学习。


图3: 用户特征编码模块示意图

### Graph Convolutional Neural Networks Layer
本文采用带有多个头部的图卷积神经网络层(multihead Graph Convolutional Neural Networks Layer)来捕获异构数据中的全局依赖关系。图卷积神经网络层由多个头部组成，每个头部对节点进行更新，从而捕获不同类型的依赖关系。对于位置信息和用户特征，CroGAT分别应用图注意力机制来学习全局的依赖关系，从而获得有效的特征表示。

#### Heterogeneous Graph Attention Mechanism
异构图注意力机制利用不同模态间的关联性，即不同位置和用户之间的关系。具体来说，该机制允许位置特征在空间上的依赖性(spatial dependency)和用户特征的依赖性(temporal dependency)进行交互。首先，CroGAT将位置编码模块生成的位置特征和用户特征编码模块生成的用户特征拼接，并用具有不同权重的图注意力机制(graph attention mechanisms)进行融合，以学习位置相关的用户特征的重建。

位置特征的重建可以用来识别不同位置之间的区域间联系，从而促进位置相关的用户特征的重建。具体来说，给定一组用户特征$u_{i}, i=1,...,N$和位置特征$l_{j}$，其中$N$是用户数量，$M$是位置数量，假设位置$l_{j}$周围有$K$个邻居位置，则位置$l_{j}$的嵌入表示$\overline{l}_{j}=\sum_{k=1}^{K}\alpha_{jk}l_{k}$，其中$\alpha_{jk}$是邻居位置权重。CroGAT将位置特征$l_{j}$和用户特征$u_{i}$映射为高维空间，然后利用多头的图注意力机制来学习区域间的位置特征依赖关系。具体来说，假设位置特征矩阵$L \in R^{N\times M\times d}$和用户特征矩阵$U \in R^{N\times f}$，其中$d$是嵌入维度，$f$是用户特征维度。那么，第$h$-th头的位置特征嵌入$Z_{h}^{\text{pos}} \in R^{N\times d}$和用户特征嵌入$Z_{h}^{\text{feat}} \in R^{N\times df}$可以通过以下公式进行计算：
$$Z^{\text{pos}}_{h}=W_{\text{pos}}^{\top}(L+\epsilon I)+b_{\text{pos}}$$
$$Z^{\text{feat}}_{h}=W_{\text{feat}}^{\top}U+b_{\text{feat}}$$
其中，$I$是一个单位矩阵，$\epsilon$是位置上下文加权系数。为了避免潜在的奇异值不稳定性，本文使用分解模型(decomposition model)对$Z^{\text{pos}}$进行编码。分解模型使用SVD分解将位置上下文特征分解为几个互相正交的基向量，然后将这些基向量作为位置嵌入向量。

为了学习位置特征之间的依赖关系，CroGAT采用了可训练的可塑性因子(trainable scaler factor)来调整各个位置的依赖程度。具体来说，对于第$m$-th位置$l_{m}$, 如果它与其他位置$l_{n}$有较大的欧氏距离，那么将$\kappa_{mn}=\frac{|l_{m}-l_{n}|}{\sqrt{(1-\lambda_{lm})d_{\text{max}}^2+\lambda_{lm}(|l_{m}|^2+|l_{n}|^2)}}$；否则，$\kappa_{mn}=0$。其中，$\lambda_{lm}$和$d_{\text{max}}$是超参数。这样一来，根据位置之间的相似度，可训练的可塑性因子会给不同的位置赋予不同的权重，从而实现更好的学习效果。

用户特征的重建可以用来帮助定位不同用户之间的行为模式的区别。具体来说，给定一组位置特征$l_{j}, j=1,...,M$和用户特征$u_{i}$，其中$M$是位置数量，$N$是用户数量，假设用户$u_{i}$参与了$K$次活动，则用户$u_{i}$的嵌入表示$\overline{u}_{i}=\sum_{k=1}^{K}\beta_{ik}u_{k}$，其中$\beta_{ik}$是参与者权重。CroGAT将位置特征$l_{j}$和用户特征$u_{i}$映射为高维空间，然后利用多头的图注意力机制来学习区域间的用户特征依赖关系。具体来说，假设位置特征矩阵$L \in R^{M\times N\times d}$和用户特征矩阵$U \in R^{N\times f}$，其中$d$是嵌入维度，$f$是用户特征维度。那么，第$h$-th头的位置特征嵌入$Z_{h}^{\text{pos}} \in R^{M\times d}$和用户特征嵌入$Z_{h}^{\text{feat}} \in R^{M\times df}$可以通过以下公式进行计算：
$$Z^{\text{pos}}_{h}=W_{\text{pos}}^{\top}L+b_{\text{pos}}$$
$$Z^{\text{feat}}_{h}=W_{\text{feat}}^{\top}(U+\epsilon I)+b_{\text{feat}}$$
其中，$I$是一个单位矩阵，$\epsilon$是用户上下文加权系数。

为了训练可训练的可塑性因子，CroGAT定义了如下损失函数：
$$L_{\text{reg}}=\lambda||\gamma_{ln}-\frac{|\kappa_{mn}|}{{\sum_{p=1}^{M}e^{\kappa_{pn}}}}||^2_2+\mu||\beta_{kn}-\frac{1}{|S_{ik}|}_+||^2_2$$
其中，$\gamma_{ln}$是第$l$-th位置$l_{l}$和第$n$-th位置$l_{n}$之间的关系的可训练的可塑性因子，$S_{ik}$是用户$i$的第$k$-th个参与活动，$\lambda,\mu$是超参数。$\gamma_{ln}$由下式更新：
$$\Delta\gamma_{ln}=-\eta\frac{(\kappa_{ln}-\alpha_{ln})(\gamma_{ln}-\beta_{ln})}{\sum_{q=1}^{Q}w_{qn}(\gamma_{nq}-\beta_{nq})}$$
其中，$\eta$是学习率，$Q$是$l_{l}$周边的位置数目。$\alpha_{ln}$和$\beta_{nl}$分别是训练样本中第$n$-th位置和第$n$-th位置的得分。$w_{qn}$是第$q$-th位置和第$n$-th位置的权重。

#### Heterogeneous Sequence Attention Mechanism
异构序列注意力机制利用不同模态之间的时间相关性，即不同位置和用户之间的轨迹。具体来说，该机制允许位置序列特征在时间上的依赖性(temporal dependency)和用户特征的依赖性(spatial dependency)进行交互。首先，CroGAT将位置编码模块生成的位置序列特征和用户特征编码模块生成的用户特征序列拼接，并用具有不同权重的序列注意力机制(sequence attention mechanism)进行融合，以学习位置相关的用户特征的预测。

位置序列特征的预测可以用来估计不同用户在不同时间段之间的轨迹变化，从而促进用户特征的预测。具体来说，给定一组用户特征$u_{i}, i=1,...,N$和位置序列特征$X_{ij}:=(x_{ijk}), i=1,...,N, j=1,...,T$,其中$N$是用户数量，$T$是时间长度。CroGAT将位置序列特征$X_{ij}$和用户特征$u_{i}$映射为高维空间，然后利用序列注意力机制来学习序列间的动态依赖关系。具体来说，假设位置序列特征矩阵$X \in R^{N\times T\times m\times d}$和用户特征矩阵$U \in R^{N\times f}$，其中$m$是嵌入维度，$d$是位置嵌入维度，$f$是用户特征维度。那么，第$h$-th头的位置序列特征嵌入$Z_{h}^{\text{seq}} \in R^{N\times T\times md}$和用户特征嵌入$Z_{h}^{\text{feat}} \in R^{N\times tf}$可以通过以下公式进行计算：
$$Z^{\text{seq}}_{h}=W_{\text{seq}}^{\top}(X+\epsilon I)+b_{\text{seq}}$$
$$Z^{\text{feat}}_{h}=W_{\text{feat}}^{\top}U+b_{\text{feat}}$$
其中，$I$是一个单位矩阵，$\epsilon$是位置上下文加权系数。

为了训练位置序列特征的预测，CroGAT采用了预测误差的最小化方法。具体来说，CroGAT训练了两种类型的预测误差：1）位置误差(Position Error): 衡量预测位置跟实际位置之间的距离；2）目标函数(Objective Function): 直接优化了预测轨迹的准确度。位置误差计算如下：
$$E_{\text{pos}}=\frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T}\sum_{k=1}^{K}||\hat{P}_{itk}^\text{pos}-\hat{P}_{itk}^\text{true}||_2^2$$
其中，$\hat{P}_{itk}^\text{pos}$和$\hat{P}_{itk}^\text{true}$分别是第$i$-th用户第$t$-th时间段第$k$-th位置的预测位置和实际位置。目标函数计算如下：
$$J=\underset{\theta}{\min} E_{\text{pos}}+\beta J_{\text{seq}}+\gamma J_{\text{feat}}$$
其中，$\theta$包括位置嵌入矩阵$W_{\text{pos}}$和特征嵌入矩阵$W_{\text{feat}}$。预测位置的预测误差包括两部分：1）位置轨迹预测误差(Trajectory Predictive Loss): 使用基于平滑L1损失函数的预测轨迹位置和真实轨迹位置之间的差距；2）位置聚类损失(Position Clustering Loss): 要求预测位置按照其出现顺序排列，防止位置误差过大而影响预测结果的有效性。位置聚类损失计算如下：
$$E_{\text{clust}}=\frac{1}{NT}\sum_{i=1}^{N}\sum_{t=1}^{T-1}\sum_{j=t+1}^{T}\delta_{ijk}$$
其中，$\delta_{ijk}=1$代表预测位置$j$在真实轨迹上出现于$i$第$t$段之后，否则为零。位置聚类损失可以看作是一种遮蔽(masking)策略，使得模型只能关注位置在实际路径中的作用。

#### Fusion Layer
在CroGAT中，位置特征的嵌入向量和用户特征的嵌入向量分别由不同的头部学习，然后通过Fusion Layer进行融合。具体来说，CroGAT将所有头部的输出进行拼接，然后用线性变换将他们映射为高维空间，并最后通过一个非线性激活函数进行归一化。


图4: 融合层示意图

### Inductive Biases of the Model
为了在多模态数据中发现模式和异质性，CroGAT引入了几种有利于学习的机制。具体来说，CroGAT的第一大优点是能够捕捉异质性的全局信息。具体来说，假设输入数据为位置序列$X=\{x_{ij}: i=1,...,N, j=1,...,T\}$和用户特征$U=\{u_{i}: i=1,...,N\}$，其中$N$是用户数量，$T$是时间长度。如果输入数据存在多种模态，那么就需要注意到它们之间可能存在高度重叠，因为这种情况下，模型可能难以从多个模态中学习有效的共同特征。例如，当输入数据既包括静态图像和语义信息，又包括静态位置和用户特征信息时，模型可能会学习到“图片+语义”和“位置+用户特征”的信息之间的联系，而不是学习到它们之间的关系。因此，CroGAT采用了很多头部来捕捉异质性的全局信息。

第二个优点是CroGAT采用了不同的图注意力机制来捕捉不同模态之间的相似性。例如，位置特征和用户特征可以共享许多相同的图注意力机制，从而提高多模态学习的效率。第三个优点是CroGAT在序列注意力机制的基础上引入了位置聚类损失，使得模型能够学习到不同用户之间的轨迹之间的相似性。最后，为了鼓励模型更好地从不同模态中学习信息，CroGAT采用了多种方法来限制其对所需信息的依赖。

### Complexity Reduction via Position Decoding Module
CroGAT考虑到了不同位置之间的关系，但这些关系不是独立的，也不是非对称的。例如，一个位置的邻居可能位于另一个位置的内部。为了减少复杂性，CroGAT采用了位置解码模块(position decoding module)，该模块将位置嵌入向量转换回其空间坐标形式。位置解码模块接收位置编码模块生成的位置特征矩阵$L \in R^{N\times M\times d}$和位置嵌入矩阵$A \in R^{M\times D}$，其中$N$是用户数量，$M$是位置数量，$d$是嵌入维度，$D$是空间维度。然后，该模块尝试对每个位置嵌入向量进行解码，以找到其对应的空间坐标。具体来说，它将每个位置嵌入向量$a_{j}$投影到空间坐标$z_{j}$，其中$z_{j}=(a_{j}^{\top}A)$。为了学习嵌入矩阵$A$，位置解码模块采用了梯度下降(gradient descent)算法，以最小化欧式距离$||Ax_j - z_j||_2^2$。

# 4.实验
## 数据集说明
作者使用了一个真实的交通移动预测数据集。数据集包含来自两个城市的408个用户的19天数据，每天产生的数据量为72000条轨迹记录。其中有25000条轨迹记录来自第一个城市，剩下的23380条轨迹记录来自第二个城市。数据集包含两类轨迹信息：静态位置信息和动态位置信息。静态位置信息只包含用户所在的位置，动态位置信息则包括用户的位置及其在某一时刻的移动速度和角速度。

## 实验设置
实验设置如下：

- 数据集划分方式：数据集被划分为训练集（80%）和测试集（20%）。
- 输入特征：静态位置信息和动态位置信息。
- 输出标签：用户的行驶轨迹。
- 网络结构：CroGAT模型结构。
- 损失函数：位置误差（最小平方损失）和位置聚类损失（交叉熵）的加权求和。
- 训练策略：使用Adam优化器，初始学习率为0.001，并且在每隔50个epoch后减小一次。
- 测试指标：均方根误差（RMSE），平均绝对误差（MAE）。

## 实验结果
### 静态位置信息
对静态位置信息进行预测任务。

#### 模型效果图


图5: 静态位置信息的预测效果

#### 最佳模型及超参数选择

在不同学习率、嵌入维度、学习速率、可训练可塑性因子、正则化项权重等参数配置下，均能获得较好的预测效果。最终，在测试集上的预测性能达到3.99（RMSE）和0.51（MAE）。而在同样的训练条件下，GRU+LSTM+Attention模型的最佳预测效果只有2.55（RMSE）和0.41（MAE）。本文的方法预测效果明显优于最佳模型。

### 动态位置信息
对动态位置信息进行预测任务。

#### 模型效果图


图6: 动态位置信息的预测效果

#### 最佳模型及超参数选择

本文方法对动态位置信息的预测效果要优于最佳模型。在不同学习率、嵌入维度、学习速率、可训练可塑性因子、正则化项权重等参数配置下，均能获得较好的预测效果。最终，在测试集上的预测性能达到5.50（RMSE）和0.77（MAE）。而在同样的训练条件下，GRU+LSTM+Attention模型的最佳预测效果只有3.55（RMSE）和0.60（MAE）。虽然动态位置信息更丰富，但由于其中的噪声和不确定性，预测效果有待提升。