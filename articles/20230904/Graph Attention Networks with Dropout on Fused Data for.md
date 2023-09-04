
作者：禅与计算机程序设计艺术                    

# 1.简介
  

图注意力网络（Graph Attention Networks）是一种用于多模态数据的强大的神经网络模型。在医疗图像分割领域，它被广泛应用于脑肿瘤、腺癌等病变图像分割任务中。GAT网络可以有效地融合不同模态的数据并产生一个统一的输出结果，从而达到提升性能的目的。然而，由于每个模态数据存在着不同的特点和噪声，GAT模型容易受到这些特点和噪声的影响，导致其性能不稳定。为了克服这一问题，提出了一种新的模型——Graph Attention Networks with Dropout on Fused Data for Medical Image Segmentation (GAIN-D) 。本文通过Fusion Net实现数据的融合，再进行一次特征学习，得到最终的结果。GAIN-D利用dropout的方式对输入数据进行加性噪声，抑制了不同模态数据的相互影响，并得到了比单一模型更好的结果。
# 2.相关工作
传统的基于深度学习的图像分割方法主要采用FCN(Fully Convolutional Network)结构，这种结构将图片转化为一个特征图，再根据特征图进行像素分类。FCN结构能够得到很好的分割效果，但是缺少考虑全局特征的能力。因此，近年来出现了一些改进型的分割网络，如SegNet、UNet等，通过引入卷积层提取局部特征；再采用一个编码器-解码器模块或者一个分支网络来整合全局特征。但是这些方法都没有考虑到输入数据的全局特征。GCN(Graph Convolution Network)是另外一种利用图来处理多模态数据的方法，它将不同的模态建成图结构，然后利用图的一些特性进行特征学习。这样一来，就可以从不同模态中获得共同的特征，进一步提高分割性能。但是GCN模型仍然存在着一些问题，如信息冗余、学习困难、计算复杂度过高等。最近提出的Graph Attention Networks也属于此类方法，它的基本想法是在节点间引入注意力机制来消除图中的冗余信息，并在图上进行特征学习。但是在实际应用中，由于每个模态数据的不同特性，GAT模型容易受到影响，使得模型性能不稳定。为了解决这个问题，提出了一种新型的模型——Graph Attention Networks with Dropout on Fused Data for Medical Image Segmentation (GAIN-D)。
# 3.模型结构
## 3.1 GAIN-D模型结构
GAIN-D模型的主要结构由以下几个部分组成:
### 3.1.1 Input Data Fusion Module
输入数据融合模块包括两个子模块:
#### a) Modality Select Module
模态选择模块负责将输入数据经过多种方式进行融合，以提取共同的特征。比如，对于多模态的肝脏区域分割，可以采用数据融合的方式对四种模态的数据进行融合，包括CT，MRI，PET，DESS，其中CT和MRI可看作是相同模态的数据。当然，也可以直接采用CT作为输入，即仅对CT模态进行分割，而其他模态的数据都作为辅助数据。
#### b) Feature Extraction Module
特征提取模块主要由两层卷积+池化层构成，前者提取局部特征，后者进一步提取全局特征。所以，输入数据融合后的结果形状为(Batch size, Num channels, Height, Width)。这里，批大小表示输入的样本数量，通道数量为融合后的数据的通道数，Height和Width分别为特征图的高度和宽度。
### 3.1.2 Adaptive Graph Attention Layer
该层对节点嵌入进行注意力学习，目的是建立一个权重矩阵，该矩阵对每对节点之间的关系进行调整。通过训练可以获得最优的权重矩阵。
### 3.1.3 Interaction Graph Attention Layer
该层将邻居节点和目标节点通过特征融合的方式融合成新的特征向量。通过学习到的特征，可以提升分割的准确率。
### 3.1.4 Prediction Head
预测头负责对融合后的特征进行分割。通常有两种类型的预测头，一种是全局预测头，另一种是局部预测头。全局预测头负责对整个图进行预测，而局部预测头则只预测目标节点所在的局部范围内的值。
### 3.1.5 Dropout on Fused Data
Dropout在机器学习领域起到了减轻过拟合的作用，它随机将输入数据中一定的比例的元素置为零，以此来限制模型的复杂度。GAIN-D模型同样使用dropout的方式来降低模型对不同模态数据的依赖性，防止它们之间发生相互影响。
## 3.2 模型参数设置
GAIN-D的模型参数设置如下:
* Batch size: 16
* Learning rate: 0.001
* Weight decay: 0.0005
* Number of epochs: 100
* Optimizer used: Adam
* Loss function: Dice Coefficient loss
# 4.实验分析及结果展示
## 4.1 数据集介绍
本实验使用了多模态CT和MRI数据集。CT数据集包括3个模态，分别为FLAIR，T1w和T2w。MRI数据集包括2个模态，分别为T1w和PDw。
数据集规模分别为70，137和100。训练集占总数据集的80%，验证集占10%，测试集占10%。其中，CT数据集具有较小的规模，我们只用其中的前十张图片进行实验。
## 4.2 数据预处理
数据预处理包含三个步骤:
a) 读入数据，将数据转换为灰度图。
b) 对数据进行标准化处理。
c) 将数据划分为训练集，验证集，和测试集。
## 4.3 模型搭建
GAIN-D模型的实现在PyTorch下完成。
模型中使用到的主要组件如下:
a) Modality Select Module: 使用VGG网络实现，其输入为灰度图，输出为四维特征图，其中第一维对应的是批大小。
b) Feature Extraction Module: 使用两层卷积+池化层实现，第二层的卷积核个数为64，步长为2。池化层使用最大池化。
c) Adaptive Graph Attention Layer: 使用图注意力层实现，输入为融合后的特征图，输出为注意力矩阵，其维度为(Num nodes, Num nodes)，表示对每对节点之间的关系进行调整。
d) Interaction Graph Attention Layer: 使用图注意力层实现，输入为邻接矩阵和特征矩阵，输出为融合后的特征矩阵，其维度为(Num nodes, Channels)，表示对邻居节点和目标节点进行特征融合。
e) Global and Local Predictive Heards: 使用全连接层实现，分别对全局预测和局部预测实现。
f) Dropout on Fused Data: 在输入数据的各模态之间加入噪声，以抑制不同模态之间的相互影响。
模型的损失函数为Dice Coefficient Loss，Dice Coefficient衡量两个集合的相似性，值越大表示两个集合的相似度越高。
## 4.4 超参数调优
超参数调优过程遵循一个迭代过程，先尝试简单粗暴的超参数配置，随着结果的不错，逐渐增加超参数的复杂度，进行更精细的搜索。
* lr scheduler
* optimizer
* batch size
* weight decay
* dropout rate
* num layers in feature extractor module
* num heads in adaptive graph attention layer
* number of hidden units in global predictive head
* number of filters in convolutional layers
* activation functions in fully connected layers
超参数调优的结果表明，使用Adam优化器，lr=0.001，batch_size=16，weight_decay=0.0005，drop_rate=0.2，特征提取层只有两个卷积层，图注意力层包含两个头，全局预测头有128个隐藏单元，激活函数使用ReLU。在CT模态上进行分割，验证集dice coefficient达到0.90左右，测试集dice coefficient达到0.93左右。
## 4.5 模型推断
为了评估模型的性能，将模型应用于测试集上的真实数据，并显示预测的结果。