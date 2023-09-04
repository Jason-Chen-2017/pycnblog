
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Person Re-Identification (ReID) is an important task in computer vision and pattern recognition that aims to match the person identity between two or more images taken from different cameras/viewpoints under diverse situations such as occlusion, lighting changes, viewpoint changes etc., where there are multiple instances of a person in each image. In recent years, deep learning models have demonstrated impressive performance on this challenging problem due to their ability to capture discriminative features from multi-scale context information. Despite its success, conventional global pooling based methods suffer from slow convergence speed during training. To address these challenges, we propose to use local covariancetopology pooling instead of conventional global pooling which has been shown to be faster during training while preserving high-dimensional feature representation. We also design a novel training strategy called Cycle Consistency Learning (CCL), which utilizes cycle consistency regularization loss term to train our model with stable gradients and effective local correlation patterns across different views. Our experiments show significant improvement over state-of-the-art methods on various datasets including Market-1501, CUHK03, and MSMT17.
本篇论文作者提出了一种新的全局协方差池化网络模型(Global Covariance Pooling Network)，并提出了一种训练策略Cycle Consistency Learning(CCL)。CCL利用循环一致性正则化损失项来训练模型，使梯度稳定并且在不同视图上保持有效的局部相关模式。实验结果表明，该方法相比于最先进的方法都有显著提升，包括Market-1501、CUHK03、MSMT17等多个数据集上的效果。因此，这是首个采用CCL策略来加速训练的全局协方差池化网络模型，可以用于极大规模的人脸识别任务。

## 2.问题阐述&方案描述
### 2.1 问题描述
ReID任务目标是从多张图片中匹配出相同身份的人。它具有挑战性，因为不同视角、光照变化、遮挡等环境因素下会出现多实例（multi-instance）情况。最近几年基于深度学习的模型已经取得了令人满意的成果，能够从丰富的上下文信息中提取出显著的特征。然而，传统基于全局池化的方法会遇到慢速收敛的问题，这主要归功于全局池化机制将不同视角的数据综合到了一起形成一个全局表示。为了解决这个问题，作者提出使用局部协方差拓扑池化代替全局池化，其在训练时速度更快并且保留了高维特征的优点。此外，作者设计了一个名为Cycle Consistency Learning (CCL)的训练策略，利用循环一致性正则化损失项来训练模型，使梯度稳定并且在不同视图上保持有效的局部相关模式。

### 2.2 方案描述
#### 2.2.1 全局协方差池化网络模型(Global Covariance Pooling Network)
首先，将输入的样本数据通过卷积神经网络(CNN)得到特征图，然后对特征图进行全局池化，得到一个固定大小的特征向量作为最终的嵌入表示。然而，这种方法忽略了不同视角的差异，而作者认为是因为它们共享相同的权重。因此，作者提出了一种全局协方差池化网络模型，即每一个视角都由它独特的权重来学习全局表示。具体地说，对于每一个视角，使用局部协方差拓扑池化，通过对邻域内的数据计算其协方差矩阵，再从矩阵中获得分量对应的特征值和特征向量，通过组合这些特征向量来构建视角之间的相似性表示。接着，将不同视角的相似性表示合并后送入全连接层完成分类或回归任务。这样做可以提高特征的泛化能力，同时可以使得模型在处理多视角数据时具有更好的鲁棒性。



#### 2.2.2 CCL训练策略
首先，当训练过程中遇到困难时，可以采取如下方式缓解：

1. 使用较小的学习率；
2. 将权重初始化为较小的值；
3. 添加权重正则化项；
4. 使用更激进的优化器；

但是，作者发现，这些方法往往无法完全解决收敛困难的问题。因此，作者提出了Cycle Consistency Learning (CCL)训练策略，其利用循环一致性正则化项来优化模型参数。它在训练过程中不断反复地最小化两组参数的距离，目的是希望这两个参数具有相同的分布，这样就可以避免梯度消失或爆炸的问题。具体地说，模型训练时每次更新参数时，都会产生两个损失函数，即结构一致性损失函数和正则化损失函数。结构一致性损失函数目的是保证模型生成的特征图能够给每个样本赋予正确的标签；正则化损失函数是用来防止模型过拟合的损失项。循环一致性正则化项则通过引入正则化损失的差距来控制模型参数。




#### 2.2.3 模型总体结构
全局协方差池化网络模型包含三个模块：一个主干网路，两个视角网络，一个联合学习模块，以及最后的分类或回归头。

主干网路由卷积神经网络(CNN)模块构成，主要负责抽取局部和全局特征。

每个视角网络由局部协方差拓扑池化模块和可学习的偏置项构成，从不同视角获取相似性特征。

联合学习模块由多个视角网络的输出特征沿通道维度联合组成，并将它们连接到一个全局变量空间中。

最后的分类或回归头模块输出最终预测结果。






## 3.实验结果及分析
### 3.1 实验设置
#### 数据集
作者采用了多种数据集进行试验，包括Market-1501、CUHK03、MSMT17，共计12,936、7,325和7,635个标注样本。

其中，Market-1501是一个著名的行人重识别数据集，包含410个训练集和126个查询集图像，其中1501个人参与标注。提供了128x64、128x128、256x128三种尺寸的图像。

CUHK03是一个校园环境下的行人重识别数据集，包含403个训练集和119个查询集图像，包含70个学生和10个教师参与标注。提供了128x64、128x128两种尺寸的图像。

MSMT17是一个真实世界中的行人重识别数据集，包含3,544个训练集和500个查询集图像，提供了128x64、256x128两种尺寸的图像。

#### 模型选择
作者比较了四种不同架构的网络，分别是ResNet、VGG、MobileNetV2、GhostNet，各自的优缺点如表1所示。




作者采用了ResNet作为主干网络，共含有三个阶段的网络结构，第一阶段的卷积核数量为64，第二阶段的卷积核数量为128，第三阶段的卷积核数量为256。通过多尺度特征融合模块来融合不同尺度的特征。

#### 参数设置
作者对所有网络的参数进行统一管理，包括初始学习率、迭代次数、优化器、学习率衰减策略、权重衰减系数等。

#### 梯度裁剪
作者在训练时使用梯度裁剪，其作用是限制参数的绝对值的最大值，防止梯度爆炸或消失。

#### 正则化策略
作者在主干网络中使用L2权重衰减，对参数进行约束，以免发生过拟合现象。

#### 超参数搜索
作者利用GridSearchCV法对网络的超参数进行搜索。

### 3.2 实验结果
作者针对不同的数据集和模型，分别进行了实验。在所有数据集上的准确率、训练时间以及模型大小等指标如表2所示。




图2展示了不同数据集上的ReID准确率以及训练时间。作者观察到，在市场行人识别数据集上，GhostNet模型的性能最好，其次是ResNet模型。在CUHK03、MSMT17数据集上，除ResNet模型外，其他模型都没有超过GhostNet的性能。






图3展示了不同模型的大小。作者观察到，GhostNet模型的大小最小，接近于其他模型的一半。然而，当Embedding Size等于2048时，VGG、ResNet和MobileNetV2的模型的大小都很大。

### 3.3 分析
作者对比了CCL策略和无CCL策略的两种网络结构在几个数据集上的性能。作者发现，在三个数据集上CCL策略的网络训练速度要快很多，而且准确率也要高一些。

## 4.讨论
作者认为，本篇论文成功地提出了一种新的全局协方差池化网络模型，并提出了一种训练策略Cycle Consistency Learning，用于加速训练过程，并且在三个数据集上取得了优秀的效果。另外，作者还对四种不同网络结构进行了评估，并找出了ResNet网络结构的最佳选择。作者的研究成果还有待进一步的验证和验证。