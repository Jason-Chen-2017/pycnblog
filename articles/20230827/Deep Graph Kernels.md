
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 1.1 论文背景
Deep learning has revolutionized the field of artificial intelligence in recent years. Despite its advantages such as high accuracy and automatic feature extraction, deep neural networks (DNN) are often regarded as black boxes because they do not provide a complete understanding of how the learned features interact with each other or with the input data. In this work, we propose to address these limitations by proposing new graph kernels that operate directly on graphs rather than on raw feature vectors. We develop a framework for combining multiple graph kernels into an ensemble-based approach that improves classification performance over individual ones while preserving their ability to capture non-linear interactions between nodes. Our experiments show that our proposed methods achieve significant improvements in terms of accuracy and efficiency compared to state-of-the-art approaches based on traditional DNN models. 
## 1.2 文章主要贡献及创新点
### 1.2.1 深度学习在图数据上的应用
* 全新的模型结构——Deep graph kernel（DGK）
  * 将深度学习方法直接应用到图上来提升性能
  * 提出多种核函数组合的方式来实现集成学习，有效处理非线性关系
* 模型层面的增强——Graph attention network（GAT）
  * GAT是一个图注意力网络，可以融合不同子图的信息并对节点进行特征嵌入，从而提升性能
* 数据集的增强——基于WebKB数据集
  * 使用WebKB数据集作为基准，将现实世界的数据扩充至具有更高复杂度和多样性的网页空间中，验证模型的泛化能力
### 1.2.2 深度学习在图数据的特性
* 稀疏性：图中的节点数量通常是海量的，而节点之间的连接关系却很少出现。因此，需要一种有效的方法来表示节点间的非线性关系。
* 多样性：异构的、不规则的网络数据给深度学习带来了巨大的挑战。除了节点的属性信息外，还包括图结构的拓扑信息等。
* 异质性：图的数据具有高度的异质性。节点和边缘上的分布都可以是不同的，例如，有些节点具有多个标签，另外一些节点没有标签。因此，需要一种有效的机制来考虑节点之间的关联性。
* 时变性：动态网络数据的快速变化也为深度学习带来了难题。在每一次迭代中，所获得的数据都会产生新的局部更新，反映在图结构上就是节点的增加或删除以及边缘上的新增或消除。
* 动态性：除了静态的图结构，还有一些时间序列数据也是图数据的组成部分。传感器网络、交通流量、股市行情、微博评论等都属于动态性的范畴。
### 1.2.3 针对以上特性的深度学习方法设计
本文首次提出了针对图数据特点的深度学习方法设计。该方法可以有效解决现有的深度学习模型面临的问题，如稀疏性、多样性、异质性等。它首先用GCN、GAT等模型对图进行表示学习，然后利用不同类型的核函数来捕获不同类型的相似性。这种方式能够考虑图的非线性关系，并且通过集成多个核函数的方式来达到较好的效果。同时，在模型的训练过程中加入了重要的约束条件，比如预先计算好邻接矩阵或者根据特征相似性计算距离矩阵等。最后，作者提出了一种新的图注意力网络(Graph Attention Network)，用以融合不同子图的信息并提取节点特征。这样做能够使模型学习到更丰富的节点特征信息，提升模型的性能。作者通过实验对比展示了其有效性。最后，作者展示了其在多种基准数据上的表现，证明了其适应性。
# 2.相关工作综述
## 2.1 相关工作简介
### 2.1.1 使用神经网络来学习表示
近年来，深度学习方法在图像分类、对象检测、语言建模等领域取得了重大突破。许多研究人员提出了基于神经网络的特征提取方法，如CNNs、RNNs和GNNs，它们在性能上相当显著。这些方法基于图的结构进行学习，把图中节点的特征转化成高维的向量形式。然而，由于图数据具有节点的丰富的特征，而且图的大小和规模也会随着时间的推移而不断增加，因此，传统的深度学习方法无法应对这些挑战。为了克服这个限制，研究人员开始探索将深度学习应用到图上来解决问题。
### 2.1.2 使用神经网络来分类
传统的深度学习方法一般采用传统的分类器或回归器来实现。但是，由于图数据的离散特性，传统的分类方法往往无法有效地处理图结构中包含的关联性。为了处理这一问题，一些研究人员提出了使用神经网络来分类图结构的数据。其中，最成功的是早期的基于神经网络的图神经网络方法。随后，图卷积神经网络(GCNs)、图注意力网络(GATs)和深度信念网络(DBNs)等模型被提出。这些模型基于图的表示学习和消息传递的机制来处理图数据。虽然它们在某些方面都取得了不错的结果，但它们仍存在很多局限性，如不具备深刻理解图数据的能力、不能捕捉到图数据多样性中的全局特征等。
### 2.1.3 关系抽取与推理
另一个方向是提出能够自动化地从图中抽取出结构化的关系。其中，关系抽取是指从文本、图像、视频等各种形式的输入中提取出预定义的关系。最近，一些研究人员提出了关系抽取方法，如基于词袋的随机游走模型、关系网格模型、逻辑回归模型等。与前两种方法不同的是，这些方法都是基于统计学习方法，它们只能学习到局部和平凡的结构关系。为了能够捕捉到全局和复杂的关系，一些研究人员提出了基于神经网络的关系抽取方法。其中，最近提出的多阶自注意力模型(Multi-head Self-Attention Networks)就是代表之一。该模型能够同时捕捉到不同阶层的关系，并且在训练过程中引入约束来保证模型的稳定性。
## 2.2 本文的贡献
本文从两个方面进一步加强了图神经网络的应用。第一，本文创造性地提出了一种全新的模型结构——Deep graph kernel。它将深度学习技术直接应用到图数据上，开发了一套新的模型来提升图数据的分类性能。第二，本文将深度学习技术应用到图数据的特性上，系统地总结了这些特性。提出了一个名为“Graph Attention Networks”的模型，它利用不同阶层的信息，来融合不同子图的信息，来提升模型的性能。这些特性对于如何将深度学习技术应用到图数据上来说，提供了独特的思路。