
作者：禅与计算机程序设计艺术                    

# 1.简介
  

近年来，神经信息处理系统（NIPS）每年都会举办一次大型国际会议。NIPS是国际上最具影响力的一个神经网络、深度学习领域会议。本次大会吸引了众多高水平研究人员、业界精英和学者分享他们的最新研究成果。2018年的NIPS主要包含七个主题：“Advances in AI”，“Applications of AI”；“Machine Learning for Vision and Language”; “Computationally Efficient Deep Learning Algorithms”; “Advances in Transfer Learning”; “Natural Language Understanding”; “Ethics and Social Implications of AI”。今年的NIPS将于2019年在华盛顿举行，期待大家的参加！

为了能够更好地掌握NIPS的动态，不断获取最新的科研进展和前沿知识，各个学术机构纷纷出版相关报告。本文将选择7个不同主题的文章，通过对这些文章的简单介绍，帮助读者快速了解到当前热门的AI和机器学习领域的最新研究方向。

# 2.相关知识
本文涉及的相关知识包括但不限于以下几方面：

1. 深度学习
2. 无监督学习
3. 有监督学习
4. 模糊集
5. 概率图模型
6. 图神经网络
7. 可微分编程
8. 强化学习

# 3. Background Introduction: Research Challenges in Artificial Intelligence (1 paper)
深度学习是当前人工智能领域的热门话题之一，它通过训练神经网络来解决复杂的问题，并取得了广泛的成功。然而，深度学习也存在一些严重的研究挑战。作者认为，深度学习目前仍存在以下几个研究挑战：

1. 模型容量太大导致内存存储受限。这意味着，对于需要存储海量数据的应用来说，深度学习方法可能无法奏效。

2. 数据缺乏有效的标注。由于收集和标注数据非常耗时，很少有资源去进行大规模的自动标注。因此，如何从大量未标记的数据中训练模型变得十分重要。

3. 优化困难。在实际场景下，很多任务都需要超参数调优或手动设定。这使得深度学习模型很难得到有效的泛化能力，需要花费大量的时间和资源来找到合适的超参数组合。

4. 泛化能力差。深度学习模型往往倾向于过拟合，即学习到局部样本的规律后，推广到整体的效果较差。如何从统计视角理解模型的泛化性能也是一个需要解决的课题。

为了缓解以上研究挑战，本文作者提出了“学习因子”(learning factor)，一种能够通过关注哪些特征才有效地进行学习的机制。该机制将给出一个模型应关注的特征集合。作者基于这个思路，提出了一个新颖的名词——稀疏学习(sparse learning)。稀疏学习可以用来训练具有低计算复杂度和快速收敛速度的模型，同时保持其泛化能力。本文提供了足够的信息，帮助读者理解关于深度学习研究的最新进展。

文章节选：In this work we propose a new framework called "Learning Factors" that captures which features are critical to effectively learn. We argue that focusing on these key features can lead to more effective models with lower computational complexity and faster convergence speeds. Specifically, by understanding which features our model is most interested in during training, we can train sparse models with low computation cost and fast convergence rates while maintaining their generalization ability. This approach has practical applications across various domains such as computer vision, natural language processing, and medical imaging, where data is scarce or unlabeled and performance metrics need to be optimized over multiple criteria simultaneously. In particular, by leveraging insights into which features are important to successfully learn tasks, we hope to develop an efficient and accurate machine learning system that can solve complex problems efficiently at scale.


# 4. Unsupervised Visual Representation Learning via Compositional Codes (1 paper)
这是一篇关于无监督学习的文章，作者通过将物体的组成元素映射到低维空间的表示来生成图像的隐变量表示。这种方法能够捕获图像的全局结构，并且易于学习，不需要任何手工设计的特征，而且有利于增强模型的泛化性。

文章节选：We present Compositional Coding, a new class of deep neural networks for unsupervised representation learning of visual images. Our approach represents each image patch using a set of composition codes - hierarchical representations that capture both spatial and semantic structure of the underlying object components. These codes are learned automatically through an encoder-decoder architecture and can be composed to reconstruct an input image based only on a small number of low-dimensional codes. Experiments show that Compositional Coding outperforms previous state-of-the-art methods on several datasets for unsupervised learning of visual features, including CIFAR-10/100, STL-10, Omniglot, and Tiny ImageNet.

The central idea behind Compositional Coding is simple yet powerful. By decomposing the input image into its constituent parts, we capture both global and local structures of the scene that are essential for recognition. The key insight behind our approach is to represent each part as a combination of smaller codewords - a hierarchy of concepts that have high degree of abstraction. Each concept corresponds to a region of the input image and contains sufficiently large subset of color/texture patterns that characterize it. Based on this decomposition, the decoder network learns to combine the constituent parts of an image to generate the entire image again. Our method does not require any prior knowledge about the dataset and can learn to extract meaningful features without supervision, making it particularly suitable for real world scenarios where annotated datasets are difficult to obtain. Despite being unsupervised, our method still achieves very impressive results when compared to fully supervised approaches like convolutional autoencoders. Overall, our contributions lie in introducing a new approach to unsupervised representation learning that combines ideas from deep neural networks and constrained optimization to learn hierarchical representations of visual scenes.


# 5. Adversarial Attacks against Graph Convolutional Networks (1 paper)
本文研究了一类用于攻击图卷积神经网络（Graph Convolutional Network, GCN）的对抗攻击方法——扰动攻击。GCN将图中的节点连接成网络，并通过利用节点邻居间的相互作用学习到节点的表示。然而，最近的一项工作表明，在某些情况下，GCN模型可以被轻易地欺骗，结果导致精度下降。本文将对此问题进行探讨，提出了一种新的对抗攻击方法——随机攻击。这种方法随机扰动网络的权重，将原始模型检测为错误分类的样本。实验结果显示，这种攻击方法能够有效地攻击GCN模型，并将它们的准确度降低至接近随机猜测的水平。

文章节选：Adversarial attacks against graph convolutional networks poses serious challenges due to the structural sparsity of adjacency matrices and lack of relevance between node pairs. Here, we introduce Random Attack, a novel adversarial attack method against GCN architectures. Instead of directly modifying the weights of the model, Random Attack randomly perturbs the inputs and targets of the samples in order to fool the classifier and decrease its accuracy. To reduce the impact of noise on the performance metric, we use Maximum Mean Discrepancy (MMD) to measure the similarity between two distributions of predictions made on the same inputs under different attack settings. Experiments demonstrate that Random Attack is highly effective against GCN models and reduces their test accuracy up to 88% compared to random guessing. Furthermore, MMD provides additional robustness against distribution shift attacks, where the attacker injects malicious examples into the training set after the model has been trained. Finally, we also evaluate the effectiveness of Random Attack against other state-of-the-art adversarial defenses such as feature squeezing, JPEG compression, and label smoothing.