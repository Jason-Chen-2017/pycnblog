
作者：禅与计算机程序设计艺术                    

# 1.简介
  


Self-attention networks (SAN) 是一种新的视觉表示学习模型，它利用注意力机制来融合图像特征。 Self-attention 可以看做是 CNN 和 Transformer 的结合体，也是将两者的长处进行了结合，能够学习到图像的全局信息和局部相关信息。 SAN 可用来解决计算机视觉领域中的各种图像任务，包括分类、检测、分割、变化检测等。

传统上，卷积神经网络 (CNNs) 都是由卷积层和池化层组成，它们通过卷积操作来提取局部特征，并通过池化操作对这些特征进行整合。然而，CNN 在捕获全局信息方面存在一些限制。随着深度学习技术的进步，出现了一类新型网络结构——Transformer——它能够在保持全局性质的同时，提升模型的局部感知能力。

但是，在实际应用中，Transformer 模型往往需要很长的训练时间才能收敛到最优效果。另外，虽然 Transformer 模型能够学习到图像全局信息，但它们通常不具备高效率的计算性能，也无法直接用于图像分类任务。为了更好地解决计算机视觉领域中的图像任务，研究人员提出了很多基于 CNN 的图像表示学习模型。但是，它们通常只能学习到低层次的图像特征，缺乏足够的全局信息。因此，如何结合 CNN 和 Transformer 以获得更好的图像表示学习方案成为研究热点。

Self-attention network (SAN) 是一种新的视觉表示学习模型，它利用注意力机制来融合图像特征。 Self-attention 可看做是 CNN 和 Transformer 的结合体，也是将两者的长处进行了结合，能够学习到图像的全局信息和局部相关信息。 SAN 可用来解决计算机视觉领域中的各种图像任务，包括分类、检测、分割、变化检测等。

与 CNN 和 Transformer 相比， SAN 具有以下优势：

1. 能够充分利用全局信息：SAN 可将全局上下文信息从输入图像中捕获出来，因此可以有效地处理复杂场景下的图像。 

2. 更强大的局部感知能力：SAN 使用注意力机制来学习到全局和局部之间的映射关系，使得模型能够更好地捕获不同位置的信息。 

3. 自适应的特征抽取：SAN 不仅能够学习到低层次的特征，还可以通过自适应调整特征来获取更多的全局信息。 

4. 更快的推理速度：由于 SAN 采用局部连接的方式，因此其推理速度比 CNN 快得多。 

总之，Self-attention network (SAN) 提供了一个全新的视觉表示学习模型，其主要特点是能够利用注意力机制来学习到全局和局部之间的映射关系，并且拥有自适应调整特征的能力，有效提升模型的推理速度。



# 2.概念与术语
## 2.1 Attention Mechanism and self-attention mechanism
Attention mechanism is a powerful technique in natural language processing and machine learning that enables a neural network to focus on relevant parts of the input data while ignoring irrelevant information. It works by assigning weights or scores to each element of an input sequence, based on its relevance to other elements in the same sequence. Self-attention mechanism is a type of attention mechanism where the model learns to pay more attention to itself rather than being fixed to certain positions in the input sequence. In contrast to traditional attention mechanisms such as additive attention which involves concatenating multiple weighted representations from different positions of the input sequence, self-attention explores interdependencies between different positions using the same set of parameters across all positions. The key idea behind self-attention is to use multi-head attention, which allows the model to jointly attend to information from different representation subspaces at different positions within the input sequence.
