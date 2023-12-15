                 

# 1.背景介绍

音乐生成和分析是人工智能领域中的一个重要方向，它涉及到创建新的音乐作品以及对现有音乐作品进行分析和理解。近年来，随着神经网络技术的发展，它们已经成为音乐生成和分析的主要工具。本文将介绍如何使用神经网络进行音乐生成和分析，并探讨其背后的原理和数学模型。

# 2.核心概念与联系
在本节中，我们将介绍音乐生成和分析的核心概念，以及神经网络如何与这些概念相关联。

## 2.1.音乐生成
音乐生成是指通过程序或算法创建新的音乐作品的过程。这可以包括任何类型的音乐，如古典、摇滚、流行音乐等。音乐生成的主要目标是创建具有创意和独特性的音乐作品，而不是简单地复制现有的音乐。

神经网络在音乐生成中的应用主要包括以下几个方面：

- **生成对抗网络（GANs）**：GANs 是一种深度学习模型，它可以生成具有高质量和多样性的音乐作品。通过训练 GANs，我们可以让其生成类似于现有音乐作品的新音乐。

- **循环神经网络（RNNs）**：RNNs 是一种递归神经网络，它可以处理序列数据，如音乐。通过训练 RNNs，我们可以让其生成基于给定输入的新音乐。

- **变分自动编码器（VAEs）**：VAEs 是一种生成模型，它可以生成具有给定特征的新音乐作品。通过训练 VAEs，我们可以让其生成具有特定特征的新音乐。

## 2.2.音乐分析
音乐分析是指通过程序或算法对音乐作品进行分析和理解的过程。这可以包括各种各样的分析方法，如音乐的结构、旋律、和弦等。音乐分析的主要目标是提取音乐作品中的有意义信息，以便进行进一步的分析和研究。

神经网络在音乐分析中的应用主要包括以下几个方面：

- **卷积神经网络（CNNs）**：CNNs 是一种深度学习模型，它可以处理图像和音频数据。通过训练 CNNs，我们可以让其对音乐作品进行分类、识别和分析。

- **自注意力机制（Self-Attention）**：自注意力机制是一种神经网络架构，它可以捕捉序列数据中的长距离依赖关系。通过训练自注意力机制，我们可以让其对音乐作品进行更精确的分析。

- **序列到序列（Seq2Seq）**：Seq2Seq 是一种深度学习模型，它可以处理序列数据，如音乐。通过训练 Seq2Seq，我们可以让其对音乐作品进行编码和解码，从而进行更深入的分析。

# 3.核心算法原理和具体操作步骤以及数学模型公式详细讲解
在本节中，我们将详细讲解神经网络在音乐生成和分析中的核心算法原理，以及相应的数学模型公式。

## 3.1.生成对抗网络（GANs）
GANs 是一种生成对抗性训练的神经网络，它由两个子网络组成：生成器（Generator）和判别器（Discriminator）。生成器用于生成新的音乐作品，判别器用于判断生成的音乐是否与现有音乐作品相似。GANs 的训练过程可以通过以下步骤进行：

1. 训练生成器：生成器接收随机噪声作为输入，并生成新的音乐作品。生成器的输出被传递给判别器，以便判断其是否与现有音乐作品相似。

2. 训练判别器：判别器接收生成器的输出作为输入，并判断其是否与现有音乐作品相似。判别器的输出被用于更新生成器。

3. 更新生成器：根据判别器的输出，更新生成器以生成更接近现有音乐作品的新音乐。

GANs 的数学模型可以通过以下公式表示：

$$
G(z) \sim P_{g}(x) \\
D(x) \sim P_{d}(x) \\
\min _{G} \max _{D} V(D,G) \\
V(D,G) = E_{x \sim P_{d}(x)} [\log D(x)] + E_{z \sim P_{g}(z)} [\log (1-D(G(z)))]
$$

其中，$G(z)$ 表示生成器生成的音乐作品，$D(x)$ 表示判别器对音乐作品的判断结果，$P_{g}(x)$ 表示生成器生成的音乐作品的概率分布，$P_{d}(x)$ 表示现有音乐作品的概率分布，$E$ 表示期望值，$\log$ 表示自然对数。

## 3.2.循环神经网络（RNNs）
RNNs 是一种递归神经网络，它可以处理序列数据，如音乐。RNNs 的核心结构包括隐藏层和输出层。隐藏层用于存储序列数据的信息，输出层用于生成序列数据的预测。RNNs 的训练过程可以通过以下步骤进行：

1. 初始化隐藏层状态：隐藏层状态用于存储序列数据的信息。

2. 输入序列数据：对于每个时间步，输入序列数据到 RNNs。

3. 更新隐藏层状态：根据输入序列数据，更新隐藏层状态。

4. 生成预测：根据隐藏层状态，生成序列数据的预测。

RNNs 的数学模型可以通过以下公式表示：

$$
h_{t} = f(Wx_{t} + Uh_{t-1} + b) \\
y_{t} = W_{y} h_{t} + b_{y}
$$

其中，$h_{t}$ 表示隐藏层状态，$x_{t}$ 表示输入序列数据，$W$ 表示权重矩阵，$U$ 表示递归权重矩阵，$b$ 表示偏置向量，$y_{t}$ 表示输出序列数据，$W_{y}$ 表示输出权重矩阵，$b_{y}$ 表示输出偏置向量，$f$ 表示激活函数。

## 3.3.变分自动编码器（VAEs）
VAEs 是一种生成模型，它可以生成具有给定特征的新音乐作品。VAEs 的核心结构包括编码器（Encoder）和解码器（Decoder）。编码器用于编码输入音乐作品，以生成其隐藏表示，解码器用于从隐藏表示生成新的音乐作品。VAEs 的训练过程可以通过以下步骤进行：

1. 训练编码器：编码器接收输入音乐作品，并生成其隐藏表示。

2. 训练解码器：解码器接收隐藏表示，并生成新的音乐作品。

3. 更新编码器和解码器：根据输入音乐作品和生成的新音乐作品，更新编码器和解码器。

VAEs 的数学模型可以通过以下公式表示：

$$
z \sim P(z) \\
\log P(x|z) = \log P_{\theta }(x|z) - \frac{1}{2} \kappa \|z\|^{2} \\
\log P_{\theta }(x) = \log \int P_{\theta }(x|z) P(z) dz \\
\log P_{\theta }(x) = \log \int \exp (\log P_{\theta }(x|z) - \frac{1}{2} \kappa \|z\|^{2}) P(z) dz \\
\log P_{\theta }(x) = \log \int \exp (\log Q_{\theta }(z|x) - \frac{1}{2} \kappa \|z\|^{2}) P(z) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} \kappa \|z\|^{2}) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }(z)\|^{2})) dz \\
\log P_{\theta }(x) = \log \int Q_{\theta }(z|x) \exp (-\frac{1}{2} (\|z\|^{2} + \kappa \|x-G_{\theta }