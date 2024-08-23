                 

# 生成式AI艺术：VQGAN与Stable Diffusion解析

> 关键词：生成式AI, VQGAN, Stable Diffusion, 变分自编码器(VAE), 判别网络(Discriminator), 生成网络(Generator), 对抗训练(Adversarial Training), 多尺度混合扩散(Multiplicative Casual Sampling), 扩散过程(Diffusion Process)

## 1. 背景介绍

### 1.1 问题由来
随着深度学习技术的发展，生成式AI艺术逐渐成为了一个热门的研究领域。传统的基于规则的图形设计方法已经难以满足日益增长的艺术创作需求，生成式AI模型凭借其自动生成高质量图像的能力，为艺术创作提供了新的可能性。在这一领域中，VQGAN和Stable Diffusion作为目前最为先进的生成式AI模型，已经被广泛应用于艺术创作、图像生成等领域。

然而，尽管VQGAN和Stable Diffusion在生成艺术方面表现优异，其原理和架构相对复杂，对于初学者来说仍存在一定的门槛。因此，本文将对VQGAN和Stable Diffusion的原理、架构、以及实践技巧进行详细的解析，旨在帮助读者全面理解生成式AI艺术的魅力。

## 2. 核心概念与联系

### 2.1 核心概念概述

生成式AI艺术的核心在于生成模型能够从随机噪声中产生高质量的图像。常见的生成模型包括变分自编码器(Variational Autoencoder, VAE)、生成对抗网络(Generative Adversarial Networks, GAN)等。其中，VQGAN和Stable Diffusion是基于GAN的改进模型，具有更强的生成能力和更高效的数据处理能力。

### 2.2 核心概念原理和架构的 Mermaid 流程图

```mermaid
graph LR
    VQGAN --> Stable Diffusion
    VQGAN --> 向量量化(Quantization)
    VQGAN --> 生成网络(Generator)
    VQGAN --> 判别网络(Discriminator)
    Stable Diffusion --> 扩散过程(Diffusion Process)
    Stable Diffusion --> 多尺度混合扩散(Multiplicative Casual Sampling)
```

这个流程图展示了VQGAN和Stable Diffusion的基本架构和关键组成部分。其中，向量量化、生成网络、判别网络是VQGAN的主要组成部分，而扩散过程和多尺度混合扩散则是Stable Diffusion的关键技术。

## 3. 核心算法原理 & 具体操作步骤

### 3.1 算法原理概述

#### 3.1.1 VQGAN原理

VQGAN（Variational Query GAN）是一种基于变分自编码器的生成对抗网络。其核心思想是将输入的高维向量表示为向量量化后的稀疏向量，并利用生成网络和判别网络进行对抗训练，生成高质量的图像。

VQGAN通过学习一个低维嵌入向量集合$\{Z\}$和一个投影矩阵$W$，将高维输入向量$x$投影到一个低维空间$z$。在低维空间中，$z$被离散化为多个码本向量$k$，每个码本向量$k$对应一个离散向量$k' = Wz$，这些离散向量可以通过投影矩阵$W$和解码器$D$重构为输入向量$x$。生成网络（Generator）负责将离散向量$k'$解码为图像，判别网络（Discriminator）负责判别图像的真实性。

VQGAN通过最大化生成网络的生成能力（即生成器G的损失函数）和最小化判别网络的判别能力（即判别器D的损失函数）来进行对抗训练，从而提高生成图像的质量。

#### 3.1.2 Stable Diffusion原理

Stable Diffusion是一种基于扩散过程的生成模型。其核心思想是通过多尺度混合扩散（Multiplicative Casual Sampling）方法，将噪声逐步引入并混合，生成高质量的图像。

Stable Diffusion通过一个标准正态分布$N(0,1)$作为噪声的起点，逐渐引入噪声并混合。具体来说，假设$z_t$表示$t$时刻的噪声向量，其分布由扩散过程确定。在$t=0$时，$z_0$为标准正态分布，然后通过扩散过程将噪声向量$z_t$逐步引入并混合，最终得到生成图像。

### 3.2 算法步骤详解

#### 3.2.1 VQGAN训练步骤

1. **初始化**：随机初始化生成网络（Generator）和判别网络（Discriminator）的权重，以及低维嵌入向量集合$\{Z\}$。
2. **生成网络训练**：将生成网络的输出与真实图像进行比较，计算生成网络的损失函数。
3. **判别网络训练**：将判别网络的输出与真实标签进行比较，计算判别网络的损失函数。
4. **对抗训练**：生成网络与判别网络进行对抗训练，通过最大化生成网络的生成能力（即生成器G的损失函数）和最小化判别网络的判别能力（即判别器D的损失函数）来进行对抗训练，从而提高生成图像的质量。
5. **向量量化**：对生成网络的输出进行向量量化，生成低维嵌入向量集合$\{Z\}$。
6. **更新权重**：根据对抗训练的结果，更新生成网络和判别网络的权重，以及低维嵌入向量集合$\{Z\}$。

#### 3.2.2 Stable Diffusion训练步骤

1. **初始化**：随机初始化扩散过程的参数，如噪声的分布和扩散时间步数。
2. **噪声引入**：从标准正态分布中采样一个噪声向量$z_t$，并将其引入扩散过程中。
3. **噪声混合**：通过扩散过程将噪声向量$z_t$逐步引入并混合，生成一个中间图像。
4. **解码**：通过解码器将中间图像转换为最终的高质量图像。
5. **扩散过程优化**：根据生成图像的质量，优化扩散过程的参数，如噪声的分布和扩散时间步数。

### 3.3 算法优缺点

#### 3.3.1 VQGAN优缺点

**优点**：

- 能够处理高维输入向量，生成高质量的图像。
- 低维嵌入向量集合$\{Z\}$能够对高维输入向量进行有效的压缩和表示，减少计算成本。
- 通过向量量化和生成网络的联合训练，能够提升生成图像的质量。

**缺点**：

- 向量量化过程和生成网络训练较为复杂，需要较大的计算资源。
- 向量量化过程可能会引入一定的信息损失，影响生成图像的质量。

#### 3.3.2 Stable Diffusion优缺点

**优点**：

- 能够生成高质量、多样化的图像。
- 多尺度混合扩散方法能够提高生成图像的分辨率和质量。
- 扩散过程具有可控性，可以通过调节参数控制图像的生成过程。

**缺点**：

- 训练和生成过程较为复杂，需要较大的计算资源和时间。
- 扩散过程可能会引入一定的噪声，影响生成图像的质量。

### 3.4 算法应用领域

VQGAN和Stable Diffusion已经被广泛应用于生成式AI艺术、图像生成、数据增强等领域。具体应用场景包括：

- 艺术创作：生成高质量的画作、雕塑、设计等艺术作品。
- 图像生成：生成逼真的图像，用于游戏、影视、广告等场景。
- 数据增强：通过生成合成数据，用于数据扩充、图像去噪等任务。

## 4. 数学模型和公式 & 详细讲解 & 举例说明

### 4.1 数学模型构建

#### 4.1.1 VQGAN数学模型

VQGAN的数学模型主要包括生成网络的输出和判别网络的输出。假设生成网络的输出为$G(z)$，判别网络的输出为$D(x)$，则生成网络的损失函数为：

$$
\mathcal{L}_G = -E_{z \sim p(z)}[\log D(G(z))]
$$

判别网络的损失函数为：

$$
\mathcal{L}_D = E_{x \sim p(x)}[\log D(x)] + E_{z \sim p(z)}[\log(1-D(G(z)))
$$

其中，$z$表示低维嵌入向量，$p(z)$表示低维嵌入向量的分布，$p(x)$表示真实图像的分布。

#### 4.1.2 Stable Diffusion数学模型

Stable Diffusion的数学模型主要包括噪声向量的引入和扩散过程的优化。假设噪声向量$z_t$的分布为$\mathcal{N}(0,\sigma_t^2)$，扩散过程的参数为$\alpha_t$，则扩散过程的损失函数为：

$$
\mathcal{L}_t = -\alpha_t \log \sigma_t^2 - \frac{1}{2\sigma_t^2}||z_t - z_{t-1}||^2
$$

其中，$z_t$表示$t$时刻的噪声向量，$z_{t-1}$表示$t-1$时刻的噪声向量。

### 4.2 公式推导过程

#### 4.2.1 VQGAN公式推导

VQGAN的生成网络输出$G(z)$可以通过解码器$D$将低维嵌入向量$k'$解码为高维输入向量$x$：

$$
x = D(k') = Wz
$$

其中，$W$为投影矩阵，$z$为低维嵌入向量。生成网络的损失函数可以推导为：

$$
\mathcal{L}_G = -\frac{1}{N}\sum_{i=1}^N \log D(G(x_i))
$$

其中，$N$表示输入向量$x_i$的数量。

#### 4.2.2 Stable Diffusion公式推导

Stable Diffusion的扩散过程$z_t$可以通过一个标准正态分布$N(0,1)$进行采样，并逐步引入噪声：

$$
z_t = \sqrt{(1-\alpha_t)z_{t-1}}N(\mu,\sigma_t^2) + \alpha_t z_{t-1}
$$

其中，$\alpha_t$为扩散参数，$z_{t-1}$为$t-1$时刻的噪声向量，$N(\mu,\sigma_t^2)$为高斯噪声。

### 4.3 案例分析与讲解

#### 4.3.1 VQGAN案例分析

假设我们有一个输入向量$x$，其分布为$N(0,1)$，通过VQGAN模型进行生成：

1. **初始化**：随机初始化生成网络和判别网络的权重，以及低维嵌入向量集合$\{Z\}$。
2. **生成网络训练**：将生成网络的输出与真实图像进行比较，计算生成网络的损失函数。
3. **判别网络训练**：将判别网络的输出与真实标签进行比较，计算判别网络的损失函数。
4. **对抗训练**：生成网络与判别网络进行对抗训练，通过最大化生成网络的生成能力（即生成器G的损失函数）和最小化判别网络的判别能力（即判别器D的损失函数）来进行对抗训练，从而提高生成图像的质量。
5. **向量量化**：对生成网络的输出进行向量量化，生成低维嵌入向量集合$\{Z\}$。
6. **更新权重**：根据对抗训练的结果，更新生成网络和判别网络的权重，以及低维嵌入向量集合$\{Z\}$。

#### 4.3.2 Stable Diffusion案例分析

假设我们有一个输入向量$z_0$，其分布为$N(0,1)$，通过Stable Diffusion模型进行生成：

1. **初始化**：随机初始化扩散过程的参数，如噪声的分布和扩散时间步数。
2. **噪声引入**：从标准正态分布中采样一个噪声向量$z_t$，并将其引入扩散过程中。
3. **噪声混合**：通过扩散过程将噪声向量$z_t$逐步引入并混合，生成一个中间图像。
4. **解码**：通过解码器将中间图像转换为最终的高质量图像。
5. **扩散过程优化**：根据生成图像的质量，优化扩散过程的参数，如噪声的分布和扩散时间步数。

## 5. 项目实践：代码实例和详细解释说明

### 5.1 开发环境搭建

在进行VQGAN和Stable Diffusion的实践前，我们需要准备好开发环境。以下是使用Python进行PyTorch开发的环境配置流程：

1. 安装Anaconda：从官网下载并安装Anaconda，用于创建独立的Python环境。

2. 创建并激活虚拟环境：
```bash
conda create -n vqgan-env python=3.8 
conda activate vqgan-env
```

3. 安装PyTorch：根据CUDA版本，从官网获取对应的安装命令。例如：
```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
```

4. 安装Transformers库：
```bash
pip install transformers
```

5. 安装各类工具包：
```bash
pip install numpy pandas scikit-learn matplotlib tqdm jupyter notebook ipython
```

完成上述步骤后，即可在`vqgan-env`环境中开始VQGAN的实践。

### 5.2 源代码详细实现

以下是使用PyTorch对VQGAN进行实现的具体代码实现：

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import save_image
from torch.utils.data import DataLoader

class VQGANGenerator(nn.Module):
    def __init__(self, num_classes=10, embed_dim=256):
        super(VQGANGenerator, self).__init__()
        self.embed_dim = embed_dim
        self.encoder = nn.Sequential(
            nn.Linear(num_classes, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )
        self.coder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        z = self.encoder(x)
        z = z.view(z.size(0), -1, self.embed_dim // 8, self.embed_dim // 8)
        z = z.view(z.size(0), self.embed_dim // 8)
        z = self.coder(z)
        return z

class VQGANDiscriminator(nn.Module):
    def __init__(self, num_classes=10, embed_dim=256):
        super(VQGANDiscriminator, self).__init__()
        self.embed_dim = embed_dim
        self.encoder = nn.Sequential(
            nn.Linear(num_classes, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        z = self.encoder(x)
        z = z.view(z.size(0), -1, self.embed_dim // 8, self.embed_dim // 8)
        z = z.view(z.size(0), self.embed_dim // 8)
        z = self.decoder(z)
        return z

# 训练函数
def train_vqgan(generator, discriminator, num_classes, embed_dim, num_epochs, batch_size, device, learning_rate):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    cifar10 = CIFAR10(root='./data', download=True, transform=transform)
    dataloader = DataLoader(cifar10, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(generator.parameters()) + list(discriminator.parameters()), lr=learning_rate)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(dataloader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            z = generator(images)
            pred_labels = discriminator(z)
            discriminator_loss = criterion(pred_labels, labels)
            generator_loss = -criterion(pred_labels, labels)
            loss = discriminator_loss + generator_loss
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"Epoch {epoch+1}, batch {i+1}/{len(dataloader)}, discriminator_loss: {discriminator_loss.item():.4f}, generator_loss: {generator_loss.item():.4f}, loss: {loss.item():.4f}")
    return generator

# 测试函数
def test_vqgan(generator, num_classes, embed_dim, device, num_samples):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    cifar10 = CIFAR10(root='./data', download=True, transform=transform)
    dataloader = DataLoader(cifar10, batch_size=1, shuffle=False)
    images = []
    for images, labels in dataloader:
        images = images.to(device)
        images = generator(images)
        images = images.cpu()
        images = images.view(images.size(0), -1)
        images = images.view(images.size(0), embed_dim // 8, embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8, embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8, embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8)
        images = images.view(images.size(0), embed_dim // 8

