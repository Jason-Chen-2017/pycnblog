
作者：禅与计算机程序设计艺术                    

# 1.简介
         

         GAN（Generative Adversarial Network）是近几年火热的无监督学习方法，它可以将无标签的数据转换成合乎真实分布的假样本。一般来说，GAN模型由两个网络组成：生成器G和判别器D。G是用于生成数据的神经网络，它的任务是通过生成模型能够生成尽可能逼真的图片，即所谓的“生成”；而D是一个二分类器，它的任务是区分真实数据和伪造数据之间的差异。D和G通过博弈的方式进行训练，使得D越来越成为一个好的判断者并欺骗G，最终使得生成的假样本能够被D认为是真实的样本。
         
         在这篇教程中，我会教您如何利用PyTorch构建一个GAN模型，以及如何使用MNIST手写数字数据集训练这个GAN模型，并可视化生成的样本。除此之外，我还会提供一些关于GAN模型的优缺点、应用领域、代码实现的细节等相关知识。
         
         本文为系列教程第一篇，主要内容包括以下几个方面：
             - GAN概述及基本概念介绍
             - GAN的训练过程详解
             - 用PyTorch实现MNIST手写数字数据集的GAN模型
             - 可视化生成的MNIST手写数字样本
         
         希望能帮助到各位读者，欢迎大家在评论区与作者交流。如果喜欢或者感觉文章对您有所启发，记得点赞或转发哦！
         
         # 2.基本概念术语说明
         
         1. 对抗训练

         在GAN的训练过程中，存在两个神经网络，即生成器和判别器。它们之间互相竞争，并且都尝试模仿真实数据生成假数据。这就是所谓的对抗训练，即相互博弈达到一种平衡。

         想了解更多对抗训练的内容，请阅读这篇[知乎回答](https://zhuanlan.zhihu.com/p/97098260)。


         2. 原始图像空间与潜在变量空间

         GAN的输入输出都是图像，但是实际上输入输出都是在不同空间中的。
         
         原始图像空间（Input Space）指的是GAN模型接受输入的图像的真实值所在的空间。比如MNIST手写数字数据集，其输入图像是在[0,1]范围内的灰度值矩阵，大小是28x28像素。这个空间叫做“raw image space”。
         
         潜在变量空间（Latent Variable Space）指的是GAN模型生成图像的潜在变量所在的空间。这个空间通常比原始图像空间小很多，因此也称作“latent variable space”。典型的潜在变量空间的大小可能是2-1000维。虽然潜在变量通常看不到，但GAN可以用它作为控制参数，改变图像的风格和结构，从而生成具有多种特征的图像。
         
         我们需要注意的是，潜在变量空间并不一定是连续的，比如可以用离散的one-hot向量表示。
         
接下来，我们将着重介绍生成器G和判别器D的结构。

         3. 生成器G

         生成器G是GAN模型中最重要的一个模块。它接收潜在变量z作为输入，生成原始图像X。G网络的目标是通过优化损失函数生成逼真的图片，以达到欺骗判别器D的目的。
         
         生成器G的结构往往比较复杂，可以由多个卷积层、池化层、全连接层等组成。例如，下图是一个常用的简单结构：
         
           ```python
            nn.Sequential(
                nn.Linear(in_dim, hidden_size),
                nn.LeakyReLU(),
                nn.Linear(hidden_size, out_dim),
                nn.Tanh()
            )
           ```
          
             其中，`in_dim`代表输入的维度，`hidden_size`代表隐藏层的维度，`out_dim`代表输出的维度。这里采用了一个线性层、一个LeakyReLU激活函数和一个线性层，然后用Tanh函数进行输出归一化。
         
         另外，还有一些高级的GAN网络如DCGAN和WGAN-GP等，都融合了其他组件，提升了生成质量和效率。

         4. 判别器D

         判别器D也是GAN模型中关键的一环。它接收原始图像X作为输入，判断其是否是合乎真实分布的数据。D的目标是尽可能准确地判断出数据是真实的还是伪造的。
         
         D网络的结构与生成器G类似，可以由多个卷积层、池化层、全连接层等组成。例如，下图是一个常用的简单结构：
          
           ```python
            nn.Sequential(
                nn.Conv2d(input_channel, 64, kernel_size=4, stride=2, padding=1),
                nn.LeakyReLU(),
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(),
                nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0),
                Flatten(),
                nn.Linear(1*7*7, 1)
            )
           ```
          
        上面的结构由三个卷积层和三个全连接层组成。第一个卷积层用于处理RGB三通道的图像，输出通道数设为64；第二个卷积层用于降低空间尺寸，输入通道数设为64，输出通道数设为128，然后接一个BatchNormalization层来减少内部协variate shift；第三个卷积层用于降低通道数，输入通道数设为128，输出通道数设为1，然后接一个Flatten层将张量降至一维，再接一个线性层输出一个概率值。
          
        有时候，为了防止过拟合，可以使用Dropout层、L2正则化、数据增强等技术来加强D的能力。

         5. 损失函数和优化器

         我们需要定义两个网络G和D之间的损失函数，然后使用优化器来更新G和D的参数，使得两个网络能够互相优化。
         
         损失函数一般包含两部分，即判别器D和生成器G的损失。对于判别器D来说，它的损失可以分成两部分，即真实图像和伪造图像的损失，这两部分的权重可以不同，也可以相同。对于生成器G来说，它的损失只包含伪造图像的损失，即希望G生成更逼真的伪造图像。
         
         下面给出一些常见的损失函数：
         
          - Binary Cross Entropy Loss: 

            $$Loss = -(y\log(\hat{y}) + (1-y)\log(1-\hat{y}))$$
            
            此损失函数适用于二分类问题，其中y为标签值，$\hat{y}$为模型预测的概率值。
            
          - Wasserstein距离损失：

            如果两个分布p和q的距离越远，那么Wasserstein距离越大。Wasserstein距离用$\mathcal{W}(p, q)$来表示。在GAN的训练过程中，我们希望让判别器D尽可能准确地区分出真实的和伪造的图片，于是我们可以让D最大化损失$max E_{x \sim p}[\log D(x)] + E_{z \sim q}[\log (1 - D(G(z)))]$。
            
          - LSGAN损失：
          
            $$\min_\phi \frac{1}{2}||D_{    heta}(x) - y||^2_2+    ext{hinge}(\epsilon-\gamma)||D_{    heta}(G(z))||_1$$
            
            其中，$    heta$为D网络的参数，$(x, y)$为真实图像和标签值，$G(z)$为生成图像，$\epsilon$和$\gamma$为超参数。LSGAN损失刻画了D网络应该尽可能欺骗G网络生成的伪造图像，而把真实图像当作真实的图片分辨出来。
            
          - Hinge loss: 
          
            $$\mathbb{E}_{x \sim P_r}\left[\max(0,1-t\cdot f(x))+\max(0,-f(x)+\epsilon+t\cdot \epsilon)\right]    ag{3}$$
          
            $\epsilon$为margin。此损失函数适用于分类问题。
            
          - Least squares loss:
          
            $$\frac{1}{2} ||y-f(x)||^2_2$$
            
            此损失函数适用于回归问题。

         6. 反向传播算法

         我们需要使用反向传播算法计算梯度，更新网络的参数，以最小化损失函数。由于网络结构复杂，可能涉及到许多可微分的函数，因此需要基于链式法则来求导。
         
         反向传播算法又称为误差反向传播算法，它通过沿着模型计算图的反方向传播误差来更新模型的参数。如果模型输出y和真实值y_true非常接近，则误差就会接近0；如果模型输出y很远离真实值y_true，则误差就会非常大。因此，我们可以通过反向传播算法来更新网络参数，使得误差最小化。
         
         在计算过程中，每个节点都会计算自己的导数。我们首先计算输出节点的导数，然后沿着路径上的所有叶子节点计算其导数，最后将导数相乘即可得到最终的损失函数的导数。
         
         7. 模型评估

         在训练GAN时，我们需要用评价指标来验证模型的效果。常见的评价指标包括：
         
          - Inception Score:
          
            使用Inception V3网络来计算真实图像和生成图像的inception score。inception score越高，代表生成图像的多样性越好。
            
            Inception score的计算方式如下：
             
            $$IS(G)=exp(\frac{1}{K} \sum_{k=1}^K [\log (\frac{1}{M}\sum_{m=1}^M p_{G}(x^{(m)}))]$$
            
            $p_{G}(x^{(m)})$表示属于第m个类别的生成图像的softmax输出，$K$表示分类数，$M$表示生成图像数量。
            
          - Frechet Inception Distance: 
          
            使用Fréchetinceptiondistance来衡量生成图像和真实图像的特征差距。
            
          - KID(Kernel Inception Distance): 
           
            衡量生成图像和真实图像的特征距离，使用SVM对图像的特征向量进行聚类。
            
          - Precision and Recall:  
         
            计算生成图像中真实类别的精度和召回率。
          
         8. 数据集和分布匹配

         在训练GAN之前，我们通常需要准备好数据集和分布匹配的问题。数据集通常是无监督数据，因此不存在标签信息。我们需要使用已有的真实图像数据集，或者自己去收集图像数据。训练GAN模型前，我们需要先检查训练数据集和测试数据集的分布是否匹配。如果不匹配，我们需要通过调整数据集、数据增强的方法来解决。
         
         通过调整数据的分布和数据增强方法，我们可以得到更好的结果。

         9. 参数共享和高斯分布初始化

         有时候，我们可以在判别器D和生成器G之间共享参数，这样的话就能减少模型参数的个数，加快训练速度。另外，我们还可以通过高斯分布来初始化模型参数，来保证模型的稳定性。
         
         具体的做法是，随机初始化判别器的参数，然后将判别器的参数复制到生成器G中。然后对G和D的参数进行初始化，随机初始化为满足高斯分布的随机数，使得模型能够快速收敛。

         10. 超参数调优

         在训练GAN时，除了要训练网络参数外，还需要调整一些超参数，如学习率、Batch Size、迭代次数等。如果不对超参数进行优化，训练出的模型效果可能会变坏。超参数优化方法有网格搜索、贝叶斯优化和遗传算法等。
         
         根据任务需求，我们还可以设置冻结训练阶段，使判别器D不参与训练，只进行固定训练，或采用其他的网络架构。

         11. 小技巧和注意事项

         在训练GAN时，我们需要注意以下几个小技巧和注意事项：
         
         - 使用残差连接：

           ResNet和Inception系列的网络架构都使用了残差连接，可以增加网络容量和提高性能。

           除了在卷积层后添加残差连接外，还可以在全连接层之间添加残差连接，增强模型的非线性性。

         - 批标准化：

           批标准化（Batch Normalization）是一种流行的规范化方法，通过均值为0、标准差为1的中心化和缩放，消除模型内部的协方差变化，防止过拟合。

           Batch Normalization可以应用到生成器G和判别器D的每一层，可以有效防止梯度爆炸和梯度消失。

         - GAN模型的局限性：

           GAN模型的训练过程具有不确定性，训练过程中模型容易陷入局部最优。

           更大的深度、更复杂的网络结构，或者采用更深层次的变换，都会导致训练时间增加，难以收敛到全局最优。所以，在实际应用中，我们仍然需要考虑模型的可解释性和鲁棒性。

         - 正则化和WGAN-GP：

           GAN模型的训练中，也会引入正则化来防止过拟合。典型的正则化方法包括L2正则化、Dropout、BatchNormalization等。

           WGAN-GP是对GAN的改进，通过最小化Wasserstein距离来训练GAN。与标准的GAN不同，WGAN-GP鼓励真实样本投射到高维的潜在空间中，而不是均匀分布。


# 3.GAN的训练过程详解

## 3.1 背景介绍

本节将介绍GAN的原理和训练过程，然后用数学公式和图示展示GAN模型的训练过程。

GAN，即生成式对抗网络，是近几年热门的无监督学习模型。它由两个网络组成：生成器G和判别器D。G的任务是生成尽可能逼真的图片，即所谓的“生成”，D的任务是区分真实数据和伪造数据之间的差异，即所谓的“判别”。两个网络通过博弈的方式进行训练，使得D越来越成为一个好的判断者并欺骗G，最终使得生成的假样本能够被D认为是真实的样本。

## 3.2 GAN的训练过程

### 3.2.1 生成器G的训练

生成器G的训练可以说是GAN模型的核心部分。

G的任务是生成尽可能逼真的图片，所以G的训练与真实图片的数据分布密切相关。

#### 3.2.1.1 G的损失函数

G的损失函数定义为：

$$\min _{G}\max _{D}\mathbb{E}_{\mathbf{x} \sim p_{data}(x)}\left[\log D(\mathbf{x})\right]+\mathbb{E}_{\boldsymbol{z} \sim p_{z}(z)}\left[\log (1-D(G(\boldsymbol{z}))\right]$$

上式右边第一项是判别器D对真实数据样本的能力，右边第二项是判别器D对生成器G生成的假样本的能力。G的目的是使得生成的假样本被认为是真实的样本，所以右边第二项尽量取0，也就是希望D将假样本判别为负，即：

$$\min _{G}\max _{D}\mathbb{E}_{\mathbf{x} \sim p_{data}(x)}\left[\log D(\mathbf{x})\right]-\mathbb{E}_{\boldsymbol{z} \sim p_{z}(z)}\left[\log (1-D(G(\boldsymbol{z}))\right]$$

我们可以看到，第一项是希望判别器D把真实图片D分类正确，也就是说希望D的输出接近1；第二项是希望判别器D把G生成的假图片判别为负，也就是说希望D的输出接近0。因此，G的总损失函数就是希望通过让D更加准确地分类真实图片，使得生成的假样本被认为是负样本，即

$$\min _{G}\max _{D}\mathbb{E}_{\mathbf{x} \sim p_{data}(x)}\left[\log D(\mathbf{x})\right]-\mathbb{E}_{\boldsymbol{z} \sim p_{z}(z)}\left[\log (1-D(G(\boldsymbol{z}))\right]$$

#### 3.2.1.2 G的优化策略

G的优化策略有多种，最简单的一种是直接使用Adam优化器来最小化损失函数：

$$\arg \min _{G} \max _{D}\left[-\log D(\mathbf{x})+\log (1-D(G(\boldsymbol{z}))\right]$$

另一种优化策略是使用条件GAN，即只更新G网络，D网络固定住，同时更新G网络的参数：

$$\arg \min _{G} \max _{D_{fixed}}\left[-\log D(\mathbf{x})+\log (1-D(G(\boldsymbol{z}))\right]$$

当D固定住时，G网络的参数会更快地收敛到一个较优的值，因此这种策略可以加速收敛。

### 3.2.2 判别器D的训练

判别器D的训练可以理解为G的辅助，因为它可以提高G的生成能力。

D的训练与真实图片的数据分布没有直接关系，它只需要尝试尽可能地分辨真实图片和生成的假图片，从而能够区分出真实图片和伪造图片。

#### 3.2.2.1 D的损失函数

D的损失函数定义为：

$$\min _{D}\max _{G}\mathbb{E}_{\mathbf{x} \sim p_{data}(x)}\left[\log D(\mathbf{x})\right]+\mathbb{E}_{\boldsymbol{z} \sim p_{z}(z)}\left[\log (1-D(G(\boldsymbol{z}))\right]$$

上式左边第一项是希望D判别真实图片为正，也就是说希望D的输出接近1；右边第二项是希望D判别生成的假图片为负，也就是说希望D的输出接近0。因此，D的总损失函数就是希望通过让G生成的假图片被认为是负样本，使得真实图片被认为是正样本，即

$$\min _{D}\max _{G}\mathbb{E}_{\mathbf{x} \sim p_{data}(x)}\left[\log (1-D(\mathbf{x}))\right]+\mathbb{E}_{\boldsymbol{z} \sim p_{z}(z)}\left[\log D(G(\boldsymbol{z}))\right]$$

#### 3.2.2.2 D的优化策略

D的优化策略也有两种，最常用的就是Adam优化器：

$$\arg \min _{D}\max _{G}\left[-\log (1-D(\mathbf{x}))-\log D(G(\boldsymbol{z}))\right]$$

当然，还有一些其它优化策略，如使用SGD优化器、RMSprop、动量等，但Adam优化器在图像领域的效果更好。

### 3.2.3 GAN模型的整体训练

GAN模型的训练可以分成两步：

1. 对抗训练

   GAN的训练分为两个网络，G网络和D网络，两个网络通过博弈训练，最终使得生成的假样本被认为是真实的样本。

2. 判别器收敛策略

   当D网络训练完成后，判别器的训练开始，但G网络的训练却停滞不前，这时需要设置判别器的收敛策略，一旦D网络训练稳定，就可以关闭D网络的训练，开启G网络的训练。

## 3.3 数学公式讲解

我们可以用数学语言来描述GAN模型的训练过程。

首先，我们定义真实图片的概率分布为$p_{data}(x)$。

对于判别器$D$来说，它的损失函数可以表示为：

$$\min _{D}\max _{G}\mathbb{E}_{\mathbf{x} \sim p_{data}(x)}\left[\log (1-D(\mathbf{x}))\right]+\mathbb{E}_{\boldsymbol{z} \sim p_{z}(z)}\left[\log D(G(\boldsymbol{z}))\right]$$

其梯度由下式给出：

$$
abla_{    heta_{D}} J(    heta_{D})=\mathbb{E}_{x \sim p_{data}(x)}\left[
abla_{x} D(\mathbf{x})\right]+\mathbb{E}_{z \sim p_{z}(z)}\left[-
abla_{z} D(G(\boldsymbol{z}))\right]=\mathbb{E}_{x \sim p_{data}(x)}\left[(1-D(\mathbf{x}))
abla_{x} \log D(\mathbf{x})\right]+\mathbb{E}_{z \sim p_{z}(z)}\left[(-D(G(\boldsymbol{z}))
abla_{z} \log D(G(\boldsymbol{z}))\right]$$

其期望由下式给出：

$$\mathbb{E}_{x \sim p_{data}(x)}\left[(1-D(\mathbf{x}))
abla_{x} \log D(\mathbf{x})\right]+\mathbb{E}_{z \sim p_{z}(z)}\left[(-D(G(\boldsymbol{z}))
abla_{z} \log D(G(\boldsymbol{z}))\right]=-\frac{1}{N_{real}}\sum_{i}^{N_{real}}(1-D(x^{(i)}))
abla_{x^{(i)}}\log D(x^{(i)})-\frac{1}{N_{fake}}\sum_{j}^{N_{fake}}D(G(z^{(j)}))
abla_{z^{(j)}}\log D(G(z^{(j)}))$$

对于生成器$G$来说，它的损失函数可以表示为：

$$\min _{G}\max _{D}\mathbb{E}_{\boldsymbol{z} \sim p_{z}(z)}\left[\log (1-D(G(\boldsymbol{z}))\right]$$

其梯度由下式给出：

$$
abla_{    heta_{G}} J(    heta_{G})=-\frac{1}{N_{fake}}\sum_{j}^{N_{fake}}D(G(z^{(j)}))
abla_{z^{(j)}}\log D(G(z^{(j)}))$$

其期望由下式给出：

$$-\frac{1}{N_{fake}}\sum_{j}^{N_{fake}}D(G(z^{(j)}))
abla_{z^{(j)}}\log D(G(z^{(j)}))$$

GAN模型的整体损失函数由两个网络的损失函数之和给出：

$$J(    heta_{G},     heta_{D})=\frac{1}{N_{data}}\sum_{i}^{N_{data}} \mathbb{E}_{x^{i} \sim p_{data}(x)}\left[\log D(\mathbf{x}^{i})\right]+\frac{1}{N_{fake}}\sum_{j}^{N_{fake}} \mathbb{E}_{z^{j} \sim p_{noise}(z)}\left[\log (1-D(G(\boldsymbol{z}^{j}))\right]$$

上式右边的第一项是判别器D的损失，第二项是生成器G的损失。

# 4.用PyTorch实现MNIST手写数字数据集的GAN模型

## 4.1 导入依赖库

我们首先导入依赖库，包括numpy、torch、 torchvision、matplotlib和cuda。如果没有gpu设备，可以忽略这一步。

``` python
import numpy as np
import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
%matplotlib inline

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print('Using {} device'.format(device))
```

## 4.2 设置超参数

我们设置一些超参数，包括batch size、learning rate、num_epochs等。

``` python
batch_size = 64
lr = 0.0002
num_epochs = 200
```

## 4.3 创建数据加载器

然后我们创建数据加载器，用于加载MNIST数据集。

``` python
transforms = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST('./mnist', train=True, download=True, transform=transforms)
test_dataset = datasets.MNIST('./mnist', train=False, download=True, transform=transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
```

## 4.4 创建生成器和判别器

我们分别创建生成器G和判别器D，它们是由一个卷积层、一个全连接层和一个sigmoid函数构成的网络结构。

``` python
class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.generator = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(100, 256, kernel_size=4, stride=1, padding=0, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            torch.nn.Sigmoid())

    def forward(self, input):
        return self.generator(input)


class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.discriminator = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2, inplace=True),
            Flatten(),
            torch.nn.Linear(256 * 7 * 7, 1),
            torch.nn.Sigmoid())

    def forward(self, input):
        return self.discriminator(input)
```

## 4.5 训练模型

最后，我们将上面定义的所有模块组合起来，训练我们的GAN模型。

``` python
def train(dataloader, model, optimizer, criterion, epoch):
    for i, data in enumerate(dataloader, 0):
        imgs, labels = data
        bs = len(imgs)

        # Generate fake images
        noise = torch.randn(bs, 100, 1, 1).to(device)
        gen_imgs = model(noise)

        # Train the discriminator with both real and fake images
        real_loss = criterion(model(imgs.float().to(device)), torch.ones((bs, 1)).to(device))
        fake_loss = criterion(gen_imgs, torch.zeros((bs, 1)).to(device))
        disc_loss = real_loss + fake_loss

        optimizer.zero_grad()
        disc_loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print('[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]'
                  % (epoch, num_epochs, i, len(dataloader),
                     disc_loss.item(), gen_loss.item()))


def test(dataloader, model):
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            outputs = model(images.float().to(device))
            predicted = torch.argmax(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

    accuracy = correct / float(total)
    print("Test Accuracy of the model on the 10000 test images: {:.4f}%".format(accuracy * 100))
    
criterion = torch.nn.BCEWithLogitsLoss()
optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

for epoch in range(num_epochs):
    train(train_loader, generator, optimizer_g, criterion, epoch)
    test(test_loader, generator)
```

## 4.6 可视化生成的MNIST手写数字样本

训练完成后，我们可以用生成器G生成一些新的MNIST手写数字样本。

``` python
noise = torch.randn(64, 100, 1, 1).to(device)
generated_images = generator(noise)
img = torchvision.utils.make_grid(generated_images.detach().cpu(), normalize=True)
plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
plt.show()
```

![Generated Images](https://i.imgur.com/eEfnbvp.png)

