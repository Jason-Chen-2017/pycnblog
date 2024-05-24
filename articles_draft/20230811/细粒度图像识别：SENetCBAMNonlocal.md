
作者：禅与计算机程序设计艺术                    

# 1.简介
         

在计算机视觉领域，以往的方法主要集中在基于全连接神经网络（FCN）、深度残差网络（Deep Residual Net，DRN）等特征学习方法上进行全局图像理解。但是随着深度学习技术的发展，越来越多的人们意识到，全局图像理解往往不能很好的应对图像数据的复杂性。因此，就需要考虑更加细化的图像理解。在本文中，我们将会介绍三种新的特征学习方法——Squeeze-and-Excitation Networks (SENet)、Convolutional Block Attention Module(CBAM)和Non-Local Neural Networks(NLNet)。通过这几种新型的方法，可以提高模型的性能同时减少计算量，有效地实现细粒度的图像理解。


# 2.基本概念和术语
## Squeeze-and-Excitation Networks(SENet)
SENet是在2017年的 CVPR 上被提出的一种特征学习方法，其目标是为卷积层添加注意力机制，能够提供全局信息并改善特征学习能力。其基本想法就是通过学习到一个通道注意力模块（Channel Attention Module，CAM），该模块能够在不丢失全局信息的前提下，提升重要特征的响应强度。而在该注意力模块中，使用了两个全连接层来生成 attention map。其中一个全连接层用于生成通道的注意力权重，另一个全连接层用于生成通道之间的相互依赖关系，从而提升不同通道之间的相似性。最后，在全局平均池化之前加入注意力映射层，用以融合全局信息。如下图所示。



在SENet的基础上衍生出了其它的变体，如：SE-ResNet、SE-Inception、SEResNeXt 和 SENets4。

## Convolutional Block Attention Module(CBAM)
CBAM是在2018年的 ECCV 上被提出的一种特征学习方法，其目标是对卷积层进行注意力分配，能够提供全局信息并改善特征学习能力。与 SENet 不同的是，CBAM 采用模块化设计，即将注意力分配模块作为一组可学习的注意力机制，这些注意力机制能够自适应地调整特征图中的全局分布，并且在保持全局信息的情况下减少计算量。与 SENet 类似，CBAM 使用了两个全连接层来生成注意力权重，但 CBAM 的注意力分配模块由三个子模块组成：通道注意力模块（Channel Attention module，CAM）、空间注意力模块（Spatial Attention module，SAM）和注意力整合模块（Attention Integration module，AIM）。如下图所示。


与 SE-Net 一样，CBAM 也有自己的变体，如：CBAM-ResNet、CBAM-Inception、CBAMs4、CBAM-DenseNet 。

## Non-Local Neural Networks(NLNet)
NLNet 是一种特征学习方法，其目标是学习全局上下文信息和局部空间位置关系，并根据这些信息对输入数据进行相应的变换，从而提升模型的全局性和局部性的特性。NLNet 在 NL 指代非局部操作（Non-Local）的基础上提出，它利用空间上的非局部性来捕获全局上下文信息。具体来说，NLNet 通过对输入数据计算空间上的注意力权重，然后将这些注意力权重作用在对应的特征图上，使得模型能够更好地关注全局信息和局部空间关系。


除此之外，还有很多其他类型的特征学习方法，如：PCB（Pyramid Channel-based Blocks）、GCNN（Global Context Network）、PAGM（Position Adaptive Graphic Model）、RNN（Recurrent Neural Networks）等。这些方法都能够帮助模型更好地识别图像数据，并提高模型的泛化能力。

# 3.核心算法原理和操作步骤
## SENet
SENet的基本思路是生成一个通道注意力模块（Channel Attention Module，CAM），该模块能够在不丢失全局信息的前提下，提升重要特征的响应强度。我们首先来看通道注意力模块，这个模块的目的是为了生成注意力权重，其中一层全连接层（即注意力卷积核）是为了生成通道的注意力权重，另一层全连接层则是为了生成通道之间的相互依赖关系。CAM包含两层全连接层，分别对应于两个全连接层。第一层的全连接层是 W^c*x+b^c ，其中的 x 是输入特征图，W^c 和 b^c 分别表示第 c 个通道的卷积核和偏置项。第二层的全连接层是 W^t*theta(X)*x + b^t ，其中 theta(X) 是 Spatial Softmax 函数，用于生成注意力映射矩阵。注意力映射矩阵 A 是一个 K × C 维度的矩阵，K 为感受野大小（即卷积核大小），C 为输入通道数。注意力映射矩阵的每一行代表了一个通道的注意力映射向量，它的元素值表示与特定像素位置相关联的特征的重要程度。CAM模块首先应用第一个全连接层将特征图中的所有像素点投影到每个通道的空间上，然后使用 SpatialSoftmax 函数来生成通道之间的相互依赖关系，最后将通道注意力权重应用于特征图中。其基本操作步骤如下:

1. 定义 SENet 中的注意力卷积核以及偏置项，并应用于输入特征图中，得到通道注意力权重。

2. 生成通道之间的相互依赖关系，即 SpatialSoftmax 函数。

3. 将通道注意力权重应用于输入特征图中。

4. 最后，使用全局平均池化层进行特征整合，将注意力映射矩阵与平均池化后的特征进行拼接，得到最终的输出结果。

SENet 的优点是能够对输入特征图中的所有通道进行全局性的特征学习，提高模型的分类精度；缺点是由于 CAM 模块的设计，计算量比较大。并且 CAM 模块的设计与输入特征图的尺寸和深度耦合度较高，无法直接处理变化多样的输入。

## CBAM
CBAM 认为卷积层是一种必要的网络结构，为了充分地利用它，引入注意力分配模块（Attention Allocation Module，AAM），它能够在不丢失全局信息的前提下，提升网络的特征学习能力。AAM 将注意力分配模块分为三个子模块：通道注意力模块（CAM）、空间注意力模块（SAM）和注意力整合模块（AIM）。其中，CAM 提供了通道级的注意力分配，它可以自适应地调整特征图中的全局分布；SAM 提供了空间级的注意力分配，它能够捕获局部特征，而不需要反传梯度；AIM 对 CAM 和 SAM 的注意力分配结果进行融合，产生最终的注意力分配结果。CBAM 的基本操作步骤如下:

1. 定义 CBAM 中的注意力分配模块，包括 CAM、SAM 和 AIM。

2. 执行 CAM 操作，它利用单个卷积核函数生成通道注意力权重。

3. 执行 SAM 操作，它利用空间关联性来生成空间注意力权重。

4. 将 SAM 注意力权重与 CAM 注意力权重进行拼接，得到完整的注意力分配矩阵。

5. 执行 AIM 操作，它将 SAM 注意力映射矩阵和 CAM 注意力映射矩阵进行融合，并将它们与输入特征图进行拼接。

6. 最后，执行全局平均池化层进行特征整合，得到最终的输出结果。

CBAM 与 SENet 同样具有全局性和局部性的特点，能够通过注意力分配模块，显著提升网络的特征学习能力。虽然 CBAM 比 SENet 更加复杂，但是它能够提取更多的全局信息，并在保持计算量的情况下有效地处理图像数据。

## Non-Local Neural Networks(NLNet)
NLNet 主要利用空间上的非局部性来捕获全局上下文信息。非局部性原理是在图像内寻找与当前像素点密切相关的邻近像素点，然后借鉴这些邻近像素点的信息来估计当前像素点的值。NLNet 的具体原理是在输入图像中先使用标准卷积层提取各个通道的局部特征，然后使用一个 1×1 卷积核和两个小卷积核（称之为互感器核和范围核）将这些局部特征聚合成为全局特征。互感器核将邻近像素点之间的相似性编码到图像特征中，而范围核则对邻近区域内的相似性进行建模。最后，NLNet 将全局特征和局部特征结合起来，作为最终的输出。


# 4.代码实现及代码解析
接下来，我将展示两种典型的 CBAM 和 SENet 的代码实现。
## CBAM 代码实现
下面的代码给出 CBAM 网络的具体实现，包括构建 CBAM 模块和定义整个网络。这里，我们假设输入特征图为 (batchsize, channels, height, width)，输出的特征图为 (batchsize, outchannels, height, width)。
```python
import torch.nn as nn
from modules import Bottleneck, Conv2dReLU

class CBAM(nn.Module):
def __init__(self, inplanes, planes, reduction=16):
super().__init__()

self.channel_attention = nn.Sequential(
nn.AdaptiveAvgPool2d((1, 1)),
Conv2dReLU(inplanes, planes // reduction, kernel_size=1),
nn.Conv2d(planes // reduction, planes, kernel_size=1),
nn.Sigmoid()
)

self.spatial_attention = nn.Sequential(
Conv2dReLU(inplanes, 1, kernel_size=7, padding=3),
nn.Sigmoid()
)

self.conv = nn.Sequential(
Conv2dReLU(inplanes * 2, planes, kernel_size=1),
nn.Conv2d(planes, outplanes, kernel_size=1)
)

def forward(self, x):
channel_weight = self.channel_attention(x)
spatial_weight = self.spatial_attention(x)

weight = torch.mul(channel_weight, spatial_weight)

out = torch.cat([x, weight], dim=1)
out = self.conv(out)

return out
```
其中 `modules` 文件夹中定义了一些辅助函数，如 `Bottleneck`、`Conv2dReLU`。定义完 `CBAM` 模块后，就可以定义整个网络，比如说 ResNet 的网络结构，如下面这样：
```python
import torch.nn as nn
from models import resnet
from modules import CBAM

class MyModel(nn.Module):
def __init__(self, num_classes):
super().__init__()

# define backbone network
self.backbone = resnet.resnet50(pretrained=True, progress=True)

# modify the last layer of resnet to fit our task
del self.backbone.fc
self.backbone.fc = nn.Identity()

# add CBAM module on the output of each block
for name, m in self.backbone.named_children():
if isinstance(m, nn.Sequential):
for n, sm in enumerate(m):
setattr(m[n], 'cbam', CBAM(sm.out_channels, sm.out_channels))

# classifier head
self.classifier = nn.Linear(2048, num_classes)

def forward(self, x):
feature = self.backbone(x)
out = self.classifier(feature)

return out
```
其中 `resenet` 文件夹中定义了 ResNet 系列的网络结构，这里直接调用 torchvision 库中的实现即可。

## SENet 代码实现
SENet 的代码实现和 CBAM 基本类似，区别在于只需修改最后的分类头，使其输出多个通道而不是单个通道即可。