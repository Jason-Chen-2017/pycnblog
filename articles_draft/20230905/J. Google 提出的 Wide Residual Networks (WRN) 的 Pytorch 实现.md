
作者：禅与计算机程序设计艺术                    

# 1.简介
  

近年来随着神经网络的火热以及模型复杂度的提高，卷积神经网络 (CNN) 在图像识别、目标检测等领域表现出色。但是这些模型往往需要极大的训练样本才能取得很好的效果。然而越来越多的研究人员认识到在保证准确率的前提下，可以通过增加网络的宽度来进一步提升模型的性能。为此，Google 团队提出了一种新的网络结构——“Wide Residual Networks”，并首次在 ImageNet 数据集上展示其优越性。下面我们就来了解一下这个“Wide Residual Network”。

2.Wide Residual Network 的特点
Wide Residual Networks 是 ResNet 的变体，它主要是为了解决梯度消失和梯度爆炸的问题。ResNet 中的残差单元中，每一个残差块都存在跳层连接 (skip-connection)，使得每一层都可以直接获取输入特征图的信息。但是过多的跳层连接会导致梯度传播缓慢或者无法继续训练。而 Wide Residual Networks 通过增加每一层的输出通道数来减小跳层连接的数量，从而缓解梯度消失和梯度爆炸问题。如下图所示：
如上图所示，使用了类似于残差单元的结构，但是每一层的输出通道数远大于输入通道数。这样就可以避免梯度消失或者梯度爆炸的问题。而且Wide Residual Network 中还采用了分组卷积 (Group convolutions) 来进行特征学习。分组卷积可以帮助网络更有效地利用空间关联性，并且可以加速收敛过程。

3.Wide Residual Network 的结构
作为论文的核心内容之一，我们首先来看一下 Wide Residual Network 的结构。这里我们假设输入图片的尺寸为 $32 \times 32$ ，输出类别个数为 $10$ 。
### Bottleneck Block
第一个模块为残差模块，它由两个 3 x 3 的卷积层组成，其中第二个卷积层的输入通道数为第一个卷积层输出的 4 倍。然后再接一个 1 x 1 的卷积层用于改变通道数，如下图所示：

### Group of Blocks with Wider Channels
第二个模块是分组卷积模块，使用的是六个不同大小的卷积核（3 × 3， 1 × 1），其中第一个卷积层与第二个卷积层的输入通道数分别为第一个模块的输出通道数，而后三个卷积层的输入通道数均为每个分组的输出通道数，如下图所示：

### Fully Connected Layer and Softmax Classifier
第三个模块为全连接层和softmax分类器。对于第四层的输出进行全局池化，然后将其扁平化为一维数组，作为全连接层的输入。最后添加一个 softmax 分类器，将输出概率归一化到 0~1 之间。如下图所示：

以上就是 Wide Residual Network 的结构。

## Pytorch 实现代码
```python
import torch.nn as nn

class WRN(nn.Module):
    def __init__(self, num_classes=10, depth=28, widen_factor=10, dropRate=0.0):
        super(WRN, self).__init__()

        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0) # depth should be 6n+4
        n = (depth - 4) // 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)
```