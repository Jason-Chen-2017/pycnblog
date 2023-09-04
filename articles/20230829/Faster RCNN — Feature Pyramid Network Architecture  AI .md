
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Faster RCNN是当前非常火热的基于区域卷积神经网络(Region-based Convolutional Neural Networks)的目标检测方法。其优点是准确率高、速度快。然而，目前该方法仍处于理论阶段，不同框架实现可能存在细微差别。本文将通过对Feature Pyramid Network（FPN）的介绍，并结合论文详细阐述其原理和具体实现过程。同时，我们还会讲解FPN的训练过程，以及如何用FPN替代VGG16作为预训练网络提升检测性能。希望通过阅读本文，能够帮助读者进一步了解Faster RCNN及其底层原理。

首先，关于什么是Faster RCNN？

Faster R-CNN是基于区域卷积神经网络的目标检测方法。它在两个阶段完成对象检测：第一阶段，提取候选区域，第二阶段，利用这些候选区域来进行分类和回归。其中，第一阶段使用了Regions of Interest (RoI)，相比于传统的CNN卷积特征图，只保留了感兴趣区域的特征。其次，基于RoI池化层，在卷积特征上形成固定大小的输出，使得后续分类器更加关注图像中的重要信息。

与之相对应的还有R-FCN，它采用了更简单的结构，并减少了计算量。但是，由于R-FCN没有使用FPN，因此只能取得不如Faster R-CNN好的结果。

而Faster RCNN则通过引入FPN，有效解决了以上两个问题：

1.通过FPN模块，可以提取不同尺度的特征，从而获取多尺度信息；
2.可以通过RoIAlign层替换RoIPooling层，提升模型的推理速度。

Faster RCNN在速度方面也有很大的优势，它能在短时延时设备上实时执行，在单张GPU上的训练速度超过其他方法。

总的来说，Faster RCNN是一个非常有前景的方法，它的优点是准确率高、速度快。但是，目前该方法还处于理论阶段，不同框架实现可能存在细微差别，因此，仍需要更多研究工作来进一步完善这个技术。
# 2.关键术语及概念

## 2.1 Region of Interest (RoI)
一个候选区域用于后续的分类和回归任务。比如，在一个图片中，某个物体所在的位置可以视为候选区域。

## 2.2 RoI Pooling
由候选区域中感兴趣区域的像素组成的一个固定大小的向量，经过池化操作后得到最终的分类结果或回归值。

RoIPooling只是一种经典的池化方式，可以在不同的框架中实现。现有的RoIPooling方法一般都是固定大小的，比如7*7、3*3或者2*2。因此，当候选区域的大小不是7*7的整数倍时，就无法直接使用Pooling层。另外，当候选区域在图像边缘时，可能出现一些负担，因此，通常会选择最近邻的RoI，这又涉及到另一个问题：“如何找到最接近候选区域的真实框坐标”？

## 2.3 Anchor Boxes
候选区域的一种初始化方法。可以认为是随机生成的一组矩形框，大小和比例都可自定义。但是，如果要达到较好的效果，建议还是要参照目标物体的实际尺寸设计Anchor Box。

## 2.4 Multi-scale RoI Align
一种改进版的RoIPooling，既考虑了候选区域的尺度，又考虑了候选区域的位置偏移。同时，对特征图的每个像素点使用不同大小的窗口进行插值，最后进行平均池化得到最终的分类结果或回归值。

Multi-scale RoI Align是为了解决由于候选区域的大小导致的检测失真的问题。如果候选区域太小，就无法充分利用上下文的信息；如果候选区域太大，又会引入额外的计算量。因此，在相同的计算量下，Multi-scale RoI Align可以取得更好的检测效果。

## 2.5 Feature Pyramid Network (FPN)
一种构建深度神经网络特征金字塔的技术。FPN主要有两个作用：

1. 提取多尺度特征，可以有效地处理各种尺度的物体；
2. 使用不同尺度的特征，降低检测的难度，提升效率。

在FPN中，提取出的不同尺度特征称为FPN Levels，由浅入深排列。其中，第k个Level的特征是第k+1个Level特征的下采样。


## 2.6 ROIAlign
相比于RoIPooling，RoIAlign在精度上略微好一些，因为它对候选区域周围的像素进行插值，而RoIPooling仅仅采用窗口中最大值的形式。但也正因如此，RoIAlign的计算量要比RoIPooling多很多。

ROIAlign是Faster RCNN所使用的新型Pooling层，可以快速且准确地对候选区域进行池化。

## 2.7 Pretrained Model
预训练模型，通常指在ImageNet数据集上预先训练出来的神经网络参数，用来帮助我们快速训练新的模型。包括AlexNet、VGG、ResNet等。

使用预训练模型可以大大提高模型的精度。特别是在目标检测领域，VGG16已经被证明是一个很好的backbone模型。

但对于不同的数据集，或不同类型的物体，或对于特殊情况下（如密集物体），预训练模型的效果可能会受限。因此，在训练自己的模型之前，应该注意是否可以使用预训练模型。

# 3.原理及流程详解

## 3.1 数据准备

输入图片：首先，输入图片应该被Resize至统一大小，这里通常使用600*600或者1024*1024这样较大的分辨率。

候选区域生成：然后，需要根据种类的数量和尺度，自动生成若干个候选区域，即Region of Interest (RoI)。通常可以设定一定的步长来增加ROI的密度。

类别标签：在生成候选区域时，还需要指定相应的类别标签。比如，给狗、猫、鸟分别分配编号为1、2、3即可。

构造训练集和测试集：最后，将所有的图片按照8:2的比例划分为训练集和测试集。


## 3.2 特征提取

Faster RCNN在初始阶段会首先使用预训练的VGG16模型来提取图像特征。在卷积层之后，通过五个池化层得到特征图，包含conv5_3、pool5和fc7。我们可以把这三个特征图看作是整张图片的高层特征图，包含了图片的全局信息。

此外，还会使用RPN网络来生成候选区域（Region of Interest）。RPN的目标就是生成多个不同大小和纵横比的区域，这些区域代表了输入图片中感兴趣的区域。具体来说，RPN首先将VGG16的中间输出conv5_3送入两个全连接层来获得候选区域的先验框（anchor box）。这些先验框用两种尺寸和三种纵横比生成，分别为16*16、32*32和64*64。对于每张图片，RPN都会产生一系列的预测框（prediction box），表示预测目标（如物体、行人等）的存在区域。


基于RPN，RoI Pooling层生成候选区域的特征图。RoI Pooling层用不同尺度和位置的窗口提取候选区域的特征图。每个窗口在输入特征图上滑动一次，获得一个固定的大小的特征图。经过池化后，可以得到候选区域的特征向量。

## 3.3 目标分类与回归

候选区域的特征经过一个两分支的网络（two-branch network）后，就可以得到候选区域的分类与回归结果。两个分支分别由两个具有不同功能的全连接层构成，一个用于分类，一个用于回归。

分类分支由K个卷积核（kernel）产生，每一个卷积核产生一个分类置信度（confidence score）。通过softmax函数，可以得到每个候选区域属于各个类别的概率值。回归分支由一个全连接层产生，该层对K个类别的回归目标（regression target）进行预测。

最终，分类结果和回归结果通过NMS（non maximum suppression）算法合并到一起，以消除重复的候选区域。

## 3.4 损失函数

为了训练模型，我们需要定义损失函数。Faster RCNN使用了两种损失函数，即分类损失和回归损失。

分类损失计算的是候选区域的分类置信度与Ground Truth的IoU（交并比）之间的误差。当IoU大于0.5时，IoU越小，分类置信度越大，此时的损失较小；当IoU小于0.1时，分类置信度较小，此时的损失较大。为了平衡这两种情况，分类损失常和回归损失混合使用。

回归损失计算的是候选区域的回归目标与Ground Truth之间的误差。通常，回归目标是候选区域的中心坐标和宽高。回归损失采用Huber损失函数，该函数对异常值更敏感，但收敛速度更快。

总的来说，Faster RCNN使用了类似于YOLOv3的损失函数，可以有效地抑制负样本（background）、提升正样本的分类置信度和回归精度。

## 3.5 优化器

为了使模型能够学习到有用的特征，需要采用优化器。Faster RCNN使用SGD加速梯度下降法。

## 3.6 模型部署

Faster RCNN的输出可以直接用于对象检测，但是由于计算量过大，实时性要求不高，因此往往需要在移动端部署。为了压缩模型大小和加速推理速度，Faster RCNN提供了一些技巧，如NMS、RoIAlign等。具体的部署过程可以参照官方文档。

## 3.7 可视化

训练过程的可视化对了解模型的训练过程非常有帮助。一般来说，可以观察分类损失和回归损失随着迭代次数的变化曲线，如果分类损失明显不足（比如收敛于一个较大的负值），那么模型可能欠拟合；反之，如果回归损失明显不足，那么模型可能过拟合。

还可以通过观察验证集上AP（Average Precision）的变化来判断模型的泛化能力。AP的值越高，模型的泛化能力越好。

# 4.代码实例及解释

## 4.1 VGG16预训练模型

```python
import torch.nn as nn
from torchvision import models

class VGG16Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(*list(models.vgg16(pretrained=True).features.children())[:-1])

    def forward(self, x):
        x = self.features(x)
        return [x]
        
```

上面是实现了VGG16作为backbone模型的代码。VGG16模型由16个卷积层和3个全连接层组成。其中前四个卷积层组成特征提取层，后三个全连接层组成分类层。下面简单介绍一下代码中使用的模型。

`torchvision.models.vgg16()`函数可以返回预训练好的VGG16模型。`pretrained=True`参数告诉PyTorch加载预训练好的权重，而`models.vgg16().features`属性返回了VGG16的特征提取层。用列表切片`[:-1]`来舍弃最后一个全连接层，避免重复使用。

```python
model = VGG16Backbone()
print(model)
```

打印模型结构。

```
Sequential(
  (0): Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (1): ReLU(inplace=True)
  (2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (3): ReLU(inplace=True)
  (4): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (5): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (6): ReLU(inplace=True)
  (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (8): ReLU(inplace=True)
  (9): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (10): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (11): ReLU(inplace=True)
  (12): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (13): ReLU(inplace=True)
  (14): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (15): ReLU(inplace=True)
  (16): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (17): Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (18): ReLU(inplace=True)
  (19): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (20): ReLU(inplace=True)
  (21): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (22): ReLU(inplace=True)
  (23): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (24): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (25): ReLU(inplace=True)
  (26): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (27): ReLU(inplace=True)
  (28): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (29): ReLU(inplace=True)
  (30): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
)
```

## 4.2 RPN

下面介绍一下RPN。RPN是一个轻量级的网络，负责生成候选区域（Region of Interest）。

RPN的输入是conv5_3特征图，输出是K个先验框（anchor box）。先验框由两种尺度和三种纵横比生成，分别为16*16、32*32和64*64。每张图片的RPN输出了N个先验框（N表示图片中检测到的目标个数）。

```python
import numpy as np
import torch
import torch.nn as nn
import torchvision.ops as ops

class RPN(nn.Module):
    def __init__(self, in_channels=512, num_anchors=3):
        super().__init__()
        
        # feature map size
        self.in_h = None
        self.in_w = None

        # anchor settings
        self.num_anchors = num_anchors
        self.anchor_sizes = [(16,), (32,), (64,)]   # anchor sizes for small objects, medium and large objects respectively
        self.aspect_ratios = [[0.5, 1.0, 2.0]] * len(self.anchor_sizes)    # aspect ratios for each anchor

        # model layers
        self.conv1 = nn.Conv2d(in_channels, 512, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(512, num_anchors * 4, 1)     # convolution layer to predict the coordinates of anchors
        self.score = nn.Sigmoid()                            # sigmoid activation function to bound the output between zero and one
        
    def _create_anchors(self, feature_map_size):
        """Create anchor boxes."""
        all_anchors = []
        h, w = feature_map_size
        for i, anchor_size in enumerate(self.anchor_sizes):
            cx = np.arange(w)/w + 0.5/(w)
            cy = np.arange(h)/h + 0.5/(h)
            cxcy = np.meshgrid(cx, cy)
            cxcy = np.stack(cxcy, axis=-1)      # shape=[H, W, 2]
            
            aspect_ratio = self.aspect_ratios[i]
            if isinstance(aspect_ratio, int):
                aspect_ratio = [aspect_ratio]
                
            for ar in aspect_ratio:
                s = anchor_size / np.sqrt(ar)        # scale factor

                ws = s * np.array([np.sqrt(ar)])           # width of bounding box
                hs = s * np.array([ar])                 # height of bounding box
                    
                ws = np.round(ws.astype('float32'))       # width of bounding box (integer)
                hs = np.round(hs.astype('float32'))       # height of bounding box (integer)

                bboxes = np.concatenate((cxcy - 0.5*(ws[:, :, np.newaxis]+hs[:, :, np.newaxis])/w,
                                         cxcy + 0.5*(ws[:, :, np.newaxis]+hs[:, :, np.newaxis])/w),
                                        axis=-1)
                all_anchors.append(bboxes)

        all_anchors = np.concatenate(all_anchors, axis=1)
        all_anchors[..., ::2] *= self.in_w          # normalize by image width
        all_anchors[..., 1::2] *= self.in_h         # normalize by image height

        return all_anchors
    
    def forward(self, x):
        self.in_h, self.in_w = x.shape[-2:]
        batch_size = x.shape[0]
        
        # create anchors
        feat_size = list(x.shape[-2:])              # input feature map size
        anchors = self._create_anchors(feat_size)   # shape=[A, 4]
        A = anchors.shape[0]                       # number of anchors per location

        # apply conv layers
        out = self.relu(self.conv1(x))               # shape=[B, C, H, W]
        rpn_cls = self.conv2(out)                   # shape=[B, K*4, H, W]
        
        # reshape outputs
        rpn_cls = rpn_cls.permute(0, 2, 3, 1)        # shape=[B, H, W, K*4]
        rpn_cls = rpn_cls.reshape(-1, A, 4)          # shape=[BHW, A, 4]
        
        # convert the predicted offsets to probabilities using softmax function
        pred_cls = self.score(rpn_cls)

        # select only positive predictions
        pos_idx = (pred_cls > 0.5).nonzero()        # tuple containing indices of positive predictions
        npos = pos_idx.shape[0]                     # total number of positive predictions
        
        # initialize a tensor with negative predictions filled with zeros
        neg_idx = (pred_cls < 0.5).nonzero()        # tuple containing indices of negative predictions
        nneg = neg_idx.shape[0]                     # total number of negative predictions
        scores_neg = pred_cls[neg_idx].view(-1, 1)  # shape=[BHW', 1], initialized with negative predictions
        labels_neg = torch.zeros(scores_neg.shape[0]).long().to(device=scores_neg.device)  # shape=[BHW']

        # merge positive and negative predictions into single tensors
        if npos == 0:             # no positives found
            scores_pos = torch.empty(0).to(device=scores_neg.device)   # empty tensor
            labels_pos = torch.empty(0).to(device=labels_neg.device)   # empty tensor
        else:
            scores_pos = pred_cls[pos_idx].view(-1, 1)    # shape=[BHW, 1], positive predictions
            labels_pos = torch.full(scores_pos.shape, fill_value=1, device=scores_pos.device)   # shape=[BHW, 1], assigned label is always set to 1
            
        scores = torch.cat([scores_pos, scores_neg], dim=0)                  # shape=[BHW, 1] or [BHW', 1] depending on whether any positives were found
        labels = torch.cat([labels_pos, labels_neg], dim=0)                  # shape=[BHW, 1] or [BHW', 1] depending on whether any positives were found
        
        return scores, labels, anchors
    
```

代码中有几点需要注意。

- 在初始化的时候，设置了feature map大小为None，输入特征图的大小会动态更新。
- 设置了三种不同大小的先验框（anchor box）。每种尺度的先验框数量由`num_anchors`决定。
- 创建了两个卷积层，一个用于分类，一个用于回归。分类层输出每个先验框的置信度，回归层输出每个先验框的偏移量。
- 通过Sigmoid激活函数将分类层的输出转换成概率值。
- 根据预测的偏移量来创建候选框（bbox）。

## 4.3 RoI Pooling

RoI Pooling是一个池化层，可以从候选框（bbox）的特征图中截取感兴趣区域的特征。通常，候选框的特征图大小为7*7，因此，使用RoI Pooling可以提取固定大小的特征图。

```python
import torch
import torchvision.ops as ops

class RoIPooling(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        
    def forward(self, features, proposals):
        rois = ops.roi_align(features, proposals, output_size=self.output_size, aligned=True)
        return rois
```

RoIPooling层的输入是特征图和候选框，输出是固定大小的特征图。输入候选框需经过归一化才能被RoIAlign层理解。`aligned=True`参数保证输入的尺度比例一致，方便运算。

## 4.4 RoI Align

RoI Align也是一种池化层，与RoIPooling相似。区别是，RoI Align采用双线性插值的方式，在输入候选框附近的像素点进行插值，避免了常规的最小池化（Pooling）造成的上下文信息丢失。

```python
import torch
import torchvision.ops as ops

class RoIAlign(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.output_size = output_size
        
    def forward(self, features, proposals):
        rois = ops.roi_align(features, proposals, output_size=self.output_size, aligned=False)
        return rois
```

RoI Align的输入与RoIPooling一样，输出也是固定大小的特征图。

## 4.5 Two-Branch Network

Two-Branch Network由两部分组成——分类分支（classification branch）和回归分支（regression branch）。

分类分支由K个卷积核产生，每一个卷积核产生一个分类置信度。通过softmax函数，可以得到每个候选区域属于各个类别的概率值。回归分支由一个全连接层产生，该层对K个类别的回归目标进行预测。

```python
class ClassifierNetwork(nn.Module):
    def __init__(self, in_channels=512, k=9, num_classes=80):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.dropout = nn.Dropout(p=0.5)
        self.conv3 = nn.Conv2d(256, k*num_classes, 1)   # classification branch
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3(x)                        # shape=[B, K*C, H, W]
        
        cls_probs = x.reshape(x.shape[:2] + (-1,))            # shape=[B, K*C, HW]
        return cls_probs
    
class RegressionNetwork(nn.Module):
    def __init__(self, in_channels=512, k=9, num_classes=80):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.dropout = nn.Dropout(p=0.5)
        self.conv3 = nn.Conv2d(256, k*4, 1)                # regression branch
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv3(x)                                    # shape=[B, K*4, H, W]
        
        bbox_deltas = x.reshape(x.shape[:2] + (-1,))         # shape=[B, K*4, HW]
        return bbox_deltas
```

分类分支的输入是conv5_3特征图，输出是K个先验框的分类概率。回归分支的输入是conv5_3特征图，输出是K个先验框的4维偏移量。

## 4.6 Loss Function

Faster RCNN使用了YOLOv3的损失函数。分类损失计算的是候选区域的分类置信度与Ground Truth的IoU（交并比）之间的误差。当IoU大于0.5时，IoU越小，分类置信度越大，此时的损失较小；当IoU小于0.1时，分类置信度较小，此时的损失较大。为了平衡这两种情况，分类损失常和回归损失混合使用。

回归损失计算的是候选区域的回归目标与Ground Trick之间的误差。通常，回归目标是候选区域的中心坐标和宽高。回归损失采用Huber损失函数，该函数对异常值更敏感，但收敛速度更快。

```python
import torch
import torch.nn as nn
import torchvision.ops as ops
from utils.losses import smooth_l1_loss

class LossFunction(nn.Module):
    def __init__(self, num_classes=80):
        super().__init__()
        self.num_classes = num_classes
        
    def forward(self, img_batch, proposal_batch, roi_batch, cls_prob_batch, bbox_delta_batch, gt_label_batch, gt_box_batch):
        '''Calculate loss'''
        
        # prepare data
        batch_size = img_batch.shape[0]
        _, H, W = img_batch.shape[-3:]                    # original image size
        
        # extract ground truth classes and their corresponding bounding boxes from GT annotations
        gt_box_batch = gt_box_batch.to(dtype=proposal_batch.dtype, device=img_batch.device)
        gt_cls_batch = ops.box_convert(gt_box_batch, 'xywh', 'cxcywh')
        gt_cls_batch[..., :2] /= torch.tensor([[W, H]], dtype=proposal_batch.dtype, device=img_batch.device)
        gt_cls_batch = gt_cls_batch[..., :4]
        
        # calculate IoUs between proposed regions and their corresponding GT boxes
        overlaps = ops.box_iou(proposals=proposal_batch, gt_boxes=gt_box_batch, eps=1e-6)

        # calculate class-specific mask for computing classification losses
        fg_mask = overlaps >= 0.5
        bg_mask = overlaps < 0.4
        ignore_mask = ~(fg_mask | bg_mask)
        
        # compute classification losses for both foreground and background regions
        weights_bg = ((1 - overlaps)**2)[bg_mask]
        weights_fg = (overlaps**2)[fg_mask & ~ignore_mask]
        alpha = cfg.alpha
        gamma = cfg.gamma
        cls_losses_bg = -torch.log((1 - alpha)*weights_bg + alpha*weights_fg).sum()/batch_size
        cls_losses_fg = -torch.log(alpha*weights_bg + (1 - alpha)*(weights_fg + epsilon)).sum()/batch_size
        
        # compute localization errors for the foreground regions
        targets = ops.box_convert(gt_box_batch, 'cxcywh', 'xyxy')
        deltas = bbox_delta_batch.unsqueeze(1)
        delta_xy = deltas[..., :2]*cfg.regress_ranges[..., :2][:, None]/strides[:, None]
        delta_wh = torch.exp(deltas[..., 2:4])*cfg.regress_ranges[..., 2:4][:, None]/strides[:, None]
        targets_xywh = torch.cat([(targets[..., :2]-delta_xy)/stride, 
                                   targets[..., 2:4]+delta_xy+delta_wh,
                                   targets[..., 4:]], dim=-1)
        loc_errors = smooth_l1_loss(preds=rois_batch[:, :, 1:], targets=targets_xywh, beta=cfg.beta)/(gt_box_batch.shape[0]*len(scales))
        
        # combine classification and localization losses into a single scalar value
        loss = cls_losses_bg + cls_losses_fg + cfg.loc_weight*loc_errors
        
        return loss
    
```