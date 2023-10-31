
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


在进入技术博客的讨论之前，我先简单介绍一下关于“智能工业”这个领域的一些基本情况。一般来说，智能工业指的是基于智能技术的产业，目前国内外已经出现了很多的案例。其中，最具代表性的就是“汽车、电动车、无人驾驶汽车”等新兴产业，以及在工厂里进行自动化生产的智能化工业。

对于目前为止，大家对智能工业的认识可能还停留在一些粗浅的阶段，比如只知道它是一种新兴产业，没什么实际意义，不能解决实际的问题，也没有哪些具体应用。这就需要我们用技术手段去逐渐推进其前景和价值，真正地用科技让生活更美好！因此，“智能工业”才成为当下热门话题之一。

本文将以“智能工业”作为主题，从计算机视觉到机器学习，介绍各个领域的基本概念、原理以及其与智能工业之间的关系。文章将着重阐述所涉及的各种理论知识，并对比分析不同技术实现的差异，使读者能够准确地理解相关理论。最后，希望通过阅读此文，读者可以体会到关于智能工业的一系列新鲜感、热情以及潜力。

# 2.核心概念与联系
首先，我们要了解一下智能工业领域中的一些核心概念和相关术语。
1. 目标检测与分类：目标检测（Object Detection）和图像分类（Image Classification）是两种经典的图像处理任务。它们的主要功能是从图像中识别出特定物体或区域。目标检测通常包括多个子任务，如候选区域生成（Region Proposal Generation），特征提取（Feature Extraction），类别判定（Classification）。而图像分类则只考虑输入图片中的单一目标，并不需要探测到其他对象。

2. 实例分割：实例分割（Instance Segmentation）是目标检测任务的子任务之一。它由一个带有掩模（Mask）的实例掩盖住的图像组成，目的是帮助定位物体内部的空间分布。实例分割既可以用来预测对象的整体位置，也可以用来划分对象的每一部分。

3. 边界框（Bounding Box）：在目标检测中，我们通常需要对图像中的每个目标进行定位，这些定位信息由一个矩形框（Bounding Box）表示。矩形框通常由4个坐标参数确定，分别是左上角横坐标和纵坐标，右下角横坐标和纵坐标。

4. 锚框（Anchor Box）：在Faster R-CNN网络中，锚框（Anchor Box）作为候选区域生成的方法之一，是一种基于区域建议的方法。它可以有效减少计算量并提高精度。

5. 卷积神经网络（Convolutional Neural Network，CNN）：CNN是一种深度学习技术，可以用于处理图像数据。CNN的核心思想是通过多层次的过滤器进行特征提取。

6. 深度学习：深度学习（Deep Learning）是一种机器学习技术，可以用来训练复杂的神经网络，它使用浅层的权重，然后根据输入的样本调整权重。

7. 注意力机制：注意力机制（Attention Mechanism）被用来解决图像分割、视频分析以及其他领域的任务。它通过给予不同的像素不同的重要程度，来帮助模型聚焦到重要的区域。

8. 序列到序列（Sequence to Sequence，Seq2seq）：Seq2seq网络是一种模型，它可以用于机器翻译、文本摘要等任务。它的关键思想是利用时间上的关联性，将一个序列转换成另一个序列。

9. 循环神经网络（Recurrent Neural Network，RNN）：RNN网络是一种有记忆能力的神经网络，可以用于文本处理、音频处理以及时间序列预测等任务。

10. 生成对抗网络（Generative Adversarial Networks，GANs）：GAN是一个深度学习模型，它可以生成新的样本或者对已有样本进行评估。生成器（Generator）和鉴别器（Discriminator）两个网络互相博弈，生成器生成新的数据，并尝试欺骗鉴别器。

11. 概率图模型（Probability Graph Model）：概率图模型（Probability Graph Model）是一种统计建模方法，用来描述复杂的概率分布。它采用图结构，节点代表随机变量，边代表条件依赖关系。概率图模型可以用于一些复杂的机器学习问题，如表示概率密度函数、判定时序模型、混合高斯模型等。

# 3.核心算法原理和具体操作步骤以及数学模型公式详细讲解
下面，我们将结合以上所介绍的一些基础概念，来详细介绍几种常用的计算机视觉和机器学习算法。
## （1）目标检测与分类
目标检测（Object Detection）和图像分类（Image Classification）是两种经典的图像处理任务。下面，我们将介绍这两种任务的相关算法原理。

### 1. Faster RCNN
Faster RCNN是当前最流行的人脸检测模型之一，它的主要特点如下：
- 使用AlexNet作为底层网络，在速度和准确率方面都优于后续模型；
- 提出了边界框回归模块，能够直接学习到边界框偏移量；
- 增加ROI池化层，减小了空间尺寸，加快了计算速度；
- 设计了RoIAlign层，提升了锚框对齐的准确率；
- 将多个训练阶段联合训练，提升了模型的泛化能力。

下面，我们将介绍Faster RCNN算法的具体操作步骤。
**Faster RCNN算法操作步骤如下：**
1. 对输入图像进行预处理，缩放到固定大小，并进行BGR色彩空间变换；
2. 输入图像通过CNN得到特征图；
3. 从特征图中提取感兴趣区域（Regions of Interest，RoIs）；
4. 在提取出的RoIs上进行预测，得出每个RoI的类别置信度和边界框偏移量；
5. 将预测结果通过非极大值抑制（Non Maximum Suppression，NMS）算法消除重复检测框；
6. 通过边界框回归模块，修正边界框位置；
7. 返回最终的检测结果，包括类别标签、边界框和置信度。

这里有一个重要的数学模型公式。首先，假设我们有K个类别，C张输入图像，S个感兴趣区域（RoIs），那么需要拟合的参数有：
- K+1个类别置信度，对应预测结果的第K个元素；
- S*4个边界框偏移量，对应预测结果的前S个元素；
- C*H*W个卷积核（feature map）。

因此，总共需要拟合的参数有K*(S+C)*4 + CKHW。Faster RCNN采用随机梯度下降法（Stochastic Gradient Descent，SGD）优化网络参数，通过反向传播算法更新参数。

### 2. YOLOv1/YOLOv2/YOLOv3
YOLO（You Look Only Once，只能看到一次）是一种目标检测模型，其最初版本是YOLOv1，之后出现了YOLOv2和YOLOv3等改良版本。它主要特点如下：
- 使用Darknet-19作为底层网络，具有良好的推理速度和低内存占用；
- 使用预定义的anchor box进行检测，可避免人工设计参数；
- 不使用全连接层，采用检测输出分支，可以得到更多的感受野；
- 有IOU损失项，能有效平衡精度和召回率；
- 支持多种尺度检测。

下面，我们将介绍YOLOv1算法的具体操作步骤。
**YOLOv1算法操作步骤如下：**
1. 对输入图像进行预处理，缩放到固定大小，并进行BGR色彩空间变换；
2. 输入图像通过CNN得到特征图；
3. 从特征图中提取感兴趣区域（Regions of Interest，RoIs）；
4. 以SxSx（S是任意正整数）个anchor box作为候选区域，对每个RoI进行预测，得出该RoI属于各个类别的置信度和边界框位置；
5. 根据阈值判断是否存在物体，并将预测结果缩放至原始图像大小；
6. 返回最终的检测结果，包括类别标签、边界框和置信度。

这里有一个重要的数学模型公式。首先，假设我们有K个类别，C张输入图像，SxSx个anchor box，那么需要拟合的参数有：
- (S^2+S^2)x(CxHxW)个边界框坐标，对应预测结果的SxSx个元素；
- S*S个anchor box的尺寸，对应预测结果的SxSx个元素；
- K+1个类别置信度，对应预测结果的第K个元素；
- CxHxWxS个卷积核（feature map）。

因此，总共需要拟合的参数有SxS(C+5) + CKHW。YOLOv1采用批量梯度下降法（Batch Gradient Descent，BGD）优化网络参数，一次对所有样本进行预测。

## （2）实例分割
实例分割（Instance Segmentation）是目标检测任务的子任务之一。下面，我们将介绍实例分割算法的相关原理。

### 1. Mask R-CNN
Mask R-CNN也是一种实例分割模型，它的主要特点如下：
- 使用ResNet作为底层网络，提升了特征提取的效率；
- 引入实例分割模块，输出每个目标的掩膜（mask）；
- 提出多任务损失函数，同时学习实例分割和分类任务；
- 用RoI Align代替RoIPooling，在一定程度上提升了RoI提取效率。

下面，我们将介绍Mask R-CNN算法的具体操作步骤。
**Mask R-CNN算法操作步骤如下：**
1. 对输入图像进行预处理，缩放到固定大小，并进行BGR色彩空间变换；
2. 输入图像通过CNN得到特征图；
3. 从特征图中提取感兴趣区域（Regions of Interest，RoIs）；
4. 对每个RoI进行预测，得出该RoI属于各个类别的置信度和边界框位置；
5. 根据阈值判断是否存在物体，并将预测结果缩放至原始图像大小；
6. 在每个RoI上执行实例分割模块，输出每个目标的掩膜（mask）；
7. 返回最终的检测结果，包括类别标签、边界框和置信度。

这里有一个重要的数学模型公式。首先，假设我们有K个类别，C张输入图像，S个感兴趣区域（RoIs），那么需要拟合的参数有：
- K+1个类别置信度，对应预测结果的第K个元素；
- S*4个边界框偏移量，对应预测结果的前S个元素；
- C*H*W个卷积核（feature map）；
- CHW*K个掩膜，对应预测结果的后K个元素。

因此，总共需要拟合的参数有K*(S+C)*4 + CKHW + SKHW。Mask R-CNN采用随机梯度下降法（Stochastic Gradient Descent，SGD）优化网络参数，通过反向传播算法更新参数。

### 2. PSPNet
PSPNet是一种强大的图像分割模型，它的主要特点如下：
- 使用ResNet-101作为底层网络，提升了特征提取的效率；
- 分割输出分支采用PPM级联，提升了感受野；
- 添加类似跳跃链接的跳级金字塔（Pyramid Scene Parsing Network，PSPNet），提升了分割精度。

下面，我们将介绍PSPNet算法的具体操作步骤。
**PSPNet算法操作步骤如下：**
1. 对输入图像进行预处理，缩放到固定大小，并进行BGR色彩空间变换；
2. 输入图像通过CNN得到特征图；
3. 从特征图中提取感兴趣区域（Regions of Interest，RoIs）；
4. 以多尺度进行预测，得到不同尺度下的预测结果；
5. 将不同尺度的预测结果融合，输出最终的分割结果。

这里有一个重要的数学模型公式。首先，假设我们有K个类别，C张输入图像，N个像素分类输出，那么需要拟合的参数有：
- NxC个卷积核（feature map）；
- NC个卷积核（feature map）；
- KxNx1x1的可学习参数。

因此，总共需要拟合的参数有NC + KN + NKx1x1。PSPNet采用全局平均池化（Global Average Pooling）的方式融合不同尺度下的预测结果。

## （3）边界框回归
边界框回归（Bounding Box Regression）是目标检测任务的一个重要模块，通常用于改善边界框的定位精度。下面，我们将介绍两种常用的边界框回归方法——SSD和FRCN。

### 1. SSD
SSD（Single Shot MultiBox Detector）是一种常用的边界框回归模型。它的主要特点如下：
- 使用VGG-16作为底层网络，快速训练；
- 使用多个尺度的特征图进行预测，在检测效果和速度之间取得了平衡；
- 每个位置预测不同尺度的边界框；
- 每个类别预测不同尺度的回归系数。

下面，我们将介绍SSD算法的具体操作步骤。
**SSD算法操作步骤如下：**
1. 对输入图像进行预处理，缩放到固定大小，并进行BGR色彩空间变换；
2. 输入图像通过VGG-16得到特征图；
3. 按照不同尺度对特征图进行采样，得到不同尺度的候选区域；
4. 对于每个候选区域，执行SSD结构，得出预测结果；
5. 返回最终的检测结果，包括类别标签、边界框和置信度。

这里有一个重要的数学模型公式。首先，假设我们有K个类别，C张输入图像，S个候选区域，那么需要拟合的参数有：
- K个类别置信度，对应预测结果的前K个元素；
- S*4个边界框偏移量，对应预测结果的后S个元素；
- K*S个默认框，对应预测结果的中间K*S个元素；
- VGG-16每个卷积层输出的通道数。

因此，总共需要拟合的参数有(S+K)*4 + K*(S+1)*4 + KC。SSD采用随机梯度下降法（Stochastic Gradient Descent，SGD）优化网络参数，通过反向传播算法更新参数。

### 2. FRCN
FRCN（Fully Recongnized Convolutional Network）是一种用于边界框回归的深度神经网络，它的主要特点如下：
- 使用ResNet-50作为底层网络，相比VGG-16提升了性能；
- 引入卷积方块（Convolutional Block）模块，增强网络的感受野；
- 使用双线性插值，提供更精细的边界框预测。

下面，我们将介绍FRCN算法的具体操作步骤。
**FRCN算法操作步骤如下：**
1. 对输入图像进行预处理，缩放到固定大小，并进行BGR色彩空间变换；
2. 输入图像通过ResNet-50得到特征图；
3. 对每个像素执行卷积方块模块，得到预测结果；
4. 返回最终的检测结果，包括类别标签、边界框和置信度。

这里有一个重要的数学模型公式。首先，假设我们有K个类别，C张输入图像，W和H是图像宽和高，那么需要拟合的参数有：
- W*H*K个卷积核（feature map）；
- (W/16)*(H/16)*2xK的偏置。

因此，总共需要拟合的参数有KCH + HWKxK。FRCN采用反向传播算法（Backpropagation）优化网络参数。

# 4.具体代码实例和详细解释说明
为了能够更直观地了解上面所介绍的各种算法的具体操作步骤以及数学模型公式，下面举例说明如何实现一小部分代码。
## （1）目标检测与分类——Faster RCNN
首先，导入相关库。
```python
import cv2
from torchvision import transforms as T
from torchvison.models.detection import fasterrcnn_resnet50_fpn
```
定义目标检测模型并加载预训练模型。
```python
model = fasterrcnn_resnet50_fpn(pretrained=True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
```
读取测试图片并预处理。
```python
transform = T.Compose([T.ToTensor(),
                       T.Normalize((0.485, 0.456, 0.406), 
                                   (0.229, 0.224, 0.225))])
img = transform(img).unsqueeze(0)
```
执行目标检测。
```python
with torch.no_grad():
    pred = model([img.to(device)])[0]
```
打印输出的检测结果。
```python
for i in range(len(pred['boxes'])):
    x1, y1, x2, y2 = int(pred['boxes'][i][0]), int(pred['boxes'][i][1]), int(pred['boxes'][i][2]), int(pred['boxes'][i][3])
    label = labels[int(pred['labels'][i])]
    score = float(pred['scores'][i])
    print('{} {} {:.3f} ({}, {})'.format(label, category_id_map[label], score, x1, y1))
    cv2.rectangle(img, (x1,y1), (x2,y2), color=(0,255,0), thickness=2)
cv2.imshow('result', img)
cv2.waitKey(0)
```
## （2）实例分割——Mask R-CNN
首先，导入相关库。
```python
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.utils.visualize import vis_bbox, vis_class, vis_mask
```
定义实例分割模型并加载预训练模型。
```python
cfg.merge_from_file('configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml')
cfg.freeze()
model = build_detection_model(cfg)
checkpointer = DetectronCheckpointer(cfg, model)
_ = checkpointer.load('path/to/your/trained/model')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
model.eval()
```
定义输入图片路径和相关参数。
```python
img_path = 'path/to/your/input/image'
img = Image.open(img_path).convert("RGB")
transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
img = transforms(img).unsqueeze(0)
```
执行实例分割。
```python
with torch.no_grad():
    predictions = model(img.to(device))[0]
predictions = [o.to('cpu') for o in predictions]
```
绘制实例分割结果。
```python
img = cv2.imread(img_path)[:, :, ::-1] # BGR -> RGB
if "coco" in cfg.DATASETS.TEST[0]:
    predicted_masks = predictions.get_field("mask").numpy()
    scores = predictions.get_field("scores").tolist()
    labels = predictions.get_field("labels").tolist()
    masks = []
    idx = 0
    while True:
        try:
            mask = predicted_masks[idx]
            class_id = labels[idx].item() - 1
            score = scores[idx].item()
            rles = cocomask.encode(np.asfortranarray(mask[..., None]))[0]
            mask = cocomask.decode(rles)[..., 0] > 0
            masks.append({
                "score": score,
                "category_id": class_id,
                "segmentation": rles,
                "bbox": utils.extract_bboxes(np.expand_dims(mask, axis=-1)),
            })
            idx += 1
        except IndexError:
            break

    _, ax = plt.subplots(figsize=(16, 16))
    im = ax.imshow(img)
    
    colors = colormap(rgb=True) / 255
    fig, ax = plt.subplots(figsize=(16, 16))
    for m in masks:
        color = tuple(colors[m["category_id"]][:3])
        mask = cocomask.decode(m["segmentation"])
        bboxes = m["bbox"]

        w, h = mask.shape[:2]
        contour, hier = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        areas = [cv2.contourArea(cnt) for cnt in contour]
        max_index = np.argmax(areas)
        
        rect = cv2.minAreaRect(contour[max_index])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        ax.add_patch(PolygonPatch(box, alpha=0.5, ec="none", fc=color))
        ax.text(box[0][0]-5, box[0][1]+10, str(m["category_id"]), fontsize=15, bbox={"facecolor":"w","alpha":0.5,"pad":5})
        
    ax.imshow(im)
    
else:
    raise NotImplementedError
plt.show()
```
## （3）边界框回归——SSD
首先，导入相关库。
```python
import os
import sys
import time
import math
import random
import numpy as np
import cv2
import argparse
from collections import OrderedDict
from itertools import product as product

import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = ''
```
定义SSD模型。
```python
def ssd(num_classes):
    inputs = tf.keras.Input(shape=(None, None, 3))

    # Block 1
    x = layers.Conv2D(64, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    # Block 2
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    c2 = x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # Block 3
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    c3 = x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # Block 4
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    c4 = x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')

    # Block 5
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')
    c5 = x = layers.Activation('linear')(x)

    return tf.keras.Model(inputs, [c2, c3, c4, c5], name='ssd')


def conv_block(inputs, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_base = 'conv{}_{}'.format(stage, block)
    bn_name_base = 'bn{}_{}'.format(stage, block)

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      name=conv_name_base + '_1x1')(inputs)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '_1x1')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      name=conv_name_base + '_3x3')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '_3x3')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1), name=conv_name_base + '_1x1_increase')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '_1x1_increase')(x)

    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             name=conv_name_base + '_shortcut')(inputs)
    shortcut = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '_shortcut')(shortcut)

    outputs = layers.Add()([x, shortcut])
    outputs = layers.Activation('relu')(outputs)
    return outputs


def identity_block(inputs, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_base = 'conv{}_{}'.format(stage, block)
    bn_name_base = 'bn{}_{}'.format(stage, block)

    x = layers.Conv2D(filters1, (1, 1), name=conv_name_base + '_1x1')(inputs)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '_1x1')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      name=conv_name_base + '_3x3')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '_3x3')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1), name=conv_name_base + '_1x1_increase')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '_1x1_increase')(x)

    outputs = layers.Add()([x, inputs])
    outputs = layers.Activation('relu')(outputs)
    return outputs
```
定义输入图片路径和相关参数。
```python
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='tf_ssd', help='The model to test.')
args = parser.parse_args()

img_path = 'path/to/your/input/image'
```
加载并预处理测试图片。
```python
original_image = cv2.imread(img_path)
image_expanded = np.expand_dims(original_image, axis=0)
resized_image = cv2.resize(image_expanded, (300, 300)).astype(np.float32)
normalized_image = resized_image / 255.0
preprocessed_image = np.transpose(normalized_image, [0, 3, 1, 2])
```
执行目标检测。
```python
tflite_interpreter = tf.lite.Interpreter(model_path='tf_ssd/tflite_graph.pb')
tflite_interpreter.allocate_tensors()

input_details = tflite_interpreter.get_input_details()[0]
output_details = tflite_interpreter.get_output_details()

tflite_interpreter.set_tensor(input_details['index'], preprocessed_image)
tflite_interpreter.invoke()

locations = tflite_interpreter.get_tensor(output_details[0]['index'])
labels = tflite_interpreter.get_tensor(output_details[1]['index'])
scores = tflite_interpreter.get_tensor(output_details[2]['index'])
num_detections = tflite_interpreter.get_tensor(output_details[3]['index'])
```
绘制SSD检测结果。
```python
h, w, _ = original_image.shape
drawed_image = image_expanded.copy().squeeze()

scale = min(drawed_image.shape[0]/h, drawed_image.shape[1]/w)
new_h, new_w = int(h*scale), int(w*scale)
drawed_image = cv2.resize(drawed_image, (new_w, new_h))

threshold = 0.4

for i in range(int(num_detections)):
    if scores[i] < threshold: continue
    ymin, xmin, ymax, xmax = locations[i]
    left, right, top, bottom = int(xmin*w), int(xmax*w), int(ymin*h), int(ymax*h)
    cv2.rectangle(drawed_image, (left,top), (right,bottom), (255,0,0), 2)
    text = "{}:{:.2f}%".format(labels[i], scores[i]*100)
    cv2.putText(drawed_image, text, (left, top-5), cv2.FONT_HERSHEY_SIMPLEX,
                0.5,(255,0,0),2)

print('[INFO] Original size:', original_image.shape)
print('[INFO] Resized size:', drawed_image.shape)

cv2.imshow('Result', drawed_image)
cv2.waitKey(0)
```