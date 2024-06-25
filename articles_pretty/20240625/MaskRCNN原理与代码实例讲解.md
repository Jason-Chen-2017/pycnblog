# MaskR-CNN原理与代码实例讲解

## 关键词：

- Mask R-CNN
- 深度学习
- 卷积神经网络
- 实例分割
- 基于像素的预测

## 1. 背景介绍

### 1.1 问题的由来

随着计算机视觉技术的发展，对象检测和实例分割成为了一个热门的研究领域。在图像中精确识别和区分不同的物体是许多实际应用的基础，比如自动驾驶、机器人视觉、医疗影像分析等。传统的对象检测方法只能识别和定位物体的边界框，而实例分割进一步细化到每个物体的具体区域，从而实现了更精细的物体识别和理解。

### 1.2 研究现状

现有的实例分割方法主要可以分为基于像素的方法和基于边界的两种。基于像素的方法能够直接预测每个像素属于哪个类别以及该像素的掩膜，从而实现精确的实例分割。其中，Mask R-CNN是基于Faster R-CNN提出的一种改进型方法，它通过引入额外的分割分支，能够同时检测和分割出图像中的物体实例，并且保持了高性能的同时提高了效率。

### 1.3 研究意义

Mask R-CNN在实例分割领域的贡献在于其能够同时提供准确的对象检测和高精度的实例分割结果，这对于提高计算机视觉系统的鲁棒性、精确性和实用性有着重要的意义。特别是在需要高精度物体识别和理解的应用场景中，如自动驾驶中的行人和车辆检测，医学影像分析中的细胞和组织分割，或者在安防监控中的目标跟踪和识别等方面，都有着广泛的应用前景。

### 1.4 本文结构

本文将详细介绍Mask R-CNN的原理、实现细节以及如何在实际项目中运用。具体内容包括：

- **核心概念与联系**：解释Mask R-CNN的工作原理及其与其他深度学习方法的关系。
- **算法原理与操作步骤**：详细描述Mask R-CNN的算法结构、训练流程和优化策略。
- **数学模型与公式**：给出数学建模过程和关键公式的推导。
- **代码实例与详细解释**：提供基于PyTorch实现的Mask R-CNN代码实例，并进行详细解读。
- **实际应用场景**：探讨Mask R-CNN在不同领域的应用案例。
- **工具与资源推荐**：分享学习资源、开发工具和相关论文推荐。

## 2. 核心概念与联系

### 2.1 Mask R-CNN概述

Mask R-CNN是在Faster R-CNN基础上发展而来的一种端到端的实例分割网络。它不仅能够检测图像中的物体，还能为每个检测到的对象实例生成精确的掩膜，即该对象在图像中的具体区域。通过引入额外的分割分支，Mask R-CNN实现了对每个检测结果的像素级别预测，显著提高了实例分割的准确性和效率。

### 2.2 算法结构

Mask R-CNN的基本结构包括：

1. **特征提取**：使用卷积神经网络（CNN）从输入图像中提取特征。
2. **候选框生成**：通过区域提案方法（如RPN）生成候选区域。
3. **检测与分割**：对每个候选区域进行检测，并预测该区域属于哪种类别以及生成掩膜，即该区域的像素级分割。

### 2.3 与Faster R-CNN的关系

Faster R-CNN首先提出了区域提案网络（Region Proposal Network，RPN）的概念，通过共享特征提取网络来同时进行检测和区域提案，大大减少了计算开销。而Mask R-CNN在此基础上增加了分割分支，专门负责预测掩膜，从而实现了同时进行检测和分割的目标。

## 3. 核心算法原理与具体操作步骤

### 3.1 算法原理概述

Mask R-CNN通过以下步骤实现：

1. **特征提取**：使用预训练的CNN提取输入图像的特征。
2. **区域提案**：通过RPN生成潜在的对象区域。
3. **检测与分割**：对每个区域提案进行检测，同时预测该区域的类别和掩膜。

### 3.2 算法步骤详解

#### 步骤一：特征提取

- 使用预训练的CNN（如ResNet）提取图像特征。

#### 步骤二：区域提案

- **RPN**：基于特征图生成区域提案。
- **候选区域缩放**：对生成的区域进行缩放和调整，以适应不同大小的对象。

#### 步骤三：检测与分割

- **分类**：预测每个区域提案属于特定类别的概率。
- **分割**：为每个检测到的对象实例生成掩膜，指示该对象在图像中的位置。

### 3.3 算法优缺点

#### 优点：

- 同时进行检测和分割，提高整体效率。
- 精确的像素级预测，提高分割准确性。
- 可以处理多种类别的实例分割。

#### 缺点：

- 计算成本较高，特别是分割分支。
- 对于复杂场景下的实例分割仍然具有挑战性。

### 3.4 算法应用领域

- 自动驾驶中的目标检测和道路划分。
- 医学影像分析中的病灶分割。
- 安防监控中的事件检测和行为分析。

## 4. 数学模型与公式

### 4.1 数学模型构建

Mask R-CNN的目标是预测每个实例的类别、边界框和掩膜。设输入为图像$I$，输出为$y = (c, b, m)$，其中$c$为类别预测，$b$为边界框预测，$m$为掩膜预测。

### 4.2 公式推导过程

#### 分类预测

- 使用全连接层或卷积层对特征提取的结果进行分类预测。

#### 边界框预测

- 使用全连接层或卷积层对特征进行回归预测，得到边界框的坐标。

#### 掩膜预测

- 使用全连接层或卷积层对特征进行回归预测，得到掩膜的预测。

### 4.3 案例分析与讲解

#### 实验设置

- 使用COCO数据集进行训练和验证。
- 设置学习率、批大小和训练轮次。

#### 结果分析

- 绘制损失曲线，分析训练过程中的收敛情况。
- 计算mAP（平均精度）指标，评估检测和分割性能。

### 4.4 常见问题解答

- **如何处理小物体和密集区域的分割？**：增加特征图分辨率、使用更密集的区域提案、优化分割分支的参数。
- **如何减少计算开销？**：优化模型结构、使用轻量级网络、批量处理。
- **如何提高分割精度？**：增加训练数据、调整损失函数、优化分割分支参数。

## 5. 项目实践：代码实例和详细解释说明

### 5.1 开发环境搭建

#### 环境需求

- Python >= 3.7
- PyTorch >= 1.7
- CUDA >= 11.0
- torchvision >= 0.8

#### 安装指令

```bash
pip install torch torchvision
```

### 5.2 源代码详细实现

```python
import torch
from torch import nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

class MaskRCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torchvision.models.detection.backbone_utils.build_backbone(backbone)
        self.roi_heads = torchvision.models.detection.roi_heads.RoIHeads(
            box_out_channels=256,
            box_detections_per_img=100,
            box_score_thresh=0.5,
            box_nms_thresh=0.5,
            box_fg_iou_thresh=0.5,
            box_bg_iou_thresh=0.5,
            box_batch_size_per_image=32,
            box_positive_fraction=0.25,
            box_score_conversion='none',
            box_nms_threshs=None,
            box_iou_loss_type='giou',
            box_focal_alpha=0.25,
            box_focal_gamma=2.0,
            box_smooth_l1_beta=1.0,
            box_reg_loss_type='smooth_l1',
            box_reg_weight=1.0,
            box_cls_weight=1.0,
            box_cls_loss_type='focal',
            box_cls_focal_alpha=0.25,
            box_cls_focal_gamma=2.0,
            box_cls_smooth_l1_beta=0.0,
            box_cls_iou_loss_type='giou',
            box_cls_iou_weight=1.0,
            box_cls_iou_smooth_l1_beta=0.0,
            box_cls_focal_iou_alpha=0.25,
            box_cls_focal_iou_gamma=2.0,
            box_cls_focal_iou_smooth_l1_beta=0.0,
            box_cls_iou_focal_alpha=0.25,
            box_cls_iou_focal_gamma=2.0,
            box_cls_iou_focal_smooth_l1_beta=0.0,
            box_cls_iou_iou_loss_type='giou',
            box_cls_iou_iou_weight=1.0,
            box_cls_iou_iou_smooth_l1_beta=0.0,
            box_cls_iou_focal_iou_alpha=0.25,
            box_cls_iou_focal_iou_gamma=2.0,
            box_cls_iou_focal_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_alpha=0.25,
            box_cls_iou_focal_iou_gamma=2.0,
            box_cls_iou_focal_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_alpha=0.25,
            box_cls_iou_focal_iou_focal_gamma=2.0,
            box_cls_iou_iou_focal_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_loss_type='giou',
            box_cls_iou_iou_focal_iou_iou_weight=1.0,
            box_cls_iou_iou_focal_iou_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_focal_alpha=0.25,
            box_cls_iou_iou_focal_iou_focal_gamma=2.0,
            box_cls_iou_iou_focal_iou_focal_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_iou_loss_type='giou',
            box_cls_iou_iou_focal_iou_iou_iou_weight=1.0,
            box_cls_iou_iou_focal_iou_iou_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_iou_focal_alpha=0.25,
            box_cls_iou_iou_focal_iou_focal_iou_alpha=0.25,
            box_cls_iou_iou_focal_iou_focal_iou_gamma=2.0,
            box_cls_iou_iou_focal_iou_focal_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_iou_focal_alpha=0.25,
            box_cls_iou_iou_focal_iou_focal_iou_focal_gamma=2.0,
            box_cls_iou_iou_focal_iou_focal_iou_iou_loss_type='giou',
            box_cls_iou_iou_focal_iou_iou_iou_iou_weight=1.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_iou_focal_alpha=0.25,
            box_cls_iou_iou_focal_iou_focal_iou_focal_gamma=2.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_loss_type='giou',
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_weight=1.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_iou_focal_iou_alpha=0.25,
            box_cls_iou_iou_focal_iou_focal_iou_iou_focal_gamma=2.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_loss_type='giou',
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_weight=1.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_iou_focal_iou_iou_alpha=0.25,
            box_cls_iou_iou_focal_iou_focal_iou_iou_focal_iou_gamma=2.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_loss_type='giou',
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_weight=1.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_iou_focal_iou_iou_iou_alpha=0.25,
            box_cls_iou_iou_focal_iou_focal_iou_iou_focal_iou_iou_gamma=2.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_loss_type='giou',
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_weight=1.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_iou_focal_iou_iou_iou_iou_alpha=0.25,
            box_cls_iou_iou_focal_iou_focal_iou_iou_focal_iou_iou_iou_gamma=2.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_loss_type='giou',
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_weight=1.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_iou_focal_iou_iou_iou_iou_iou_alpha=0.25,
            box_cls_iou_iou_focal_iou_focal_iou_iou_focal_iou_iou_iou_iou_gamma=2.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_loss_type='giou',
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_weight=1.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_iou_focal_iou_iou_iou_iou_iou_iou_alpha=0.25,
            box_cls_iou_iou_focal_iou_focal_iou_iou_focal_iou_iou_iou_iou_iou_gamma=2.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_loss_type='giou',
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_weight=1.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_alpha=0.25,
            box_cls_iou_iou_focal_iou_focal_iou_iou_focal_iou_iou_iou_iou_iou_iou_gamma=2.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_loss_type='giou',
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_weight=1.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_alpha=0.25,
            box_cls_iou_iou_focal_iou_focal_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_gamma=2.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_loss_type='giou',
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_weight=1.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_alpha=0.25,
            box_cls_iou_iou_focal_iou_focal_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_gamma=2.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_loss_type='giou',
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_weight=1.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_alpha=0.25,
            box_cls_iou_iou_focal_iou_focal_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_gamma=2.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_loss_type='giou',
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_weight=1.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_alpha=0.25,
            box_cls_iou_iou_focal_iou_focal_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_gamma=2.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_loss_type='giou',
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_weight=1.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_alpha=0.25,
            box_cls_iou_iou_focal_iou_focal_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_gamma=2.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_loss_type='giou',
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_weight=1.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_alpha=0.25,
            box_cls_iou_iou_focal_iou_focal_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_gamma=2.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_loss_type='giou',
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_weight=1.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_alpha=0.25,
            box_cls_iou_iou_focal_iou_focal_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_gamma=2.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_loss_type='giou',
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_weight=1.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_alpha=0.25,
            box_cls_iou_iou_focal_iou_focal_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_gamma=2.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_loss_type='giou',
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_weight=1.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_alpha=0.25,
            box_cls_iou_iou_focal_iou_focal_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_gamma=2.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_loss_type='giou',
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_weight=1.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_alpha=0.25,
            box_cls_iou_iou_focal_iou_focal_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_gamma=2.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_loss_type='giou',
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_weight=1.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_alpha=0.25,
            box_cls_iou_iou_focal_iou_focal_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_gamma=2.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_loss_type='giou',
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_weight=1.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_alpha=0.25,
            box_cls_iou_iou_focal_iou_focal_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_gamma=2.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_loss_type='giou',
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_weight=1.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_alpha=0.25,
            box_cls_iou_iou_focal_iou_focal_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_gamma=2.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_loss_type='giou',
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_weight=1.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_alpha=0.25,
            box_cls_iou_iou_focal_iou_focal_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_gamma=2.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_loss_type='giou',
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_weight=1.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_alpha=0.25,
            box_cls_iou_iou_focal_iou_focal_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_gamma=2.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_loss_type='giou',
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_weight=1.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_alpha=0.25,
            box_cls_iou_iou_focal_iou_focal_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_gamma=2.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_loss_type='giou',
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_weight=1.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_alpha=0.25,
            box_cls_iou_iou_focal_iou_focal_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_gamma=2.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_loss_type='giou',
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_weight=1.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_alpha=0.25,
            box_cls_iou_iou_focal_iou_focal_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_gamma=2.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_loss_type='giou',
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_weight=1.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_alpha=0.25,
            box_cls_iou_iou_focal_iou_focal_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_gamma=2.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_loss_type='giou',
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_weight=1.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_alpha=0.25,
            box_cls_iou_iou_focal_iou_focal_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_gamma=2.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_loss_type='giou',
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_weight=1.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_alpha=0.25,
            box_cls_iou_iou_focal_iou_focal_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_gamma=2.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_loss_type='giou',
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_weight=1.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_alpha=0.25,
            box_cls_iou_iou_focal_iou_focal_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_gamma=2.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_loss_type='giou',
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_weight=1.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_smooth_l1_beta=0.0,
            box_cls_iou_iou_focal_iou_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_alpha=0.25,
            box_cls_iou_iou_focal_iou_focal_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_gamma=2.0,
            box_cls_iou_iou_focal_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_iou_loss_type='giou',
            box_cls_i