
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 1.1 文章概述
目标检测（Object Detection）一直是一个计算机视觉领域重要且具有挑战性的问题。近几年来，卷积神经网络（CNNs）在图像分类、目标检测等任务上都取得了令人瞩目的成就。然而，对于某些特定场景下的目标检测问题，CNN 模型往往无法得到理想的效果。比如，当我们需要检测小目标时，很少有模型能够达到SOTA水平。另外，很多复杂场景下表现出来的检测问题也比较难处理。例如，一些行人穿过障碍物后仍然可以保持距离甚至跟随的行为非常困难。本文将介绍一种新的目标检测模型——CornerNet。它与最先进的基于 Anchor 的检测器（如 SSD 和 YOLOv3）相比，CornerNet 有以下优点：
- CornerNet 可以解决小目标检测、更具尺度感知的检测器、更高精度的性能以及针对目标类别不均衡问题的鲁棒性；
- CornerNet 在各种检测数据集（如 PASCAL VOC 和 COCO）上的精度超过 state-of-the-art 方法；
- CornerNet 设计简单、速度快、占用内存小。因此，它可以在资源受限的设备上快速部署。


## 1.2 相关工作
为了提升目标检测的准确率，已经提出了许多有效的方法。其中，基于 Anchor 的检测方法（如 SSD 和 YOLOv3）通常采用不同的特征图对不同大小的物体进行检测。这些方法大多采用设置不同的 Anchor Boxes 来检测不同大小的物体，从而提高检测精度。但是，这些方法只能检测固定数量的种类的目标，并且对目标类别的分布缺乏鲁棒性。因此，对于较难识别的物体，它们的检测能力可能不足。另一方面，对于同一个类别的多个目标，这些方法往往会产生重复的检测结果。最后，训练过程耗费时间长，并需要大量的标注数据，这是限制其推广范围的因素。

最近，CornerNet 模型被提出来作为替代方案。它与最先进的基于 Anchor 的检测器相比，主要有以下优点：
- 使用多角度关键点（Corner）的思路解决了小目标检测的问题；
- 提出了一个新的损失函数 CornerNet Loss，该函数可以更好地刻画边界框与关键点之间的关系；
- 利用边界框回归中的特征图对不同尺寸的物体进行检测，避免了固定的 Anchor Boxes 设置；
- 通过消除严重错误或不相关的预测框，提高了检测的鲁棒性；
- 可微分，使得计算梯度更加容易，并可在线训练。

然而，当前的 CornerNet 仅仅局限于目标检测任务中，没有考虑到其它的任务，比如分割、实例分割以及增强学习。因此，本文着重讨论了 CornerNet 对小目标检测、更具尺度感知的检测器、更高精度的性能以及针对目标类别不均衡问题的鲁棒性。除此之外，还阐述了 CornerNet 的相关理论基础、开源代码实现以及未来研究方向。

## 2.核心概念及术语说明
### 2.1 什么是 CornerNet？
CornerNet 是一种基于多角度关键点（Corner）的目标检测模型，通过引入关键点辅助定位的方式，解决小目标检测问题。它由两个部分组成：首先，它借鉴了 Faster RCNN 中的区域建议网络 (Region Proposal Network) 技术，生成候选区域并利用 RoI Pooling 池化层进行特征提取；其次，它结合了 Heatmap 相似度函数和 Cornerness-Aware Regression Loss 函数，生成准确的预测框。整个系统的结构如下所示：
其中，Corners 是 CornerNet 中使用的关键点集合，包括左上、右上、右下、左下四个顶点。Corners 通过旋转角度和平移位置两种方式进行描述，最终用于预测对象边界框及其目标分类。网络的输入输出分别是：输入图片 x ，输出特征图 f 。输出特征图中，每个像素点对应了一个 Corners 的集合，因此特征图的通道数就是 Corners 的个数。接下来，我们将详细介绍 CornerNet 的关键组件。

### 2.2 Region Proposal Network （RPN）
首先，我们来看 RPN（Region Proposal Network）。RPN 是一种轻量级的目标检测框架，可以产生候选区域 (Region proposal)。不同于其他检测方法，如 SSD 或 Faster R-CNN，它不需要使用 anchor box 来进行检测，而是直接使用任意形状和大小的候选区域。RPN 根据滑动窗口（类似于传统的 sliding window 检测方法）来生成候选区域，并在计算时忽略与 ground truth 框 (ground truth bounding box) 重叠的区域。RPN 会生成一定数量的锚框 (Anchor boxes)，每个锚框都会与一个 ground truth 绑定。然后，通过非极大值抑制 (Non Maximum Suppression) 操作，移除一些冗余的锚框，保留那些与 ground truth 高度重合的锚框。之后，可以使用预训练的卷积神经网络或全连接层对这些锚框进行分类和回归。
如上图所示，RPN 会产生一系列的候选区域，其中一些区域可能会与 ground truth 框重叠，也有可能与其它候选区域有重叠。接着，我们再来看一下 CornerNet 的整体结构。

### 2.3 Corner Detection Module（CM）
CornerNet 中的 CM （Corner Detection Module）用来产生 corner point（也称作 corners） 。它将候选区域 (Regions of Interest, RoIs) 输入到 CNN 网络中，得到相应的 feature map 。然后，它采用预定义的模板（templates）对 feature map 上的每个像素点进行定位，即判断它属于哪个 corner point 。对于 corner detection 模块来说，其输出是一个四维 tensor ，它给出了每个像素点在图像坐标系中的相对坐标和相对角度。如下图所示：
如上图所示，CornerNet 中的 CM 接收到一个输入，其中包含候选区域 RoI 和对应的 CNN 特征图。CM 通过一个三层的卷积网络进行特征提取，它将候选区域 (RoI) 输入到 CNN 中，获得了对应的 feature map 。CM 将特征图上每个位置的像素映射到空间坐标系中的点 P （其中 x，y 分别表示特征图中相应的横轴坐标和纵轴坐标），同时也确定了 corner points （通过预定义的模板）。然后，CM 生成了一个四维张量，每一维分别代表了坐标位置及角度信息，这样就完成了对 corner points 的描述。

### 2.4 CornerNet Loss Function
在 CornerNet 里，其 loss function （也称作 objective function ）包含两部分，第一部分为 Heatmap Similarity Loss ，第二部分为 Cornerness Aware Regression Loss 。Heatmap Similarity Loss 的作用是鼓励 Corners 生成热力图（heatmap），并且保证了 heatmap 的全局一致性。而 Cornerness Aware Regression Loss 的目的是在保证 Corners 的坐标精度的同时，保障其 Corners 的一致性。这一部分的公式如下所示：
$$ L_{hm} = \frac{1}{K_t}\sum^K_{k=1}L_{\text{HM}}(\hat{y}_k, y_k), $$
$$ L_{off} = \frac{1}{C_t}\sum^{C_t}_{i=1}\sum^{K_t}_{j=1}[\lambda_{ij}(x-\hat{x})^2+\lambda_{ij}(\theta-\hat{\theta})^2+\lambda_{ij}^2(p_i-\hat{p}_i)^2], $$
其中 $ K_t$ 为真实 target boxes 的个数，$ C_t$ 为真实 corner points 的个数。 $\lambda_{ij}$ 表示权重参数，用以控制不同 corner points 的影响。这里的 $ p_i$ 是第 i 个 corner points 的置信度。$\hat{p}_i$ 则是在 CornerNet 网络训练过程中根据真实位置计算得到的修正后的置信度。 $\hat{y}_k$ 和 $y_k$ 分别表示真实的分类得分和修正后的分类得分，如采用 softmax 作为激活函数。$\hat{x}$, $\hat{\theta}$ 分别表示根据 CornerNet 修正后的坐标位置和角度信息。$L_{\text{HM}}$ 为热力图 Similarity Loss 。

### 2.5 Non Maxima Suppression
在 CornerNet 的输出中，有许多重叠的预测框，因此需要做一次非极大值抑制 (Non Maxima Suppression) 操作。假设原始输入图片为 I，输出的预测框为 P，那么 Non Maxima Suppression 的流程如下：
1. 对 P 中每个预测框，计算其 IOU （Intersection over Union，交并比）和置信度（confidence score）。
2. 只留下置信度最大的预测框，排除掉 IOU 较大的预测框。
3. 如果剩余的预测框个数大于 N，则选择置信度最小的 N 个预测框，并丢弃置信度第二低的预测框。
4. 返回最终的 N 个预测框。
这样一来，虽然有许多重叠的预测框，但是只返回其中置信度最高的一个或几个预测框。

### 2.6 数据集和评价指标
与其它最新模型相比，CornerNet 与较早的基于 Anchor 的检测器相比，有着更高的精度。因此，为了评估 CornerNet 的效果，我们使用 PASCAL VOC 2007+2012 两个数据集，并评价指标有两种：AP（平均精度）和 AR（平均 recall）。前者计算的是多个IoU阈值下的平均精度，而后者计算的是所有ground truth的平均召回率。

## 3.核心算法原理和具体操作步骤以及数学公式讲解

### 3.1 CornerNet 组件
我们首先要明确几个概念：
1. 关键点 (corner): 每个关键点是一个锚点，它确定某个物体的四条边的端点。
2. 框 (bounding box): 框是矩形物体的一种二维表达形式。
3. 一张图片: 二维数组的形式，表示照片中的所有像素值。
4. 格子 (grid cell): 一个小的正方形框，其中一个单位格子代表了整个图片的一小部分。
5. anchor: 每个 anchor box 代表了一个特定的目标，即物体的特定类别。
6. 模板 (template): 一个四阶张量，用于代表物体边缘的位置和角度。

#### 3.1.1 Region Proposal Network （RPN）
首先，我们来看一下 Region Proposal Network。在 RPN 中，我们使用滑动窗口 (Sliding Window) 来生成候选区域。在每一个滑动窗口内，生成 9 个锚框，并与一个 ground truth 进行绑定。然后，利用预训练的网络对每个锚框进行分类和回归。如果一个锚框与 ground truth 框 IoU 大于阈值，则认为这个锚框是 positive sample，否则为 negative sample。然后，利用非极大值抑制 (Non Maximum Suppression) 操作，去掉一些重叠的负样本。最后，利用分类和回归损失函数，优化 RPN 的参数。

#### 3.1.2 Corner Detection Module（CM）
然后，我们来看一下 Corner Detection Module。在 CM 中，我们使用卷积神经网络 (CNN) 来对 candidate regions 进行特征提取。我们定义几个固定长度的模板，将候选区域分割成四个角点。然后，利用卷积网络将模板映射到相应的特征图上，来生成关于每个角点的坐标信息。

#### 3.1.3 Corners and Bounding Boxes
最后，我们将多个角点连接起来，构建多个边界框 (bounding boxes)。


如上图所示，当 CornerNet 成功生成边界框时，它既可以检测不同大小的物体，也可以检测小目标。所以，这个模型实际上是一种 multi-task detector，可以同时用于检测不同种类的物体。

### 3.2 CornerNet Loss Function
CornerNet 的损失函数包括两项：
1. Heatmap Similarity Loss。它使得 Corners 生成的 heatmap 更加一致，并且预测的类别更加正确。
2. Cornerness Aware Regression Loss。它在保证 Corners 预测精度的同时，帮助 Corners 保持稳定的形状。

##### Heatmap Similarity Loss
首先，我们来看一下 heatmap similarity loss。它鼓励 Corners 生成的热力图更加一致，预测的类别更加正确。具体地，对于每一个真实的 target box $ t_i$，它与生成的热力图 $ \hat{H}_i$ 的距离应该尽可能的小。



其中，$ p_i(u, v)$ 为第 i 个 target box 中是否包含格子 $(u, v)$。$ h(u, v)=1 $ 当且仅当 $ p_i(u, v)$ 为 true，表示该格子在第 i 个 box 中。$ H_i $ 为真实的热力图，它将处于不同的值，有利于区分不同类型的物体。$\hat{h}(u, v)$ 为 CornerNet 生成的热力图，它预测的类别，有利于估计目标的边界框。

我们的目标是希望 $h(u, v)$ 与 $\hat{h}(u, v)$ 尽可能的接近，即 $\|\|h- \hat{h}\|\|$ 小。一种简单的损失函数是 Smooth L1 Loss。它使得 $h$ 和 $\hat{h}$ 之间拟合得更加光滑。但是，Smooth L1 Loss 不能够直接用于预测目标类别，因此我们需要使用 cross entropy loss 或者 weighted cross entropy loss。

在 CornerNet 的源码里，默认使用 Weighted Cross Entropy Loss。Weighted Cross Entropy Loss 根据样本权重来计算 Cross Entropy Loss，这个权重的计算依赖于 GT 框和预测框的 iou，只有当 IoU > threshold 时才进行赋值。如下所示：
```python
    def forward(self, output, label, mask):
        hm_loss = self._sigmoid_cross_entropy_loss(output['hm'], label['hm'],
                                                    mask)
        wh_loss = smooth_l1_loss(output['wh'], label['reg_mask'],
                                 label['ind'], label['wh']) / 2.0
        off_loss = smooth_l1_loss(output['reg'], label['reg_mask'],
                                  label['ind'], label['reg']) / 2.0

        loss = {'hm': hm_loss, 'wh': wh_loss, 'off': off_loss}
        return loss

    @staticmethod
    def _sigmoid_cross_entropy_loss(pred, gt, mask):
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        neg_weights = torch.pow(1 - gt[neg_inds], 4)

        loss = 0

        pos_loss = torch.log(pred[pos_inds]) * torch.pow(1 - pred[pos_inds],
                                                       2) * pos_inds
        neg_loss = torch.log(1 - pred[neg_inds]) * torch.pow(pred[neg_inds],
                                                             2) * neg_weights

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss
```

##### Cornerness Aware Regression Loss
然后，我们来看一下 cornerness aware regression loss。它在保证 Corners 预测精度的同时，帮助 Corners 保持稳定的形状。

我们可以认为 cornerness 的定义为: “指向元素周围的元素有相似的几何结构”。这种观察结果可以解释为什么一般的回归方法对目标形状的估计往往不准确。

因为我们在 RPN 中生成的候选框往往不是完全的正方形，因此我们需要将 Corners 的预测结果转换为更合适的边界框。但是，这种转换对 Corners 的预测结果不一定有很好的改善，尤其是在存在较大的边界遮挡的情况下。因此，我们提出了 Cornerness Aware Regression Loss 来增强 Corners 的预测结果。

具体地，对于一个 target box $t_i$ 中的每个 Corners $\phi(u_i, v_i)$，我们定义 $\psi(u_i, v_i)$ 为该 Corners 是否为局部极值点。若 $\psi(u_i, v_i)$ 为 true，则表示该 Corners 在 $t_i$ 中为局部极值点。


显然，Corners 在某个方向上越是局部极值点，说明该 Corners 应保持更小的偏离。而 Cornerness 正是由该 Corners 在其他方向上的偏差决定的。因此，我们可以考虑使用 $\psi$ 作为一个权重，调整 Corners 的预测结果。

但目前的CornerNet 的实现方式只是简单地将 cornerness 视为独立的预测值，因此这种权重并不会起到额外的约束作用。因此，我们提出了一种新的损失函数 Cornerness Aware Regression Loss。

对于真实的 target box $t_i$ 中的每个 Corners $ (\phi(u_i, v_i), c_i)$，我们定义预测值为 $\hat{c}(u_i, v_i, k_i)$。其中，$k_i$ 用来指定 Corners 的类型。$\hat{c}$ 用来描述 Corners 的内聚程度。


其中，$c_i$ 表示真实的 Corners 的置信度。$\hat{c}(u_i, v_i, k_i)$ 表示根据 CornerNet 修正后的 Corners 的置信度，根据 CornerNet 输出的角度信息预测。如上图所示，对于边界处的 Corners，我们期望其置信度大于其他位置的 Corners。为了实现这种目标，我们提出了下面的损失函数：

$$
L_{cor}=\frac{1}{C_i}\sum^{C_i}_{j=1}[\psi_{ij}(w_j-w_\hat{j})(h_j-h_\hat{j})\cdot\left(k_j\cdot\hat{c}-\mu_k^2\right)+(\psi_{ij}(w_j-w_\hat{j}))^2]
$$

其中，$C_i$ 为 $t_i$ 中的 Corners 数量。$\psi_{ij}$ 为 $i$ 和 $j$ 号 Corners 的内核函数，表示 $i$ 号 Corners 是否落在边界中。$k_j$ 为 $j$ 号 Corners 的类型。$\hat{c}(u_i, v_i, k_j)$ 为 $j$ 号 Corners 的修正后的置信度。$\mu_k^2$ 为 Corners 类型的均值。

$$\psi_{ij}(r)(w_j-w_\hat{j})\cdot (h_j-h_\hat{j})\cdot (k_j\cdot \hat{c}(u_i, v_i, k_j)-\mu_k^2}$$

因此，该损失函数由四部分构成。第一个部分是 Corners 局部敏感函数，表示 Corners 在各个维度上的敏感度。第二个部分是 Corners 内聚度，表示 Corners 内聚度的高低。第三个部分是置信度偏差，表示预测置信度与真实置信度之间的差距。第四部分是 Corners 类型偏差，表示不同类型的 Corners 预测的置信度之间的差距。

目前，已有的 CornerNet 损失函数都是预测的中心点、宽高和置信度，因此只能考虑物体中心点的偏移。而 CornerNet Loss Function 还能更好地考虑物体边界上的点，因此可以更准确地预测物体边界。