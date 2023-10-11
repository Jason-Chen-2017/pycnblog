
作者：禅与计算机程序设计艺术                    

# 1.背景介绍



YOLO(You Look Only Once)是一个目标检测模型，其名字含义就是只看一遍。它由两部分组成，第一部分是卷积神经网络（CNN），第二部分是预测模块（Detection Module）。其主要优点如下：

1. 实时性：YOLO可以实时的对视频图像进行目标检测。相对于其他复杂的模型，YOLO可以在每秒处理几百帧图片，而且它的速度非常快。

2. 速度快：YOLO的速度快之处在于，它并没有使用复杂的特征提取方法或多种锚框。它的主干网络结构是基于ResNet-50，精简而不失精确性。并且使用两次降采样来获取特征图。

3. 准确率高：YOLO的准确率相当高，仅有轻微的欠拟合问题。在COCO数据集上，YOLOv4达到了37.9%的mAP。

4. 可扩展性：YOLOv4还可以方便的被迁移到其他任务上，比如行人检测、车辆检测等。

本篇文章将带领读者从头实现一个轻量级目标检测YOLOv4，包括了YOLOv4的各个组件原理、功能实现、性能评估、测试结果和未来方向等内容。

# 2.核心概念与联系
## 2.1 YOLOv4相关术语
YOLOv4的创新点很多，这里简单介绍一下YOLOv4相关术语。

1. Anchor Boxes：YOLOv4使用了一种名叫Anchor Box的特殊检测框，与真实框不重叠，可用于帮助网络学习更好的分类。不同尺寸和宽高比的Anchor Box会对应着不同的分辨率和感受野，从而有助于网络学习全局的信息。

2. NMS：Non-Maximum Suppression，在YOLOv4中用来过滤掉置信度较低的边界框，保证每个目标都有一个对应的检测框。

3. IoU：Intersection over Union，用于衡量两个框的重叠程度。

4. Multi-Scale Training：YOLOv4同时训练多个尺度的数据增强方法，从而增加模型的鲁棒性。

5. Batch Normalization：Batch Normalization是一种正则化的方法，可以在每个隐藏层之前加上，以减少梯度消失和梯度爆炸。

6. Dropout：Dropout是一种正则化的方法，可以防止过拟合，并通过丢弃一定比例的权重实现。

7. Focal Loss：Focal Loss是另一种代价函数，用来处理样本不均衡的问题。

8. Cosine Annealing Scheduler：Cosine Annealing Scheduler是一种学习率衰减策略，可以有效的避免模型陷入局部最优。

9. One-Stage Detector：One-Stage Detector指的是不需要使用Region Proposal Network（RPN）来预先选定候选区域。YOLOv4就是这种类型。

## 2.2 模型结构
YOLOv4由以下几个模块构成：

1. Backbone CNN：是一个ResNet-50，用于抽取图像的特征。

2. Neck：即上采样模块，包括一个1×1卷积和三个3×3卷积，它用于减少特征图的空间分辨率并扩张通道数，使得后面的输出层具有更多的通道。

3. Head：包括一个1×1卷积、三个3×3卷积、三个线性层，用于预测类别和边界框坐标。


YOLOv4的输出大小是7×7，其通道数为$C_b\times 3$，其中$C_b$表示边界框的数量。对于单个像素点，其预测结果有四个值：$(p_0, b_x, b_y, b_w, b_h)$，分别代表概率、边界框中心横坐标、纵坐标、宽度、高度。

其中，$p_0$是该位置是否包含物体的概率；$b_{x}, b_{y}$是边界框中心的坐标；$b_{w}, b_{h}$是边界框的宽度和高度。

## 2.3 数据集
本文选择了COCO数据集来训练我们的模型。COCO数据集共有80个类别、25万张训练图片、2万张验证图片、413万个标注框。COCO数据集主要用于目标检测、图像分割、人脸检测等任务。

# 3.核心算法原理和具体操作步骤以及数学模型公式详细讲解
## 3.1 预处理
### 3.1.1 训练集数据增广
首先，对训练集进行数据增广，这样可以提升模型的泛化能力。常用的数据增广方式有随机裁剪、翻转、色彩抖动、等。除此外，还可以使用上下文信息，如图像中的固定物体。除了增强训练样本外，我们还需要计算新的anchors boxes。

### 3.1.2 测试集数据预处理
对测试集进行预处理的目的是为了使其与训练集一致，同时缩放到统一大小。通常的预处理包括归一化、中心裁剪、短边补齐、调整图像尺寸等。

## 3.2 网络搭建
YOLOv4网络的骨干是基于ResNet-50的，因此先对输入的图像进行Backbone CNN的特征提取。然后，把输出传入Neck模块，该模块包括一个1×1卷积和三个3×3卷积，将特征图的空间分辨率降低到原来的1/32，并扩充通道数，输出特征图的尺寸为7×7。接着，输出特征图传入Head模块，该模块包括一个1×1卷积、三个3×3卷积和三个线性层，用于预测类别和边界框坐标。


### 3.2.1 Backbone CNN
Backbone CNN是一个ResNet-50结构，其作用是提取图像的特征。由于ResNet-50采用了残差连接，因此具有良好的特征提取能力。

### 3.2.2 Neck
Neck模块包括一个1×1卷积和三个3×3卷积。第一个1×1卷积用于降维，扩张通道数，之后三个3×3卷积用于提取特征。在YOLOv4中，Neck的输出为7×7，通道数为$256\times 3$。

### 3.2.3 Head
Head模块包括一个1×1卷积、三个3×3卷积和三个线性层。第一个1×1卷积用于降维，扩张通道数，之后三个3×3卷积用于提取特征，提取的特征有三个，用于预测类别和边界框坐标。每个anchor box的输出为$N_a\times (C+5)$，其中$C$表示类别数量。最终输出为$S\times S\times (C+\frac{1}{2})\times N_a$，其中$S$是网格大小，默认为7。

## 3.3 Loss Function and Regularization
YOLOv4的损失函数由两个部分组成：置信度损失函数和定位损失函数。

### 3.3.1 概率损失函数
置信度损失函数用来判断当前位置是否有物体。其计算方法如下：

$$L_{\text {conf}}=-\left[\sum_{i=0}^{N} \sum_{j=0}^{\text {C } -1} y_{ij}\left(p_{ij}-\text { obj}_j\right)+\lambda\left(\sum_{i=0}^{N} \sum_{j=0}^{\text { C } -1}(1-y_{ij}) p_{ij}\right)\right] / N$$

其中，$N$是anchor box的数量，$\text{C}$是类别数量，$y_{ij}=1$表示第i个anchor box包含物体，$y_{ij}=0$表示第i个anchor box不包含物体；$p_{ij}$是第i个anchor box属于第j类的置信度，$obj_j$是第j类的背景置信度阈值。$\lambda$是平衡系数。

### 3.3.2 定位损失函数
定位损失函数用来计算边界框的中心坐标和尺寸。其计算方法如下：

$$L_{\text {coord }}=\sum_{i=0}^{N} \sum_{j=0}^{\text { C } -1} L_{\text {coord } j}^{i}$$

其中，$L_{\text {coord } j}^{i}$表示第i个anchor box关于第j类的坐标的回归损失。

最后，YOLOv4的总损失函数为：

$$\begin{array}{l}{\mathcal{L}_{total}=\alpha \cdot L_{\text {conf }}+\beta \cdot L_{\text {coord }}} \\ {\text { with }\alpha,\beta>0}\end{array}$$

其中，$\alpha$和$\beta$是平衡系数。

## 3.4 训练过程
### 3.4.1 预热期
在训练初期，YOLOv4权重初始化参数较小，可能导致无法有效收敛。因此，YOLOv4采用预热期的方式，先对整个模型进行冻结，不更新权重，待权重更新完成后再解冻训练模型。预热期一般为10k步。

### 3.4.2 解耦骨干网络
由于YOLOv4的Backbone CNN是一个深层的网络，因此容易过拟合。所以，YOLOv4提出了一个解耦骨干网络的思路，将backbone CNN提取到的特征图作为预测网络的输入。这样做的好处是在训练初期，可以防止网络过早地被深层特征所主导，从而快速收敛到局部最优。在训练过程中，backbone CNN的参数是固定的，只有预测网络的参数才会进行更新。

### 3.4.3 多尺度训练
YOLOv4采用了Multi-Scale training的方式，即训练多个尺度的数据增强方法。这样做的原因是YOLOv4采用了anchor box机制， anchor box 的面积大小不同，尺度也不同。如果仅用一幅图像进行训练，那么势必造成对象很小或者很大的物体无法被检测出来。但是，通过多尺度的数据增强方法，可以解决这一问题。

### 3.4.4 预测网络
YOLOv4中的预测网络包括三个网络层：1. 1×1卷积，2. 3×3卷积，3. 3线性层。预测网络的作用是根据anchor box预测类别及其偏移量。首先，通过1×1卷积将输入特征图降维。然后，对降维后的特征图应用3×3卷积，提取特征。最后，通过3线性层预测类别及其偏移量，输出形式为$(S\times S\times (C+\frac{1}{2})\times N_a)$。

### 3.4.5 损失函数
YOLOv4的损失函数包括置信度损失函数和定位损失函数。置信度损失函数用来判断当前位置是否有物体，定位损失函数用来计算边界框的中心坐标和尺寸。除此外，还有一项正则化损失函数，用来控制网络的复杂度。

### 3.4.6 学习率策略
YOLOv4采用了Cosine Annealing Scheduler作为学习率策略。主要原因是YOLOv4使用了预热期，在此期间，YOLOv4权重参数较小，因此学习率应随着训练进行变大。同时，Cosine Annealing Scheduler能够帮助模型逐渐恢复到初始的学习速率，并稳定到局部最优。

### 3.4.7 Data Augmentation
YOLOv4使用的Data Augmentation技巧主要有：

1. Flip augmentation: 对图像做水平翻转，可以减少过拟合。
2. Color augmentation: 在RGB空间里添加随机噪声，有利于模型鲁棒性。
3. Mosaic augmentation: 将图像拼接成不同比例的小图像，混合在一起送入网络，可以增加更多的训练数据。

## 3.5 推理过程
首先，对测试集的图像进行预处理，包括归一化、中心裁剪、短边补齐、调整图像尺寸等。然后，通过预测网络对图像进行预测，得到输出结果。最后，对输出结果进行后处理，包括非极大值抑制、门限筛选、解码得到最终结果。

# 4.具体代码实例和详细解释说明
## 4.1 模块定义
首先，导入必要的库。
```python
import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
```
然后定义网络结构：
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # backbone
        self.base = resnet()

        # neck
        self.conv1 = conv_bn(256, 512, kernel_size=1, stride=1)
        self.conv2 = conv_bn(512, 256, kernel_size=3, stride=1)
        self.conv3 = conv_bn(256, 512, kernel_size=1, stride=1)
        self.conv4 = conv_bn(512, 256, kernel_size=3, stride=1)
        self.conv5 = conv_bn(256, 512, kernel_size=1, stride=1)

        # head
        self.conv6 = conv_bn(512, 128, kernel_size=1, stride=1)
        self.conv7 = conv_bn(128, 256, kernel_size=3, stride=1)
        self.conv8 = conv_bn(256, 128, kernel_size=1, stride=1)
        self.conv9 = conv_bn(128, 256, kernel_size=3, stride=1)

    def forward(self, x):
        # backbone
        x = self.base(x)
        
        # neck
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        route = self.conv4(x)
        x = self.conv5(route)

        # head
        output = self.conv6(x)
        output = self.conv7(output)
        output = self.conv8(output)
        output = self.conv9(output)

        return [route, output]

def yolo(num_classes, anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                             [59, 119], [116, 90], [156, 198], [373, 326]], input_size=(416, 416)):
    model = Net()
    for param in model.parameters():
        param.requires_grad = False
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    return model
```
这里，我们定义了一个`Net`类，继承自`nn.Module`，用于构建YOLOv4网络的骨干。网络结构包含`backbone`、`neck`、`head`。`backbone`是一个ResNet-50结构，`neck`是一个串联的`ConvBN`结构，`head`是一个串联的`ConvBN`结构。网络结构定义完毕。

接着，定义模型中的卷积层和BatchNorm层，如下所示：
```python
class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x
    
def conv_bn(in_channels, out_channels, kernel_size, stride, padding=0, groups=1):
    block = nn.Sequential(
        ConvBN(in_channels, out_channels, kernel_size, stride, padding, groups),
    )
    return block
```
`ConvBN`类用于构建卷积层和BatchNorm层，`conv_bn()`函数用于构建`ConvBN`块。

最后，定义`yolo()`函数，用于构建YOLOv4网络。
```python
model = yolo(num_classes)
```
调用`yolo()`函数，即可得到训练模型。

## 4.2 数据集准备
数据集准备包括两个方面：

1. 获取数据集，包括训练集和验证集，从原始数据集获取图片及其标签，转换成`PIL Image`对象。
2. 创建训练集的数据加载器，包含训练数据增广。
3. 创建验证集的数据加载器，包含验证数据增广。

这里，我们以COCO数据集为例，演示如何准备数据集。首先，安装PyTorch中用于处理COCO数据的库`pycocotools`。
```python
!pip install pycocotools
```
然后，下载数据集。
```python
!wget http://images.cocodataset.org/zips/train2017.zip
!unzip train2017.zip
```
COCO数据集的标注文件存放在`annotations`文件夹下，我们只需下载其中`instances_train2017.json`文件。
```python
!mkdir data
!cp annotations/instances_train2017.json./data/coco.json
```
然后，加载数据集。
```python
import json
import os
import cv2
from PIL import Image
from skimage import io, transform
import random
import torch
import torchvision
from torchvision.transforms import functional as TF
import matplotlib.pyplot as plt
import numpy as np

# 读取数据集
def get_annotation(ann_file):
    ann_list = []
    with open(ann_file,'r') as f:
        data = json.load(f)
        images = data['images']
        annotations = data['annotations']
        categories = data['categories']

        img_dict = {}
        cat_dict = {}
        for item in categories:
            id = item["id"]
            name = item["name"]
            cat_dict[str(id)] = str(name)
            
        for item in images:
            file_name = item['file_name']
            height = item['height']
            width = item['width']
            
            img_dict[str(item['id'])] = {'file_name':file_name,
                                          'height':height,
                                          'width':width}
            
        for item in annotations:
            bbox = item['bbox']
            category_id = item['category_id']
            image_id = item['image_id']

            ymin, xmin, ymax, xmax = bbox
            w = abs(xmax - xmin)
            h = abs(ymax - ymin)
            cx = xmin + w // 2
            cy = ymin + h // 2
            label = cat_dict[str(category_id)]
            
            try:
                ins = img_dict[str(image_id)].copy()
                
                ins['label'] = label
                ins['cx'] = int(round(cx))
                ins['cy'] = int(round(cy))
                ins['w'] = max(int(round(w)),1)
                ins['h'] = max(int(round(h)),1)
                
                ann_list.append(ins)
                
            except KeyError:
                pass
                
    print('Annotation list length:', len(ann_list))
    return ann_list

# 读取图像
def load_image(filename, mode='color'):
    """
    Load an image into the given mode
    """
    assert mode in ['grayscale', 'color'], "Unsupported image type"
    im = None
    if mode == 'grayscale':
        im = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    elif mode == 'color':
        im = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
    return im

# 显示图像
def show_image(im, figsize=(10, 10)):
    fig = plt.figure(figsize=figsize)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(im)

# 图像数据增广
class YoloTransform:
    def __init__(self, resize, mean, std):
        self.resize = resize
        self.mean = mean
        self.std = std
        
    def __call__(self, sample):
        img, insts = sample['img'], sample['insts']
        H, W = img.shape[:2]
        img = cv2.resize(img, tuple(reversed(self.resize)))

        cxywh_boxes = [(inst['cx']/W, inst['cy']/H, inst['w']/W, inst['h']/H) for inst in insts]
        labels = [cat_dict[inst['label']] for inst in insts]

        cxywh_boxes = np.asarray(cxywh_boxes).reshape(-1, 4)
        labels = np.asarray([cat_dict.index(lbl) for lbl in labels]).reshape(-1,)

        target = {}
        target['boxes'] = torch.as_tensor(cxywh_boxes, dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels, dtype=torch.int64)

        img = TF.to_tensor(img)/255.0
        img = TF.normalize(img, self.mean, self.std)

        return {"img": img, "target": target}
        
# 数据集
class CocoDataset(torch.utils.data.Dataset):
    def __init__(self, root, ann_file, transform=None):
        self.root = root
        self.transform = transform
        
        # 获取图像列表
        self.filenames = sorted(os.listdir(os.path.join(root, 'train2017')))
        self.ids = [int(fname[:-4]) for fname in self.filenames]
        
        # 获取标注信息
        self.anno_list = get_annotation(ann_file)
            
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        # 获取图像路径
        filepath = os.path.join(self.root, 'train2017', filename)
        
        # 读取图像
        img = load_image(filepath)
        
        # 获取标注信息
        anno_info = [info for info in self.anno_list if info['file_name']==filename][0]
        insts = [{'cx': anno_info['cx'],
                  'cy': anno_info['cy'],
                  'w': anno_info['w'],
                  'h': anno_info['h'],
                  'label': anno_info['label']} for _ in range(100)]
                
        # 数据增广
        if self.transform is not None:
            sample = {'img': img, 'insts': insts}
            sample = self.transform(sample)
            img, target = sample['img'], sample['target']
            
        return img, target
```
这里，我们定义了`CocoDataset`类，用于管理COCO数据集，包括读取图像、读取标注信息、数据增广、获取指定索引下的图像及其标注。

## 4.3 训练过程
训练过程包括三步：

1. 配置超参数，设置模型训练的超参数，如学习率、权重衰减等。
2. 初始化模型，加载预训练权重，设置优化器。
3. 训练模型，每隔一段时间（epoch）进行一次验证集的评估和保存模型。

首先，配置超参数。
```python
input_size = (416, 416)  # 网络输入大小
batch_size = 32         # mini-batch size
lr = 0.001             # 学习率
momentum = 0.9         # momentum
weight_decay = 5e-4    # weight decay
epochs = 30            # 训练轮数
save_dir = './models/' # 保存模型目录
```
然后，初始化模型，加载预训练权重，设置优化器。
```python
# 初始化模型
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('GPU number:', torch.cuda.device_count())
else:
    device = torch.device("cpu")

model = yolo(num_classes, anchors=[[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                                 [59, 119], [116, 90], [156, 198], [373, 326]])
model.to(device)

pretrained_weights = '/path/to/pretrained/weights'  # 预训练权重路径
state_dict = torch.load(pretrained_weights)['state_dict']
for k in list(state_dict.keys()):
    if re.search(r'^module\.', k):
        state_dict[k[7:]] = state_dict[k]
        del state_dict[k]
model.load_state_dict(state_dict, strict=False)
print('Pretrained weights loaded successfully.')

optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr,
                      momentum=momentum, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss().to(device)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs*len(train_loader))
```
这里，我们设置了一些超参数，如`input_size`、`batch_size`、`lr`、`momentum`、`weight_decay`等。我们初始化了模型，并载入了预训练权重，设置了优化器、损失函数、学习率调节器。

接着，编写训练代码，每隔一段时间（epoch）进行一次验证集的评估和保存模型。
```python
# 训练过程
best_loss = float('inf')
for epoch in range(epochs):
    loss_history = []
    
    # Train step
    model.train()
    for i, batch in enumerate(train_loader):
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)[1].view(-1, num_classes+5, input_size[0]/32, input_size[1]/32)
        loss = criterion(outputs[:,:-1,:,:]*outputs[:,[-1],:,:],targets['labels']) + \
               criterion(outputs[:,:-1,:,:]*outputs[:,[-1],:,:], targets['labels'].unsqueeze(1))/0.5
        loss += sum([torch.sqrt((para**2).sum()).to(device) for para in model.parameters()]) * 0.0005
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        loss_history.append(loss.item())
        
        if i % 100 == 0:
            avg_loss = sum(loss_history)/len(loss_history)
            print('[Epoch:%d/%d, iter:%d/%d] loss=%.5f' % 
                  (epoch+1, epochs, i+1, len(train_loader), avg_loss))
            writer.add_scalar('Train Loss', avg_loss, epoch*len(train_loader)+i+1)
    
    # Validate step
    model.eval()
    val_loss_history = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)[1].view(-1, num_classes+5, input_size[0]/32, input_size[1]/32)
            loss = criterion(outputs[:,:-1,:,:]*outputs[:,[-1],:,:], targets['labels'])
            loss_history.append(loss.item())
            
    avg_loss = sum(val_loss_history)/len(val_loss_history)
    print('[Epoch:%d/%d] Val Loss=%.5f' % 
          (epoch+1, epochs, avg_loss))
    writer.add_scalar('Val Loss', avg_loss, epoch+1)
    
    # Save best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        save_path = os.path.join(save_dir, '%s_%.5f.pth'%(model.__class__.__name__, best_loss))
        torch.save({
            'epoch': epoch+1,
           'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            }, save_path)
        print('Best Model Saved!')
```
这里，我们定义了训练过程。在每次迭代时，我们对训练集和验证集进行迭代，并计算损失函数。我们对学习率调节器进行更新。每次保存模型时，我们将最新模型与最佳模型进行比较，若最新模型效果更好，则替换最佳模型。

训练结束后，模型会保存在`./models/`目录下，名字为`Net_<loss>.pth`，其中`<loss>`表示该模型在验证集上的损失值。

## 4.4 推理过程
推理过程包括两步：

1. 使用训练好的模型对测试集的图像进行预测，得到输出结果。
2. 对输出结果进行后处理，得到最终结果。

首先，加载训练好的模型。
```python
checkpoint = torch.load('/path/to/saved/model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```
然后，对测试集的图像进行预测，得到输出结果。
```python
with torch.no_grad():
    outputs = model(inputs)[1].view(-1, num_classes+5, input_size[0]/32, input_size[1]/32)
```
这里，我们直接使用训练好的模型对测试集的图像进行预测。

接着，对输出结果进行后处理，得到最终结果。
```python
def postprocess(outputs, threshold=0.5, nms_iou=0.45):
    detections = []
    
    # Compute class probabilities
    scores, classes = torch.max(outputs[:, :-1, :, :], dim=-1)
    
    # Apply thresholding
    mask = scores > threshold
    scores = scores[mask]
    classes = classes[mask]
    wh = outputs[0, :, mask]
    
    # Convert bounding boxes to absolute coordinates
    xywh = xywh[:, mask]
    xywh[..., 0] *= grid_sz / input_size[0] 
    xywh[..., 1] *= grid_sz / input_size[1]
    xywh[..., 2] *= input_size[0] / 32
    xywh[..., 3] *= input_size[1] / 32
    
    centers = xywh[..., :2]
    sizes = xywh[..., 2:]
    
    top_left = centers - sizes/2
    bot_right = centers + sizes/2
    
    boxes = torch.stack((top_left[..., 0], top_left[..., 1],
                        bot_right[..., 0], bot_right[..., 1]), axis=-1)
    boxes /= scale_factor
    
    # Non-maximum suppression
    keep = torchvision.ops.nms(boxes, scores, iou_threshold=nms_iou)
    keep = keep[:keep_top_k]
    det_boxes = boxes[keep]
    det_scores = scores[keep]
    det_classes = classes[keep]
    
    for box, score, clss in zip(det_boxes, det_scores, det_classes):
        detections.append({'box':box.tolist(),
                          'score':score.item(),
                           'clss':clss.item()-1})
                           
    return detections

detections = postprocess(outputs)
```
`postprocess()`函数用于后处理，其中包括计算类别概率、阈值化、转换边界框坐标、非极大值抑制。

最终，得到最终结果。