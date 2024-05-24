
作者：禅与计算机程序设计艺术                    

# 1.简介
  

视差（optical flow）是指物体在图像中运动所引起的光流场，通过计算两帧图像间位移信息、角度信息或速度信息等来描述这些运动，可以有效识别和跟踪目标。目前已经有多种基于视差的方法被提出，但都存在一些局限性。特别是在处理复杂场景下，很多现有的基于视差的方法在训练时难以适应新的数据集及处理新任务。例如，由于目标的运动不规则，有的目标在同一个位置上经过长时间的移动而另一些目标可能在短时间内迅速运动并出现分离甚至消失等，导致很多现有的基于视差的方法在训练时无法学习到有效的特征表示，使得检测性能较低。另一方面，有的现有方法对于学习到的特征没有很好的区分能力，对于检测不同类别的目标，有时候会产生较大的误报率。因此，作者提出了一种新的基于视差的目标检测方法——基于Optical Flow的Pedestrian Detection，旨在提升基于视差的方法在复杂环境下的性能。
# 2.基本概念
## 2.1 Optical Flow
Optical flow是指物体在图像中运动所引起的光流场，它是基于相机特性和传感器技术设计的，能够提供从当前像素点到其他像素点的光流矢量（flow vector），用来反映图像中的空间运动。根据光流方向的不同，分为垂直光流(vertical optical flow)和水平光流(horizontal optical flow)。
上图展示了垂直光流(vertical optical flow)，左侧矩形代表图像$I(x,y)$，右侧矩形代表图像$I(x+\Delta x, y+\Delta y)$，$\Delta x$代表偏移x轴方向上的距离，$\Delta y$代表偏移y轴方向上的距离，蓝色箭头的粗细代表了光流的方向和大小。而水平光流(horizontal optical flow)则相反。

## 2.2 Dense Optical Flow
Dense optical flow是指多个连续的像素点的光流情况，它通过计算在两个图像之间沿着各个方向进行的光流，所以它具有三维性质。其基本思路是通过图像金字塔的形式对图像进行预处理，每一层对应着图像某一小区域的光流信息，如图2.1所示。然后对每个像素点，分别在三层图像中搜索最近邻，得到的光流作为其最终结果。
图2.1 基于高斯金字塔预处理的dense optical flow。

## 2.3 Pedestrian Detection with Optical Flow
基于视差的方法往往采用检测边缘、提取特征、分类等方式，但是由于不同类别目标的光流变化可能相似，这就导致它们可能共享相同的特征，导致分类效果较差。基于视差的方法往往需要大量训练数据才能学习到有效的特征，而且对于目标的位置检测也不是太准确。另外，基于视差的方法对目标的形状判断较弱，容易受到场景复杂度的影响。为了解决以上问题，作者提出了一种基于视差的目标检测方法——基于Optical Flow的Pedestrian Detection，该方法利用相邻帧的光流信息来获取目标的位置信息和运动信息，并通过深度神经网络进行目标检测。具体流程如下：

1. 数据准备阶段: 从视频序列中提取固定长度的连续帧作为输入，将其分成若干个子序列。其中，第一段作为基准序列，用来产生目标流动的模拟轨迹。其余部分作为待测序列，待测序列与基准序列的光流应该保持一致。

2. 光流计算阶段: 对基准序列中的所有帧，分别计算其与后一帧之间的光流，并裁剪出流域(flow region)，再将剩余部分与该帧拼接。

3. 特征抽取阶段: 通过光流场求取目标的空间分布和方向分布。先对流域内所有像素点计算其梯度(gradient)，然后求取流域内的直方图，并通过聚类等方式获得目标的中心点。最后，对每个目标，抽取其在空间域中密集的区域作为候选区域。通过提取局部特征的方式，对候选区域进行特征学习，并存储在待检测的序列的数据库中。

4. 检测阶段: 对待测序列中的所有帧，使用深度卷积神经网络对其进行检测。首先，对输入帧中的流域进行光流估计，然后将其与数据库中的已知对象进行匹配。如果匹配成功，就获取该对象的位置信息，并将其与输入帧中的检测框进行比较，得到检测结果。

5. 评价阶段: 根据精度、召回率等指标对检测结果进行评价。

# 3.算法原理和具体操作步骤
## 3.1 数据集生成
数据集生成是该算法最重要的一个环节，因为它会直接决定着整个算法的精度。一般来说，目标检测算法的数据集都应满足以下几个条件：

1. 大量数量的数据: 在实际应用场景中，大量且真实的目标数据是十分重要的。尤其是在复杂的环境下，光流变化与目标运动的关系才是最重要的信息。

2. 充分的标注信息: 有些情况下，除了拥有大量数据外，还需要相应的标注信息，如目标类型、位置、大小等。

3. 良好的背景信息: 目标检测算法也需要考虑背景信息，比如阴影、斑马线等。在光流检测时，需要仔细考虑到背景信息。

本文提出的算法基于真实的视频数据进行测试，因而使用真实视频中的目标信息作为数据集。作者使用Baidu HiQ摄像头收集的高清视频作为数据集，原始数据集共有400多条高清视频，视频中的人行走行为均为匀速运动，平均帧率为30fps。作者按照如下的方式进行数据集的生成：

1. 将原始视频进行剪辑，每隔5秒取10帧，每条视频保留100条帧。这样可以保证视频中只有人行走时刻对应的帧被提取出来。

2. 使用Matlab自带的函数VideoRead函数读取每一条视频的帧，将其转化为灰度图，并进行图像大小的缩放，统一为500x300的大小。

3. 使用Flownet2进行光流估计，输入图像为500x300大小的灰度图，输出为光流图。计算光流图上的流场，并记录所有帧的光流数据。

4. 每条视频记录100条帧的数据，包括6个维度：光流场的两个坐标值、目标所在的列索引、目标所在的行索引、光流场的强度值、光流场的梯度值(角度信息)、是否目标区域的掩码值(标记目标区域为1，否则为0)。

5. 将所有的视频数据按顺序排好，并保存到文本文件中。

## 3.2 模型构建
模型结构如下图所示：


网络由四个模块组成，分别为光流编码器(Flow Encoder)、特征提取器(Feature Extractor)、目标检测器(Object Detector)和分类器(Classifier)。光流编码器用于对光流场进行编码，将它压缩成一系列可学习的特征。特征提取器通过学习特征，将其投射到特征空间。目标检测器是一个两层的卷积网络，它的作用是对输入帧中的对象进行定位，即确定目标的位置和大小。分类器是一个二元分类器，用于判断输入帧中的对象属于哪个类别。整个模型可以同时处理一段连续的光流序列，如图3.2所示。


### 3.2.1 光流编码器(Flow Encoder)
光流编码器是一个两层的卷积神经网络，由一个卷积层和一个池化层构成。卷积层的结构如下图所示：


其中，输入是光流图，输出维度为64。第二个卷积层的结构如下图所示：


其中，输入是第一次卷积之后的特征，输出维度为128。然后，把特征图缩减成一个特征向量，用于后面的特征提取。

### 3.2.2 特征提取器(Feature Extractor)
特征提取器是一个两层的卷积神经网络，用于从输入的帧中提取目标的特征，并投射到特征空间。网络的结构如下图所示：


其中，输入是目标区域的特征向量，输出为64维特征。第二个卷积层的结构如下图所示：


其中，输入是第一个卷积层后的特征向量，输出为128维特征。

### 3.2.3 目标检测器(Object Detector)
目标检测器是一个两层的卷积神经网络，用于对输入帧中的对象进行定位。网络的结构如下图所示：


其中，输入是光流编码器和特征提取器的输出，输出为四个值的回归结果，即目标的中心点和大小。

### 3.2.4 分类器(Classifier)
分类器是一个二元分类器，用于判断输入帧中的对象属于哪个类别。网络的结构如下图所示：


其中，输入是目标检测器的输出，输出为类别的预测概率。

## 3.3 Loss Function
作者用了一个Joint Learning的策略，在训练阶段，采用两个损失函数，即Reconstruction Loss和Classification Loss。Reconstruction Loss用于对光流图的预测结果进行编码，用以评价光流场的逼近程度。Classification Loss用于对分类器的预测结果进行编码，用以评价分类器的性能。Joint Learning的策略源自于一个观察结果——单独训练分类器和光流编码器是不够的，必须结合两种模型的参数，才能取得更好的性能。Joint Learning的损失函数表达式如下：

$$\mathcal{L}_{joint} = \alpha_{rec}\cdot L_{rec}(y,\hat{y}) + \beta_{cls}\cdot L_{cls}(t,o) $$

其中，$\alpha_{rec}$和$\beta_{cls}$为权重系数，$L_{rec}$和$L_{cls}$分别是Reconstruction Loss和Classification Loss。

### 3.3.1 Reconstruction Loss
Reconstruction Loss用于评价光流场的逼近程度。论文中采用的Reconstruction Loss是基于信息瓶颈理论的无监督交叉熵损失函数。假设输入图像为$x$，输出图像为$y$，则光流场为$(u,v)$，那么两者的重建误差为：

$$L_{rec}(y,\hat{y})=\frac{1}{nhw}||F(x)[u,v]+F(\hat{y})^{-1}[u,v]||^2_2$$

其中，$F$表示由$x$生成的光流场的函数，$F^{-1}$表示由$y$生成的光流场的逆函数，即$F^{-1}=F^\ast$。$[u,v]$表示光流场的两维坐标值，$n$, $h$, $w$分别表示batch size、高度、宽度。

### 3.3.2 Classification Loss
Classification Loss用于评价分类器的性能。分类器的输出是预测的目标类别的概率，标签是真实的目标类别。因此，分类Loss定义为交叉熵损失函数：

$$L_{cls}(t,o)=-\log o_{gt}^t$$

其中，$o_{gt}$是标签对应的one-hot向量，$t$表示真实的目标类别。

### 3.4 训练过程
训练过程由三个步骤构成：

1. Pretrain Step: 首先对光流编码器和分类器进行预训练，预训练期间只优化它们的网络参数。这一步有助于网络收敛更快地进入更困难的优化空间。

2. Joint Train Step: 在预训练的基础上，采用Joint Learning的策略，利用Reconstruction Loss和Classification Loss训练整个网络。优化目标是最小化总的损失函数：

   $$\mathcal{L}_{joint} = \alpha_{rec}\cdot L_{rec}(y,\hat{y}) + \beta_{cls}\cdot L_{cls}(t,o) $$
   
   其中，$t$为真实的目标类别，$o$为分类器预测的概率分布。Joint Training的目的是将特征提取器的权重更新为使分类器的预测分布接近$o_{gt}$。

3. Finetune Step: 在Joint Training的基础上，微调网络的前几层参数，主要是针对目标检测器进行微调。微调的目的是为了提升网络的泛化能力，避免模型在训练过程中过拟合。

# 4.具体代码实例和解释说明
## 4.1 数据加载和预处理
数据加载和预处理的代码如下所示：
```python
import cv2
from glob import glob
import numpy as np
import os 

def load_data():
    # train data path
    train_path = "PATH to training dataset"

    X_data = []
    y_data = []
    
    filelist = sorted([os.path.join(train_path, f) for f in os.listdir(train_path)])
        
    print("Loading {} files...".format(len(filelist)))
            
    # load and preprocess each video sequence
    for filename in filelist:
        cap = cv2.VideoCapture(filename)
        
        if not cap.isOpened():
            continue
            
        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        count = 0
        
        while True:
            
            ret, img = cap.read()

            if not ret or count == frameCount - 1:
                break
                
            u, v, mask = calculate_flow(np.expand_dims(img, axis=-1), np.expand_dims(img, axis=-1))
                
            sample = [normalize(img).transpose((2,0,1)), normalize(mask)]
            label = (int(filename[-5])-1)*frameCount + count + 1
                
            X_data.append(sample)
            y_data.append(label)
                
            count += 1
                
    return X_data, y_data
    
def calculate_flow(prev_frame, curr_frame):
    grayPrevFrame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    grayCurrFrame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
    # Calculate dense optical flow using FlowNet2
    cmd = "./flownet2 --model=FlowNet2 --input={} --output={}".format(grayPrevFrame, grayCurrFrame)
    os.system(cmd)
        
    # Read flow image generated by FlowNet2
    flow_image = cv2.imread('flownet2-pytorch_out.flo', cv2.IMREAD_UNCHANGED).astype('float')
        
    hsvImage = np.zeros_like(curr_frame)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

    hsvImage[...,1] = 255
    hsvImage[...,0] = ang*180/np.pi/2
    hsvImage[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        
    rgbImage = cv2.cvtColor(hsvImage, cv2.COLOR_HSV2RGB)
    bgrImage = cv2.cvtColor(rgbImage, cv2.COLOR_RGB2BGR)
    
    # Generate binary mask for pedestrains
    blur_bgr = cv2.GaussianBlur(bgrImage,(11,11),0)
    _,binary_bgr = cv2.threshold(blur_bgr,10,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    binary_bgr = cv2.morphologyEx(binary_bgr,cv2.MORPH_CLOSE,kernel)
    contours, _ = cv2.findContours(binary_bgr.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    
    masks = np.empty((*binary_bgr.shape[:2], len(contours)))
    
    for i, contour in enumerate(contours):
        xmin, ymin, w, h = cv2.boundingRect(contour)
        xmax = xmin + w
        ymax = ymin + h
        mask = np.zeros_like(binary_bgr)
        mask[ymin:ymax,xmin:xmax] = cv2.drawContours(mask,[contour],-1,255,-1)
        masks[:,:,i] = mask.astype('uint8')
    
    return flow[...,0], flow[...,1], masks
        
def normalize(x):
    """Normalize input images"""
    norm_x = (x / 255.).astype('float32')
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    norm_x -= mean
    norm_x /= std
    return norm_x

if __name__=="__main__":
    X_data, y_data = load_data()
    print(len(X_data))
```

## 4.2 模型构建
模型构建的代码如下所示：
```python
import torch
import torchvision.models as models
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # Load pre-trained ResNet model
        resnet18 = models.resnet18(pretrained=True)

        # Remove the last layer of classifier
        modules = list(resnet18.children())[:-1]
        self.encoder = nn.Sequential(*modules)

        # Create convolutional layers for feature extraction from ResNet encoder output
        self.conv1 = nn.Conv2d(in_channels=512, out_channels=128, kernel_size=(3,3), padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=128)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)

        self.conv2 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(3,3), padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)

        self.fc1 = nn.Linear(in_features=3136, out_features=1024)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.relu4 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.5)
        
        self.object_detector = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(3,3), padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=64, out_features=4),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=2048, out_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=2),
        )

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Set the learning rate of the object detector layers
        params_to_update = []
        update_params_names = ['conv1','bn1','relu1','pool1','conv2','bn2','relu2','pool2']
        for name,param in resnet18.named_parameters():
            if any(layer_name in name for layer_name in update_params_names):
                param.requires_grad = False
                
        for name, param in self.object_detector.named_parameters():
            if 'linear' not in name:
                param.requires_grad = True
                params_to_update.append(param)
            else:
                param.requires_grad = True
        
        optimizer = torch.optim.Adam(params_to_update, lr=0.0001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    def forward(self, inputs):
        batch_size = inputs['base'].shape[0]

        base = self.encoder(inputs['base'])
        feat = self.conv1(base)
        feat = self.bn1(feat)
        feat = self.relu1(feat)
        feat = self.pool1(feat)

        feat = self.conv2(feat)
        feat = self.bn2(feat)
        feat = self.relu2(feat)
        feat = self.pool2(feat)

        feat = feat.view(batch_size, -1)

        feat = self.fc1(feat)
        feat = self.relu3(feat)
        feat = self.dropout1(feat)

        feat = self.fc2(feat)
        feat = self.relu4(feat)
        feat = self.dropout2(feat)

        pred = self.object_detector(feat)
        prob = self.classifier(feat)

        return {'pred': pred, 'prob': prob}
```

## 4.3 Joint Training
Joint Training的主函数如下：
```python
import torch
import torch.optim as optim
import torch.utils.data as Data

BATCH_SIZE = 64
EPOCHS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Build network
model = Net().to(device)

# Load training data
trainset = Dataset(X_train, y_train)
trainloader = DataLoader(dataset=trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)

for epoch in range(EPOCHS):
    running_loss = 0.0
    total = 0
    correct = 0
    
    for idx, data in enumerate(trainloader):
        inputs, labels = data[0].to(device), data[1].to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        class_loss = criterion(outputs['prob'], labels)
        reg_loss = torch.mean((torch.abs(outputs['pred'][..., :2]-labels[:, :, 2:])).sum()/torch.tensor(reg_factor))
        joint_loss = class_loss + reg_loss
        
        joint_loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(outputs['prob'], 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        running_loss += joint_loss.item() * BATCH_SIZE
        
    print('[%d/%d] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
          (epoch + 1, EPOCHS, running_loss / len(trainloader.dataset), 100. * float(correct) / total, correct, total))
```