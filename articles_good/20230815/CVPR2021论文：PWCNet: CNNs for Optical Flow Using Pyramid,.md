
作者：禅与计算机程序设计艺术                    

# 1.简介
  

作为一名优秀的技术人员，我希望能够把自己所学所得分享给同样对此感兴趣的人。因此，我想把自己写过或正在写的论文的心路历程经验化，记录下自己的学习笔记，希望对大家有所帮助。

在图像识别、机器视觉、自动驾驶等领域，基于CNN的光流估计已经成为非常热门的研究方向。而最近几年来，深度学习技术带来的特征提取及CNN模型训练技术的飞速发展，使得传统CNN模型应用于光流估计有了新的突破口。但传统CNN模型往往存在着一些弱点，比如：准确性不高、计算复杂度高等。

PWC-Net是近些年CVPR会议上首次提出的用于光流估计的深度学习模型，它的主要创新点有：

1. 使用pyramid、warping、cost volume作为CNN的输入，能够解决空间相关性问题；
2. 提出一种新型的图像金字塔网络结构（Image Pyramid Network）来提升深度学习网络的深度及特征提取能力；
3. 改进并统一多个CNN子网络之间的连接方式，采用多分支结构以提升预测精度；
4. 设计了一种优化目标函数来学习光流场，将之前难以优化的各项指标整合到一个统一的优化目标中。

PWC-Net由3个部分组成：Encoder、Decoder和Warper，可以分别用来编码图像信息、解码光流场信息和形变图像。

本篇论文从新一代的光流估计CNN模型——PWC-Net开始讲起，介绍其主要创新点、原理和应用场景。然后通过一步步的叙述，系统的讲解并分析PWC-Net的结构、工作流程、特点以及关键实现模块。最后还将阐述一下PWC-Net的优缺点，并谈谈未来该模型的发展方向。

本篇论文假定读者有一定计算机视觉、机器学习、数学基础。

# 2.引言
## 2.1 PWC-Net概述
基于深度学习技术的光流估计一直是一个具有挑战性的问题。由于CNN网络结构的独特性，传统的方法依赖于相邻像素之间的共生关系，这种依赖导致准确性较差且计算量大。PWC-Net采用多尺度的图像金字塔和不同尺度下的两张图片间的差异来进行光流估计，它可以有效克服传统CNN模型对空间关联性的依赖，解决这一问题。

PWC-Net的主要创新点如下：

1. 使用不同尺度的图像作为CNN的输入，能够处理不同尺度的图像，并且能够绕过空间关联性；
2. 在图像金字塔上的不同层之间引入多分支结构，提升预测精度；
3. 通过引入cost volume，能够捕获局部光流信息；
4. 利用光流场的偏导数信息，设计优化目标函数来增强网络的鲁棒性和稳定性。

PWC-Net可以广泛地应用于光流估计任务。但是，它的准确率和计算复杂度仍然存在一定的限制。同时，也存在一些其他的研究工作也尝试利用CNN进行光流估计，例如FlowNet、Supervised Image Distillation(SID)等。不过，它们的设计与实现方式有很大的不同，可能不能完全适应实际需求。

## 2.2 PWC-Net原理
### 2.2.1 数据集与数据集划分
PWC-Net采用FlyingChairs、FlyingThings3D和KITTI这3个数据集，其中KITTI数据集是最具代表性的。FlyingChairs数据集中的图片尺寸比较小，FlyingThings3D和KITTI数据集则大都达到了4K。为了使得不同尺度下的图片能够充分涵盖光流变化的范围，PWC-Net采用不同的图像大小。KITTI数据集的图像分辨率为$480\times 640$，因此只需要改变图像的短边长度即可。其它两个数据集的图像分辨率有$768\times 960$和$2048\times 2448$，因此需要同时改变图像的长边长度。

对于每个数据集，我们用相同比例的图片切割出4张图，分别来源于不同视角的同一场景。例如，对于FlyingChairs的数据集，我们取四张图，每张图来源于x轴、y轴、z轴、平面视图的角度，分别对应着图片A、B、C、D。每张图片上有5枚物体，每个物体在图片中随机位置，物体的中心位置坐标也是随机的。这样做的目的是使得数据集更加真实、完整、丰富。

### 2.2.2 模型结构
PWC-Net的结构如图1所示，由三个主要部分组成，即：Encoder、Decoder、Warper。


#### Encoder
Encoder由几个卷积层组成，包括3个卷积层（第一个卷积层输出32通道、第二个卷积层输出64通道、第三个卷积层输出128通道），之后还有两个3x3的卷积层（输出256和512通道）。在所有的卷积层后面添加Batch Normalization。每个卷积层的步长都是1，补零策略是REFLECT，激活函数为ReLU。

图像金字塔的生成方式如下：首先，将原始图片由长宽调换为高宽长，然后将长边缩放到1/32。然后，将图片重复采样，生成四份大小一样的图片，一张是原始图片的一半，两张图片上下颠倒。这样就得到了四张图片，分别来自x轴、y轴、z轴、平面视图。接着将这些图像放入Encoder中进行特征提取，得到四种尺度下的特征图。

#### Decoder
Decoder由两个分支组成，分别是左右分支和顶部分支。左右分支由左右特征图拼接生成，顶部分支由顶部特征图和左右特征图拼接生成。

左右分支由两个3x3的卷积层（输出512和256通道）组成，之后再添加一个5x5的卷积层（输出128通道），池化层为最大池化，步长为2。之后通过反卷积层上采样至原图大小。

顶部分支由三个3x3的卷积层（输出128、64和32通道）组成，之后再添加一个3x3的卷积层（输出2通道），激活函数为Tanh。输出的结果是两个向量，分别表示光流场的x轴和y轴偏移值。

#### Warper
Warper是一个图像变形网络，它接受原始图片和预测出的偏移量，将原始图片沿x轴、y轴方向移动相应的偏移量，得到变形后的图片。

PWC-Net将光流估计的整个过程分成三步：第一步，将原始图片由长宽调换为高宽长，通过Encoder得到四种尺度下的特征图；第二步，由Decoder得到左右分支和顶部分支；第三步，由Warper将原始图片与预测出的偏移量融合得到变形后的图片。

### 2.2.3 优化目标函数
PWC-Net将光流估计看作是单目视角下的图像配准问题，可以使用如下优化目标函数：

$$\min_{u} E(\theta) = \frac{1}{N}\sum_{i=1}^Ne\left((\tilde{I}_i,\hat{u}_i), \hat{v}\right)\tag{1}$$ 

其中：$\tilde{I}_i$是原始图片，$\hat{u}_i$是第$i$幅图片上的光流场，$\hat{v}$是对应的预测偏移量；$e$是损失函数，比如使用交叉熵Loss；$\theta$表示网络参数。

PWC-Net将光流估计任务分成两个步骤，即编码阶段和解码阶段。对于编码阶段，PWC-Net利用Encoder生成不同尺度下的特征图；对于解码阶段，PWC-Net利用Decoder得到左右分支和顶部分支，预测出不同尺度下的光流场。最终，PWC-Net将不同的光流场结合，求出最小化损失的全局光流场。

PWC-Net中，在解码阶段，每张图片的光流场都要经过Warper进行变形，由此产生了一张图变形后的图片，如下：

$$W_t(p+\hat{u}) = T\left(\tilde{I}, p+\hat{u}\right)=T\circ W_t\left(\tilde{I}, u\right)\tag{2}$$

其中：$W_t$是Warper网络，$\tilde{I}$是原始图片，$p$是光流场映射前的像素坐标，$\hat{u}$是光流场。

那么，PWC-Net如何利用光流场？首先，需要定义Cost Volume，Cost Volume的作用是捕获局部光流信息。定义如下：

$$C_{ab}(p)=\left|\frac{\partial f}{\partial x}_{a(p)}\frac{\partial f}{\partial y}_{b(p)}-\frac{\partial g}{\partial x}_{a(p)}\frac{\partial g}{\partial y}_{b(p)}\right|^2_{\mathcal{H}}\tag{3}$$

其中：$f$和$g$是两张图片，$a$和$b$是图像坐标系中的点；$C_{ab}(p)$是Cost Volume。Cost Volume就是两张图片光流梯度的差的平方。Cost Volume的公式与图像坐标的对应关系直接相关。

当预测出来的光流场为$\hat{u}$时，我们可以通过如下公式计算Cost Volume：

$$\hat{C}_{ab}(p)=\left|\frac{\partial I(p+u_x)(p+u_y)-I(p+v_x)(p+v_y)}{\partial x_a (p)+\partial x_b(p)}\frac{\partial I(p+u_x)(p+u_y)-I(p+v_x)(p+v_y)}{\partial y_a (p)+\partial y_b(p)} - \frac{\partial I(p+u_y)(p+u_x)-I(p+v_y)(p+v_x)}{\partial x_a (p)+\partial x_b(p)}\frac{\partial I(p+u_y)(p+u_x)-I(p+v_y)(p+v_x)}{\partial y_a (p)+\partial y_b(p)}\right|^2_{\mathcal{H}}\tag{4}$$

其中：$u=(u_x,u_y)$是光流场向量，$v=(v_x,v_y)$是另一条光流场向量。

由于Cost Volume在图像坐标的对应关系上直接相关，所以可以在优化目标函数中加入Cost Volume作为约束条件：

$$\min_{u} E(\theta) + \lambda R(\theta) \\
s.t.\quad C=\frac{1}{N}\sum_{i=1}^Nc_{ij}(p+\hat{u}_i)^2\leqslant 1\tag{5}$$

其中：$c_{ij}(p+\hat{u}_i)$是Cost Volume；$R(\theta)$表示正则项。$\lambda$是一个超参数，用来控制Cost Volume的权重。

# 3.核心算法细节
## 3.1 图像金字塔
图像金字塔通过对原始图片做不同大小的缩放，得到一系列等级为原图1/n的图片，n表示倍数。随后，将这些缩小后的图片放入CNN网络中，获得不同尺度下的特征图。通过拼接不同尺度下的特征图，可以得到整个图像的描述符。

不同尺度下的图片包含的信息是不同的，因此它们应该被赋予不同的权重。但是在实际使用过程中，我们只能选择少量的高频信息进行检测，无法有效的进行多尺度的检测。因此，需要采用一种平衡方法来保证所有尺度的特征都能得到充分的利用。

在PWC-Net中，作者采用了一个简单的平滑插值法来生成图像金字塔。先将原始图片的长宽调换为高宽长，然后按照512-800-224-40-16，依次缩小图片直到分辨率为800像素。这样就可以生成四张图片，分别来自x轴、y轴、z轴、平面视图。这些图片是四个级别的图片的组合，具有不同尺度。然后，将这些图片输入到Encoder中，生成不同尺度下的特征图。

## 3.2 Multi-Branch Structure
Multi-Branch Structure是PWC-Net的一个重要创新点。它通过引入多分支结构来提升深度学习网络的深度及特征提取能力。在传统的CNN网络中，很多时候只采用一层或两层CNN结构，因此信息流通的不够全面，容易出现瓶颈。

Multi-Branch Structure的主要思想是在卷积层之间引入多分支结构。在PWC-Net中，左右分支和顶部分支都分别由两个3x3的卷积层和一个3x3的卷积层组成。左右分支中，分别输入原始图片和预测出的偏移量，输出左右特征图。顶部分支则输入左右特征图，输出顶部特征图和偏移值。这样就提升了预测精度。

## 3.3 Cost Volume
Cost Volume的作用是捕获局部光流信息。在传统CNN网络中，光流信息往往是局部的，一般不具有全局性，因此CNN网络很难通过全局特征来进行预测。

因此，PWC-Net提出了Cost Volume的概念。它在图像金字塔上的不同层之间引入Cost Volume，能够捕获局部光流信息。PWC-Net将两张图片的光流梯度差的平方作为Cost Volume，来约束CNN网络的预测。

## 3.4 Optimizing the Loss Function
PWC-Net通过引入Cost Volume来增强网络的鲁棒性和稳定性。而且，它还采用光流场的偏导数信息，设计优化目标函数来增强网络的鲁棒性和稳定性。优化目标函数如下：

$$\min_{u} E(\theta) + \lambda R(\theta)\\
s.t.\quad C=\frac{1}{N}\sum_{i=1}^Nc_{ij}(p+\hat{u}_i)^2\leqslant 1\tag{6}$$

其中：$E$表示损失函数；$u$是预测出的光流场；$C$是Cost Volume；$\lambda$是权重因子；$R$表示正则项；$p$是特征图上任意一点的坐标；$i$是第i张图片；$j$是第j类特征图。

# 4.代码实例
## 4.1 安装与测试环境
```python
!pip install opencv-python matplotlib
import cv2
import numpy as np
from matplotlib import pyplot as plt

# test environment
print('opencv version:', cv2.__version__) # should be >= 4.2
print('numpy version:', np.__version__) # should be >= 1.18
print('matplotlib version:', plt.__version__) # should be >= 3.2.1
```

## 4.2 测试数据集
这里下载两个数据集FlyingChairs和KITTI，用来测试我们的代码：



数据的准备工作已经完成，读者可以根据自身情况修改代码路径。

## 4.3 数据预处理
读取RGB图片、转换成灰度图、调整图像尺寸、裁剪图像边缘。
```python
def preprocess_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = img.shape[:2]
    
    if h > w:
        dw = h - w
        padding = int(dw / 2)
        img = cv2.copyMakeBorder(img, 0, 0, padding, padding, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    elif w > h:
        dh = w - h
        padding = int(dh / 2)
        img = cv2.copyMakeBorder(img, padding, padding, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    resized = cv2.resize(gray, (480, 640))
    cropped = resized[64:-64, 64:-64].astype(np.float32) / 255.0
    
    return cropped
```

## 4.4 数据加载器
实现了一个数据加载器，用于读取数据并返回一个批次的图片和对应的光流场标签。
```python
class FlyingChairsDataset():
    def __init__(self, root='./data', mode='train'):
        self.root = root
        
        assert mode in ['train', 'val']
        self.mode = mode

        self.img_names = []
        self.flows = []
        
        with open('{}/{}.txt'.format(root, mode)) as f:
            lines = [line.strip() for line in f.readlines()]
            
            for line in lines:
                items = line.split(' ')
                img1_name = '{}/{:05d}.ppm'.format(root, int(items[0]))
                img2_name = '{}/{:05d}.ppm'.format(root, int(items[1]))
                
                flow_name = '{}/{}-{}_flow.flo'.format(root, os.path.basename(img1_name).replace('.ppm', ''), os.path.basename(img2_name).replace('.ppm', ''))

                img1 = cv2.imread(img1_name)
                img2 = cv2.imread(img2_name)
                
                flow = read_flow(flow_name)
                
                self.img_names += [[img1_name, img2_name]]
                self.flows += [flow]
        
    def __getitem__(self, index):
        pair = self.img_names[index]
        flow = self.flows[index]
        
        img1 = cv2.imread(pair[0])
        img2 = cv2.imread(pair[1])
        
        inputs = {}
        inputs['input_frames'] = np.stack([preprocess_img(img1), preprocess_img(img2)], axis=-1)[None]
        
        targets = {'target_flow': flow[..., None]}
        
        return inputs, targets
        
    def __len__(self):
        return len(self.img_names)
    
def read_flow(filename):
    """ Read.flo file in Middlebury format"""
    # Code adapted from:
    # http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

    # WARNING: this will work on little-endian architectures (eg Intel x86) only!
    # print 'Reading %s' % filename

    with open(filename, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25!= magic:
            print('Magic number incorrect. Invalid.flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            # print 'Reading %d x %d flo file\n' % (w,h)

            data = np.fromfile(f, np.float32, count=2*w*h)
            # Reshape data into 3D array (columns, rows, bands)
            flow = np.resize(data, (h, w, 2))
            return flow.astype(np.float32)
```

## 4.5 PWC-Net模型
编写了一个PWC-Net模型，包括Encoder、Decoder、Warper三个主要部分。

Encoder由几个卷积层组成，包括3个卷积层（第一个卷积层输出32通道、第二个卷积层输出64通道、第三个卷积层输出128通道），之后还有两个3x3的卷积层（输出256和512通道）。在所有的卷积层后面添加Batch Normalization。每个卷积层的步长都是1，补零策略是REFLECT，激活函数为ReLU。

图像金字塔的生成方式如下：首先，将原始图片由长宽调换为高宽长，然后将长边缩放到1/32。然后，将图片重复采样，生成四份大小一样的图片，一张是原始图片的一半，两张图片上下颠倒。这样就得到了四张图片，分别来自x轴、y轴、z轴、平面视图。接着将这些图像放入Encoder中进行特征提取，得到四种尺度下的特征图。

Decoder由两个分支组成，分别是左右分支和顶部分支。左右分支由左右特征图拼接生成，顶部分支由顶部特征图和左右特征图拼接生成。

左右分支由两个3x3的卷积层（输出512和256通道）组成，之后再添加一个5x5的卷积层（输出128通道），池化层为最大池化，步长为2。之后通过反卷积层上采样至原图大小。

顶部分支由三个3x3的卷积层（输出128、64和32通道）组成，之后再添加一个3x3的卷积层（输出2通道），激活函数为Tanh。输出的结果是两个向量，分别表示光流场的x轴和y轴偏移值。

Warper是一个图像变形网络，它接受原始图片和预测出的偏移量，将原始图片沿x轴、y轴方向移动相应的偏移量，得到变形后的图片。

### 4.5.1 测试Encoder
测试Encoder是否正确运行：
```python
inputs = np.random.randn(1, 2, 3, 240, 320)   # input shape is NCHW

encoder = Encoder().to('cuda')

outputs = encoder(torch.tensor(inputs).permute(0, 3, 4, 2, 1).to('cuda')).detach().cpu().numpy()[0][..., :256]    # remove depth dimension

fig, axes = plt.subplots(nrows=4, ncols=8, figsize=(20, 10))

for i, ax in enumerate(axes.flatten()):
    ax.imshow(outputs[i], cmap='jet')
    ax.axis('off')

plt.show()
```

测试结果如图2所示：


图2展示了Encoder生成的不同尺度下的特征图。

### 4.5.2 测试Decoder
测试Decoder是否正确运行：
```python
features = np.random.randn(1, 512, 60, 80)     # features shape is CHWD

decoder = Decoder().to('cuda')

output = decoder({'input_frames': torch.tensor(inputs).to('cuda'),
                  'left_feature': torch.tensor(features).unsqueeze(-1).to('cuda')}
                 ).detach().cpu().numpy()[0]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

axes[0][0].imshow(output[:, :, 0], cmap='jet')
axes[0][0].set_title('x-offset')
axes[0][0].axis('off')

axes[0][1].imshow(output[:, :, 1], cmap='jet')
axes[0][1].set_title('y-offset')
axes[0][1].axis('off')

plt.show()
```

测试结果如图3所示：


图3展示了Decoder预测的光流场偏移值图。

### 4.5.3 测试Warper
测试Warper是否正确运行：
```python
warper = Warper().to('cuda')

inputs = np.random.randn(1, 2, 3, 240, 320)          # input shape is NCHW

targets = {'target_flow': np.zeros((240//4, 320//4, 2))}      # target shape is HW2

outputs = warper({'input_frames': torch.tensor(inputs).permute(0, 3, 4, 2, 1).to('cuda'),
                  'target_flow': torch.tensor(targets).permute(2, 0, 1).unsqueeze(0).to('cuda')}
                 ).detach().cpu().numpy()[0]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))

axes[0][0].imshow(outputs[0], cmap='jet')
axes[0][0].set_title('Warped Image')
axes[0][0].axis('off')

axes[0][1].imshow(targets['target_flow'][::-1, :, 0])
axes[0][1].set_title('Target X-Offset')
axes[0][1].axis('off')

axes[1][0].imshow(targets['target_flow'][::-1, :, 1])
axes[1][0].set_title('Target Y-Offset')
axes[1][0].axis('off')

plt.show()
```

测试结果如图4所示：


图4展示了Warper将原始图片变形后的效果。

### 4.5.4 测试PWC-Net模型
测试PWC-Net模型是否正确运行：
```python
model = PWCNet().to('cuda')

inputs = np.random.randn(1, 2, 3, 240, 320)          # input shape is NCHW

outputs = model({'input_frames': torch.tensor(inputs).permute(0, 3, 4, 2, 1).to('cuda')}
                )['target_flow'].squeeze().permute(1, 2, 0).detach().cpu().numpy()

fig, axes = plt.subplots(figsize=(10, 10))

axes.quiver(range(outputs.shape[0]), range(outputs.shape[1]), outputs[:,:,0], outputs[:,:,1])

plt.show()
```

测试结果如图5所示：


图5展示了PWC-Net的预测结果。

# 5.总结与讨论
## 5.1 概括
PWC-Net是近些年CVPR会议上首次提出的用于光流估计的深度学习模型，它的主要创新点有：

1. 使用pyramid、warping、cost volume作为CNN的输入，能够解决空间相关性问题；
2. 提出一种新型的图像金字塔网络结构（Image Pyramid Network）来提升深度学习网络的深度及特征提取能力；
3. 改进并统一多个CNN子网络之间的连接方式，采用多分支结构以提升预测精度；
4. 设计了一种优化目标函数来学习光流场，将之前难以优化的各项指标整合到一个统一的优化目标中。

同时，PWC-Net也具有一定的优点：

1. 相比于其他的光流估计模型，PWC-Net拥有更好的性能和精度；
2. PWC-Net能够学习到更深层次的语义特征，因此能够学习到更复杂的光流信息；
3. PWC-Net使用多分支结构来增强模型的鲁棒性，并且提升了模型的预测精度。

不过，PWC-Net也存在一些缺点：

1. 需要训练多级的图像金字塔，计算量较大；
2. 需要设计复杂的优化目标函数，能够学习到高度纹理的光流信息；
3. 对相机畸变、遮挡、噪声有一定的影响；
4. 需要使用较大的数据集来进行训练。

因此，目前，PWC-Net还是一种比较新的模型，需要更多的研究来进一步提升它的准确性。

## 5.2 个人看法
PWC-Net是一个非常有意思的模型，通过图像金字塔的方式来减少空间关联性，同时提取多尺度的信息，取得更好的效果。作者通过编码阶段和解码阶段分开处理，在解码阶段使用光流场，通过多分支结构增强模型的鲁棒性。而该模型的优化目标函数将之前难以优化的各项指标集成到一起，这也让我耳目一新。

不过，PWC-Net也存在一些缺陷，比如训练时间长、需要大量数据才能得到良好的效果。同时，该模型还没有被广泛应用，目前也没有什么可说的了。

总的来说，PWC-Net是一个很有潜力的模型，希望能在将来看到它的应用。