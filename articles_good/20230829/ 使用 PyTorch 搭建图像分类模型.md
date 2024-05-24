
作者：禅与计算机程序设计艺术                    

# 1.简介
  

PyTorch是一个基于Python语言的开源机器学习库，用于构建和训练神经网络。它具备强大的GPU加速计算能力、自动求导和模型可移植性等优点。在本教程中，我们将用PyTorch搭建一个简单的图像分类模型，从头到尾完成整个过程。

## 1.背景介绍
图像分类就是给输入的图片预测其类别。常见的图像分类任务有多分类和二分类，如识别图片是否为猫或者狗、图片是否为衣服等；也有多标签分类，即识别出图片上所有的物体及其对应位置，如人脸识别。

人们需要对图像进行分类是因为我们每天都在接收、处理、分析大量的图像数据。然而，如果没有有效的方法对图像进行分类，那么我们的大量数据很可能会被无意义地分散掉，造成资源浪费和信息孤岛，最终导致我们无法从中获益。所以，如何提高图像分类的准确率，是一项非常重要的工作。

当前最流行的图像分类方法主要有卷积神经网络（CNN）、循环神经网络（RNN）和深度置信网络（DNN）。这些方法的特点是能够学习到图像的全局特征和局部关联，从而实现图像的高效分类。但是，通过构建复杂的模型结构和超参数优化，这些方法仍然存在一些瓶颈。因此，本文将介绍一种更简单但更有效的图像分类方法——VGGNet。

## 2.基本概念术语说明

### 2.1 Pytorch
PyTorch是一个基于Python的开源机器学习框架，可以实现高效的并行计算，并且可以轻松地与NumPy进行交互。PyTorch的主要特性包括：

- Tensor：类似于Numpy中的数组，但支持GPU加速。
- Autograd：可以自动执行反向传播算法，在训练神经网络时非常有用。
- nn module：用于构建神经网络。
- optim module：用于定义和优化神经网络的优化器。
- Cuda Tensors：可以使用GPU加速计算。
- DataLoader：加载和预处理数据集。

除了以上基础知识外，本文还会用到以下几个概念：

- 数据集：训练模型所需的数据集合，包括训练集、测试集、验证集等。
- 模型：由输入层、输出层和隐藏层构成，每层都是线性或非线性函数。
- 损失函数：用来衡量模型预测结果与真实值之间的差距。
- 优化器：用于更新模型参数，使得损失函数最小化。
- Epoch：一个完整的训练过程称为一个Epoch，每个Epoch完成一次迭代。
- Batch：每次训练时的样本批次，通常是32、64、128或256个样本。
- 分类：将图片划分为不同类别。
- 训练集：模型在训练过程中用于拟合的数据集。
- 测试集：模型在训练结束后用于评估模型性能的数据集。

## 3.核心算法原理和具体操作步骤以及数学公式讲解

### 3.1 VGGNet
VGGNet是2014年ImageNet比赛的冠军，它在准确率方面取得了卓越的成绩。它的设计思路是利用多个小卷积核代替全连接层，并堆叠深度多层的卷积网络，增加网络深度和宽度，提升性能。VGGNet有三种类型：

1. VGG16
2. VGG19
3. VGG-Face

本文采用VGG16作为模型结构，它的网络结构如下图所示：


VGG16模型由五个模块组成，分别是卷积层(convolutional layers)，池化层(pooling layers)，全连接层(fully connected layers)，dropout层(dropout layer)和softmax层(softmax layer)。

#### 3.1.1 卷积层
对于卷积层，首先是利用一个具有3×3大小的过滤器，跟随一个步长为1、零填充的卷积层。然后再施加两个3×3大小的过滤器，这两个滤波器共享同一个偏移值，跟随一个步长为1、零填充的卷积层。接着再施加三个3×3大小的过滤器，这三个滤波器共享同一个偏移值，跟随一个步长为1、零填充的卷积层。最后再施加三个3×3大小的过滤器，这三个滤波器共享同一个偏移值，跟随一个步长为1、零填充的卷积层。这样一来，最后的特征图就包含了输入图像的信息。

#### 3.1.2 池化层
对于池化层，它是对前一层特征图的采样过程，一般选择最大值池化层。池化层的大小一般为2×2，步长为2，也就是说它不仅仅保留输入特征图的部分信息，而且还减少了特征图的尺寸。

#### 3.1.3 全连接层
对于全连接层，它可以看作是两层的神经网络，第一层接收一维向量输入，第二层输出也是一维向量。它的权重矩阵是按照卷积的方式进行初始化的。全连接层一般都使用ReLU激活函数，并且设置较小的学习率。

#### 3.1.4 dropout层
dropout层是为了防止过拟合，它随机忽略某些神经元，使得神经网络在训练过程中能够集中注意力于其他神经元。

#### 3.1.5 softmax层
softmax层是指输入向量的值被限制在0到1之间，并且所有元素之和等于1，是一个归一化的概率分布。

### 3.2 数据准备

#### 3.2.1 数据集

对于图像分类，我们通常会使用多个不同的数据库来训练模型。这些数据库会提供不同的类型的图像数据，比如，CIFAR-10、CIFAR-100、MNIST、ImageNet等。这里，我们使用的是ImageNet数据集，它是包含1000类的共10万张图像的高质量数据集。

#### 3.2.2 数据加载

当准备好数据集之后，我们就可以将其加载进内存中。由于ImageNet数据集比较大，所以我们需要考虑到数据的加载速度。我们可以先随机选择一些图像进行测试，然后将剩下的图像分成训练集和验证集。

#### 3.2.3 数据预处理

在加载图像数据并将其转化为PyTorch能够识别的格式之前，我们需要对图像做一些预处理。我们通常需要缩放到固定大小，然后对图像进行中心裁剪。除此之外，还有许多其它预处理方法，比如颜色标准化、旋转变换等。

```python
import torchvision.transforms as transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
```

这里使用的预处理方式是从图像中随机裁剪一个224x224大小的区域，然后随机水平翻转图像。然后再将其转化为PyTorch能够识别的张量形式。另外，我们还对图像进行了颜色标准化。

### 3.3 模型构建

#### 3.3.1 创建模型

我们将创建的模型称为VGGNet，它由卷积层、池化层、全连接层和softmax层组成。

```python
import torch.nn as nn

class VGGNet(nn.Module):

    def __init__(self, num_classes=1000):
        super().__init__()

        self.features = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第二个卷积块
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第三个卷积块
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第四个卷积块
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第五个卷积块
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.shape[0], -1)
        logits = self.classifier(x)
        probas = nn.functional.softmax(logits, dim=-1)
        return logits, probas
```

这里我们实现了一个VGGNet模型，该模型包含五个卷积块，每个卷积块包含三个卷积层。每个卷积层的大小是3×3，激活函数是ReLU。然后使用平均池化层进行特征映射的降采样。

然后，我们创建了一个分类器，它包含三个全连接层。第一个全连接层接收512*7*7的输入，然后输出4096维的特征向量。第二个全连接层接收4096维的输入，然后输出4096维的特征向量。第三个全连接层接收4096维的输入，然后输出num_classes维的输出，即图像的类别。输出层采用Softmax函数，输出概率分布。

#### 3.3.2 参数初始化

对于卷积层和全连接层，我们都采用Xavier初始化法，均值为0，方差为$gain \times \frac{2}{\text{fan\_in} + \text{fan\_out}}$。其中，gain为0.01，fan\_in为输入单元数量，fan\_out为输出单元数量。

```python
for m in self.modules():
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)
```

对于池化层，我们直接采用默认的初始化方式。

### 3.4 训练模型

#### 3.4.1 配置训练参数

在训练模型之前，我们需要配置一些训练参数。例如，我们希望批量大小为128，学习率为0.01，权重衰减系数为0.0005，动量参数为0.9，共训练50个epoch。

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VGGNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
num_epochs = 50
batch_size = 128
```

#### 3.4.2 创建数据加载器

```python
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

train_dataset = ImageFolder('path/to/training/set', transform=train_transform)
val_dataset = ImageFolder('path/to/validation/set', transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
```

#### 3.4.3 训练模型

```python
best_acc = 0.0

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    scheduler.step()
    
    train_loss = 0.0
    model.train()
    for images, labels in tqdm(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs, _ = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()*images.size(0)
        
    train_loss /= len(train_loader.dataset)
    
    val_loss = 0.0
    val_acc = 0.0
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs, _ = model(images)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()*images.size(0)
            val_acc += torch.sum(preds == labels.data).double()/labels.size(0)
            
    val_loss /= len(val_loader.dataset)
    val_acc /= len(val_loader.dataset)
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), f'models/{args.name}.pth')
        
print(f"Best accuracy on validation set: {best_acc:.4f}")
```

在训练模型的时候，我们定义了一些训练策略。首先，我们使用步长为30、乘法因子为0.1的学习率下降策略。然后，我们使用SGD作为优化器，并设定了动量为0.9、权重衰减系数为0.0005。然后，我们打印每一个epoch的训练和验证误差、正确率和最佳正确率。最后，我们保存最佳模型的参数。

#### 3.4.4 测试模型

最后，我们使用测试集测试模型的性能。

```python
model.load_state_dict(torch.load(f'models/{args.name}.pth'))
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for data in test_loader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs, _ = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))
```

模型测试完毕之后，我们计算得到精度。

## 4.具体代码实例和解释说明

### 4.1 数据预处理

```python
import os
import random
from PIL import Image
from shutil import copyfile

def preprocess_image(src_dir, dest_dir):
    '''
    src_dir: directory containing raw image files to be preprocessed
    dest_dir: destination directory where processed image will be saved
    This function takes a directory containing raw images and saves them
    in another directory after performing some preprocessing operations such
    as resizing, center cropping, randomly rotating and flipping images. The 
    processed images are also split into training and testing sets by placing
    70% of the images into training folder and remaining 30% into testing folder.
    '''
    img_list = []
    dirs = sorted(os.listdir(src_dir))
    class_dirs = {}

    for d in dirs:
        path = os.path.join(dest_dir, d)
        os.mkdir(path)
        class_dirs[d] = {'train': [], 'test': []}

    for subdir, dirs, files in os.walk(src_dir):
        for file in files:
                continue

            filepath = os.path.join(subdir, file)
            filename = os.path.basename(filepath)

            try:
                im = Image.open(filepath)
                w, h = im.size

                if min(w, h) < 224 or max(w, h) > 448:
                    raise ValueError("Invalid dimensions")
                
                new_im = resize_and_center_crop(im, 224)
                new_filename = '_'.join([str(random.randint(0, 100000)), 
                                          str(random.randint(0, 100000)),
                
                dir_name = os.path.split(subdir)[-1]
                save_path = os.path.join(dest_dir, dir_name, new_filename)
                new_im.save(save_path)
                
                if dir_name!= "Testing":
                    class_dirs[dir_name]['train'].append(new_filename)
                else:
                    class_dirs[dir_name]['test'].append(new_filename)
                    
            except Exception as e:
                print(f"{filename}: {e}")
                
    ratio = int(len(class_dirs['Training']['train'])/(len(class_dirs['Training']['train']) + len(class_dirs['Training']['test'])))
    move_files('Training', class_dirs, ratio, src_dir, dest_dir+'/Training/')
    
def resize_and_center_crop(img, size):
    '''
    Resize an image to specified size and then center crop it to desired output size.
    If height is less than width, then adjust the output width accordingly. Otherwise,
    adjust the output height accordingly.
    '''
    w, h = img.size
    new_h = size
    new_w = round(size/h*w)
    
    resized_img = img.resize((new_w, new_h))
    left = round((new_w - size)/2)
    top = round((new_h - size)/2)
    right = left + size
    bottom = top + size
    
    cropped_img = resized_img.crop((left, top, right, bottom))
    
    return cropped_img
    
def move_files(folder, directories, ratio, src_dir, dst_dir):
    '''
    Move files from source directory to destination directory based on their distribution
    within each category. For example, we want to move only 70% of Training images into
    training folder while moving rest into testing folder.
    '''
    i = 0
    j = 0
    
    indices = list(range(len(directories[folder][ratio])))
    random.shuffle(indices)
    
    for index in indices:
        if i >= len(directories[folder]['train']):
            break
            
        name = directories[folder][ratio][index]
        src_file = os.path.join(src_dir+'/'+folder, name)
        dst_file = os.path.join(dst_dir, name)
        copyfile(src_file, dst_file)
        
        i += 1
        
    for index in indices[i:]:
        if j >= len(directories[folder]['test']):
            break
            
        name = directories[folder][ratio][index]
        src_file = os.path.join(src_dir+'/'+folder, name)
        dst_file = os.path.join(dst_dir[:-1]+'/Testing/', name)
        copyfile(src_file, dst_file)
        
        j += 1
```

### 4.2 模型训练与测试

```python
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from utils import preprocess_image


parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='../data', help='Dataset root.')
parser.add_argument('--workers', type=int, default=8, metavar='N', help='Number of workers.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay.')
parser.add_argument('--seed', type=int, default=1, help='Seed for reproducibility.')
parser.add_argument('--model', type=str, default='', help='Pretrained model checkpoint.')
parser.add_argument('--evaluate', action='store_true', help='Evaluate model on evaluation dataset instead of training dataset.')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Define preprocessing transformations
train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
train_data = ImageFolder(args.root+'/Train', transform=train_transform)
val_data = ImageFolder(args.root+'/Val', transform=test_transform)

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

# Create model
model = models.densenet161(pretrained=True)
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 2)
model = model.to(device)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay)
criterion = nn.CrossEntropyLoss()

# Load pretrained weights if provided
if args.model!= '':
    state_dict = torch.load(args.model)
    model.load_state_dict(state_dict)

# Train model or evaluate on eval set
if not args.evaluate:
    # Train model
    best_accuracy = 0.0
    epochs = 50
    patience = 10
    
    for epoch in range(1, epochs+1):
        train_loss = 0.0
        train_acc = 0.0
        model.train()
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            predictions = torch.argmax(outputs, axis=1)
            acc = sum(predictions==labels).item()/len(predictions)
            
            train_loss += loss.item()
            train_acc += acc
            
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        val_loss = 0.0
        val_acc = 0.0
        model.eval()
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                predictions = torch.argmax(outputs, axis=1)
                acc = sum(predictions==labels).item()/len(predictions)
                
                val_loss += loss.item()
                val_acc += acc
            
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        print(f"[Epoch {epoch}] Train Loss={train_loss}, Train Acc={train_acc}, Val Loss={val_loss}, Val Acc={val_acc}")
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), 'checkpoints/checkpoint'+str(epoch)+'.pt')
            count = 0
        else:
            count += 1
        
        if count >= patience:
            break
            
    print(f"Best Accuracy: {best_accuracy}")
    
else:
    # Evaluate on eval set
    checkpoint = torch.load('checkpoints/checkpoint'+str(n_epoch)+'.pt')
    model.load_state_dict(checkpoint)
    model.eval()
    
    y_pred = None
    y_true = None
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            predictions = torch.argmax(outputs, axis=1)
            
            if y_pred is None:
                y_pred = predictions
                y_true = labels
            else:
                y_pred = torch.cat((y_pred, predictions))
                y_true = torch.cat((y_true, labels))
            
    target_names = ['No Dementia', 'Mild Dementia', 'Moderate Dementia', 'Severe Dementia']
    report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
    conf_mat = confusion_matrix(y_true, y_pred)
    
    print("Classification Report:\n", report)
    print("\nConfusion Matrix:\n", conf_mat)
    
    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(conf_mat, cmap="YlGnBu")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.figure.colorbar(im, cax=cax)
    tick_marks = np.arange(len(target_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(target_names, rotation=45)
    ax.set_yticklabels(target_names)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    fmt = 'd'
    thresh = conf_mat.max() / 2.
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax.text(j, i, format(conf_mat[i, j], fmt), ha="center", va="center", color="white" if conf_mat[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
```

### 4.3 混淆矩阵可视化

```python
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# Initialize variables
target_names = ['No Dementia', 'Mild Dementia', 'Moderate Dementia', 'Severe Dementia']
conf_mat = [[3844,  100,   30,    0], 
            [ 124,  224,   26,    0], 
            [  50,   20,  351,    0], 
            [  15,    1,    0,   13]]

# Convert confusion matrix to DataFrame
df_cm = pd.DataFrame(conf_mat, columns=target_names, index=target_names)

# Plot heatmap using Seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(df_cm, annot=True, cmap="YlGnBu", vmin=0, vmax=None, square=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Actual Value')
plt.title('Confusion Matrix')
plt.show()
```