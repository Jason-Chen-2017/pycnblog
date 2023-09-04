
作者：禅与计算机程序设计艺术                    

# 1.简介
  

目前大多数深度学习框架都提供了模型的保存和加载功能，例如TensorFlow、Keras等，而PyTorch也不例外。本文将基于PyTorch的官方文档，详细地讲述在PyTorch中加载已经训练好的模型并进行预测的整个流程。
# 2.准备工作
首先需要准备一个已经训练好的PyTorch模型（可以是自己训练的模型或者从别处下载的预训练模型）。假设我们已经有一个名为“model”的PyTorch模型，并且它已经经过了至少一次的训练，其权重参数保存在文件“weights.pth”。
# 3.加载模型
加载模型的方式主要有两种，一种是从本地加载模型的参数文件，另一种是直接加载预训练的模型。
## 从本地加载模型的参数文件
如果已经训练好的模型的权重参数保存在单独的文件里，那么可以通过下面的方式直接加载到PyTorch模型中：

```python
import torch

# 创建一个新的空白模型
model = MyModel()

# 加载模型参数
checkpoint = torch.load('weights.pth')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['optimizer'])
epoch = checkpoint['epoch']
```

这里创建了一个新的空白模型`MyModel`，然后用`torch.load()`方法从文件`weights.pth`中读取模型的权重参数。接着通过`model.load_state_dict()`方法加载模型参数，同样的方法也可以用来加载优化器的状态字典。最后得到的`epoch`变量可以用于记录当前训练到了哪个epoch。注意，当模型中含有自定义的网络结构时，需要相应修改上面的代码。
## 直接加载预训练的模型
另外一种加载模型的方式是直接从网上下载已经训练好的预训练模型，例如ImageNet上被广泛使用的ResNet-18模型。这样不需要再从头训练模型，而只需要加载预训练的模型权重即可。下面以加载ResNet-18模型为例：

```python
import torchvision.models as models

# 下载并加载预训练的ResNet-18模型
resnet18 = models.resnet18(pretrained=True)
```

这里用`torchvision.models`中的`resnet18()`函数直接加载了预训练的ResNet-18模型，此函数会自动下载模型权重并加载到模型参数中。
# 4.预测
通过上面两种方式，我们已经成功地加载了模型参数，现在可以对模型进行预测了。通常情况下，对于图像分类任务，一般有两类输入：一组图像；一组标签，表示每个图像的类别。因此，预测过程可能涉及到两个数据集之间的交叉验证，所以一般把图像划分成多个小批量进行预测，然后再合并结果。下面举一个具体的例子：

```python
from torchvision import transforms, datasets
import torch

# 使用图像增强（数据扩充）和归一化（标准化）
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# 加载测试集
testset = datasets.ImageFolder('/path/to/test', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False)

# 对测试集预测
def predict():
    resnet18.eval()   # 把模型设置为评估模式

    predictions = []    # 存放预测结果
    with torch.no_grad():
        for inputs, labels in testloader:
            outputs = resnet18(inputs).numpy()  # 用模型预测出图像对应的类别概率
            pred = np.argmax(outputs, axis=1)     # 根据概率最大的类别作为最终预测
            predictions += list(pred)            # 将预测结果加入列表

    return predictions

predictions = predict()
```

这里使用`datasets.ImageFolder`模块加载测试集，使用`torch.utils.data.DataLoader`模块构造了一个数据迭代器，用于在每次迭代时加载固定数量的数据。然后调用`predict()`函数对测试集进行预测。在`predict()`函数内部，先把模型设置为评估模式`resnet18.eval()`, 这是为了防止反向传播的发生，因为我们只是用模型做预测，不需要更新模型参数。然后遍历测试集的数据，逐步取出每张图像对应的输入数据和标签，送入模型中进行预测，得到输出结果后用`np.argmax()`求得概率最大的类别作为预测结果。最后返回预测结果列表。