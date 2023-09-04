
作者：禅与计算机程序设计艺术                    

# 1.简介
  

 PyTorch是一个开源的、基于Python的科学计算平台，具有以下特性：
-   GPU支持：可以利用GPU加速神经网络运算，大幅提升计算速度；
-   深度学习API：提供丰富的机器学习组件，包括神经网络、优化器、损失函数等；
-   可扩展性：通过动态计算图和自动微分，PyTorch可以很容易地构建复杂的神经网络；
-   灵活性：支持多种编程语言，包括Python、C++、Java、Julia、Scala等；
-   便携性：可运行于Windows、Linux、macOS等平台；
-   生态系统：有强大的社区资源支持和丰富的第三方库支持；

为了方便安装和体验PyTorch，作者建议用户在自己的本地环境下安装Anaconda，它是开源数据科学处理工具包Conda的升级版，提供一个跨平台的Python开发环境。

本教程将带领大家完成在本地环境下的PyTorch安装和简单实践，主要包括以下内容：

1. 检查系统环境是否满足安装要求
2. 配置Anaconda
3. 创建并激活虚拟环境（Virtual Environment）
4. 安装PyTorch及其依赖库
5. 在Jupyter Notebook中测试安装结果

## 准备工作
首先，请确保您的电脑上已经安装了以下软件：

- Python >= 3.7 (64-bit)
- CUDA Toolkit >= 9.0 （推荐从NVIDIA官网下载）
- cuDNN SDK (>= v7.0 for CUDA 9.0）
- Anaconda 3 (64-bit)

如果还没有安装以上所需软件，请参考相应的安装指南进行安装。

## 检查系统环境是否满足安装要求
确认CUDA Toolkit 和 cuDNN SDK 是否安装成功，并按照以下命令检查系统环境是否满足安装要求：

```python
import torch

print(torch.__version__)
```

如果出现版本号信息则表明安装成功。

## 配置Anaconda
Anaconda是一个开源的数据分析平台，它包含了很多有用的工具包。由于Anaconda自带的包管理器pip在国内可能存在一些问题，我们选择用Anaconda安装pytorch。


配置Anaconda后，我们创建一个名为`pytorch_env`的环境，用来存放pytorch相关的包：

```python
conda create -n pytorch_env python=3.7 anaconda
source activate pytorch_env
```

激活环境后，就可以继续按照教程安装pytorch及其依赖库。

## 安装PyTorch及其依赖库
在激活环境后，我们可以使用conda或者pip命令安装pytorch。以下命令用于安装最新版的PyTorch及其依赖库：

```python
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
```

该命令会同时安装CPU版本和GPU版本的PyTorch。默认情况下，CPU版本被安装。如要安装GPU版本，请将`-c pytorch`改成`-c pytorch-nightly`。

安装完成后，运行如下代码测试安装结果：

```python
import torch
x = torch.rand(5, 3)
y = torch.rand(5, 3)
z = x + y
print(z)
```

如果输出了一个随机张量，则说明安装成功。

## 在Jupyter Notebook中测试安装结果
您也可以在Jupyter Notebook中测试安装结果。请先启动jupyter notebook命令：

```python
jupyter notebook
```

打开浏览器，访问http://localhost:8888，然后新建一个notebook文件。输入以下代码测试安装结果：

```python
import torch
x = torch.rand(5, 3)
y = torch.rand(5, 3)
z = x + y
print(z)
```

如果输出了一个随机张量，则说明安装成功。