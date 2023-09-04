
作者：禅与计算机程序设计艺术                    

# 1.简介
  

PyTorch是由Facebook AI Research（FAIR）基于Python语言开发的一款开源深度学习框架。PyTorch可以说是当前最火的深度学习框架之一了。主要有以下特点：

1. GPU加速计算：PyTorch可以利用GPU进行高效的计算运算，提升训练、推断等任务的执行速度；

2. 动态计算图：PyTorch基于动态计算图，将神经网络模型表示为一种拓扑结构图，使得计算图可以根据输入数据进行实时修改，并提供自动微分功能；

3. 模块化设计：PyTorch对各个模块都进行了高度的模块化设计，能够实现不同层次的功能组合；

4. 灵活的数据处理：PyTorch提供了丰富的数据处理函数接口，包括图像处理、文本处理、音频处理等；

5. 轻量级API：PyTorch提供了轻量级的API接口，使得开发人员可以使用简单方便的方式快速上手；

6. 可移植性强：PyTorch的可移植性非常好，可以在Linux、Windows、MacOS等多种平台运行；

总的来说，PyTorch是一个强大的开源深度学习框架，具有广泛的应用场景。本文就从2019年7月份发布的1.0版本开始，带领大家快速上手PyTorch。首先让我们来看一下PyTorch的安装和环境配置吧！

# 2. 安装及环境配置
## 2.1 安装
### 2.1.1 Python版本要求
PyTorch目前支持Python 2.7，3.5，3.6和3.7四种版本。如果没有特殊需求，强烈建议使用最新版本的Python。

### 2.1.2 Anaconda安装方式
PyTorch可以通过Anaconda集成环境进行安装。Anaconda是一个开源的Python发行版本，包含了conda、pip和其他一些包管理工具。


然后在命令提示符或终端中输入以下命令安装PyTorch：
```
conda install pytorch torchvision -c pytorch
```
其中-c选项指定channel，默认为pytorch，即从PyTorch官方镜像源获取。如果需要从国内源获取，则需要设置清华源或者中科大源：
```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/menpo/
conda config --set show_channel_urls yes
```
然后再重新运行`conda install pytorch torchvision -c pytorch`，即可从中科大源获取安装包安装。

由于网络原因，第一次安装可能比较慢，因此需要耐心等待。成功安装后，在命令提示符或终端中输入`python`命令进入交互式命令行界面，输入`import torch`命令验证是否安装成功。如果出现`ImportError: No module named 'torch'`错误，代表环境中没有安装PyTorch。

### 2.1.3 源码安装方式

解压源码包，进入根目录，运行`python setup.py install`。如果系统中没有Python环境，则需要先安装Python，推荐使用Anaconda。

## 2.2 配置环境变量
为了方便起见，我们可以把安装路径下的bin文件夹和lib文件夹加入到环境变量中，这样就可以省略掉绝对路径。

**Windows下:** 在系统变量Path下追加安装路径bin目录，如：C:\Program Files\Anaconda3；

**Linux下:** 执行命令`export PATH=/path/to/anaconda3/bin:$PATH`将Anaconda安装路径下的bin文件夹加入PATH环境变量，如：`export PATH=/home/user/anaconda3/bin:$PATH`。

为了验证是否配置成功，可以打开命令行窗口，输入`which python`或`whereis python`查看Python安装位置。如果返回类似于`/usr/bin/python`、`~/anaconda3/bin/python`之类的路径信息，则表示配置成功。