
作者：禅与计算机程序设计艺术                    

# 1.简介
  

PyTorch是一个开源的深度学习框架，它由Facebook AI Research团队开发，支持多种机器学习任务，主要包括图像分类、视频分析、语言模型、音频处理等。该框架能够在GPU上进行高速计算。
本文介绍如何在系统中已安装了CUDA的情况下，通过pip命令行工具安装PyTorch。由于PyTorch并不是官方支持的Linux发行版中的默认包，因此如果用户没有将PyTorch添加到默认软件源中，则需要先安装下面的依赖项。
# 2.准备工作
（1）确认系统中是否已安装CUDA
可以使用以下命令检查系统中是否已经安装CUDA：
```
$ nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2019 NVIDIA Corporation
Built on Sun_Jul_28_19:07:16_PDT_2019
Cuda compilation tools, release 10.1, V10.1.243
```
如果系统中已经安装CUDA，则显示以上信息；否则提示需要先安装CUDA。

（2）配置Python环境变量
首先需要确保Python已经安装，然后配置PATH和PYTHONPATH环境变量，使得当前用户可以执行python命令和import相应的库。编辑或新建~/.bashrc文件，加入以下内容：
```
export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
export CUDA_HOME=/usr/local/cuda-10.1
export PKG_CONFIG_PATH=/usr/local/cuda-10.1/pkgconfig${PKG_CONFIG_PATH:+:${PKG_CONFIG_PATH}}
```
刷新bashrc文件：
```
$ source ~/.bashrc
```
（3）安装依赖项
为了安装PyTorch，需要先安装下面的依赖项：
- Python 3.x及以上版本
- numpy >= 1.16.6
- cffi >= 1.12.3
- pyyaml >= 5.1
- typing_extensions
- future

```
sudo add-apt-repository ppa:deadsnakes/ppa # 添加 deadsnakes PPA 以获取最新版本的 Python 3
sudo apt update && sudo apt upgrade # 更新软件源
sudo apt install python3.x python3.x-dev # 安装 Python 3.x 版本
sudo pip3 install numpy==1.16.6
sudo pip3 install https://download.pytorch.org/whl/cu101/torch-1.6.0%2Bcu101-cp37-cp37m-linux_x86_64.whl
```
其中torch-1.6.0+cu101对应PyTorch 1.6.0，cu101表示用的是CUDA 10.1。注意替换3.7为实际的Python版本号。运行上述命令后会自动安装依赖项，若提示找不到相关包，则可能是系统库版本不匹配，可通过指定版本安装。

# 结论
通过本文介绍的安装方法，可以成功在已安装了CUDA的系统中安装PyTorch。由于PyTorch并非官方支持的Linux发行版中的默认包，因此需要从源码编译安装。另外，安装PyTorch可能会遇到一些依赖项问题，请根据提示解决。最后，感谢您的阅读，欢迎您对我们的产品提出宝贵意见。