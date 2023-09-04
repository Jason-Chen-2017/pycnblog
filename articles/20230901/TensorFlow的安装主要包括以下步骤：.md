
作者：禅与计算机程序设计艺术                    

# 1.简介
  

TensorFlow是一个开源机器学习库，可以用于深度学习、图像识别等领域。本文将从零开始介绍如何在Ubuntu系统上安装TensorFlow及其相关依赖包。
# 2.准备工作
## 2.1 安装CUDA
首先，我们需要下载并安装NVIDIA CUDA Toolkit，CUDA Toolkit包含了很多开发环境的组件和工具，包括C/C++编译器、GPU驱动程序、cuDNN(神经网络的深度学习加速库)、NCCL(用于并行计算的消息传递接口)，还有一些示例和教程程序。CUDA Toolkit一般配合英伟达驱动一起安装，所以如果没有安装显卡驱动，那么CUDA Toolkit也无法正常运行。CUDA Toolkit可以在官网下载安装包后，根据系统环境选择不同的安装方式。我这里用的是Ubuntu 16.04系统，通过apt-get命令安装CUDA Toolkit:
```bash
sudo apt-get install cuda
```
注意：如果没有安装Ubuntu内置驱动的话，可以参考官方文档安装最新版显卡驱动。

## 2.2 安装cuDNN
```bash
tar xvf libcudnn7_7.3.1.20-1+cuda10.0_amd64.deb #解压安装包
sudo dpkg -i libcudnn7*.deb #安装cuDNN
rm libcudnn7*.deb #删除解压安装包
```

## 2.3 安装TensorFlow
TensorFlow提供了两种安装方式：
1. 使用pip命令安装（推荐）：这个方法安装简单快捷，只需在终端执行一条命令即可安装TensorFlow。但是这种安装方式可能存在版本兼容性的问题，因此，如果你使用Python虚拟环境，建议使用第2种安装方式。
```bash
export PATH=/usr/local/cuda-10.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
./configure
```
根据你的硬件情况，相应的设置可能有所不同，比如是否开启CUDA支持，是否启用XLA优化等。然后执行下面的命令编译、安装：
```bash
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
sudo pip install /tmp/tensorflow_pkg/tensorflow-1.12.0-cp35-cp35m-linux_x86_64.whl
```

至此，TensorFlow的安装已经完成！