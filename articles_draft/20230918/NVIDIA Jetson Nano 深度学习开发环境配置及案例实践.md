
作者：禅与计算机程序设计艺术                    

# 1.简介
  

近几年随着移动终端的普及，深度学习在移动端的应用越来越火热。由于移动设备的计算资源限制，深度学习框架对计算性能的需求也越来越高。而NVIDIA推出了NVIDIA Jetson Nano平台，其具有低功耗、高性能、嵌入式系统等特点，可以满足用户对端侧机器学习的需求。作为一个具有自主知识产权的公司，NVIDIA一直以来致力于开放GPU编程接口，方便第三方开发者基于Jetson Nano构建自己的深度学习产品和服务。本文将以实例化的方式，展示如何在Jetson Nano上进行深度学习开发，并演示一些具体的案例。希望通过本文的分享，能够帮助更多的开发者了解如何利用Jetson Nano平台进行深度学习开发，并快速地在端侧设备上实现自己的目标。
## Jetson Nano 是什么？
NVIDIA Jetson Nano是一款迷你的、自主可控的AI计算模块，由英伟达（Nvidia）自主研发。Jetson Nano搭载了2GB GDDR5X内存、512MB eMMC存储和尺寸紧凑的大小。并且采用ARM Cortex-A57处理器，具备强大的CPU性能。它的价格在399美元左右，相较于其他的高端主流笔记本电脑更加值得关注。除了帮助用户在移动端深度学习应用外，Jetson Nano还拥有完善的硬件支持和驱动库，可以很好地配合不同传感器或外设进行集成。
## 目录结构
在本教程中，我们将通过三个章节来详细阐述如何在Jetson Nano上进行深度学习开发。

在每个章节的最后，还有相关资源和下载链接供读者参考。
## 环境准备
首先需要准备一台Jetson Nano开发板和一根网线，如下图所示：
在运行Deep Learning任务之前，需要安装好Jetpack SDK，这个SDK包含了编译、调试、部署应用程序的所有工具和文件。JetPack SDK包括了CUDA Toolkit、cuDNN Library、TensorRT、OpenCV、CUDA Visual Profiler以及许多其它工具和库。这里，我们下载的是JetPack 4.3版本。
### 安装Ubuntu系统
第一步，我们需要从NVIDIA官网上下载Jetson Nano系统镜像文件，然后烧写到SD卡上。接着，我们把SD卡插入开发板，等待它完成启动过程。当屏幕提示“Welcome to Ubuntu”时，按下Esc键，选择Try Ubuntu without installation，这样可以进入Ubuntu命令行界面。
```
sudo apt update && sudo apt upgrade -y
sudo apt install openssh-server vim
```
### 配置SSH访问权限
为了使得Jetson Nano可以通过网络远程访问，我们需要配置SSH服务器。SSH (Secure Shell)协议是一种网络安全传输协议，可以让两个网络节点之间建立加密的通道，进行远程通信。我们可以使用以下命令开启SSH服务：
```
sudo service ssh start
```
为了允许远程访问，我们还需要修改SSH配置文件`/etc/ssh/sshd_config`，设置监听地址为任何ip地址(`AddressFamily any`)，并关闭root登录(`PermitRootLogin no`)，设置允许的密码登录方式(`PasswordAuthentication yes`)，禁止X11转发功能(`X11Forwarding no`)，保存退出，重启SSH服务使之生效：
```
sudo sed -i's/^#ListenAddress 0\.0\.0\.0$/ListenAddress 0.0.0.0/' /etc/ssh/sshd_config
sudo sed -i's/^#PermitRootLogin prohibit-password$/PermitRootLogin no/' /etc/ssh/sshd_config
sudo sed -i's/^#PasswordAuthentication yes$/PasswordAuthentication yes/' /etc/ssh/sshd_config
sudo sed -i's/^#X11Forwarding yes$/X11Forwarding no/' /etc/ssh/sshd_config
sudo systemctl restart sshd
```
为了安全起见，建议修改默认SSH端口号(22)，最佳方法是创建一个新的SSH端口映射，然后仅在本地通过该端口访问Jetson Nano：
```
sudo ufw allow 2202
```
以上命令会在防火墙允许TCP端口2202上的SSH连接。
### 设置静态IP地址
为了确保Jetson Nano在每次开机后都保持同样的IP地址，我们需要配置静态IP地址。首先，我们需要查看Jetson Nano当前的MAC地址：
```
ifconfig eth0 | grep ether
```
然后编辑网络配置文件`/etc/network/interfaces`：
```
auto lo
iface lo inet loopback

allow-hotplug eth0
iface eth0 inet static
    address 192.168.0.100 # replace with your IP address
    netmask 255.255.255.0
    network 192.168.0.0
    broadcast 192.168.0.255
    dns-nameservers 8.8.8.8 8.8.4.4
```
其中，`address`选项指定了Jetson Nano的静态IP地址；`dns-nameservers`指定了DNS服务器的IP地址。保存退出，重新加载网络配置文件使之生效：
```
sudo ifup eth0
```
此时，Jetson Nano应该已经获取到静态IP地址，可以通过以下命令测试：
```
ping baidu.com
```
如果出现类似“ping: unknown host baidu.com”的错误信息，可能是DNS服务器的问题，需要手动指定DNS服务器地址：
```
ping -S <dns server ip> baidu.com
```
### 准备开发环境
接下来，我们要准备好Jetson Nano的开发环境。
#### 安装CUDA Toolkit
CUDA是由NVIDIA提供的一套用于高性能GPU运算的开发工具包。我们可以从NVIDIA官网上下载CUDA Toolkit，并按照提示安装：
```
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux
sudo sh cuda_9.0.176_384.81_linux --silent
```
安装成功后，我们可以在`/usr/local/cuda-9.0/bin/`目录找到CUDA Toolkit的可执行文件。
#### 安装cuDNN Library
cuDNN是NVIDIA针对CUDA开发的一个神经网络库。我们可以从NVIDIA官网上下载cuDNN Library，并按照提示安装：
```
tar xvzf cudnn-9.0-linux-x64-v7.tgz
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h /usr/local/cuda/lib64/libcudnn*
```
#### 安装TensorRT
TensorRT是一个用于高性能机器学习推理的加速引擎。我们可以从NVIDIA官网上下载TensorRT并按照提示安装：
```
wget https://developer.nvidia.com/compute/machine-learning/tensorrt/secure/5.1/GA/x86_64/nv-tensorrt-repo-ubuntu1604-ga-cuda9.0-trt5.1.3.0-20180406_1-1_amd64.deb
sudo dpkg -i nv-tensorrt-repo-*.deb
sudo apt-key add /var/nv-tensorrt-repo-<version>/7fa2af80.pub
sudo apt-get update
sudo apt-get install tensorrt
```
#### 安装OpenCV
OpenCV是一个开源计算机视觉库。我们可以从源代码编译安装最新版的OpenCV：
```
sudo apt-get install build-essential cmake git unzip pkg-config libgtk2.0-dev
git clone https://github.com/opencv/opencv.git
cd opencv
mkdir release
cd release
cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D WITH_TBB=ON \
      -D BUILD_NEW_PYTHON_SUPPORT=ON \
      -D ENABLE_PRECOMPILED_HEADERS=OFF \
      -D CUDA_NVCC_FLAGS="-gencode;arch=compute_30,code=sm_30;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_61,code=sm_61;-gencode;arch=compute_70,code=sm_70"..
make -j$(nproc)
sudo make install
ldconfig
```
#### 安装PyTorch
PyTorch是一个开源深度学习框架。我们可以从GitHub仓库克隆最新版的PyTorch源码，并按照提示编译：
```
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
export USE_CUDA=1
export CUDA_HOME=/usr/local/cuda-9.0
python setup.py install
```
安装成功后，我们可以在Python环境下通过`import torch`语句导入PyTorch模块。
#### 安装其他库
除上面介绍的依赖项外，还有一些常用的库需要额外安装。如：
```
sudo pip install numpy scipy matplotlib ipython pandas sympy nose future h5py tensorflow scikit-learn pillow plotly
```
至此，我们的Jetson Nano开发环境就准备好了。