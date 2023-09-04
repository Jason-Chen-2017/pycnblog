
作者：禅与计算机程序设计艺术                    

# 1.简介
         

如果您刚接触到神经网络编程（如用TensorFlow、PyTorch或Keras开发神经网络模型），那么如何在云计算平台上训练机器学习模型就成为一个值得思考的问题了。本文将教您如何利用亚马逊EC2云服务器来快速地训练神经网络模型，并获得更快的训练速度。

在过去几年里，基于GPU硬件的云服务器产品已经十分火爆。这类产品可以提供高度可靠性和超高的性能，对于训练神经网络模型来说，显然是个非常好的选择。本文将以Keras为例，介绍如何利用亚马逊EC2云服务器快速地训练神经网络模型。

# 2.基本概念术语说明
## 2.1 概念介绍
为了能够更好地理解本文的内容，我们需要了解一些基础概念和术语。以下内容仅作一般性介绍，具体内容还需要参考相关文档进行查阅。

- GPU (Graphics Processing Unit) 是一种由英伟达（NVIDIA）、ATI 或其他供应商设计、制造和生产的并行处理器，用来加速图形渲染和建模运算工作。
- CPU (Central Processing Unit)，即中央处理器，又称主板上的控制单元，是一个集成电路板上的主机电脑芯片。它通常是微控制器或单芯片计算机芯片。
- CUDA (Compute Unified Device Architecture) 是一种用于编写并运行在 NVIDIA CUDA™平台和独立于 CUDA 的异构系统架构之间的应用编程接口。它是一种开源框架，为嵌入式系统、科学计算、数据中心应用程序和游戏引擎提供了计算能力。
- cuDNN (CUDA Deep Neural Network) 是 CUDA 提供的一组高级神经网络库。它包含卷积神经网络、循环神经网络、GRU 和 LSTM 等神经网络层。cuDNN 通过向现有的 CUDA 平台添加新的功能，提升了神经网络模型训练和推断的性能。
- Kubernetes 是 Google 开发的一个开源容器集群管理系统，其目标是在任何环境下都能运行分布式应用。
- EC2 (Elastic Compute Cloud) 是亚马逊旗下的一项云计算服务，提供按需付费的虚拟服务器资源。

## 2.2 技术选型建议
如果您刚开始搭建自己的机器学习模型训练环境，那最佳的方式可能就是自己购买一台服务器，然后安装必要的软件环境，配置好GPU驱动，并且配置好相应的Python依赖包。但是这种方式可能会花费相当多的时间精力。因此，我们推荐采用亚马逊EC2云服务器作为训练环境。

### EC2概述
Amazon Elastic Compute Cloud (Amazon EC2) 是一款提供按需计算能力的云计算服务。用户可以在所需的规格下创建虚拟机 (VM) ，通过互联网访问这些 VM 。该服务为用户提供了简单而灵活的计算服务，包括计算资源、存储资源、数据库、负载均衡、安全和网络组件等。根据用户需求，可以自由选择部署虚拟机的区域、数量及配置。另外，Amazon EC2 提供了丰富的计费选项，支持按使用量付费、包年包月付费或竞价付费。

### AWS EC2 安装 TensorFlow GPU
这里我们会在AWS上安装Tensorflow GPU版本。

1. 在AWS控制台登陆https://aws.amazon.com/ec2/, 使用AWS账户登录；
2. 在导航栏点击"服务->EC2"，进入EC2主页面，点击左侧菜单栏中的"启动实例"按钮创建新实例；
3. 在"选择Amazon Machine Image (AMI)"选择"Deep Learning AMI (Ubuntu 18.04) Version 34.0";
4. 在"选择实例类型"设置实例规格和数量，比如t2.medium;
5. 在"配置实例详细信息"选择密钥对，保持默认即可；
6. 配置"标签"，可跳过；
7. 配置"安全组规则"，允许TCP流量，端口范围为6006~6015，以及SSH流量(方便远程登录)。点击"下一步:添加存储";
8. 配置"磁盘"，可以选择"默认设置"，或者直接选择"自定义"添加磁盘，大小可以根据需要设置；
9. 配置"启动脚本"，可跳过；
10. 配置"审核"，选择"启动实例"即可。

等待实例准备完毕，在"查看实例状态"页面可以看到实例的状态，一般状态如下:
- pending - 被排队等待
- running - 正常运行中
- stopping - 正在停止
- stopped - 已停止

### 运行 GPU 实例
1. SSH连接实例。
```bash
ssh -i "keypair_name.pem" ubuntu@instance_public_ip
```
2. 更新软件源。
```bash
sudo apt update && sudo apt upgrade
```
3. 安装 CUDA toolkit。
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt-get update
sudo apt-get install cuda
```
4. 安装 cuDNN。
```bash
wget https://developer.nvidia.com/cudnn-download-survey?formid=NVSHFCT_FORM
sudo dpkg -i libcudnn7_7.6.5.32-1+cuda10.2_amd64.deb
sudo dpkg -i libcudnn7-dev_7.6.5.32-1+cuda10.2_amd64.deb
```
5. 设置环境变量。
```bash
export PATH=/usr/local/cuda-10.2/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```
6. 测试CUDA是否正确安装。
```bash
cd /usr/local/cuda/samples
make
./deviceQuery
```