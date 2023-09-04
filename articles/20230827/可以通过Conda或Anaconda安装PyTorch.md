
作者：禅与计算机程序设计艺术                    

# 1.简介
  

最近，深度学习领域的火热也越来越多，尤其是在计算机视觉、自然语言处理、强化学习等方向上取得了惊人的成果。为了帮助机器学习爱好者更好的了解并掌握深度学习框架PyTorch，本文将向大家介绍如何通过Conda或Anaconda快速在各个操作系统下安装并运行PyTorch。

# 2. Conda/Anaconda介绍
Conda是一个开源的包管理工具，用于管理Python及其他软件包及环境。Conda可以帮助我们轻松地安装、卸载、更新各种Python库和程序。Anaconda则是基于Conda的一整套数据科学相关的开发环境和包集合。其中包括Python、NumPy、SciPy、pandas、matplotlib、sympy、jupyter、TensorFlow、Keras等许多先进的科学计算和数据分析工具。

# 3. PyTorch简介
PyTorch是一个开源的深度学习框架，可以帮助我们轻松地进行深度学习实验和编程。它有着简单易懂的API接口和高度模块化设计，并提供对GPU加速的支持。本文将介绍如何通过Conda或Anaconda在各个操作系统下安装并运行PyTorch。

# 4. 在Windows平台安装PyTorch
由于PyTorch尚未在Windows平台上正式发布，因此需要通过conda-forge渠道进行安装。以下是通过Conda在Windows 10系统中安装PyTorch的步骤。

1. 安装Miniconda

   Miniconda是一个轻量级的conda安装包，只包含conda、python和pip三个主要包，占用空间很小。我们可以在<https://docs.anaconda.com/anaconda/install/windows/>下载并安装该软件。

2. 创建conda环境

   执行如下命令创建名为pytorch的conda环境：
   
   ```
   conda create -n pytorch python=3.7
   ```

3. 添加conda-forge源

   执行如下命令添加conda-forge源：
   
   ```
   conda config --add channels conda-forge
   ```

4. 安装PyTorch

   执行如下命令安装PyTorch：
   
   ```
   conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
   ```

   其中`-c`参数指定安装包的源，这里我们选择的是PyTorch的官方源`pytorch`。如果你的网络连接较差，可以使用清华大学、北京理工大学、Anaconda官网等源下载安装包安装。

5. 测试PyTorch是否安装成功

   通过执行如下命令测试PyTorch是否安装成功：
   
   ```
   import torch
   
   print(torch.__version__)
   ```

   如果输出版本号信息，表示PyTorch已成功安装。

# 5. 在Linux平台安装PyTorch
由于PyTorch尚未在Linux平台上正式发布，因此需要通过conda-forge渠道进行安装。以下是通过Conda在Ubuntu Linux系统中安装PyTorch的步骤。

1. 更新apt软件源

   Ubuntu Linux默认软件源已经比较老旧，一些软件包可能无法正常工作，所以我们需要手动更新软件源。执行如下命令：
   
     ```
     sudo apt update
     ```

2. 安装Miniconda

   执行如下命令安装Miniconda：
   
   ```
   wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
   bash Miniconda3-latest-Linux-x86_64.sh
   ```

3. 创建conda环境

   执行如下命令创建名为pytorch的conda环境：
   
   ```
   conda create -n pytorch python=3.7
   source activate pytorch
   ```

4. 添加conda-forge源

   执行如下命令添加conda-forge源：
   
   ```
   conda config --add channels conda-forge
   ```

5. 安装PyTorch

   执行如下命令安装PyTorch：
   
   ```
   conda install pytorch torchvision torchaudio cpuonly -c pytorch
   ```

   `-c`参数指定安装包的源，这里我们选择的是PyTorch的官方源`pytorch`，`-cpuonly`参数安装CPU版PyTorch。如果你想同时安装CUDA版PyTorch，也可以使用如下命令：
   
      ```
      conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
      ```

   如果你的网络连接较差，可以使用清华大学、北京理工大学、Anaconda官网等源下载安装包安装。

6. 测试PyTorch是否安装成功

   通过执行如下命令测试PyTorch是否安装成功：
   
   ```
   python
   import torch
   
   print(torch.__version__)
   ```

   如果输出版本号信息，表示PyTorch已成功安装。

# 6. 在Mac OS平台安装PyTorch
由于PyTorch尚未在Mac OS平台上正式发布，因此需要通过conda-forge渠道进行安装。以下是通过Conda在MacOS系统中安装PyTorch的步骤。

1. 安装Homebrew

   Homebrew是一个开源的包管理器，可以方便地安装、更新各种软件。首先，访问<http://brew.sh>，下载并安装Homebrew。

2. 安装Miniconda

   执行如下命令安装Miniconda：
   
   ```
   brew cask install miniconda
   ```

3. 创建conda环境

   执行如下命令创建名为pytorch的conda环境：
   
   ```
   conda create -n pytorch python=3.7
   source activate pytorch
   ```

4. 添加conda-forge源

   执行如下命令添加conda-forge源：
   
   ```
   conda config --add channels conda-forge
   ```

5. 安装PyTorch

   执行如下命令安装PyTorch：
   
   ```
   conda install pytorch torchvision torchaudio cpuonly -c pytorch
   ```

   `-c`参数指定安装包的源，这里我们选择的是PyTorch的官方源`pytorch`，`-cpuonly`参数安装CPU版PyTorch。如果你想同时安装CUDA版PyTorch，也可以使用如下命令：
   
      ```
      conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
      ```

   如果你的网络连接较差，可以使用清华大学、北京理工大学、Anaconda官网等源下载安装包安装。

6. 测试PyTorch是否安装成功

   通过执行如下命令测试PyTorch是否安装成功：
   
   ```
   python
   import torch
   
   print(torch.__version__)
   ```

   如果输出版本号信息，表示PyTorch已成功安装。