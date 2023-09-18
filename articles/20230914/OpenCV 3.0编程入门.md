
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## OpenCV（Open Source Computer Vision Library）是一个基于BSD许可证发行的跨平台计算机视觉库。它由Intel、WetaBoards公司和其他开源贡献者开发维护。OpenCV的目标是在实时计算环境下提供简单易用的计算机视觉接口，旨在帮助读者快速开发出具有高效率的应用软件。它的API简单易用，封装了底层复杂算法，使得开发人员可以集中精力分析和解决计算机视觉相关的问题。
## 优势
- **开源免费**：OpenCV完全基于开源协议发布，任何人都可以自由使用该库。
- **跨平台**：OpenCV可以在Windows、Linux、Mac OS X等多种操作系统上运行，并支持多种硬件平台。
- **C/C++语言开发**：OpenCV采用C++语言编写，具有丰富的函数接口和模块化设计，适用于需要复杂图像处理功能的高性能应用程序。
- **功能强大**：OpenCV提供了包括图像处理、视频分析、机器学习及几何变换等众多领域的算法实现。同时还提供了如GPU加速、深度学习等新功能特性。
## 下载安装
OpenCV目前最新版本为3.1.0，您可以通过两种方式获取该库：编译源码或者下载预编译好的二进制文件。下面我们将介绍如何编译源码。
### Windows平台下编译
#### 前期准备工作
首先，确保您的电脑已经安装Visual Studio。如果您没有安装 Visual Studio，请到微软官方网站下载相应的版本进行安装。
其次，您需要安装CMake。CMake是一个开源项目，是跨平台的构建工具，用来生成各种各样的Makefiles和工程文件。CMake安装包可从官网下载:http://www.cmake.org/download/.下载后双击以安装。
最后，您需要安装Git。Git是一个开源的分布式版本控制系统，可以跟踪记录一个文件或多个文件的所有历史版本。安装Git的方法很简单，打开命令提示符(cmd)进入到任意路径，输入以下命令：
```git clone https://github.com/opencv/opencv.git```
然后等待下载完成即可。
#### 安装OpenCV
如果您已经下载了OpenCV的源代码，那么就应该从GitHub上克隆源代码。完成克隆后，切换到`build`目录，然后创建一个新的文件夹，命名为`Release`，执行以下命令：
```cmake -G "Visual Studio 14 Win64"..\opencv```
在上面的命令中，`-G "Visual Studio 14 Win64"`表示指定编译器为Visual Studio 2015，`-A x64`表示编译为64位。也可以不用指定编译器，CMake会根据系统自动选择编译器。
然后运行MSBuild，生成OpenCV.sln项目文件。在Solution Explorer窗口右键单击`INSTALL`项目，选择`Build Solution`。等待编译完成即可。
如果编译过程中出现错误，可以尝试以下方法排除：
- 检查是否已安装所有必备软件。
- 删除掉CMake缓存文件，重新运行CMake指令。
- 切换到其他分支，如master分支，以便获取最新的更新。
- 查阅错误信息，搜索关键词，寻求帮助。
当编译完成之后，将得到一个名为`opencv_world310d.dll`的文件，拷贝到您的项目中就可以使用了。
### Linux平台下编译
同样，对于Linux平台，您也需要安装相应的依赖库，包括CMake、Git和OpenCV源码。如下所示：
```sudo apt-get install cmake git libgtk2.0-dev pkg-config libglib2.0-dev libavcodec-dev libavformat-dev libswscale-dev```
然后按照Windows平台下的编译方式编译即可。
### Mac OS X平台下编译
Mac OS X上的编译相对比较麻烦，因为缺少Visual Studio，而且Apple禁止直接安装该软件，所以，这里只给出手动安装方法。
首先，您需要安装Xcode，并启动终端。然后通过Homebrew安装cmake、pkg-config和glib库：
```brew install cmake pkg-config glib```
接着，从GitHub上克隆OpenCV的源代码：
```git clone https://github.com/opencv/opencv.git opencv```
切换到该目录，创建`build`目录并进入：
```mkdir build && cd build```
运行cmake指令：
```cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D WITH_TBB=OFF \
      -D BUILD_NEW_PYTHON_SUPPORT=ON \
      -D WITH_V4L=ON \
      -D INSTALL_C_EXAMPLES=OFF \
      -D INSTALL_PYTHON_EXAMPLES=OFF \
      -D OPENCV_ENABLE_NONFREE=OFF \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D PYTHON_EXECUTABLE=$(which python) \
     ../opencv```
其中，`-D CMAKE_BUILD_TYPE=RELEASE`设置编译模式为release；`-D CMAKE_INSTALL_PREFIX=/usr/local`设置安装路径；`-D WITH_TBB=OFF`禁止使用Intel TBB库；`-D BUILD_NEW_PYTHON_SUPPORT=ON`开启Python支持；`-D WITH_V4L=ON`打开视频设备支持；`-D INSTALL_C_EXAMPLES=OFF`禁止安装C示例程序；`-D INSTALL_PYTHON_EXAMPLES=OFF`禁止安装Python示例程序；`-D OPENCV_ENABLE_NONFREE=OFF`禁止安装非自由版；`-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules`指定OpenCV contrib模块位置；`-D PYTHON_EXECUTABLE=$(which python)`指定Python解释器。
接下来，运行make指令：
```make -j8```
可能出现的问题有：
- `ld: library not found for -lpython2.7`，解决办法是删除掉CMakeCache.txt文件，重新运行cmake指令。
- `fatal error: 'numpy/arrayobject.h' file not found`，解决办法是安装numpy。
- `no member named 'imread' in namespace cv`，解决办法是编译时启用WITH_QT选项。
- 没有权限访问目录等错误，解决办法是使用管理员身份运行terminal。
编译完成后，运行make install命令将OpenCV安装到指定路径。
至此，OpenCV安装成功！