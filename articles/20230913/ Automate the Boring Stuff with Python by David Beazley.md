
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Python编程语言成为了人们研究计算机科学、数据科学及机器学习的主要工具。许多大型互联网公司如Google、Facebook、Twitter等都在使用Python进行应用开发。作为一门高级语言，Python具有简单而易于学习的特性，其语法清晰、可读性强，能够简化编码工作。同时，Python拥有庞大的生态系统，几乎涵盖了所有应用领域，包括科学计算、Web开发、数据分析、机器学习、图像处理、视频处理等。因此，Python已成为各行各业广泛使用的编程语言。

本书将教会你如何使用Python自动化重复性任务。第1版的Python编程语言简介提供了有关Python及其生态环境的全面概述。第2章“命令行与脚本”阐述了如何从命令行运行Python程序，以及如何用脚本文件保存复杂的程序指令。第3至5章分别介绍了文件的读写、字符串处理、列表和元组、字典、条件语句和循环控制语句。第6章探讨了函数、异常处理、模块化编程、设计模式以及调试技巧。第7至9章详细讲述了处理文本、CSV（Comma Separated Value，逗号分隔值）文件、JSON（JavaScript Object Notation，JavaScript对象表示法）数据的标准库。第10至12章详细讲述了处理Excel、Word、PDF文档、数据库和电子邮件的数据提取、处理、存储。最后，第13章结束语介绍了Python在不同应用领域的应用场景，并给出了一个计划。

本书适用于有一定Python基础的人员，包括但不限于工程师、数据科学家、学生和科研人员等。如果你已经掌握了Python语言的基本知识，或正在准备学习Python，那么这个书籍正适合你阅读。

作者：<NAME>，美国加利福尼亚州伯克利的计算机科学家、软件工程师和开源倡导者。他于2010年创建了Python，现在担任Python Core Developer之一。他曾就职于世界500强企业IBM，担任高级系统工程师；也曾担任过创新办主任，负责研发工程师招聘与管理。除了技术文章外，他还撰写了多部与Python相关的书籍和课程。他热爱分享知识，并致力于帮助其他人掌握Python编程技巧。你可以通过email联系他：davidbeazley AT gmail DOT com 。此外，他还拥有自己的个人网站www.dbader.org ，分享一些Python代码片段和教程。

# 2.基本概念术语说明
## 2.1 Python语言概述
Python是一种高层次的解释型动态编程语言。它的设计具有简单性、易用性、可读性和可维护性，特别适合作为脚本语言来使用，尤其是在各种应用程序中需要快速原型的开发阶段。它支持多种编程范式，包括面向对象的、命令式、函数式、逻辑、脚本化以及面向过程的编程风格。Python的解释器被称为CPython，可以运行在 Linux、Windows 和 macOS 操作系统上。它还可以在嵌入式系统、移动设备和网络设备上运行。Python版本目前有两个系列：Python 2和Python 3。截止2021年1月，Python的最新版本是3.10。由于Python 3具有完全向后兼容性，所以目前仍然可以使用Python 2，但是推荐使用更高版本的Python。

Python最初由Guido van Rossum于1989年创建，主要用来进行网络编程。在2000年底，Python获得了图灵奖，并推广到整个互联网行业。Python使用一种交互式命令提示符，让用户输入命令、查看输出结果、打印输出信息。Python支持多种编程范式，其中包括面向对象的、命令式、函数式、逻辑、脚本化以及面向过程的编程风格。

## 2.2 安装配置Python环境
安装Python有两种方式：直接下载安装包或源码编译安装。对于Windows平台，建议下载安装包安装；对于Linux平台，建议源码编译安装。下面，我将分别展示源代码编译安装Python环境的过程。

### Windows平台安装
#### 从官网下载安装包
访问https://www.python.org/downloads/windows/，选择合适的版本下载。下载完毕后双击exe文件即可安装。安装过程非常简单，勾选Add Python to PATH(添加Python目录到PATH)选项即可，这样就可以在任意位置打开命令窗口并调用python命令。


#### 源码编译安装
如果由于网络原因不能从官网下载安装包，可以通过源码编译的方式安装Python。首先下载源码压缩包。然后解压到指定路径，比如D:\Python39\。接下来，需要配置编译参数。


打开cmd，进入Python安装目录，执行如下命令：

	cd D:\Python39\
	./configure --prefix=C:/Users/YourUserNameHere/AppData/Local/Programs/Python/Python39 --enable-optimizations

--prefix指定安装目录；--enable-optimizations开启优化，即生成优化后的字节码。配置完成后，执行以下命令：

	nmake install

等待编译完成即可。安装成功后，打开命令行，输入python，如果看到以下画面，则表示安装成功：


此时，Python默认安装到了你的C盘根目录Program Files下的Python文件夹里。当然，如果修改安装目录的话，只需要更改configure的参数就可以了。另外，Python安装完成后，还需要安装第三方库才能使用。

### Linux平台安装
#### 使用包管理器安装
如果您的系统上已经设置了包管理器，比如yum、apt-get或pacman，可以直接使用包管理器安装Python。比如，Ubuntu系统可以通过如下命令安装：
	
	sudo apt-get update && sudo apt-get install python3
	
Fedora系统可以通过如下命令安装：

	sudo dnf install python3

Debian/Ubuntu系统下可以使用下面的命令安装：

	sudo aptitude install python3
	
Archlinux系统下可以使用下面的命令安装：

	sudo pacman -S python
	
其他基于RPM系的系统，比如CentOS、openSUSE等，也可以使用类似命令安装。

#### 源码编译安装
如果没有设置包管理器或者下载安装包速度比较慢，可以选择源码编译安装。这种方法适合没有系统管理员权限的情况，因为安装过程不需要管理员权限。首先，下载源码压缩包，然后解压到指定路径，比如~/Downloads/Python39。接下来，编译安装：

```bash
tar xzf Python-3.9.5.tgz
cd Python-3.9.5
./configure --prefix=/usr/local/python3.9
make && make install
```

这里，--prefix指定安装路径。等待编译完成即可。安装成功后，打开命令行，输入python3，如果看到以下画面，则表示安装成功：

```bash
Python 3.9.5 (default, May  4 2021, 03:33:11)
[GCC 10.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>>
```