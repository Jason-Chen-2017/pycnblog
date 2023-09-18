
作者：禅与计算机程序设计艺术                    

# 1.简介
  


Python（英文全称：Python programming language）是一个开源、跨平台的高级程序设计语言，被誉为“拼车”语言。Python 具备以下特征：

1.易学习：Python 具有简单性、易读性、明确的语法，可以让初学者快速上手并掌握其特性。
2.丰富的数据结构：包括列表、字典、集合等数据类型。
3.动态语言：像其他语言一样，Python 可以轻松地进行动态编程，你可以在运行时改变程序的行为。
4.可移植性：你可以把 Python 程序编译成不同平台上的可执行文件或共享库。
5.开放源代码：Python 是开源项目，它的源代码可以在网上免费获取。

# 2.下载安装

## 2.1 安装过程

本文假设您已安装了Python3环境。

### Windows系统

1. 进入Python官网https://www.python.org/downloads/windows/
2. 根据自己电脑系统选择适合版本进行下载，比如64位系统下载64-bit version的安装包，下载后双击运行，点击 Install Now 安装Python3到本地目录。

安装完成后，默认安装了IDLE这个图形化编程工具，用来编写和运行Python程序。如果不需要IDLE可以使用命令提示符，打开cmd输入`python`，开始Python交互环境。

```
C:\Users\Administrator>python
Python 3.7.9 (tags/v3.7.9:13c94747c7, Aug 17 2020, 18:58:18) [MSC v.1900 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> print("Hello World!")
Hello World!
>>> 
```

### Linux系统

由于Python已经安装在Linux系统中，所以直接使用pip进行安装即可，不再赘述。

```
sudo apt update && sudo apt install python3-pip
```

### macOS系统

在macOS上安装Python主要有两种方式：Homebrew 和 Anaconda。

#### Homebrew

Homebrew 是 Mac OS 下一个强大的软件包管理器，用它可以很容易地安装 Python。

```bash
brew install python
```

#### Anaconda

Anaconda 是基于 Python 的数据科学计算平台，提供了超过 300 个预先构建好的科学包及其依赖关系，使得用户可以零配置安装这些包。同时它还提供了一个 Jupyter Notebook 用于交互式数据分析。

从 https://www.anaconda.com/download/#macos 上下载对应系统的安装包并安装即可。

```bash
curl -O https://repo.continuum.io/archive/Anaconda3-2020.11-MacOSX-x86_64.sh
chmod +x Anaconda3-2020.11-MacOSX-x86_64.sh
./Anaconda3-2020.11-MacOSX-x86_64.sh # 一路回车默认设置
source ~/.bashrc   # 使环境变量生效
conda create --name myenv python=3    # 创建名为myenv的环境
conda activate myenv                     # 激活myenv环境
```

安装好后，就可以在终端中使用Python了，如下所示：

```
(base) ➜  ~ which python
/usr/local/bin/python
(base) ➜  ~ python
Python 3.7.9 (default, Sep  3 2020, 21:48:36) 
[Clang 10.0.0 ] :: Anaconda, Inc. on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> print('hello world')
hello world
>>> exit()
```

这里需要注意的是，Anaconda会将所有包都放在单独的环境路径下，如上面的 `/usr/local/anaconda3/` ，因此默认情况下激活环境后，仍然可以通过 `pip` 来安装新的包，不会影响系统的全局环境。

# 3. Hello World!

大家应该都知道，计算机程序的入门都是打印`Hello World!`作为测试。现在，我们就用Python实现一下这个程序。

```python
print('Hello World!')
```

保存为`helloworld.py`, 在命令行中输入：

```
python helloworld.py
```

如果一切顺利，就会看到屏幕输出：

```
Hello World!
```

说明程序运行成功。