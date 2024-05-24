
作者：禅与计算机程序设计艺术                    

# 1.简介
  

随着数据科学领域的蓬勃发展，基于Python开发的数据分析、处理工具越来越多，在保证数据质量的同时，还可以方便地进行数据的分析、可视化、预测等。对于大数据应用场景下的数据分析需求，目前开源社区中最常用的Python语言版本是3.x系列，最新发布的正式版是Python 3.7，为了更好地实现数据分析任务，需要掌握相关数据科学环境的配置与管理。本文将以Linux系统和Windows系统为例，介绍如何配置与管理基于Python 3.7的数据科学环境。
# 2.基本概念术语说明
## 2.1 Linux系统安装Python 3.7
1.下载安装包

   根据自己使用的Linux发行版本，从python官网（https://www.python.org/downloads/）下载适合自己系统的安装包。
   
   **Ubuntu/Debian/Mint**
   
   ```shell
   sudo apt-get update
   sudo apt-get install python3.7
   ```
   
   **CentOS/Redhat/Fedora**
   
   ```shell
   sudo yum install epel-release
   sudo yum upgrade
   sudo yum install python37
   ```

2.设置Python3.7为默认版本

   在安装完成之后，将Python3.7设置为默认版本。
   
   ```shell
   sudo update-alternatives --install /usr/bin/python3 python3 $(which python3.7) 1 
   ```

## 2.2 Windows系统安装Python 3.7
1.下载安装包
   
   从python官网（https://www.python.org/downloads/）下载适合自己系统的安装包并安装。
   
   **注意**：windows下安装python时，为了避免出现编码问题，最好选择安装“Add python to environment variables”选项，这样会将python安装到系统环境变量Path中，并且python可执行文件名为“python”，而不是“python.exe”。

2.配置pip源

   ```shell
   pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
   ```
   
   设置pip源为清华大学TUNA镜像站，速度较快。

# 3.核心算法原理和具体操作步骤
## 3.1 安装第三方库模块

1.使用conda安装第三方库模块
   
   如果安装了anaconda，则可以使用conda命令安装第三方库模块。
   
   ```shell
   conda install pandas scikit-learn matplotlib seaborn scipy sympy pillow
   ```
   
   使用conda命令安装的模块都存储在conda环境中的，不受影响系统全局模块的污染。
   
2.直接通过pip安装第三方库模块
   
   通过pip命令安装的模块，不受conda管理，如果系统全局安装了其他版本的python或模块，可能会导致冲突。

   ```shell
   pip install numpy tensorflow keras torch
   ```

   **注意**：tensorflow官方只支持python2和python3.4及以上版本，因此强烈建议安装anaconda或者miniconda。而keras则支持python2、3、4、5、6、7、8、9版本，所以也可以选择安装anaconda或者miniconda。

## 3.2 配置jupyter notebook

1.启动jupyter notebook

   在终端运行以下命令，启动jupyter notebook服务器。
   
   ```shell
   jupyter notebook
   ```
   
   此时，系统浏览器会打开jupyter notebook主页。
   
2.创建并编辑ipynb文件

   创建一个新的ipynb文件（注意不要带后缀），然后用编辑器打开该文件，就可以编写和运行python代码了。
   
   每个单元格可以被视作是一个独立的python脚本，你可以按需在多个单元格中输入代码，组合成完整的python脚本。单元格类型分为代码和文本两种。你可以在右上角选择单元格类型。
   
   当你在一个单元格中输入代码，点击左上角运行按钮，当前单元格的代码就会被执行。
   
3.配置jupyter notebook自动保存功能

   默认情况下，jupyter notebook仅仅在编辑时才保存文件，需要手动点击菜单栏File->Save and Checkpoint才能保存文件。为了让jupyter notebook自动保存，你需要修改配置文件。
   
   ```shell
   vim ~/.jupyter/jupyter_notebook_config.py
   ```
   
   将c.NotebookApp.checkpoint_dir = ''改为：
   
   ```python
   c.NotebookApp.checkpoint_dir = 'C:/Users/<username>/Documents/' # 将<username>替换成你的用户名
   ```
   
   上面的路径指的是jupyter notebook保存文件的位置，需要根据自己的情况修改。
   
   重启jupyter notebook服务器即可生效。
   
4.启用外网访问

   如果需要让jupyter notebook在外网被访问，可以在服务器上开启端口转发功能。
   
   方法一：利用ssh tunnel转发
   
   在本地终端运行如下命令，将本地端口转发至远程服务器：
   
   ```shell
   ssh -N -L localhost:port:localhost:8888 username@serverip
   ```
   
   把`port`换成你希望监听的端口号，把`username@serverip`换成你的远程服务器地址和用户名。
   
   然后在浏览器中输入：http://localhost:port/，就能看到远程服务器上的jupyter notebook主页。
   
   方法二：利用ngrok托管服务
   
   ngrok是一个开源的反向代理工具，它可以将本地的web服务映射到公开的网络上，你可以通过公网访问该web服务。
   
   激活ngrok账户并下载客户端，安装并启动客户端。登录ngrok网站，创建一个新http服务，选择http和tcp端口为8888，然后复制该服务的链接。
   
   在本地终端运行如下命令，将本地端口转发至ngrok的公网链接：
   
   ```shell
   ssh -N -R port:localhost:8888 username@serverip -p port # `port`是你的ngrok服务所在的端口号
   ```
   
   然后在浏览器中输入：http://ngrok.io/，粘贴刚才复制的ngrok http服务链接，点击start，就可以看到该web服务被映射到公网上，外部世界可以通过这个公网链接来访问本地服务。