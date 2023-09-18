
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 1.1 课程概述
该课程从零开始教授Python编程语言、数据结构、算法和编程技巧，是一个高级工程师必备的编程入门课程。本课程适合有一定编程基础的人群学习，帮助其掌握Python编程语言的基本语法、函数、模块、类等特性及应用场景。通过该课程的学习，可以帮助学生更加充分地理解Python编程语言并运用其在实际开发中遇到的问题进行解决。同时本课程也会结合实践项目带领同学们更进一步地了解Python编程的一些内在联系，以及如何利用Python解决一些实际的问题。
## 1.2 目的
通过“课程+实践”的方式培养学生对于Python编程的熟练程度、能力和思维能力，为求职面试提供良好的编程素质指标，促进全栈工程师的进阶。
## 1.3 主要内容
- 数据类型（int、float、bool、string、list、tuple、set、dict）
- 操作符（算数运算符、比较运算符、逻辑运算符、赋值运算符、条件运算符、成员运算符、身份运算符）
- 控制语句（if-else、for-while-break、try-except-finally）
- 函数定义、调用
- 模块导入、导包
- 面向对象编程、异常处理
- 文件读写、正则表达式、系统交互、线程、多进程、数据结构
- Pythonic编程规范、代码优化、单元测试、文档生成
- Web编程、爬虫、数据分析、机器学习、深度学习、图像处理、视频处理、GIS编程
## 1.4 时长
课程预计5个小时左右。由于实践环节较多，因此建议面试前完成课程实践。
## 1.5 考核方式
- 通过编程任务达到作业要求
- 面对面或者网络上编程项目展示，同学们根据项目要求实现功能并提交代码
## 1.6 报名方式
注册报名即可，一般都是双选一。网址为 https://study.163.com/provider/openAuth.htm?#/register 。
## 2.准备工作
## 2.1 Python语言历史
Python是一种面向对象的动态脚本语言，最初由Guido van Rossum编写。它的设计目标就是像英语那样易于阅读和理解，并且具有强大的可拓展性。

它的第一个版本0.9.0发布于1991年，并在10年后发布了2.0版。Python几乎成为了最受欢迎的脚本语言之一，被用于科学计算、Web编程、自动化运维、游戏开发等领域。

目前已知Python有着超过四万多种库和框架，各种功能可以轻松实现，例如Web开发、数据分析、机器学习、GUI编程、游戏开发、自然语言处理等。

## 2.2 安装Python
Python可以在Windows、Linux、macOS平台上运行。如果您已经安装了相应版本的Python，可直接进行下一步，否则，请根据您的操作系统进行安装。

### Windows
从https://www.python.org/downloads/windows/下载安装文件，安装过程会自动配置环境变量。

### Linux
从https://www.python.org/downloads/linux/下载安装包，将下载后的安装包上传至服务器并解压。进入解压目录，执行命令`sudo./configure`，然后输入`make && sudo make install`。

### macOS
从https://www.python.org/downloads/macos/下载dmg文件并安装。

## 2.3 安装相关模块
除了默认安装的Python标准库外，还有很多第三方库可以使用，比如NumPy、pandas、matplotlib、seaborn、scikit-learn、Django、Flask等。

要安装第三方库，可以使用pip命令，命令行窗口（cmd或PowerShell）输入以下命令：

```
pip install numpy pandas matplotlib seaborn scikit-learn Django Flask
```

这样就可以安装这些库了。当然，你也可以选择手动安装，方法很简单，就是把压缩包解压，然后把里面的文件复制到某个目录下，修改PATH环境变量使得这个目录能够找到模块。