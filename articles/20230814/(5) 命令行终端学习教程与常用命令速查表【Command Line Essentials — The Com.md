
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 概述
命令行（Command-Line Interface）即键盘输入命令和显示输出结果的方式。对于程序员来说，命令行可以提高工作效率和解决实际问题，并且可以使用脚本语言自动化管理复杂任务。本课程将介绍如何在Windows、Mac OS X和Linux上安装并配置命令行环境，熟练掌握命令行中的基本操作技巧，从而提升日常工作效率和效率。

为什么要学习命令行？因为命令行是无处不在的，经常被用于各种各样的应用场景。无论是在工作、生活中，还是学习编程、网站开发、运维管理等方面，都离不开命令行工具。因此，掌握命令行技能对于个人职业生涯和职场发展都非常重要。

本教程适合刚入门者，具有一定计算机基础。希望通过本教程，您能够掌握命令行的相关知识，并成为一名成功的命令行工程师。

## 本教程目标读者
本教程面向所有想要学习命令行技能的个人，包括但不限于：程序员、系统管理员、Web开发人员、DevOps工程师、UI设计师、视频编辑者、游戏制作人等。

## 本教程结构
本教程共分为五章：
1. 安装配置命令行环境
2. 操作命令行基本命令
3. 文件管理
4. shell脚本
5. 命令行终端小技巧

每章均提供关于该章的内容介绍，以及推荐的阅读材料，供用户自主选择。

最后附上常见问题与解答。

# 2.安装配置命令行环境
## Windows系统下安装配置命令行环境
### 安装过程
安装过程分为两步：下载安装包、安装配置。

1. 从以下链接获取适用于Windows的命令行工具：https://gitforwindows.org/。本课程使用Git Bash作为命令行工具。下载完后直接运行安装程序进行安装即可。

2. 配置Git Bash终端环境变量：
    - 在Windows资源管理器中，打开`此电脑`，在右侧找到`属性`选项卡，然后点击`高级系统设置`。
    - 点击左侧菜单栏中的`环境变量`，在弹出的窗口中找到`系统变量`，点击“编辑系统环境变量”按钮。
    - 点击`新建`，输入`Path`作为变量名称，把之前安装的Git路径追加到值中，比如：C:\Program Files\Git\cmd。然后点击确定。

3. 测试是否成功安装：打开命令提示符（Win + R，输入cmd），输入`git`，如果出现以下信息，表示已成功安装：

   ```
   Usage: git.exe [options] command [args]
  ...
   Additional help topics:

    "gittutorial" and "giteveryday" get you started with using Git.
    "gitworkflows" describes various workflows that people use with Git.
    "gitcvs" shows you how to migrate from CVS to Git.
    "gitcli" shows you what commands are available in the command-line tool.
    "gitebookshelf" is a book on Git that will teach you all you need to know.
   ```

### 命令行窗口快捷键
| 快捷键      | 功能描述                   |
| ----------- | -------------------------- |
| Ctrl + A    | 移动光标到行首             |
| Ctrl + E    | 移动光标到行尾             |
| ↑↓         | 上下移动光标               |
| Page Up     | 向上翻页                   |
| Page Down   | 向下翻页                   |
| Home        | 移动光标到屏幕顶部         |
| End         | 移动光标到屏幕底部         |
| Delete      | 删除当前字符               |
| Backspace   | 删除前一个字符             |
| Ctrl + K    | 删除到行尾                 |
| Ctrl + U    | 删除到行首                 |
| Alt + →     | 向右扩展单词               |
| Alt + ←     | 向左缩减单词               |
| Tab         | 插入四个空格               |
| Shift + Tab | 删除一个空格               |
| F1          | 帮助                       |
| F2          | 同义词                     |
| F3          | 命令历史记录              |
| F5          | 更新屏幕                  |
| F9          | 设置断点                  |
| Ctrl + C    | 中止正在执行的命令         |
| Ctrl + Z    | 暂停正在执行的命令，返回到上次暂停的地方继续执行 |
| Ctrl + D    | 退出命令行                |