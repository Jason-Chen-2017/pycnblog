
作者：禅与计算机程序设计艺术                    

# 1.简介
  

PyCharm 是一款 Python IDE，具有强大的编辑、运行、调试功能。在实际项目开发中，它可以提升代码编写效率和质量，帮助开发人员解决各种各样的问题。PyCharm 可以有效地管理多种项目文件，包括 Python 文件、测试脚本、配置文件等。此外，还提供代码自动补全、语法检查、编译运行等功能。另外，还有许多插件可用于扩展 PyCharm 的功能，如 Django 插件、Google App Engine 插件等。因此，掌握 PyCharm 的使用技巧十分重要。
# 2.安装与配置
## 2.1 安装
从 JetBrains 官网下载适合自己操作系统的 PyCharm 安装包，然后根据安装向导一步步完成安装即可。PyCharm 安装后默认安装了 Python 插件，如果需要用到其他版本的 Python，则还需要安装相应版本的插件，比如要用到 Python 3.7，则需要安装 Python37 plugin。安装完毕后，打开 PyCharm 之后会出现欢迎页面，点击 “Configure” 按钮，进入设置界面。
## 2.2 配置
### 2.2.1 默认编码
在 Preferences -> Editor -> File Encodings 中选择你的项目源码文件的编码方式，建议使用 UTF-8。
### 2.2.2 tab 键的宽度
在 Preferences -> Editor -> Code Style 中将 Tab and Indent -> Spaces 的值设置为 4 个空格。
### 2.2.3 设置默认文档类型
在 Preferences -> Editor -> File Types 中添加新的文档类型，如 Django 模板文件（*.html.djt）、Python testcase 文件（*.py.test）等。
### 2.2.4 添加解释器路径
Preferences -> Project:projectName -> Project Interpreter 中设置你的项目使用的 Python 解释器路径。
### 2.2.5 启用代码分析提示
在 Preferences -> Editor -> Inspections 中开启 Python PEP 8 Convention 和 Type-checking 两个选项。