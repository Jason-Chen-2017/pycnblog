                 

# 1.背景介绍


在编程中，模块(Module)是对具有一定功能或相关属性的代码集合，一般来说，一个模块封装了一种功能或相关属性，可以被其他程序所引用和调用。

例如，面向对象编程中的类，模块化编程中的函数库等都属于模块。Python 中使用 import 语句来导入模块。

为了更好的管理项目中的模块，减少命名冲突、提高代码复用率，Python 提供了包（Package）这一概念。包是一组模块的集合，按照一定规则组织起来并提供访问接口。

比如，标准库就是 Python 的一个包。如果安装了 Python ，那么就可以使用标准库中的所有模块，它们都是独立的包，包含多个模块。

本文将详细介绍模块导入和定义相关知识，包括：

1. 模块导入语法；
2. 普通模块的导入方式；
3. 使用 from...import 语句导入模块成员；
4. 使用 as 来给模块别名；
5. 包的导入方式；
6. 查找路径、搜索路径、相对导入和绝对导入；
7. 自定义模块查找顺序。

# 2.核心概念与联系
## 模块
模块是对具有一定功能或相关属性的代码集合，一般来说，一个模块封装了一种功能或相关属性，可以被其他程序所引用和调用。

模块分类：

1. 内置模块 (Built-in modules): Python 自带的模块，主要用于实现语言内部功能。如：math、sys、json 等模块。

2. 第三方模块 (Third-party Modules): 通过 pip 或 easy_install 安装的外部模块，通常由开发者编写，可直接使用的功能非常广泛。如：requests、beautifulsoup 等模块。

3. 用户自己编写的模块 (.py 文件)。

## 包
包是一组模块的集合，按照一定规则组织起来并提供访问接口。

包通常是多层目录结构，每个目录下有一个 `__init__.py` 文件，该文件使得该目录成为一个包。

包的作用：

1. 避免命名冲突：不同包下的同名模块不会冲突，可以分别导入，但不建议这样做。

2. 管理模块：包能提供统一的接口，便于模块之间的交互。

3. 提高代码复用率：可以把相同的函数、类等打包成一个包，方便其他地方引用。

## 模块导入语法
模块导入语法有两种形式：

- import module: 在当前脚本中引入某个模块的所有成员，不需要使用模块名前缀。
- from module import member: 只引入某个模块的一个成员，需要用模块名前缀。

## 导入路径
当我们通过 `import` 语句导入模块时，实际上是在告诉 Python 要导入哪个模块及其里面的成员。Python 首先会在当前脚本所在目录下寻找对应的模块文件，如果没有找到，则会在环境变量 PYTHONPATH 指定的目录列表中依次搜索。如果还是没有找到，则报错 ModuleNotFoundError。

PYTHONPATH 是指当前用户的环境变量，它指定了 Python 在搜索模块时应该检查的目录列表。可以设置这个环境变量，让 Python 在导入时优先从指定的目录中查找模块。

可以通过以下命令查看当前环境变量：
```python
>>> import os
>>> print(os.environ['PYTHONPATH'])
```

也可以手动设置环境变量：
```python
>>> import os
>>> os.environ['PYTHONPATH'] = '/path/to/module' # 设置环境变量
```

注意：对于第三方模块，不要修改环境变量，否则可能会影响到其他依赖该模块的程序。

## 普通模块导入方式
Python 支持三种普通模块导入方式：

1. `import module`: 把模块所有成员全部导入到当前脚本中。
2. `from module import member`: 从模块中导入单个成员到当前脚本中，可以简化代码。
3. `import module as alias`: 为模块指定别名，可以简化代码。

下面以 sys 模块为例，演示各个导入方式的区别。

### 用 import 导入模块
使用 `import module` 将整个模块全部导入到当前脚本中，可以使用模块名称或者模块文件的完整路径作为别名。

#### 通过模块名称导入
举例如下：

file1.py
```python
import math
print(math.pi) # 输出 3.141592653589793
```

#### 通过模块文件导入
当模块在当前目录中时，可以直接通过文件名作为别名进行导入。

举例如下：

file1.py
```python
import sys
print(sys.__file__) # 当前运行脚本的绝对路径
```

当模块不在当前目录中时，可以通过模块文件的完整路径作为别名进行导入。

举例如下：

file1.py
```python
import requests
response = requests.get('https://www.baidu.com')
print(response.text[:10]) # 获取百度首页的内容
```

### 用 from...import 导入模块成员
使用 `from module import member` 可以只导入模块的一部分成员，并简化代码。

举例如下：

file1.py
```python
from random import choice
print(choice(['apple', 'banana', 'orange'])) # 随机选择水果
```

导入了 `random.choice()` 方法后，就不需要使用 `random.` 前缀了。

### 用 import...as 给模块指定别名
使用 `import module as alias` 可以给模块指定别名，简化代码。

举例如下：

file1.py
```python
import time as t
t.sleep(5) # 休眠五秒
```

导入了 `time` 模块并指定别名 `t`，可以缩短代码长度。