                 

# 1.背景介绍


在日常开发中，我们经常会遇到需要调用其他函数或者模块的情况。那么我们该如何引用这些模块呢？本文将带领大家学习Python的模块管理机制、模块导入方式、Python模块搜索路径等知识，并通过实例来加深对相关概念的理解。

Python的模块分为两种类型：标准库模块和第三方模块。标准库模块已经在安装Python时就预装了，而第三方模块则需要通过pip命令进行安装，当然也可以手动下载安装包然后手动添加到环境变量Path里面。

其中，官方文档给出的模块分类如下图所示:

1.标准库模块（Built-in Modules）：
这种模块都是由Python语言提供，直接使用即可，不需要安装额外模块。如random、math、collections等模块都是标准库模块。

2.第三方模块（Third-Party Modules）：
这种模块一般由Python社区或者其他人维护，安装使用前，需要先配置环境。一般可以通过pip命令安装。

# 2.核心概念与联系
## 模块的概念
Python中，模块(Module)是一个相对独立的代码文件。它定义了一个相关功能集合并且可以被其它地方使用。每一个模块都包含一些定义函数、类、变量和常量的语句。这些语句的执行使得模块中的代码能够起作用，从而完成特定的功能。

模块是封装数据的一种方式。每一个模块都有一个名称，这个名称用来标识模块的用途。模块名只能包含字母数字字符、下划线或点号。模块名应当短小、描述性强。如果模块名比较长，可以使用别名简化导入。

## __init__.py 文件
每个目录下都会存在一个__init__.py的文件。这个文件可以为空，但是不能缺失。当导入某个目录下的模块时，Python解释器会首先查找__init__.py文件。如果找到__init__.py文件，那么Python解释器会将这个目录视为一个模块。否则，只会导入这个目录下面的 *.py* 文件。

## import 语句
import语句用于在当前模块中引入其他模块中的定义对象。import语句通常出现在模块的最开始位置，一般放在文件的开头。模块中的import语句的语法如下：
```
import module1[, module2[,... moduleN]]
```
import语句可以一次引入多个模块，模块之间用英文半角逗号（,）隔开。导入的模块在当前模块中成为可用的名称，就可以使用模块中定义的函数、类、变量。

## from... import 语句
from... import语句用于从模块中导入指定的对象。from...import语句的语法如下：
```
from module import name1[, name2[,... nameN]]
```
from...import语句仅导入模块中的指定对象，可以提高代码的精简度。

## dir() 函数
dir() 函数用于获取模块内定义的属性和方法列表。该函数的参数是模块对象或者字符串形式的模块名称。
```
>>> import math
>>> dir(math)
['__doc__', '__file__', '__loader__', '__name__', '__package__', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atan2', 'atanh', 'ceil', 'comb', 'copysign', 'cos', 'cosh', 'degrees', 'dist', 'e', 'erf', 'erfc', 'exp', 'expm1', 'fabs', 'factorial', 'floor', 'fmod', 'frexp', 'gamma', 'gcd', 'hypot', 'inf', 'isclose', 'isfinite', 'isinf', 'isnan', 'ldexp', 'lgamma', 'log', 'log10', 'log1p', 'log2','modf', 'nan', 'perm', 'pi', 'pow', 'radians','remainder','sin','sinh','sqrt', 'tan', 'tanh']
```