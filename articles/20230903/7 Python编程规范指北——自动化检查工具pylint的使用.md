
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Pylint是一个开源的Python的代码分析工具，可以检测出Python代码中的错误、不规范或过于复杂的地方。它提供了多种检查功能，从复杂性检查（循环复杂度，函数的最大嵌套层数等）到错误处理检查（检查try...except...finally块是否都正确关闭）。同时，它还支持第三方插件扩展其功能。由于Pylint的开放式架构，使得它能够应用在各种不同的项目中。在全球范围内，有许多公司都采用了Pylint进行代码质量管理，以确保代码风格一致性、可读性和健壮性。本文将对Pylint的基本概念、安装配置及用法做简单介绍，并结合实际案例分享自己的体会。本篇博文基于Pylint 2.9版本，适用于Python2.x/3.x。
# 2.基本概念术语说明
## 2.1 Pylint简介
Pylint (Python linter) 是一种源自Python语言的源代码分析工具。其目标是在源代码文件中发现潜在错误、不规范或过于复杂的代码，并提醒开发人员改进这些代码的结构和方式。Pylint 可以帮助检测代码中存在的语法、逻辑和设计缺陷，以及帮助查找bug和难以发现的编码规范问题。 Pylint 以模块形式提供给用户，可以通过命令行执行或者集成到其它开发环境如Eclipse、IDEA等。 Pylint 被广泛地应用在开源项目上，例如，Mozilla Firefox、Zope、Google App Engine等。

## 2.2 Pylint架构
Pylint 的整体架构由两部分组成：
- **Linter engine** - 负责解析Python源代码并从中识别出语法错误、错误的变量引用、违反的编码规则等问题，生成报告。
- **Reporters** - 报告生成器，根据linter engine的结果生成对应的报告。Pylint 提供了许多预设的报告样式，也允许用户自定义报告样式。

## 2.3 Pylint配置文件
Pylint 支持读取配置文件，配置文件可以用来控制Pylint的行为，包括开启和禁止检查项、设置检查严重级别、设置输出报告格式等。Pylint 默认配置文件为 ~/.pylintrc 。

## 2.4 Pylint检查项分类
### 2.4.1 Pylint官方检查项
Pylint 自带一些检查项，可以满足绝大多数用户的需求。详细的检查列表及其详细描述，可查看官网文档 https://www.pylint.org/features.html#checks 。

### 2.4.2 Pylint第三方检查项
Pylint 的生态系统不断丰富，目前已有多款 IDE 或编辑器插件支持 Pylint ，它们往往针对某些特定的工程模式、编程习惯等，可提供更高级的检查能力。如PyCharm Professional 提供了 Python 源码分析插件 Pylint 。

第三方检查项也可以通过 pip 安装。以 pylint_django 来举例，该检查项用于分析 Django 框架相关的代码。pip install pylint_django 即可安装该检查项。

## 2.5 Pylint常见命令行参数
### 2.5.1 检查指定文件
```shell
$ pylint mymodule.py myotherfile.py
```
此命令将对mymodule.py 和 myotherfile.py进行代码风格检查。

### 2.5.2 忽略某些警告
```shell
$ pylint --ignore=C0111 mymodule.py
```
此命令将忽略mymodule.py中所有关于missing docstring的警告。如果要忽略多个警告，则逗号分隔。比如：--ignore=C0111,W0612。

### 2.5.3 只显示错误警告
```shell
$ pylint -E mymodule.py
```
此命令只会显示mymodule.py中所有的错误警告信息，不会显示信息提示。

### 2.5.4 指定输出报告类型
```shell
$ pylint --reports=y mymodule.py
```
此命令会将mymodule.py中所有检查结果以y表示形式输出，可选择xml、text、json、colorized等几种形式。

## 2.6 Pylint默认警告
Pylint 会针对不同类型的错误或警告生成对应的默认警告消息，当用户不再需要某个默认警告时，可以忽略掉。可以通过 --disable 参数来屏蔽默认警告。比如：
```shell
$ pylint --disable=missing-docstring,invalid-name mymodule.py
```
这条命令将会屏蔽 mymodule.py 中 missing-docstring 和 invalid-name 两个默认警告。