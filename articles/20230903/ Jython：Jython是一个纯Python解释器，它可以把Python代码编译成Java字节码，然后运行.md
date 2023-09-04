
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Jython是一个纯Python解释器，同时支持CPython、IronPython和JYTHON三个不同的Python实现版本。它可以在Java平台下执行标准的Python代码，并且可以在JVM之上提供一个快速、可移植、易于使用、跨平台的Python环境。其主要特性如下：

1.易于学习：Jython由Python编程语言编写而成，语法与标准Python相似，使用者很容易就能掌握。通过学习Jython教程和参考文档，就可以快速上手使用Jython。

2.跨平台：Jython能够自动将Python源代码编译成能在各种JVM上的字节码文件，因此无需重新编译即可在多个平台上运行。

3.高性能：Jython采用一种高度优化过的JIT编译器，使其具有比CPython更快的执行速度。

4.集成开发环境：Jython有着完善的集成开发环境（IDE）支持，包括PyDev、Eclipse插件等。

5.易于使用：Jython具有丰富的库，能够轻松处理日常任务中的复杂场景。

6.社区支持：Jython有一个活跃的社区，提供了众多的扩展模块、工具以及资源，极大的促进了该语言的发展。

7.脚本语言：Jython既可以作为命令行应用工具，也可以嵌入到其他应用中，用于实现快速脚本化开发。

总的来说，Jython是一款非常优秀的Python语言实现，具备良好的易用性及跨平台能力，适合用于各种各样的应用场景，有着广泛的应用潜力。同时，Jython也有着自己的一些限制，比如无法使用C/C++的一些高级语法特性。不过对于绝大多数场景来说，Jython都可以胜任。
# 2.基本概念和术语
Jython相关概念和术语介绍，以便阅读本文的读者可以快速了解相关知识。
## 2.1 Python解释器类型
Jython有三种类型的Python解释器：CPython、IronPython和JYTHON。
### CPython
CPython是最常用的Python解释器，通常被称作默认Python解释器。CPython是Python官方实现，是官方发布的最新版Python解释器。CPython的特点是基于栈结构的虚拟机，占用内存少，启动速度快。
### IronPython
IronPython是一个纯动态Python解释器，能够直接与.NET Framework和Mono一起运行。它是微软针对Python的重新实现，能够在没有安装CPython的情况下运行Python程序。IronPython的特点是可以直接调用.NET类库、COM对象和其他.NET功能。
### Jython
JYTHON是一个纯Python解释器，可以把Python代码编译成Java字节码，然后运行在JVM上。JYTHON的作用是在Java平台上运行Python脚本程序，并提供了一个快速、可移植、易于使用的Python环境。JYTHON的开发人员声称它比标准CPython和IronPython更快、更易于使用。
## 2.2 Python虚拟机
Python虚拟机(VM)指的是在某种计算机系统上，用软件模拟构成该计算机系统的实际硬件，并允许在该虚拟机上运行的指令集合。Python程序通过虚拟机执行时，首先需要被翻译成机器代码，然后才能在物理机器上执行。不同的Python实现版本都对应着不同的Python虚拟机。
Jython使用Java平台的JVM来运行Python程序。JVM是Java Virtual Machine的缩写，是一种运行在操作系统之上的java虚拟机，它的工作方式类似于硬件，因为它能够加载字节码文件并执行它们。JVM运行Python代码的方式是把Python源代码编译成字节码，然后再由JVM来执行。
## 2.3 Java字节码
Java字节码是一种中间代码，通过Java编译器或解释器生成，可用于任何兼容Java虚拟机的平台。Python代码经过Jython编译器后会产生相应的Java字节码，之后可以被Java虚拟机运行。
Jython编译器是专门针对Python语言设计的，可以将Python代码转换成相应的Java字节码文件，这样就可以在JVM上运行Python代码。Jython提供了两种编译器版本：
1. CPythonCompiler：用于生成CPython对应的字节码文件；
2. Javacompiler：用于生成Jython对应的字节码文件。