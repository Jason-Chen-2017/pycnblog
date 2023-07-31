
作者：禅与计算机程序设计艺术                    

# 1.简介
         
随着 Java 发展历史的不断推进，其版本更新的频率越来越快。早在 2017 年年初，OpenJDK 项目就宣布，将在今年中旬召开一次大会，讨论 Java 的下一个 LTS 长期支持（Long Term Support）版本：Java 17，并计划于 2022 年 9 月发布正式版。

Java 17 是 OpenJDK 开发团队在 JDK 16 中引入的一系列特性的集合，其中包括多项语言特性改进、性能提升等功能，增强了 Java 编程的能力。同时，OpenJDK 项目也宣布与 Eclipse Adoptium 社区合作，将在明年底或后年初发布与 Java 17 兼容的 Java SE Development Kit (JDK) 。

本文重点介绍 Java 17 中的新增和变更模块及相关知识。

# 2.基本概念术语说明
## 2.1.概述
Java SE Development Kit (JDK) 是一种开源软件，用于创建、编译、运行和调试 Java 应用程序。它提供了 API 和运行时环境，让开发者可以开发、部署和运行跨平台 Java 程序。从某种意义上说，JDK 是整个 Java 生态系统中的基础，其他组件都需要通过调用 JDK 提供的 API 来实现各种功能。

OpenJDK 是由 Oracle Corporation 创建的基于 HotSpot JVM 的免费、开放源码 Java 开发工具包。由于 Oracle 对该项目拥有商标权，因此称之为 OpenJDK。OpenJDK 的目标是在没有 Oracle 许可的情况下获得完整的 Java SE 规范的源代码并构建出符合标准的 JVM。OpenJDK 在源代码方面做出了许多修改，使得它可以在不同平台上运行，并且修复了一些已知的问题。

Java 17 是 Java SE 7 到 Java SE 16 的最新版本，其目标是在保持向前兼容性的同时，提升性能和实现更多功能。

## 2.2.JVM 简介
Java Virtual Machine （JVM）是位于用户计算机上的软件，作为计算设备与操作系统之间的接口。它负责执行字节码指令，解释执行 Java 源代码。JVM 将字节码转换成平台无关的代码，然后将其映射到本地 CPU 上。JVM 可以在几乎任何平台上运行，包括 Windows、Linux、macOS、Solaris 操作系统以及 Android 手机。

JVM 有三个主要组件：类加载器、垃圾收集器和运行时数据区。

1.类加载器：类加载器是 JVM 的组成部分，用于将字节码编译成实际的机器码。当 JVM 启动时，它会加载类文件，并且会创建一个 ClassLoader 对象。ClassLoader 根据指定的位置加载类的字节码文件，并对这些类进行验证、准备和解析。

2.垃圾回收器：JVM 使用垃圾回收器来释放堆内存中不再被引用的对象占用的空间。垃圾回收器主要有两种类型：复制回收器和标记-清除回收器。复制回收器在堆内存分成两个较大的区域，每次只用其中一个区域，当第一个区域填满时，将其所有内容复制到第二个区域，然后再把第一个区域清空。标记-清除回收器首先标记所有的活动对象，然后清除掉未被标记的对象占用的空间。

3.运行时数据区：JVM 在运行时都会使用一块内存，即运行时数据区。它包括方法区、虚拟机栈、堆和程序计数器。方法区用于存放类信息、常量、静态变量、即时编译器编译后的代码等数据；虚拟机栈用于存储局部变量、操作数栈、动态链接、方法出口等信息；堆用于存放对象实例、数组、字符串等内存；程序计数器用于存放正在执行的线程的字节码指令地址。

## 2.3.JDK、JRE 和 JVM
JDK 是 Java Development Kit 的缩写，是用于开发 Java 应用的开发工具包。包含 JRE 和开发工具，如 Java 编译器 javac、Java 调试器 jdb、单元测试框架 junit、类浏览器 javadoc、集成开发环境 IDE 等。

JRE 是 Java Runtime Environment 的缩写，是运行 Java 程序所需的运行环境。它是 Java SDK 的子集，仅包括 Java 类库和必备的 Java 命令行环境。

JVM 是 Java Virtual Machine 的缩写，是一个抽象的概念，指运行 Java 字节码的虚拟机。

总结一下就是：JDK = JRE + development tools，JVM 是运行 Java 字节码的虚拟机。

