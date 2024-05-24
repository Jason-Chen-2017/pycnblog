
作者：禅与计算机程序设计艺术                    

# 1.简介
  

在现代信息化时代，GUI界面越来越重要。因此，JavaFX(JavaFx)技术成为当下最流行的Java GUI框架之一。本文将通过对JavaFX的详细介绍，带领读者了解其工作机制、特点及优缺点，以及如何使用它进行图形用户界面编程。

## JavaFX的介绍
JavaFX是一个开源的Java平台的技术框架，用于开发富客户端应用程序，具有简单易用、跨平台特性、可扩展性强等特点。

JavaFX由两部分组成：JavaFX API 和 JavaFX运行时环境（JRE）。API定义了一组用来开发GUI应用程序的类和接口，运行时环境负责渲染GUI组件并提供基础功能支持，包括图形处理、输入输出、事件处理、动画、多媒体支持、本地化等。

JavaFX被设计用于支持广泛的设备类型，包括桌面、台式机、移动设备和嵌入式系统。JavaFX不仅适合移动设备，也适用于桌面应用和服务端应用程序。同时，JavaFX还提供了丰富的控件库，可以帮助开发人员快速创建出色的UI。

### JavaFX历史版本
JavaFX的第一个版本于2007年发布。截至目前，JavaFX已迭代了多个版本，每隔6个月发布一个新版本，当前最新版本为JavaFX 16。JavaFX 16在GUI编程方面的能力已经达到了一个相当高的水平，但还有很多地方需要改进。例如，UI布局组件仍然很少，尤其是在手机、平板电脑等小屏幕设备上。另外，线程模型、网络连接等功能也仍处于相对完善阶段，这都给JavaFX 16提升了一个很大的空间。

### JavaFX特性
JavaFX具有以下主要特性：

1. 可移植性：JavaFX基于开源的JRE，可以部署到各种操作系统平台上，并且具有良好的移植性。
2. 模块化：JavaFX采用模块化设计，每个模块都独立维护自己的更新计划和生命周期，这使得开发人员能够灵活选择所需的功能。
3. 用户友好：JavaFX的语法和特性简单易懂，使得初学者容易学习和使用。
4. UI组件丰富：JavaFX内置了一系列常用的UI组件，例如Button、TextField、ComboBox等，还可以方便地自定义组件。
5. 多平台兼容性：JavaFX可以打包成跨平台应用程序，并可以在不同平台上运行，包括桌面平台、移动平台、浏览器等。
6. 轻量级框架：JavaFX占用的资源较少，适合于桌面应用程序和移动应用程序的开发。
7. 易用性：JavaFX提供了丰富的API和工具来加速GUI应用程序的开发。
8. 性能优化：JavaFX对内存管理和线程模型做了充分优化，可以提升GUI应用程序的运行速度。
9. 支持多语言：JavaFX可以支持多种语言，如英语、法语、德语等。

### JavaFX API概览
JavaFX API的主要类、接口和注解如下表：

| 名称 | 描述 |
| --- | --- |
| javafx.application | 提供Application基类，用于创建JavaFX应用程序。|
| javafx.scene | 用于构建用户界面，包括节点、场景、画布、动画、效果、调色板和样式。|
| javafx.stage | 包含了窗口（Stage）、对话框（Dialog）和应用程序（Applet）的主要构件。|
| javafx.fxml | 是一种声明性的XML标记语言，用于定义JavaFX用户界面。|
| javafx.css | 提供了一种机制，用于管理CSS样式表，并应用于JavaFX组件。|
| javafx.media | 为音频、视频、图像等提供了支持。|
| javafx.web | 提供了JavaFX的Web视图支持。|
| javafx.controls | 提供了一些常用控件，如Button、ListView、TextArea等。|
| javafx.fxml | 是一种声明性的XML标记语言，用于定义JavaFX用户界面。|
| javafx.graphics | 提供了对图像、光栅图形、字体和颜色的支持。|

### JavaFX编程模型
JavaFX编程模型分为四层：

- 窗体模型：提供各种窗口类型的支持，包括透明、弹出窗口、复合窗口、模式窗口、对话框等；
- 图形模型：提供各种形状、路径、画笔、渐变、阴影、滤镜等图形元素的支持；
- 控制器模型：提供MVC架构中的控制器支持，包括FXML、CSS样式表和代码支持；
- 数据模型：提供各种数据结构的支持，包括集合、树、图、日期时间、格式化、国际化、验证、加密、存储等。

## JavaFX开发环境配置
要开发JavaFX应用程序，首先需要安装Java Development Kit (JDK)和JavaFX SDK。下面分别介绍安装过程。

### 安装JDK
如果没有安装JDK，可以通过Oracle官网下载并安装JDK。



安装完成后，在命令提示符或PowerShell中输入`javac -version`，查看是否安装成功。如果显示版本号，则表示安装成功。

```powershell
C:\Users\username> javac -version
javac 1.8.0_211
```

### 安装JavaFX SDK
JDK安装成功后，就可以安装JavaFX SDK了。JavaFX SDK是针对Java SE 8及更高版本的JavaFX相关组件的开发工具包，其中包含JavaFX运行时环境（JRE）、JavaFX API及示例代码。



下载完成后，将压缩包解压到任意位置，然后设置系统PATH环境变量，这样就可以直接在命令提示符或PowerShell中执行JavaFX相关命令了。

设置系统PATH环境变量的方法如下：

打开注册表编辑器，找到`Computer\HKEY_LOCAL_MACHINE\System\CurrentControlSet\Control\Session Manager\Environment`。双击`Path`属性，在`编辑`菜单中选择`新建`，然后输入`%JAVA_HOME%\bin`（Windows系统）或`$JAVA_HOME/bin`（Linux系统），保存修改即可。

```powershell
C:\Users\username> java --module-path "%PATH_TO_SDK%\lib" --add-modules javafx.base,javafx.controls,javafx.fxml -m HelloWorld
```

如果出现以下错误：

```powershell
Error: Could not find or load main class HelloWorld
Caused by: java.lang.NoClassDefFoundError: javafx/application/Application
```

那么可能是因为在PATH环境变量中没有添加正确的JavaFX运行时环境（JRE）。在命令提示符或PowerShell中输入`java -version`，检查Java版本号是否符合要求。

```powershell
C:\Users\username> java -version
openjdk version "11.0.7" 2020-04-14
OpenJDK Runtime Environment AdoptOpenJDK (build 11.0.7+10)
Eclipse OpenJ9 VM AdoptOpenJDK (build openj9-0.23.0, JRE 11 Mac OS X amd64-64-Bit Compressed References 20200415_285 (JIT enabled, AOT enabled)
OpenJ9   - 7c6d2c4a1
OMR      - b31a8b4db
JCL      - 2a7af5a47e based on jdk-11.0.7+10)
```

如果Java版本号低于JavaFX SDK要求的版本（1.8.0_211），则需要安装更新的Java JDK。

最后，为了能正常运行JavaFX应用程序，还需要设置一下JVM参数。

```powershell
C:\Users\username> java --module-path "%PATH_TO_SDK%\lib" --add-modules javafx.base,javafx.controls,javafx.fxml -m HelloWorld --add-exports javafx.graphics/com.sun.javafx.tk=ALL-UNNAMED --add-opens javafx.graphics/com.sun.javafx.tk.quantum=ALL-UNNAMED --add-exports javafx.graphics/com.sun.glass.ui=ALL-UNNAMED --add-opens javafx.graphics/com.sun.glass.ui.win=ALL-UNNAMED
```

其中，`--add-exports`参数允许JavaFX API使用其内部类的导出。`--add-opens`参数允许JavaFX API使用内部类的非导出开放。`--add-exports`和`--add-opens`参数一般只需要在命令行中设置一次，并不需要重复设置。

这样就可以启动JavaFX应用程序了！