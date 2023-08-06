
作者：禅与计算机程序设计艺术                    

# 1.简介
         

         JavaFX（Java Desktop UI）是一个用于创建桌面应用程序、图形用户界面（GUI）的API。它在2014年发布了它的8th版本，可以运行于Windows、macOS及Linux系统平台上。JavaFX框架完全基于Java语言，并提供了丰富的控件和组件，这些控件和组件可以用来开发高性能的、出色的桌面应用程序。本教程将向读者介绍JavaFX的基础知识，包括安装配置、布局管理器、事件处理机制、皮肤定制等方面的知识。另外，本教程还会着重介绍一些有趣实用的新功能，例如Canvas、SVG画布、Web视图、OpenGL图形渲染以及动画效果。
         本教程将从以下六个部分进行阐述：

         1. 安装配置
         2. 布局管理器
         3. 事件处理机制
         4. 活动区域（Stage）
         5. Canvas
         6. SVG画布

         在学习本教程之前，需要确保读者已经具备基本的编程能力。如对Java编程不熟悉，建议先阅读《Java Programming for Beginners》或其他相关书籍。

         如果读者无法安装Java开发环境或者环境变量设置错误，请参考以下教程进行安装：

         Windows：https://www.java.com/en/download/help/windows_install.html

         macOS: https://www.java.com/en/download/help/mac_install.html

         Linux: https://www.java.com/en/download/faq/linux_repositories.xml

         建议读者根据自己平台选择适合自己的安装方式。

         # 2. JavaFX 安装配置

         ## 下载JDK 和 JRE

         JDK（Java Development Kit）是Java开发工具包，包括JRE（Java Runtime Environment）和Java编译器（javac）。JRE仅支持运行Java程序；而JDK除了支持运行Java程序外，还支持开发Java程序、调试Java程序、集成第三方工具等。需要注意的是：不同的JDK版本只能运行特定的JRE版本，比如JDK7只能运行JRE7，不能运行JRE8。因此，如果有多个Java开发环境，建议安装一致的JRE和JDK。

         ### JDK下载

         从Oracle官网下载JDK：http://www.oracle.com/technetwork/java/javase/downloads/index-jsp-138363.html

         根据自己平台下载对应版本的JDK，点击“接受许可协议”，然后按照提示安装即可。

         ### JRE下载

         JDK下载完成后，可以直接安装JRE，也可以单独下载JRE。JRE相比JDK小很多，仅支持运行Java程序，不会影响开发Java程序。从Oracle官网下载JRE：http://www.oracle.com/technetwork/java/javase/jre8-downloads-2133155.html

         根据自己平台下载对应版本的JRE，点击“接受许可协议”，然后按照提示安装即可。

         ## 配置Java开发环境

         ### 检查Java是否已安装

         打开命令行窗口，输入命令“java –version”检查Java是否已安装。

         ```
         C:\Users\username> java --version
         java version "1.8.0_151"
         Java(TM) SE Runtime Environment (build 1.8.0_151-b12)
         Java HotSpot(TM) 64-Bit Server VM (build 25.151-b12, mixed mode)
         ```

         如果看到类似的输出信息，证明Java已安装成功。

         ### 设置JAVA_HOME和PATH环境变量

         为了方便调用Java开发工具，需要设置JAVA_HOME环境变量。在命令行窗口输入以下命令，设置JAVA_HOME变量值为jdk的安装目录。

         ```
         setx JAVA_HOME "C:\Program Files\Java\jdk1.8.0_151" /m
         ```

         此命令的含义如下：

         * `setx` 是Windows下的一个设置环境变量的命令。
         * `JAVA_HOME` 是设置的环境变量名。
         * `"C:\Program Files\Java\jdk1.8.0_151"` 是设置的环境变量的值，表示JDK的安装目录。
         * `/m` 表示修改系统的全局环境变量。

         命令执行成功后，重新打开命令行窗口，输入`echo %JAVA_HOME%`，查看环境变量是否生效。

         ```
         C:\Users\username> echo %JAVA_HOME%
         C:\Program Files\Java\jdk1.8.0_151
         ```

         ### 配置CLASSPATH环境变量

         CLASSPATH环境变量用来告诉Java编译器查找类库的路径。在命令行窗口输入以下命令，设置CLASSPATH变量值为`.;%;%JAVA_HOME%\lib;%JAVA_HOME%\lib    ools.jar`。

         `.;`表示当前目录，`;%JAVA_HOME%\lib`表示JDK的lib文件夹，`;%JAVA_HOME%\lib    ools.jar`表示JDK的lib/tools.jar文件。

         ```
         setx CLASSPATH ".;%;%JAVA_HOME%\lib;%JAVA_HOME%\lib    ools.jar" /m
         ```

         命令执行成功后，重新打开命令行窗口，输入`echo %CLASSPATH%`，查看环境变量是否生效。

         ```
         C:\Users\username> echo %CLASSPATH%
        .;%;C:\Program Files\Java\jdk1.8.0_151\lib;C:\Program Files\Java\jdk1.8.0_151\lib    ools.jar
         ```

         ### 测试JavaFX是否可用

         通过以下命令测试JavaFX是否可用：

         ```
         C:\Users\username> java --module-path "%PATH_TO_ javafx-sdk-11.0.2\lib" --add-modules javafx.controls,javafx.fxml HelloWorld
         ```

         上面的命令指定了JavaFX SDK的lib文件夹，添加了javafx.controls和javafx.fxml两个模块，并执行HelloWorld类。如果成功执行，应该能看到一个空白的JavaFX应用窗体。

         ```
         May 29, 2018 4:11:04 PM javafx.fxml.FXMLLoader$ValueExpressionCollector computeValueExpressions
         INFO: Loading FXML document from '/HelloWorld.fxml'
         May 29, 2018 4:11:05 PM javafx.fxml.FXMLLoader loadImpl
         INFO: Loading 'Hello World'
         May 29, 2018 4:11:05 PM javafx.scene.Parent load
         WARNING: CSS URL was not found. Stylesheets will be loaded but may not render correctly.
         Welcome to the JavaFX hello world sample!
        ```

         执行失败的话，可能是由于没有正确配置PATH环境变量，导致无法找到java命令。可以通过在命令行下输入`where java`来定位java的位置。如果找不到java命令，则需要配置PATH环境变量。