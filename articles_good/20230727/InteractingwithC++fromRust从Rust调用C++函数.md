
作者：禅与计算机程序设计艺术                    

# 1.简介
         
 在本教程中，我们将会学习如何从Rust语言环境中调用C++编译器生成的库文件，并利用这些库中的函数。通过学习示例，可以快速掌握从Rust调用C++函数的方法。
          # 2.相关知识点
          1.Cargo: 它是一个Rust包管理工具，类似于npm或maven。它允许我们在Rust项目中管理多个依赖项，包括本地依赖和crates.io上的依赖项。
          2.CPP Binding: 即“C++绑定”，它可以让我们在Rust代码中使用C++编译器生成的库。
          3.FFI（Foreign Function Interface）: 它是一种定义外部函数接口标准的方式，它可以使我们的Rust代码和非Rust代码互相交流。
          4.bindgen: 是rust社区提供的一个库，它可以将C头文件转换成Rust代码。
          5.NDK（Native Development Kit）: 它是一个用于开发Android应用的软件开发包。
          # 3.为什么要用Rust调用C++？
          1.性能上优于C/C++语言。Rust语言编译成机器码效率高、运行速度快、内存安全性高等特点。而C/C++语言则需要经过编译才能运行。
          2.更安全。Rust提供了所有权系统来保证内存安全性，而C/C++则没有这一机制。
          3.更容易编写多线程、异步编程。Rust拥有所有权系统和任务系统，这使得编写多线程程序变得非常简单。同时还可以使用线程池来提升效率。
          4.更容易理解和维护代码。Rust拥有完善的类型系统，而且其设计模式也较C/C++更接近纯粹语言。
          # 4.准备工作
          ## 安装Cargo
          1.首先，确保您的电脑上安装了Rust编译器和Cargo包管理工具。如果您尚未安装过，请参考官方网站的[下载页面](https://www.rust-lang.org/tools/install)进行安装。
          2.运行以下命令查看是否成功安装：
             ```shell
             $ cargo --version
             ```
          3.如果显示版本号，则证明Cargo安装成功。
          ## 安装NDK
          如果要在Android平台上运行Rust，还需要安装NDK。
          [下载地址](https://developer.android.com/ndk/downloads?hl=zh-cn)。
          ### Linux
          1.下载压缩包并解压到指定目录：
             ```shell
             $ tar -xvf ndk-<version>-linux-x86_64.bin
             $ mv ndk-<version> ~/path/to/ndk
             ```
            <version>: ndk版本号，如r17c等。
          2.配置环境变量：
             ```shell
             $ export ANDROID_NDK_HOME=$HOME/path/to/ndk
             $ export PATH=$PATH:$ANDROID_NDK_HOME
             ```
          3.验证是否安装成功：
             ```shell
             $ ndk-build
             ```
             如果显示Usage信息，则证明安装成功。
          ### MacOSX
          1.下载dmg包并安装：
             ```shell
             $ open Downloads/<version>.dmg
             ```
            <version>: ndk版本号，如r17c等。
          2.配置环境变量：
             ```shell
             $ export ANDROID_NDK_HOME=/Users/$USER/Library/Android/sdk/ndk/<version>
             $ export PATH=$PATH:$ANDROID_NDK_HOME
             ```
          3.验证是否安装成功：
             ```shell
             $ ndk-build
             ```
             如果显示Usage信息，则证明安装成功。
          ### Windows
          1.下载exe安装包并安装：
             ```shell
             $./<version>/android-ndk-<version>-windows-x86_64.exe
             ```
            <version>: ndk版本号，如r17c等。
          2.配置环境变量：
             ```shell
             > setx ANDROID_NDK_HOME "C:\Program Files (x86)\Android\android-ndk\<version>"
             ```
            将"<version>"替换为实际安装的版本号。
          3.验证是否安装成功：
             ```shell
             > cd %ANDROID_NDK_HOME%
             > ndk-build
             Usage: ndk-build [-h] [--file FILE]
                            [--dir NATIVE_DIR]
                            [--arch {arm,arm64,x86,x86_64}]
                            [--toolchain TOOLCHAIN]
                            target...
             ```
             如果显示Usage信息，则证明安装成功。
          ## 安装CMake
          为了方便CMake与Rust绑定，需要安装CMake。
          ### Linux
          ```shell
          sudo apt install cmake
          ```
          ### MacOSX
          ```shell
          brew install cmake
          ```
          ### Windows
          下载安装包并安装即可：[Download CMake](https://cmake.org/download/)。
          ## 配置cargo.toml文件
          在你的项目根目录下创建一个名为"Cargo.toml"的文件，添加以下内容：
          ```toml
          [package]
          name = "your_crate_name"
          version = "0.1.0"
          authors = ["you"]
          edition = "2018"

          [dependencies]
          cc = "1.0.58"
          bindgen = "0.51.1"
          cmake = "0.1.44"
          ```
          此处的"[dependencies]"部分内容会在后面使用到。
          # 5.实战案例
          下面，我们举一个简单的例子，介绍如何用Rust调用C++函数。假设有一个C++文件叫做"calculator.cpp"，它的内容如下：
          calculator.cpp
          ```c++
          int add(int a, int b) {
              return a + b;
          }

          double subtract(double a, double b) {
              return a - b;
          }
          ```
          在这个文件中，我们定义了两个函数："add()"和"subtract()"，它们分别实现整数加法和浮点数减法功能。
          创建新文件夹，名称为"rust_call_cpp",进入该目录，创建一个新的Cargo项目：
          ```shell
          mkdir rust_call_cpp && cd rust_call_cpp
          cargo new my_project
          ```
          修改"my_project/Cargo.toml"文件，添加以下内容：
          my_project/Cargo.toml
          ```toml
          [package]
          name = "my_project"
          version = "0.1.0"
          authors = ["you"]
          edition = "2018"

          #[dependencies]
          #cc = "1.0.58"
          #bindgen = "0.51.1"
          #cmake = "0.1.44"
          bindgen = "^0.53.2"
          ```
          使用Cargo构建项目：
          ```shell
          cargo build
          ```
          ## 生成C接口
          为生成可供C++使用的C接口，我们需要使用到bindgen库。先修改Cargo.toml文件，增加"links"项：
          my_project/Cargo.toml
          ```toml
          [package]
         ...
          links = "calculator"
          crate-type = ["cdylib", "staticlib"]
          
          [lib]
          name = "calculator"
          path = "../rust_call_cpp/src/lib.rs"

          [[bin]]
          name = "main"
          path = "./src/main.rs"
          ```
          上述设置告诉Cargo链接名为"calculator"的动态库或静态库，并生成一个可被C++语言调用的动态库文件。
          编辑"rust_call_cpp/src/lib.rs"文件，加入以下代码：
          lib.rs
          ```rust
          use std::os::raw::{c_char, c_int};
          use std::ffi::CString;
          extern "C" {
              fn add(a: c_int, b: c_int) -> c_int;
              fn subtract(a: f64, b: f64) -> f64;
          }

          pub fn call_add() -> i32 {
              unsafe {
                  let result = add(1, 2);
                  println!("called add(), result is {}", result);
                  result as _
              }
          }

          pub fn call_subtract() -> f64 {
              unsafe {
                  let result = subtract(3.0, 2.0);
                  println!("called subtract(), result is {:.2}", result);
                  result
              }
          }
          ```
          这里声明了两个extern块，分别声明了两个C接口："add()"和"subtract()"，它们均接受整型参数并返回整型结果。
          在"rust_call_cpp"文件夹下执行以下命令：
          ```shell
          bindgen --no-layout-tests --with-derive-partialeq \
                 --whitelist-function add --whitelist-function subtract \
                 --whitelist-var INT_MAX --whitelist-var INT_MIN -- \
                ../rust_call_cpp/calculator.cpp -o src/bindings.rs
          ```
          执行成功后，会在"rust_call_cpp/src/"目录下生成一个名为"bindings.rs"的文件。
          ## 编写Rust代码
          在"my_project/src/main.rs"中，编写Rust代码：
          main.rs
          ```rust
          #[link(name = "calculator")] // 指定要链接的动态库文件名
          extern "C" {
              fn add(a: i32, b: i32) -> i32;
              fn subtract(a: f64, b: f64) -> f64;
          }

          fn main() {
              unsafe {
                  let res1 = add(1, 2);
                  assert_eq!(res1, 3);

                  let res2 = subtract(3.0, 2.0);
                  assert!((res2 - 1.0).abs() < f64::EPSILON);

                  println!("all tests passed.");
              }
          }
          ```
          通过对"calculator"库的链接和调用，在Rust代码中，我们可以调用C++库中的函数。
          ## 编译
          最后，在"my_project"文件夹下执行以下命令，编译成可执行文件：
          ```shell
          cargo run
          ```
          成功执行后，控制台输出"all tests passed."，说明Rust调用C++函数成功。至此，我们完成了一个从Rust调用C++函数的完整流程。
          # 6.未来发展方向
          本教程主要介绍了Rust调用C++函数的基本方法。但是随着Rust生态系统的不断发展，越来越多的库和框架已经支持与Rust调用，比如serde，rocket等。因此，可以将目光投向这些生态，学习相应的调用方法。另外，Rust正在逐步成为主流编程语言，越来越多的创业公司都采用Rust作为主要开发语言，所以在涉及Rust的项目中，应该适当关注相关的技术更新。
          # 7.总结
          本教程旨在帮助读者了解从Rust调用C++函数的基本方法。文章首先介绍了Rust调用C++的一些原因，然后给出了调用过程中需要注意的问题。通过实际案例演示，读者可以快速熟悉Rust调用C++的过程。希望本文对大家的Rust学习、工作有所帮助！

