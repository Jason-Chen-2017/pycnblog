
作者：禅与计算机程序设计艺术                    

# 1.简介
         

## GPU编程模型简介
在GPU编程中，通常会使用到线程并行(thread-parallel)和数据并行(data-parallel)两种方式。通过多线程，可以同时执行多个任务；而通过数据并行，可以对同一个数据进行并行处理，提升处理效率。数据的组织形式也有很多种，如数组、矩阵、图像等。

通常情况下，GPU上的编程模型和CPU上有所不同。首先，GPU上没有可执行的代码，所有的代码都需要编译成机器语言，然后通过设备驱动程序(device driver)加载到显存中运行。另外，GPU上并没有寄存器，所有的数据都需要从主内存(主存)拷贝到显存中。这样做的结果就是，GPU编程模型的运行速度要比CPU快得多，因为它不受限于寄存器的数量和运行时间，而且可以充分利用并行计算能力。

GPU编程模型之所以如此复杂，主要是因为为了满足并行计算的需求，GPU硬件设计者们考虑了许多因素。其中，最重要的一点就是内存访问模式。由于GPU具有高度并行性，它要求线程不能相互冲突地访问同一块内存，否则就会造成冲突。因此，为了保证数据访问的正确性，GPU硬件设计者们采用了内存调度策略来管理内存访问。

在内存调度策略下，每个线程只能访问其私有的地址空间。也就是说，不同的线程只能读或写自己的数据，不能直接访问其他线程的数据。虽然不同线程之间可以通信，但实际上还是需要通过主机内存(host memory)进行数据共享。因此，为了保证多个线程之间的内存访问不会出现冲突，GPU开发者们在编程的时候要注意以下几点：

1. 不要声明两个指针指向相同的数据块。即使是在同一个程序中，如果两个指针指向同一块数据，也不要对其中任何一个指针赋值，因为这种行为可能导致其他线程无法正常工作。
2. 使用volatile关键字标记的变量只能被读取，不能被写入。volatile变量可以帮助编译器对程序进行优化，使得程序执行更加高效。
3. 将数组切片时，要确保每个线程访问到的元素数量相同。否则，可能会发生内存访问冲突，导致错误的结果。
4. 不要声明局部变量，而应尽量减少使用局部变量。局部变量的生命周期较短，容易造成数据竞争，也无法进行有效的内存分配。
5. 函数的参数应该尽量避免传递大对象。这对于GPU来说非常重要，因为函数参数拷贝到GPU显存中需要额外的时间开销。如果参数过大，那么传递的时间开销就很难接受。
6. 不要将变量作为输入输出参数。这样做的结果往往就是多个线程同时修改同一份变量，造成冲突，最终导致不可预测的结果。
7. 在编写OpenCL/CUDA程序时，务必熟悉并理解不同内存访问模式带来的影响。特别是对内存共享的关注，要全面而深入。

## CUDA编程模型的特点及限制
### 数据类型
CUDA编程语言支持两种数据类型——浮点型(float)和整型(int)。单精度浮点型对应C语言中的float类型，双精度浮点型对应C语言中的double类型。GPU上整数运算比起CPU上要快很多，因此一般用32位整型(int)表示数据。

CUDA还支持两种结构化数据类型——矢量(vector)，存储在矢量寄存器中，用于向量运算；阵列(array)，类似于C语言中的数组。

### 内存管理
CUDA编程语言提供了三个内存管理机制：统一虚拟地址空间(unified virtual address space)，页锁定(page locking)，和占位符(placeholder)。

统一虚拟地址空间机制让GPU上的所有线程都能访问整个虚拟地址空间，从而降低了内存管理的复杂度。但是，它也意味着只能从主机访问内存，而不能直接访问GPU显存。因此，需要通过GPU驱动程序(cudaDriver API)来访问显存。

页锁定机制允许用户锁定一段内存，使其不能被其他线程访问。虽然页锁定的功能很强大，但是如果频繁申请、释放锁，就会严重影响性能。因此，适当地使用页锁定机制可以提高应用程序的性能。

占位符机制允许用户将一段内存分配为页大小的倍数，并且指定该内存为只读或只写。只有满足特定条件才能读取或写入占位符内存。占位符机制可以用来实现缓存系统，能够极大地提升性能。

### 动态内存分配
CUDA编程语言支持动态内存分配，用户可以在运行时调用函数malloc()和free()来申请和释放内存。但是，这种方法需要程序员自行负责内存管理，容易产生内存泄漏的问题。

### 并行性
CUDA编程语言支持多种并行编程模型。最简单的并行模型是内核(kernel)，它是指在GPU上并行执行的小段代码。CUDA编程语言还支持集体(warp)、线程束(block)和网格(grid)等并行模型。集体(warp)是一个固定大小的线程组，支持单指令流多数据流(SIMD)或多指令流多数据流(MIMD)的并行计算。线程束(block)是一个由多条线程组成的集合，它们共享相同的全局内存和本地内存，支持同时在多个线程间通信。网格(grid)是一个由多块线程束组成的集合，它们共享相同的全局内存和本地内存。

CUDA编程语言不支持进程(process)或线程间的同步，因此必须由主机程序来管理并行线程之间的同步。CUDA编程语言提供了一系列的同步机制，包括内存屏障(memory barrier)、事件(event)和同步块(syncthreads)等。

除了支持多种并行编程模型之外，CUDA编程语言还提供一些其他特性，包括自动矢量化(auto vectorization)、自动分层(auto tiling)、动态 shared memory、缓存机制(caching mechanism)、断言机制(assertion mechanism)等。

### 系统调用
CUDA编程语言不支持系统调用，所有系统调用必须在主机程序中实现。不过，CUDA编程语言提供了几个库函数，可以方便地调用系统调用。例如，libcurand提供了随机数生成相关的函数，libcublas提供了线性代数相关的函数，libculibos提供了一些并行编程相关的函数。

### 兼容性
CUDA编程语言兼容于NVIDIA图形处理单元(GPGPU)的硬件，包括 NVIDIA GTX系列、RTX系列和 Tesla V100等主流芯片。但是，目前还不完全兼容AMD卡。

## CUDA编程环境配置
CUDA编程环境包括NVCC（NVIDIA C++ Compiler）、cuDNN、CUTLASS、 thrust、 nvprof工具包。下面分别介绍各个工具的安装与配置。

### NVCC
NVCC（NVIDIA C++ Compiler）是一个开源的C++编译器，可以将C++源文件编译为GPU设备可识别的二进制代码。NVCC可以用于编译单独的文件或者整个工程，也可以与CMake集成。


配置安装完毕后，将`bin/`目录添加至环境变量PATH中，然后检查nvcc版本是否成功。

```shell
$ nvcc --version
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2020 NVIDIA Corporation
Built on Wed_Jul_22_19:09:09_PDT_2020
Cuda compilation tools, release 11.0, V11.0.221
Build cuda_11.0_bu.TC445_37.28845127_0
```

### cuDNN
cuDNN(Convolutional Neural Network DNN Library)是一个深度学习框架，是专门针对CUDA平台的卷积神经网络的高性能库。它包括卷积操作的优化实现、池化操作的优化实现、归一化层的实现等。


下载cuDNN安装包之后，根据安装文档一步步进行安装。

配置安装完毕后，将`include/`、`lib64/`和`bin/`目录添加至环境变量CPATH、LIBRARY_PATH和PATH中，然后验证cuDNN版本是否成功。

```shell
$ cudnn_configures=$(find /usr/local/cuda-11.0/targets/x86_64-linux/ -name "libcudnn*")
$ echo $cudnn_configures
/usr/local/cuda-11.0/targets/x86_64-linux/lib/libcudnn.so.8.0.5
$ strings $cudnn_configures | grep cudnnVersion
cudnnVersionMajor = 800
cudnnVersionMinor = 5
```

### CUTLASS
CUTLASS是NVIDIA推出的用于GPU的通用矩阵类库。它定义了一套高性能的数学函数接口，包括用于矩阵乘法和切片的batched matrix multiplication/slice kernels，用于向量加法的reduction kernels，以及用于softmax激活函数的device-wide kernels。CUTLASS可以与Thrust库或CUDA编程模型配合使用。


下载源码之后，根据README.md里面的说明配置并安装CUTLASS。

```shell
$ mkdir build && cd build
$ cmake.. # 配置与安装
```

配置安装完毕后，将`include/`、`tools/`和`library/`目录添加至环境变量CPATH和PATH中，然后验证CUTLASS版本是否成功。

```shell
$ cutlass_version=1.2.0
$ cp cutlass_${cutlass_version}/include/* /usr/local/cuda-11.0/include/.
$ cp cutlass_${cutlass_version}/build/lib*/libcutlass_* /usr/local/cuda-11.0/lib64/.
$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.0/lib64:$PWD/cutlass_${cutlass_version}/build/lib/
$ strings $(which test_cutlass_install) | grep Version
Version: ${cutlass_version}.0;
```

### Thrust
Thrust是NVIDIA推出的用于GPU编程的开源C++模板库。它封装了CUDA编程模型中涉及到的方方面面，包括初始化和终止运行时系统，内存管理，并行计算，错误处理等。Thrust的目的是为GPU编程提供易用的高级API，让开发人员只需关注算法逻辑即可快速地开发应用。


下载源码之后，根据INSTALL.txt里面的说明配置并安装Thrust。

```shell
$ git clone https://github.com/thrust/thrust
$ cd thrust
$./configure.py
$ make all install
```

配置安装完毕后，将`include/`和`bin/`目录添加至环境变量CPATH和PATH中，然后验证Thrust版本是否成功。

```shell
$ strings $(which test_thrust_install) | grep Thrust
Thrust Version: 1.9.9
```

### nvprof
nvprof是NVIDIA提供的性能分析工具，可以帮助用户查看GPU上程序的运行时间，包括Host端和Device端的时间。


下载完成后，将`bin/`目录添加至环境变量PATH中，然后验证nvprof版本是否成功。

```shell
$ nvprof --version
nvprof (Cuda Toolkit 11.0)
```