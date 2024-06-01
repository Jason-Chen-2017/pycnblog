
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


Java I/O 是一种用来处理输入输出的机制，在 Java 中对文件、网络、数据库等的读写操作都依赖于 I/O 操作。由于 Java 的跨平台特性，使得 Java 可以运行在各种各样的平台上，因此 Java I/O 提供了一致的开发环境，能够更好地适应各种应用场景。同时，I/O 的性能也非常出色，能够实现高速的数据交换，提升系统的处理能力。

本教程的目标是在阐述 Java I/O 相关知识点时，既不仅限于语法或基础知识，还要兼顾实际需求和实际操作，并通过丰富的代码实例，向读者展示 I/O 流操作中常用的方法和用法，帮助读者更加全面地理解 Java I/O 操作。

本教程面向具有一定编程基础的技术人员，包括但不限于 Java 初级到高级开发者，熟悉计算机基本知识、数据结构和算法，有实际工作经验或者在做相关工作。文章主要包括以下四个部分：

1. I/O 概念及其关系；
2. Java I/O 流的分类及其区别；
3. 文件、字节数组、字符数组、字符串之间的相互转换；
4. Java I/O 流的主要方法和用法。

# 2.核心概念与联系
## 2.1 I/O 概念及其关系
计算机程序中输入输出（Input/Output，I/O）是指计算机从外围世界接受信息、处理信息、输出结果的过程，也是计算机系统之间通信的接口。I/O 有两个重要的方面：数据的输入和数据的输出。通常来说，输入数据的种类可以分为文件输入、网络输入、键盘输入等；输出数据的种类可以分为文件输出、网络输出、显示器输出等。I/O 设备又可分为两大类：存储设备（如硬盘、光盘、磁盘）和通信设备（如打印机、扫描仪、调制解调器）。因此，I/O 设备可以看作计算机系统中不可缺少的一部分。

I/O 和计算机系统间的通信方式有多种，最常见的两种通信方式分别是串口（Serial Ports）和并行端口（Parallel Ports），如下图所示。


串口通信方式就是利用串行通讯线路连接在一起的几个设备，一个设备就像一条电话线那样发送信息给其他的设备，反之亦然。串口通信时常使用 UART （Universal Asynchronous Receiver Transmitter ），它由 RX、TX 和 CTS / RTS 三根引脚组成，如下图所示。


并行端口通信方式就是直接与主板上的多个设备连接，这种方式虽然比串口通信速度快很多，但是信号极容易受到干扰。并行端口一般由 DB-25、DB-9 或 DB-232 等标准化的接口提供，如下图所示。


## 2.2 Java I/O 流的分类及其区别
Java 中存在着五种不同类型的 I/O 流：

1. 控制台输入输出流：InputStreamReader 和 OutputStreamWriter ，分别负责将控制台输入流（键盘输入、文件读取）转换为字符流，和将控制台输出流（显示器输出、文件写入）转换为字符流。
2. 数据源输入输出流：FileInputStream 和 FileOutputStream ，分别用于从文件中读取数据、将数据写入文件。
3. 基于内存缓冲的流：ByteArrayInputStream 和 ByteArrayOutputStream ，用于缓存字节数据和将缓存中的字节数据读出到内存中。
4. 对象流：ObjectInputStream 和 ObjectOutputStream ，用于序列化对象到文件或网络，反序列化对象从文件或网络中恢复。
5. 字符编码流：FileReader 和 FileWriter ，用于文件操作的编码转换。

下表汇总了这些流类型之间的关系和区别。

|    |控制台输入输出流|数据源输入输出流|基于内存缓冲的流|对象流|字符编码流|
|---|---|---|---|---|---|
|**方向**|控制台 → 字节流或字符流|数据源 → 字节流|字节流 → 内存缓存|对象 → 字节流|字符流 → 字节流|
|**操作对象**|**字符流**|**字节流**|**字节数组**|**对象**|**字节流**|
|**描述**|控制台输入输出流用于控制台读入、输出字符。|数据源输入输出流用于文件读写操作。ifstream 和ofstream 是 C++ 中的同名类，用于文件的读写操作。|基于内存缓冲的流用于缓存字节数据或将缓存中的字节数据读出到内存中。 ByteArrayInputStream 和 ByteArrayOutputStream 都是 Java 中的同名类，用于缓存字节数组。|对象流用于序列化、反序列化 Java 对象。|字符编码流用于文件操作的编码转换。FileReader 和 FileWriter 使用特定编码（如 GBK 或 UTF-8）将字符流读入或写出到文件。|
|**使用场合**|控制台界面、命令行程序、文本编辑器等。|文件系统操作、数据压缩等。|网络协议传输、图片处理等。|持久化保存 Java 对象、远程调用服务等。|处理编码错误、文本文件读写等。|.NET Framework 系统编程中使用的文件输入输出流 System.IO.|

## 2.3 文件、字节数组、字符数组、字符串之间的相互转换
由于文件读写操作和字符流操作需要不同的处理方式，因此在进行文件操作之前，首先需要将文件数据转换成字节流或字符流。字节流包括 FileInputStream 和 FileOutputStream ，它们可以读取和写入 byte[] 数组，而字符流包括 FileReader 和 FileWriter ，它们可以读取和写入 char[] 数组。除了以上介绍的常用文件、字节数组、字符数组之间的相互转换，Java I/O 流中还有一些其它有用的工具，比如以下几种：

1. DataInputStream 和 DataOutputStream ：处理二进制数据流。DataInputStream 和 DataOutputStream 可用于处理基本数据类型（如 int、long、float、double）、String 和 BigDecimal 。使用 DataInputStream 和 DataOutputStream 时，不需要考虑平台差异性，因为它们都是字节流。
2. ObjectInputStream 和 ObjectOutputStream ：用于序列化和反序列化 Java 对象。序列化是将 Java 对象转换成字节流的过程，反序列化则是将字节流重新构造成 Java 对象。ObjectInputStream 和 ObjectOutputStream 通过写入和读取特殊的字节序列来完成序列化和反序列化。使用 ObjectInputStream 和 ObjectOutputStream 时，需要注意平台差异性，因为它们不是标准的字节流。
3. Charset ：用于指定编码集。在 Java NIO 中，ByteBuffer 支持多种编码集，可以通过 Charset 指定使用的编码集。
4. BufferedInputStream 和 BufferedOutputStream ：为了提高效率，增加缓冲区大小。BufferedInputStream 和 BufferedOutputStream 在使用底层 InputStream 和 OutputStream 后，包装了一层缓冲区，读写速度更快。
5. PipedInputStream 和 PipedOutputStream ：允许线程间通过管道通信。PipedInputStream 和 PipedOutputStream 分别在两个线程间提供了一个双向通道。
6. InputStreamReader 和 OutputStreamWriter ：用于将字节流转换成字符流。 InputStreamReader 和 OutputStreamWriter 可根据指定的编码集将字节流转换为字符流，并且提供了字符编码转换功能。

## 2.4 Java I/O 流的主要方法和用法
### 2.4.1 文件相关的方法
Java 中常用的文件相关的类是 java.io.File 类。

**创建文件**：

```java
// 创建文件，如果父目录不存在，则创建父目录
File file = new File("test.txt");
if (!file.exists()) {
    if (file.mkdirs()) {
        System.out.println("创建成功！");
    } else {
        System.out.println("创建失败！");
    }
} else {
    System.out.println("文件已存在！");
}
```

**创建目录**：

```java
// 创建目录，如果父目录不存在，则创建父目录
boolean success = false;
if (!dir.exists() && dir.isDirectory()) {
    success = dir.mkdirs();
}
if (success) {
    System.out.println(dir + " 创建成功！");
} else {
    System.out.println(dir + " 创建失败！");
}
```

**删除文件或目录**：

```java
// 删除文件或目录，如果存在的话
if (fileOrDir.exists()) {
    boolean success = true;
    // 如果是目录，递归删除所有子文件和子目录
    if (fileOrDir.isDirectory()) {
        String[] children = fileOrDir.list();
        for (int i=0; i<children.length; i++) {
            success &= delete(new File(fileOrDir, children[i]));
        }
    }
    if (success && fileOrDir.delete()) {
        System.out.println(fileOrDir + " 删除成功！");
    } else {
        System.out.println(fileOrDir + " 删除失败！");
    }
} else {
    System.out.println(fileOrDir + " 不存在！");
}
```

**获取文件属性**：

```java
// 获取文件属性
System.out.println("文件名：" + file.getName());
System.out.println("绝对路径：" + file.getAbsolutePath());
System.out.println("文件大小：" + file.length() + " bytes");
System.out.println("是否存在：" + file.exists());
System.out.println("是否是文件：" + file.isFile());
System.out.println("是否是目录：" + file.isDirectory());
System.out.println("最后修改时间：" + new Date(file.lastModified()));
```

### 2.4.2 字节数组相关的方法
在 Java 中，字节数组是用来表示二进制数据的容器。

**新建字节数组**：

```java
byte[] data = new byte[(int) file.length()];
```

**从文件读取字节数组**：

```java
try (InputStream in = new FileInputStream(file)) {
    int len;
    while ((len = in.read(data))!= -1) {}
} catch (IOException e) {
    e.printStackTrace();
}
```

**从字节数组写入文件**：

```java
try (OutputStream out = new FileOutputStream(file)) {
    out.write(data);
} catch (IOException e) {
    e.printStackTrace();
}
```

### 2.4.3 字符数组相关的方法
在 Java 中，字符数组是用来表示 Unicode 字符的数据容器。

**新建字符数组**：

```java
char[] chars = new char[(int) file.length()];
```

**从文件读取字符数组**：

```java
try (Reader reader = new FileReader(file)) {
    int len;
    while ((len = reader.read(chars))!= -1) {}
} catch (IOException e) {
    e.printStackTrace();
}
```

**从字符数组写入文件**：

```java
try (Writer writer = new FileWriter(file)) {
    writer.write(chars);
} catch (IOException e) {
    e.printStackTrace();
}
```

### 2.4.4 字符串相关的方法
在 Java 中，字符串是用来表示 Unicode 字符序列的数据容器。

**新建字符串**：

```java
StringBuilder sb = new StringBuilder((int) file.length());
```

**从文件读取字符串**：

```java
try (BufferedReader br = new BufferedReader(new FileReader(file))) {
    String line;
    while ((line = br.readLine())!= null) {
        sb.append(line).append("\n");
    }
} catch (IOException e) {
    e.printStackTrace();
}
```

**从字符串写入文件**：

```java
try (PrintWriter pw = new PrintWriter(new FileWriter(file), true)) {
    pw.print(sb);
} catch (IOException e) {
    e.printStackTrace();
}
```