
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


在Kotlin语言中，对文件的读写操作都是通过标准库中的kotlin.io包来完成的，其主要类包括BufferedReader、BufferedWriter、File、InputStreamReader、OutputStreamWriter等。但是，这些类都不是线程安全的，因此，为了更加安全地并发访问文件，官方建议使用 kotlinx.coroutines中的coroutine APIs或其他线程安全机制（如java.nio.channels包）进行封装。本文将阐述Kotlin文件操作及IO相关的基本知识和技巧。


# 2.核心概念与联系
## 文件路径和目录
### Linux/Unix下的文件路径和目录结构
一个典型的Linux/Unix文件系统的目录结构如下图所示:


根目录(/)是整个文件系统的顶层文件夹，里面包含两个重要的子目录：bin 和 lib 。其中 bin 是存放可执行二进制文件（包括命令行工具）的地方，而 lib 是存放各种共享库（动态链接库）的地方。其他所有的目录都可以看作是分层次的，各个目录按照功能或用途划分成不同的层级。例如，usr 目录存放所有用户相关的文件，其中包含了可安装的软件包和应用程序，var 目录存放运行时产生的数据，log 目录存放系统日志。

在Linux/Unix下，每个文件都有一个路径名（pathname），表示它在文件系统中的位置。比如，"/usr/local" 表示本地计算机上的 "/usr/local" 目录。路径名分为绝对路径和相对路径两种。

1. 绝对路径：从根目录(/ )开始，表示文件的完整确切地址。例如，"/usr/local/bin/ls" 表示本地计算机上的 "/usr/local/bin/" 目录下的 ls 命令所在的绝对路径。
2. 相对路径：不以斜杠开头，表示文件相对于当前工作目录的位置。例如，"./foo" 表示当前工作目录下的 "foo" 文件，"../bar" 表示上级目录下的 "bar" 文件。

### Windows下的文件路径和目录结构
Windows操作系统的文件路径和目录结构也遵循着树状层次结构。每台PC机硬盘的根目录被称为磁盘代号C:\ (C:代表盘符)，之后被称为 C 盘。同样，其他盘符也可以用来标识别盘。

在Windows下，路径名由盘符、冒号(:)和斜杠(/)组成，比如，"D:\Program Files\Java\jdk-9\bin\java.exe" 表示的是 D 盘的 "Program Files" 目录下面的 "Java" 子目录下面的 jdk-9 目录下面的 bin 目录里面的 java.exe 执行文件。与Linux/Unix不同，Windows下的文件路径使用反斜杠(\ )作为分隔符，而不是正斜杠(/)。

## File类
File类是一个抽象类，提供了很多实用的方法用于处理文件及目录，其中包括创建、删除、重命名、检查是否存在、获取大小、读取文本、写入文本、修改权限等。

创建File对象的方法有三种：

1. 通过构造函数传入路径名字符串：

```kotlin
val file = File("/path/to/the/file")
```

2. 在当前路径下创建一个新文件：

```kotlin
val file = File("new_file.txt").apply {
    writeText("Hello World!") //写入文本
}
```

3. 使用系统默认目录创建一个新文件：

```kotlin
val homeDir = System.getProperty("user.home")
val newFile = File("$homeDir/new_file.txt").apply {
    createNewFile()   //创建新的文件
    appendText(", welcome to the world of files.")    //追加文本到文件末尾
}
```

除了以上三个方式外，还有一些其它的方法，可以通过调用相应方法的属性或者调用对象的方法来实现，如：`isFile()`、`exists()`、`length()`、`canWrite()`等。

## FileInputStream和 FileOutputStream类
FileInputStream类和FileOutputStream类是用来读写文件的输入流和输出流，分别继承自InputStream和OutputStream类。

打开文件流的方法有两种：

1. 通过构造函数传入File对象：

```kotlin
val input = FileInputStream(File("/path/to/the/file"))
val output = FileOutputStream(File("/path/to/the/output/file"))
```

2. 传入路径名字符串：

```kotlin
val input = FileInputStream("/path/to/the/file")
val output = FileOutputStream("/path/to/the/output/file")
```

与File类的创建类似，还可以使用open()函数打开一个文件：

```kotlin
val inputStream = openStream("/path/to/the/file", "r") as InputStream
val outputStream = openStream("/path/to/the/output/file", "w") as OutputStream
```

通过read()方法读取字节，然后写入ByteArrayOutputStream中，或者写入BufferedWriter中，最后再把字节数组转换成字符串：

```kotlin
fun readBytesAndConvertToString(): String? {
    val buffer = ByteArrayOutputStream()

    try {
        val bytesRead = ByteArray(1024)
        var byteCount = inputStream.read(bytesRead)

        while (byteCount > -1) {
            buffer.write(bytesRead, 0, byteCount)

            byteCount = inputStream.read(bytesRead)
        }

        return buffer.toString("UTF-8")
    } catch (e: Exception) {
        e.printStackTrace()
        return null
    } finally {
        buffer.close()
    }
}
```

写入字节的方法与之类似，只不过这里需要指定要写入的字节数组：

```kotlin
fun writeStringAsBytes(input: String): Boolean {
    if (!outputFile.exists()) {
        outputFile.createNewFile()
    }

    try {
        val outputStream = FileOutputStream(outputFile)
        outputStream.write(input.toByteArray())
        outputStream.flush()
        outputStream.close()

        println("Successfully wrote $input to ${outputFile.absolutePath}")
        return true
    } catch (e: IOException) {
        e.printStackTrace()
        return false
    }
}
```