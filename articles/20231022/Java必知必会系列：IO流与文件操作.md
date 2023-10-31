
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


## 概述
计算机系统中需要处理很多种输入输出的数据，比如文本、图像、声音、视频等等，这些数据通常通过文件或网络的方式存储在磁盘上或者通过互联网传输到另一台计算机中。程序需要读取、写入、修改这些文件中的数据，这就需要对文件的读写进行管理。对于IO操作来说，关键点包括打开、关闭、读、写、定位、复制、删除等操作。通过这几乎所有的文件系统操作，都可以实现对文件的各种操作。理解如何利用Java的IO操作机制进行文件的操作非常重要。本文将详细阐述Java I/O的相关知识。

## 功能概览
Java SE从1.0版本起提供了标准的I/O库（Input/Output，即输入输出），其主要用于处理各种输入输出的数据，如文件、网络、控制台等。常用的I/O类及方法如下表所示：

| 类名 | 方法 | 描述 |
|---|---|---|
| java.io.File | exists() | 判断文件或目录是否存在 |
| | mkdirs() | 创建目录（若不存在） |
| | delete() | 删除文件或目录 |
| | renameTo() | 重命名文件或目录 |
| | listFiles() | 列出目录下的文件列表 |
| java.io.RandomAccessFile | read() | 从文件中读取字节数据 |
| | seek() | 设置当前位置 |
| | write() | 将字节数据写入文件 |
| | close() | 关闭随机访问文件 |
| java.nio.channels.FileChannel | open() | 打开文件输入输出通道 |
| | transferFrom() | 从源通道向目标通道传输字节数据 |
| | transferTo() | 从源通道向目标通道传输字节数据 |
| | force() | 强制刷新输出缓冲区 |
| java.io.InputStreamReader | read() | 读取字符数据 |
| | close() | 关闭输入流 |
| java.io.BufferedReader | readLine() | 读取一行字符串 |
| | close() | 关闭输入缓冲流 |
| java.io.OutputStreamWriter | write() | 写入字符数据 |
| | close() | 关闭输出流 |
| java.io.BufferedWriter | newLine() | 写入换行符 |
| | flush() | 清空输出缓存 |
| | close() | 关闭输出缓冲流 |
| javax.swing.JFileChooser | showOpenDialog() | 打开选择文件对话框 |
| | showSaveDialog() | 打开保存文件对话框 |
| | setFileSelectionMode() | 设置文件选择模式 |
| | addChoosableFileFilter() | 添加可选文件过滤器 |
| javax.xml.transform.stream.StreamResult | StreamResult(java.io.Writer) | 指定结果用Writer输出 |
| javax.xml.transform.stream.StreamSource | StreamSource(java.io.InputStream) | 指定要解析XML文档的输入流 |
| org.w3c.dom.DocumentBuilder | parse(java.io.InputStream) | 使用DOM构建Document对象 |

## 文件操作的特点
文件操作在应用程序开发过程中占有重要的地位。文件的操作包括创建、打开、关闭、读写、定位、复制、删除等。其中，读写、定位、删除等基本操作依赖于文件指针的操作。下面简要描述这些操作的特点。
### 打开、关闭文件
Java中使用File类的`exists()`、`mkdirs()`、`delete()`、`renameTo()`方法可以实现文件的打开、关闭、创建、删除、重命名等操作。例如，创建一个新文件，并写入一些字符：
```java
import java.io.*;

public class FileDemo {
    public static void main(String[] args) throws Exception{
        // 创建一个文件，若该文件已存在则覆盖原文件
        FileOutputStream fos = new FileOutputStream("newfile");
        byte b[] = "hello world".getBytes();
        for (int i=0;i<b.length;i++) {
            fos.write(b[i]);
        }
        fos.close();

        // 验证该文件是否已经存在
        if (new File("newfile").exists()) {
            System.out.println("newfile is created successfully!");
        } else {
            System.out.println("Error: failed to create newfile.");
        }
        
        // 删除新建的文件
        boolean success = new File("newfile").delete();
        if (success) {
            System.out.println("newfile is deleted successfully!");
        } else {
            System.out.println("Error: failed to delete newfile.");
        }
    }
}
```
此外，还可以使用try-with-resources语句自动关闭资源，如：
```java
import java.io.*;

public class AutoCloseDemo {
    public static void main(String[] args) throws Exception{
        try (FileInputStream fis = new FileInputStream("oldfile");
             FileOutputStream fos = new FileOutputStream("newfile")) {

            int data;
            while ((data = fis.read())!= -1) {
                fos.write(data);
            }
            
            System.out.println("Copy file completed!");
            
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
```
### 读文件
Java中使用`RandomAccessFile`类可以实现文件的读操作，该类提供两个主要的方法`seek()`和`read()`，分别用来设置当前读取位置和读取字节数组。例如，读取一个二进制文件的内容：
```java
import java.io.*;

public class ReadBinaryFile {
    public static void main(String[] args) throws Exception{
        RandomAccessFile raf = new RandomAccessFile("binary", "r");
        
        long len = raf.length();
        System.out.println("The length of the file is:" + len);
        
        long pos = 0;
        StringBuilder sb = new StringBuilder((int)len);
        
        while (pos < len) {
            char c = (char)(raf.read());
            sb.append(c);
            pos++;
        }
        
        String content = sb.toString();
        System.out.println("Content of the file:\n" + content);
        
        raf.close();
    }
}
```
也可以使用`BufferedReader`来按行读取文件内容，但速度较慢。
```java
import java.io.*;

public class ReadTextFile {
    public static void main(String[] args) throws Exception{
        BufferedReader br = new BufferedReader(new FileReader("text"));
        
        String line;
        while ((line = br.readLine())!= null) {
            System.out.println(line);
        }
        
        br.close();
    }
}
```
### 写文件
Java中使用`RandomAccessFile`类可以实现文件的写操作，该类也提供了两个主要的方法`seek()`和`write()`，分别用来设置当前写入位置和写入字节数组。例如，往一个二进制文件中写入一些数据：
```java
import java.io.*;

public class WriteBinaryFile {
    public static void main(String[] args) throws Exception{
        RandomAccessFile raf = new RandomAccessFile("binary", "rw");
        
        long len = raf.length();
        System.out.println("The length of the file is:" + len);
        
        long pos = len;
        StringBuffer sb = new StringBuffer("Hello World!\n");
        
        raf.seek(pos);
        raf.write(sb.toString().getBytes());
        
        raf.close();
    }
}
```
如果要向一个文件写入多行字符串，建议使用`PrintWriter`类，它支持自动添加换行符。
```java
import java.io.*;

public class WriteTextFile {
    public static void main(String[] args) throws Exception{
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter("text")));
        
        pw.println("This is a text file.");
        pw.print("Here's some more data:");
        pw.println("Some text...");
        
        pw.flush();
        pw.close();
    }
}
```