
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


在编写复杂的软件程序时，需要处理大量的数据输入输出操作。比如读取用户数据、向数据库写入数据、保存文件等等。但是计算机运行环境中的IO操作，其底层都是基于字节流、字符流实现的。对于不同的程序应用场景而言，有两种类型的IO流：字节流和字符流。
字节流（Byte Stream）就是指从一个源（如磁盘、网络等）读取一个字节到内存中进行处理。字节流经过编码和解码后就可以直接作为程序中的数据了。例如：图片文件的读写，视频文件的播放、下载等等。字节流提供了一些基本的方法来读取和写入字节数据，包括：InputStream、OutputStream、ByteArrayInputStream、ByteArrayOutputStream、FileOutputStream、FileInputStream等。
字符流（Character Stream）是基于字节流构建的。不同之处在于它是以字符的方式进行操作，所以可以方便地处理文本文件、xml文档、数据库查询结果等等。在Java语言中，最常用的字符流就是BufferedReader和PrintWriter。它们可以将字节流转换成字符流，从而方便地对文本文件、数据库记录等进行读取和写入。
# 2.核心概念与联系
## 字节流类


java.io包里面的所有输入输出流都属于字节流类。输入输出流都扩展自InputStream和OutputStream抽象类。

- InputStream是所有的输入流类的父类，它只提供了读取方法read()。子类有 ByteArrayInputStream, FileInputStream, PipedInputStream, StringBufferInputStream等。
- OutputStream是所有的输出流类的父类，它只提供写入方法write()。子类有 ByteArrayOutputStream, FileOutputStream, PrintStream, FileOutputStream等。
- Reader是所有的字符输入流类的父类，它只提供了读取方法read()。子类有 BufferedReader, InputStreamReader, StringReader等。
- Writer是所有的字符输出流类的父类，它只提供写入方法write()。子类有 BufferedWriter, FileWriter, PrintWriter, OutputStreamWriter等。

这四个类的关系：InputStream -> ByteArrayInputStream -> ByteArrayInputStreamImpl

如果要从InputStream读取一个字节数组，可以用toByteArray()方法：
```java
public byte[] toByteArray(){
    int size = available(); // 获取当前缓冲区大小
    if (size == 0) {
        return new byte[0];
    }

    byte[] buf = new byte[size]; // 创建字节数组
    int offset = position;

    synchronized(this){
        try{
            System.arraycopy(buf, 0, backingArray(), arrayOffset()+position, size); // 拷贝缓冲区数据到字节数组
        } finally {
            clear(); // 清空缓冲区
        }
    }
    
    return buf;
}
```
这个方法会将输入流的所有剩余字节拷贝到一个新的字节数组中。

如果要从InputStream读取到一个临时文件，可以使用BufferedInputStream和 FileOutputStream一起完成：
```java
import java.io.*;

public class ReadToTempFile {
    public static void main(String[] args) throws IOException {
        String inputFilePath = "input.txt";

        InputStream inputStream = null;
        FileOutputStream outputStream = null;

        try {
            inputStream = new FileInputStream(inputFilePath);

            // 设置缓冲区大小为1M
            inputStream = new BufferedInputStream(inputStream, 1024*1024);

            long startTimeMillis = System.currentTimeMillis();

            String tempFileName = inputFilePath + ".temp";
            outputStream = new FileOutputStream(tempFileName);
            
            byte[] buffer = new byte[1024*1024];
            int len;

            while ((len = inputStream.read(buffer))!= -1) {
                outputStream.write(buffer, 0, len);
            }

            long endTimeMillis = System.currentTimeMillis();

            System.out.println("Temp file: " + tempFileName);
            System.out.println("Read time: " + (endTimeMillis - startTimeMillis) + " ms");
            
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (outputStream!= null) {
                outputStream.close();
            }
            if (inputStream!= null) {
                inputStream.close();
            }
        }
    }
}
```
这个例子演示了如何从InputStream读取到一个临时文件，并获取读取耗时。

## 文件操作
### 读取文件
#### 从文件中读取字节
为了从文件中读取字节数据，可以使用BufferedInputStream或者直接调用InputStream的read()方法。

假设有一个txt文件，里面存储了二进制数据，可以使用以下代码读取：
```java
public class ReadFromFile {
    public static void main(String[] args) throws IOException {
        String filePath = "/path/to/file.txt";
        
        InputStream inputStream = null;

        try {
            inputStream = new FileInputStream(filePath);

            // 设置缓冲区大小为1M
            inputStream = new BufferedInputStream(inputStream, 1024*1024);

            int b;
            while((b=inputStream.read())!=-1){
                System.out.print((char)b);
            }
            
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (inputStream!= null) {
                inputStream.close();
            }
        }
    }
}
```
这里创建了一个FileInputStream对象，并设置缓冲区大小为1MB。然后调用read()方法逐个读取文件中的字节，并打印出来。

#### 从文件中按行读取
如果要从文件中按行读取字节数据，可以使用BufferedReader。

示例如下：
```java
public class ReadLineFromFile {
    public static void main(String[] args) throws IOException {
        String filePath = "/path/to/file.txt";
        
        BufferedReader reader = null;

        try {
            reader = new BufferedReader(new FileReader(filePath));

            String line = "";
            while ((line = reader.readLine())!= null) {
                System.out.println(line);
            }
            
        } catch (Exception e) {
            e.printStackTrace();
        } finally {
            if (reader!= null) {
                reader.close();
            }
        }
    }
}
```
创建一个BufferedReader对象，传入FileReader对象作为参数。然后调用readLine()方法一次读取一行数据，并打印出来。