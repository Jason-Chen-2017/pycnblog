
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


## 概述
Java语言的I/O流用于对文件、网络、控制台等各种输入输出源进行输入输出操作。InputStream和OutputStream是java.io包中最基本的抽象类，它们分别表示输入流和输出流，处理字节数据的类别。Java I/O流支持数据的序列化和反序列化，本文将从流的分类、应用场景、数据类型、主要方法等方面详细介绍java的IO流。
## 定义及相关术语
### 流(Stream)
数据流动是指不同计算机之间传送的数据叫做数据流。数据流在计算机中起到一个中转站的作用，它包括的信息单位称为数据元素（Data Element）。数据流有两种流向：单向（只能从输入方向传输，或者只能从输出方向传输）和双向。双向数据流既可以向输入方向传输，又可以向输出方向传输。
### 字节流(Byte Stream)
字节流是一个基于字节的流，数据以字节形式进行读写。它的特点是以字节为基本单元，能够提供高效的数据传输。字节流包括InputStream和OutputStream两个接口，实现了基于字节的数据输入输出操作。InputStream继承自Object类，代表了字节输入流。OutputStream继承自Object类，代表了字节输出流。InputStream和OutputStream分别用于读取和写入字节数据。

### 字符流(Character Stream)
字符流是一个基于字符的流，数据以字符形式进行读写。它处理的是字节数据转换成字符的过程，因此只能用于处理文本数据。字符流包括Reader和Writer两个接口，分别继承于InputStream和OutputStream。InputStream和OutputStream的父类都是Object类，而Reader和Writer则继承于BufferedReader和BufferedWriter类，BufferedReader和BufferedWriter实现了带缓冲区的字符输入输出。 BufferedReader是从Reader中按行读取字符，BufferedWriter则是把字符写到Writer中。

### 文件输入输出流(File Input/Output Stream)
文件输入输出流用于处理文件，其父类都实现了Closeable接口，可自动关闭相应的资源，不需要手动关闭。主要包括FileInputStream、FileOutputStream、FileReader、FileWriter四个类。FileInputStream从文件读取字节数据，FileOutputStream从字节数组或其它输入流写入文件。FileReader和FileWriter类分别从文件或字符串中读取或写入字符数据。

### 数据报输入输出流(Datagram Input/Output Stream)
数据报输入输出流用于处理网络中的数据报。主要包括DatagramSocket、DatagramPacket、DatagramInputStream和DatagramOutputStream四个类。DatagramSocket用于接收和发送数据报，DatagramPacket是代表数据报的容器。DatagramInputStream和DatagramOutputStream类分别用于从字节流或其它输入流读取和写入数据报。

### 对象序列化流(Object Serialization Stream)
对象序列化流用于将对象的状态信息转换为字节序列，便于存储或传输。主要包括ObjectOutputStream和ObjectInputStream两类。ObjectOutputStream用于序列化对象，ObjectInputStream用于反序列化对象。通过序列化，可以保存对象的状态信息，并用它创建相同的对象。

## 分类
### 字节流和字符流
字节流和字符流是流的两种主要类型，每个类都有一个对应的父类，它们之间的区别就是它们处理的单位不同。字符流主要处理文本数据，只能处理单字节的字符编码；字节流则能处理任意字节数据。一般情况下，字节流用于处理二进制数据，如图片、视频和音频等；而字符流通常用于处理文本数据，如文档、网页、数据库记录等。

### 同步和异步流
同步流（也称为阻塞流）是指调用线程必须等待读写操作完成后才能继续执行；异步流（也称为非阻塞流）是指调用线程无需等待读写操作完成即可继续执行。同步流就像老式的电话机一样，一条命令要执行完毕之前不能接着执行下一条命令；而异步流则像手机一样，可以同时进行多条命令的执行。

### 节点流和装饰器流
节点流是指直接操作底层资源，比如文件的读写、网络通信等；装饰器流则是对已有的流进行额外的功能增强，比如缓冲流、加密流、压缩流等。

### 操作方式
流提供了不同的操作方式，如按字节读写、按字符读写、按块读写等。按字节读写的方式对应于 ByteArrayInputStream 和 ByteArrayOutputStream；按字符读写的方式对应于 FileReader 和 FileWriter；按块读写的方式则需要实现自己的 BufferedInputStream 和 BufferedOutputStream。

## 应用场景
### 文件复制
```
public class CopyFile {
    public static void main(String[] args) throws IOException {
        String src = "D:\\test\\src.txt"; //源文件路径
        String dest = "D:\\test\\dest.txt"; //目的地路径

        FileInputStream fis = new FileInputStream(src);
        FileOutputStream fos = new FileOutputStream(dest);
        
        int c;
        while ((c = fis.read())!= -1) {
            fos.write(c);
        }
        
        fis.close();
        fos.close();
    }
}
```
### 文件比较
```
import java.io.*; 

public class CompareFile { 
    public static void main(String[] args) throws Exception{ 
        String file1 = "D:/temp/file1.txt";//第一个文件的位置  
        String file2 = "D:/temp/file2.txt";//第二个文件的位置  
          
        BufferedReader br1=new BufferedReader(new FileReader(file1));  
        BufferedReader br2=new BufferedReader(new FileReader(file2));  
          
  
        String line1="",line2="";  
        boolean flag=true;//用来标志是否所有行均相等  
  
        while((line1=br1.readLine())!=null){  
            if(!flag)//判断是否所有行均相等  
                break;  
            while((line2=br2.readLine())!=null){  
                if(line1.equals(line2)){  
                    System.out.println("第"+(br1.getLineNumber()-1)+"行的内容相同");  
                    continue;  
                }else{  
                    System.out.println("第"+(br1.getLineNumber()-1)+"行的内容不相同");  
                    flag=false;  
                    break;  
                }  
            }  
        }  
        if(flag&&!"".equals(line2))//如果所有行均相等但是最后一行只有一半  
            System.out.println("第"+(br1.getLineNumber()-1)+"行的内容相同");  
          
        br1.close();  
        br2.close();  
    }  
}
```