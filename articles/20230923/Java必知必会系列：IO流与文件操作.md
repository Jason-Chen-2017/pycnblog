
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## IO(Input/Output)流概述
IO(Input/Output)流是java中用于处理输入输出设备的基础类库。在java中提供了四种流类型：InputStream、OutputStream、Reader、Writer，分别负责读入和写入字节、字符序列数据；通过相应的方法可以从某个地方读取数据到内存中或者将内存中的数据写入到磁盘或网络中。每个流都实现了Closeable接口，使其在使用的过程中可以自动关闭资源。当不需要访问某个流时，应该调用它的close()方法释放系统资源，否则可能导致程序运行不稳定或资源泄露等问题。
## 文件操作
文件操作主要涉及到三个类：File、RandomAccessFile和FileInputStream/FileOutputStream。其中，File是一个抽象类，表示文件或目录路径名，提供对文件的各种属性和操作方法；RandomAccessFile则是一个扩展类，继承于java.io.InputStream和java.io.OutputStream，并实现了seek()方法，可随机读写文件；FileInputStream和FileOutputStream则是用来打开一个文件输入/输出流的类，支持字节和字符两种数据类型。
## File类的常用方法
- public boolean exists()：检查文件是否存在。
- public String getPath()：获得文件的绝对路径名。
- public long length()：获得文件的长度（以字节为单位）。
- public boolean delete()：删除文件。
- public boolean mkdirs()：创建多级文件夹。
- public boolean renameTo(File dest)：重命名文件。
- public File[] listFiles()：列出当前目录下的文件和目录。
- public static boolean createNewFile()：创建一个新的空文件。
## RandomAccessFile类常用方法
- public void seek(long pos)：移动到指定位置处读取或写入。
- public int read()：读取单个字节的数据。
- public int read(byte b[])：批量读取字节数组。
- public int write(int b)：向文件写入单个字节。
- public int write(byte b[])：批量写入字节数组。
## FileInputStream类常用方法
- public FileInputStream(String name) throws FileNotFoundException：构造方法，传入文件路径名。
- public int read()：读取单个字节的数据。
- public int read(byte b[], int off, int len)：读取字节数组的一部分到缓冲区。
## FileOutputStream类常用方法
- public FileOutputStream(String name) throws FileNotFoundException：构造方法，传入文件路径名。
- public void write(int b)：写入单个字节。
- public void write(byte b[], int off, int len)：写入字节数组的一部分。
# 2.基本概念术语说明
## 文件
文件，是一种存放在存储设备上的信息，它由一组数据且能被识别的二进制符号集合组成，其唯一标识符就是文件的名称。文件通常分为两大类：文本文件和二进制文件。文本文件是以人类可读的方式进行记录的，比如纯文本文档、歌词、电子邮件等。而二进制文件是指计算机只能识别、操纵和处理的机器语言指令、图形图像和其他形式的无结构化数据。所有的文件都属于磁盘上存储空间中的一个逻辑实体，具有独立于存储媒体的格式、大小、结构和位置的特征。
## 文件操作
文件操作是在计算机中对文件进行创建、修改、删除、复制、查找等操作的一系列过程，如创建新文件、删除已有文件、读取文件内容、写入文件内容、打开或关闭文件等。文件操作的基本单位是文件，即一次文件操作往往涉及多个相关联的文件。例如，拷贝文件前需要确认目标文件不存在，才能确保文件内容的正确性和完整性。同样，文件的保存、删除和修改也须遵循合理的目录组织方式，防止数据丢失或损坏。
## 字节
字节（Byte），也称八位字节，是计算机中最小的数据单位。一个字节的计量单位是bit（比特），常用的位数有7、8或9个。通常情况下，使用unsigned char定义一个字节变量，所以字节变量的取值范围就是0～255，即0x00~0xFF。
## 字符编码
字符编码，是将计算机内部使用的二进制编码表示的信息转换为可读文字形式的过程。字符编码的目的是为了能够实现跨平台的文本处理。目前通行的字符编码有ASCII编码、GBK编码、UTF-8编码、Unicode编码等。
### ASCII编码
ASCII编码是最早的字符编码标准，采用7位二进制表示一个字符。其中，英文字母、数字和一些特殊符号都可以使用对应的十进制编码来表示。这种编码虽然简单但不能显示所有字符，仅适用于英文环境。
### GBK编码
GBK编码是中国国家标准 GB 18030 的变体，采用两个字节表示一个字符。GBK编码集中在汉字字符、日文字符、韩文字符、俄文字符等中文应用中。GBK编码与ASCII编码兼容，但不包括所有字符。因此，GBK编码适用于少数中国地区的使用。
### UTF-8编码
UTF-8编码是一种针对Unicode的可变长字符编码，它可以表示世界上几乎所有语言的字符。它的编码规则和GBK编码类似，但也加入了更多的编码方式。
### Unicode编码
Unicode编码是计算机科学领域里的一项业界标准，也是唯一推荐的国际标准。它对每一个字符都分配一个唯一的编号，这样就可以表示全世界所有语言和文化了。
## 流
流是计算机中信息传输的渠道。输入流一般是指数据源提供给计算机处理的过程；输出流一般是指计算机处理后把结果输出的过程。由于输入输出设备之间的差异性很大，因此，流的实现方式也有所不同。常见的流有字节输入流InputStream、字节输出流OutputStream、字符输入流Reader、字符输出流Writer等。
## 消息Digest
消息摘要算法，又称哈希函数、散列函数或信息认证码，是由密码学家乌索普·维吉尼亚创立，用来计算数据（又称“消息”）的固定长度值，并用此值来鉴别、验证数据完整性的方法。常用的消息摘要算法有MD5、SHA-1等。