
作者：禅与计算机程序设计艺术                    

# 1.简介
         
2010年9月Java5发布，在Java中引入了NIO（New I/O）和新的I/O包java.nio.*。这个包主要包含了一系列类用于处理各种各样的、高效的、面向缓冲区的I/O操作，包括网络I/O、文件I/O和内存映射I/O等。其中最重要的类就是ByteBuffer、CharBuffer、ShortBuffer、IntBuffer、LongBuffer、FloatBuffer和DoubleBuffer这些缓存区类型。缓冲区是一个容器对象，可以保存要传输的数据，并提供对数据的结构化访问。缓冲区主要用于支持NIO，它提供了一种机制来访问本地磁盘上的文件或通过网络发送数据。缓冲区也可用于在JVM内部进行通信，例如读写SocketChannel中的数据或者与内存映射的文件的交互。
          
          在本篇文章中，我将详细介绍Java NIO中的Buffer缓冲区及其相关概念、用法和特性，并给出一些重要的原理和算法。希望读者能够从中获益。
          
         # 2.基本概念及术语
         1. Buffer 概念：
         Buffer是一个容器对象，提供存取固定大小的连续内存块的方法。每个Buffer有一个当前位置指针，在缓冲区的任一端都可以标记一个元素的位置，从而可以使用不同的指针进行读写操作。对于数据结构而言，Buffer通常是一个字节序列，但也可以是一个其他结构，如字符串、数组或者对象。通俗地说，Buffer就是一个存储器，用来临时存放数据。当需要读取或写入某个特定的类型时，就可以用相应类型的Buffer。例如，如果要读取或写入字节数组，就应该使用ByteBuffer；如果要读取或写入字符串，就应该使用CharBuffer；如果要读取或写入整数数组，就应该使用IntBuffer等等。Buffer类定义了一个getInt()方法，用于从Buffer中读取一个int值，再用一个索引值确定要读取的位置。与此类似，Buffer还提供了putInt()方法，用于向Buffer中写入一个int值，再用另一个索引值确定写入的位置。下图展示了Buffer的组织结构：
        ![](https://upload-images.jianshu.io/upload_images/729281-c8a6d8f4cd2c8e4b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
         2. 术语表:
          - Channel: 是2个进程间通信的端点。Channel代表了从源到目标的流动方向，即只能单向传输信息。常见的Channel有FileChannel、SocketChannel、DatagramChannel等。
          - Selector：Selector是一个多路复用的抽象类。Selector会监听注册在自己身上的Channel，并根据Channel上发生的事件（比如新连接、数据到达等），通知相应的Handler来处理。
          - Buffer：Buffer是一个容器对象，可以保存要传输的数据，并提供对数据的结构化访问。Buffer包括三个核心属性：capacity、position和limit。capacity表示Buffer的总容量；position表示当前已经读写的位置，初始值为0，最大值为capacity – 1；limit表示Buffer中不能进行读写操作的位置，初始值为capacity，最大值为capacity。
          - Scattering Reads / Gathering Writes：Scattering Reads和Gathering Writes都是一种I/O操作模式，可以将数据分散或集成到多个Buffer中。
          - Mapped File：Mapped File也叫内存映射文件，它是一种将磁盘上的文件直接加载到内存的技术。
          
          除以上介绍的术语外，还有一些Buffer的子类。如ByteBuffer、CharBuffer、ShortBuffer、IntBuffer、LongBuffer、FloatBuffer和DoubleBuffer等。它们共同继承自父类Buffer，但是拥有独有的功能，如字节编解码和字符集编码。
          
          # 3.核心算法原理及具体操作步骤
          从上面介绍的基本概念和术语后，我们知道Buffer可以理解为一段连续的内存，可以通过相应的API对其进行操作。下面我们以ByteBuffer为例，来介绍一下Buffer的一些核心算法。
          1. 创建ByteBuffer
             ByteBuffer buf = ByteBuffer.allocate(10); //创建一个容量为10的ByteBuffer
             
          为什么创建ByteBuffer的时候需要指定它的容量呢？因为ByteBuffer中的内容是无限的，如果不指定它的容量，则无法分配足够的内存空间。当然，我们也可以选择从现有的内存中分配ByteBuffer，如下所示：
             
             
             
          ByteBuffer buf = ByteBuffer.wrap("Hello World".getBytes()); //从字节数组中创建一个ByteBuffer
              
              
          通过ByteBuffer.wrap()方法，我们可以把一个字节数组转换成ByteBuffer。
          
          另外，ByteBuffer还有一种比较高效的方式——使用allocateDirect()方法，该方法可以在非堆内存中分配ByteBuffer，这样可以提高性能。但是需要注意的是，这种方式需要调用unsafe类的功能，所以并不是所有的平台都支持。
          
          2. 填充数据到ByteBuffer
            int length = inStream.read(buf.array()); //从输入流读取数据到字节数组
            if (length == -1){ //若没有更多数据可读
                break;
            }
            buf.rewind(); //重设position指针为0，即准备再次读取数据
            buf.limit(length); //调整limit指针为实际读取的数据长度
            
            上述代码演示了如何从输入流中读取数据到ByteBuffer。首先使用inStream.read()方法从输入流中读取数据到字节数组中。然后再调用buf.wrap()方法将字节数组包装成ByteBuffer。由于ByteBuffer自带了数组，所以不需要额外的内存分配，这里使用的只有字节数组的一小部分。

            如果我们想读取整个字节数组的话，可以直接调用以下代码：
            
            
            
          byte[] bytes = new byte[buf.remaining()];
          buf.get(bytes);
          
          将读取的数据保存到一个字节数组中。
          
          3. 读取数据 from ByteBuffer to 其他类型
          
          使用flip()方法可以反转ByteBuffer的limit和position指针，使得可读指针指向第一个未读数据，并将可写指针指向最后一个空闲位置。
          
          
          byte[] dest = new byte[buf.limit()];
          buf.get(dest); //从Buffer中读取数据到byte数组
          
          将字节数组中的数据复制到别处。
          
          
          char[] chars = new char[buf.limit()];
          ((CharBuffer)buf).get(chars); //从CharBuffer中读取数据到char数组
          
          CharBuffer中也可以使用get()方法来读取数据。
          
          
          int[] ints = new int[buf.limit()/4];
          IntBuffer ibuf = buf.asIntBuffer();
          ibuf.get(ints); //从IntBuffer中读取数据到int数组
          
          只要知道读取数据的格式，就可以使用对应的ByteBuffer的方法来读取。
          
          4. 写入数据到ByteBuffer
          
          与读取数据一样，写入数据也是按照相同的过程进行。
          
          
          outStream.write(buf.array(), 0, length); //将数据从字节数组中写入输出流
          flipAgain(); //将buffer的内容反转，以便让新的数据覆盖旧的数据
          
          上述代码演示了如何从输出流中写入数据到ByteBuffer。首先使用outStream.write()方法将ByteBuffer中的内容写入到输出流中。然后再调用flipAgain()方法，将ByteBuffer的内容反转，以便让新的数据覆盖旧的数据。
          
          
          public void flipAgain(){
              int pos = buffer.position();
              int limit = buffer.limit();
              for(int i=pos;i<limit;i++){
                  System.out.print((char)(buffer.get(i)&0xff)); //先将数据拷贝到字节数组中
              }
              buffer.clear(); //清空buffer的内容
          }
          
          5. 分散/聚合 Reads and Writes

          分散/聚合是指一次读取多个缓冲区的数据到单个缓冲区，或者一次写入多个缓冲区的数据到单个缓冲区。下面给出Scattering Reads和Gathering Writes两种模式的Java实现。
          
          Scattering Reads
          
          ByteBuffer src1 =...;
          ByteBuffer src2 =...;
          ByteBuffer dst =...;
          ByteBuffer[] bufs = {src1, src2};
          channel.read(bufs);
          for(ByteBuffer b : bufs){
              while(b.hasRemaining()){
                  byte data = b.get();
                  dst.put(data);
              }
          }
          通过调用channel.read(bufs)，从SocketChannel或从管道中读取两个ByteBuffer中的数据。遍历两个ByteBuffer，分别读取它们的数据并写入到dst中。
          
          Gathering Writes
          
          ByteBuffer src =...;
          ByteBuffer dst1 =...;
          ByteBuffer dst2 =...;
          ByteBuffer[] bufs = {dst1, dst2};
          src.position(0);
          channel.write(bufs);
          src.position(src.limit());
          当SocketChannel或SocketWritableByteChannel需要发送数据时，可以将数据划分到两个ByteBuffer中。调用channel.write(bufs)，将数据写入到SocketChannel或SocketWritableByteChannel中。
          
        下面列出一些关键代码：
        
        ```
        ByteBuffer allocate(int capacity); //分配一个新的ByteBuffer，容量为capacity
        static ByteBuffer wrap(byte[] array); //包装一个字节数组为ByteBuffer
        void clear(); //重置position指针为0，重置limit指针为capacity
        boolean hasRemaining(); //判断是否还有剩余的字节，即position < limit
        int remaining(); //返回limit指针的值减去position指针的值
        void mark(); //记录当前position指针位置
        void reset(); //将position指针移回标记的位置
        void rewind(); //将position指针移回0
        long getFilePointer(); //获取当前文件的指针位置
        void setAutoExpand(boolean value); //设置自动扩展，默认为true
        boolean isAutoExpand(); //返回是否自动扩展
        void put(byte x); //将byte x放入buffer
        byte get(); //从buffer中取出一个byte
        ```
        
        6. 流水线模型

        NIO中还实现了一种流水线模型，可以让某些操作同时运行在多个Channel上。例如，可以通过Selector将SocketChannel读入缓存区，然后再将缓存区中的数据批量写入SocketChannel。下面是StreamPiplineDemo的简单实现。
        
        StreamPipelineDemo.java
        
        ```
        import java.io.*;
        import java.net.*;
        import java.nio.*;
        import java.nio.channels.*;
        import java.util.Arrays;
        
        public class StreamPipelineDemo {
        
            private static final String HOSTNAME = "localhost";
            private static final int PORT = 8888;
            private static final int BUFFERSIZE = 1024;
        
            public static void main(String[] args) throws Exception {
                
                Socket socket = new Socket(HOSTNAME, PORT);
            
                InputStream inputStream = socket.getInputStream();
                OutputStream outputStream = socket.getOutputStream();
        
                //声明多个缓冲区，用于读写
                ByteBuffer[] inputBuffers = new ByteBuffer[BUFFERSIZE];
                Arrays.setAll(inputBuffers, n -> ByteBuffer.allocate(BUFFERSIZE));

                ByteBuffer outputBuffer = ByteBuffer.allocate(BUFFERSIZE);
        
                //声明一个线程安全的Selector，负责监控各个通道
                Selector selector = Selector.open();
                ServerSocketChannel serverSocketChannel = socket.getChannel();
        
                SelectionKey key = serverSocketChannel.register(selector, SelectionKey.OP_ACCEPT);
        
                while (true) {
                    int readyChannels = selector.select();
                    
                    if (readyChannels == 0) continue;

                    Iterator<SelectionKey> selectedKeysIterator = selector.selectedKeys().iterator();
                    
                    while (selectedKeysIterator.hasNext()) {
                        SelectionKey sk = selectedKeysIterator.next();
                        
                        if (!sk.isValid()) {
                            selectedKeysIterator.remove();
                            continue;
                        }
        
                        if (sk.isAcceptable()) {
                            SocketChannel clientChannel = ((ServerSocketChannel)sk.channel()).accept();
                            
                            clientChannel.configureBlocking(false);
                            
                            SelectionKey ck = clientChannel.register(selector, SelectionKey.OP_READ | SelectionKey.OP_WRITE);
                            ck.attach(new InputOutputPair(clientChannel));
                            
                        } else if (sk.isReadable()) {
                            InputOutputPair pair = (InputOutputPair)sk.attachment();
                            SocketChannel clientChannel = pair.getClientChannel();
                        
                            int readBytesCount = clientChannel.read(pair.getInputBuffer());
                            
                            if (readBytesCount > 0) {
                                synchronized (outputBuffer) {
                                    outputBuffer.put(pair.getInputBuffer());
                                    outputBuffer.flip();
                                    
                                    clientChannel.write(outputBuffer);
                                
                                    outputBuffer.compact();
                                }
                            } else {
                                clientChannel.close();
                                sk.cancel();
                            }
                        }

                        selectedKeysIterator.remove();
                    }
                    
                }
                
            }
            
        }
        
        
        /**
         * 此类保存了SocketChannel和对应的ByteBuffer
         */
        class InputOutputPair{
            private SocketChannel clientChannel;
            private ByteBuffer inputBuffer;
            private ByteBuffer outputBuffer;
            
            public InputOutputPair(SocketChannel sc){
                this.clientChannel = sc;
                this.inputBuffer = ByteBuffer.allocate(BUFFERSIZE);
                this.outputBuffer = ByteBuffer.allocate(BUFFERSIZE);
            }
            
            public SocketChannel getClientChannel() {
                return clientChannel;
            }
            
            public ByteBuffer getInputBuffer() {
                return inputBuffer;
            }
            
            public ByteBuffer getOutputBuffer() {
                return outputBuffer;
            }
        }
    
        ```

