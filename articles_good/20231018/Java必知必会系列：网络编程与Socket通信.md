
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


计算机网络（英语：Computer network）或互联网（Internet），是指由计算机及其通信设备组成的分布式系统，利用通信技术实现信息传输、电子商务、远程管理、网上交流等功能。Internet是一个世界范围内跨地域、跨国界的计算机网络系统，由众多网络服务提供商所共同构建和维护，是全球公用的基础设施网络。

从本质上来说，网络就是一个连接各个计算机节点（计算机设备、主机、路由器）的一组规则，这些规则使得不同节点之间可以交换数据、发送消息、共享资源和协作工作。而Socket（套接字）则是实现网络通信的一种方式，它是应用层与TCP/IP协议族间的一个抽象层，应用程序使用该接口通过 sockets 向网络发出请求或者应答网络请求，并接收返回的数据。因此，理解 Socket 的概念对于理解网络编程至关重要。

网络编程主要涉及两种基本操作：客户端-服务器模型和分布式系统模型。在客户端-服务器模型中，服务端提供可供客户端访问的服务，客户端则可以通过 socket 将请求发送给服务端，服务端处理请求后将结果返回给客户端；在分布式系统模型中，客户端直接跟踪服务端资源，不需要知道真正提供服务的是哪台机器，而且可以随时添加或删除服务端，甚至可以通过负载均衡提高服务质量。

本文将着重介绍 Java 中如何进行 Socket 编程，以帮助读者更好地理解网络通信及相关知识。

# 2.核心概念与联系
## 2.1 Socket概述
Socket 是支持 TCP/IP 协议的网络通信组件，通常用于两台主机之间的数据交换。简单来说，每一条 TCP/IP 连接唯一对应于两个 Socket，即，每个 Socket 都有自己的本地 IP 地址和端口号，唯一标识这个 Socket 在某个特定的 IP 地址的网络环境下。


图 1 示意图

## 2.2 Socket通信流程
### 服务端开启监听端口
首先，服务端需要先创建 ServerSocket 对象，指定绑定的端口号，然后调用它的 listen() 方法开始监听客户端的连接请求。一般情况下，服务器程序运行到此处会卡住，等待客户机的连接。

```java
import java.io.*;
import java.net.*;

public class TcpServer {
    public static void main(String[] args) throws IOException {
        // 创建服务器端Socket，指定绑定的端口号
        ServerSocket server = new ServerSocket(8080);
        
        System.out.println("服务器已启动...");
        
        while (true){
            // 监听客户端的连接请求
            Socket client = server.accept();
            
            System.out.println("有新的客户端连接：" + client.getInetAddress());
            
            try{
                InputStream is = client.getInputStream();
                
                byte[] bytes = new byte[1024];
                int len;
                
                OutputStream os = new FileOutputStream("client_" + client.getPort() + ".txt");
                
                while ((len=is.read(bytes))!=-1){
                    os.write(bytes, 0, len);
                    
                }
                
                os.close();
                is.close();
                client.shutdownInput();
                client.shutdownOutput();
                client.close();
            } catch (Exception e){
                System.err.println("数据读取异常！");
                e.printStackTrace();
            }
            
        }
        
    }
}
```

### 客户端发起连接请求
客户端程序首先创建一个 Socket 对象，指定要连接的服务器 IP 地址和端口号，然后调用它的 connect() 方法尝试建立到服务器的 TCP 连接。如果连接成功，客户端就可以像对待任何其他输入/输出流一样，通过它与服务器进行数据交换。

```java
import java.io.*;
import java.net.*;

public class TcpClient {
    public static void main(String[] args) throws IOException {
        // 创建客户端Socket，指定连接的目标地址和端口号
        Socket socket = new Socket("localhost", 8080);
        
        System.out.println("客户端已连接到服务器：" + socket.getRemoteSocketAddress());
        
        try {
            // 获取客户端写入数据的输出流
            PrintWriter writer = new PrintWriter(socket.getOutputStream(), true);
            
            String message = "Hello World";
            
            // 通过输出流向服务器发送数据
            writer.println(message);
            
            System.out.println("客户端已发送数据：" + message);
            
            // 从服务器获取回应数据
            BufferedReader reader = new BufferedReader(new InputStreamReader(socket.getInputStream()));
            
            String response = null;
            
            while((response = reader.readLine())!=null){
                System.out.println("服务器回复：" + response);
            }
            
            writer.close();
            reader.close();
            socket.shutdownInput();
            socket.shutdownOutput();
            socket.close();
            
        } catch (Exception e) {
            System.err.println("数据读取异常！");
            e.printStackTrace();
        }
        
    
    }
    
}
```

## 2.3 Socket类型
### 面向连接型 Socket
面向连接型 Socket 就是客户端与服务器建立了持久性连接之后才能进行通信的 Socket。典型场景如：文件传输、多媒体聊天等。

当客户端主动发起连接请求时，服务器端便为该客户端创建了一个 Socket 对象。如果连接成功建立，则服务器端和客户端两边都可以开始进行通信，直到双方都关闭连接。

### 非阻塞 Socket
非阻塞 Socket 是指在不能立刻得到结果之前，该函数不会阻塞线程，而会立即返回错误码或空值。在一些网络环境下，阻塞可能会导致严重的延迟和超时。

当设置为非阻塞模式后，需要轮询查看状态是否可用，并采用相应的方式进行处理。

### 可靠性 Socket
可靠性 Socket 是指确保数据能完整、顺序地传输。它能够自动重传丢失的包、保证数据不被破坏、超时重传等机制，确保数据可靠地传输。

当出现网络分区故障、通信线路瘫痪、中途路由器崩溃等情况时，可靠性 Socket 会尝试进行各种恢复措施，确保数据能正常传输。

### 数据报型 Socket
数据报型 Socket 是一种无连接的 Socket，它不保持长时间的 TCP 连接，只在数据传送过程中使用 UDP 协议。

当某个数据包丢失时，数据报型 Socket 仍然可以重新传输，但不会保持长时间的连接。对于要求高速传输的数据，可以采用数据报型 Socket 提高效率。

# 3.核心算法原理和具体操作步骤以及数学模型公式详细讲解
本节将详细描述 Socket 编程中的一些重要的算法原理。

## 3.1 分配端口号
每个 Socket 都有一个唯一的本地端口号，用来标识自己在某个特定网络环境中的身份。因此，在绑定本地端口号之前，需要首先确定未使用的端口号。

分配端口号的过程也称为端口复用（port reuse）。为了避免重复，通常采用动态分配的方式，即程序运行时由操作系统决定端口号。

```java
ServerSocket ss = new ServerSocket(0);   // 为当前 JVM 中的第一个未使用的端口号分配一个 Socket
int port = ss.getLocalPort();             // 获取分配到的端口号
ss.close();                                // 不再使用这个 Socket 时，释放占用的端口资源
```

```python
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)    # 创建 TCP Socket
sock.bind(('localhost', 0))                                  # 为当前主机中的第一个未使用的端口号分配一个 Socket
address, port = sock.getsockname()                            # 获取分配到的地址和端口号
sock.listen(backlog)                                          # 设置最大挂起连接数
```

## 3.2 字节流 I/O
Socket 编程中最常见的 I/O 模式就是字节流（byte stream）I/O 模式。字节流 I/O 是指利用 Socket 收发二进制数据。

字节流 I/O 使用 DataInputStream 和 DataOutputStream 来读取和写入 Socket 中的数据。

```java
DataInputStream dis = new DataInputStream(socket.getInputStream());
DataOutputStream dos = new DataOutputStream(socket.getOutputStream());

// 从输入流中读取数据
byte[] input = new byte[1024];
dis.readFully(input);

// 把数据写入输出流
dos.write(output);

// 关闭 Socket 流
dis.close();
dos.close();
```

## 3.3 字符流 I/O
字符流 I/O 对字节流 I/O 的扩展，它提供了方便的文本数据读写方法。

字符流 I/O 使用 BufferedReader 和 PrintWriter 来读取和写入 Socket 中的文本数据。

```java
BufferedReader br = new BufferedReader(new InputStreamReader(socket.getInputStream()));
PrintWriter pw = new PrintWriter(socket.getOutputStream(), true);

// 从输入流中读取文本数据
String line = "";
while ((line = br.readLine())!= null) {
  System.out.println("接收到的文本数据：" + line);
}

// 把文本数据写入输出流
pw.println("Hello World!");

br.close();
pw.close();
```

## 3.4 超时设置
某些时候，在客户端或服务器处于低带宽、慢速连接等原因造成的阻塞时，可能需要设置超时时间。超时时间设置的 API 比较复杂，这里举例 Java 中的超时设置：

```java
Socket s = new Socket("localhost", 8080);

s.setSoTimeout(5000);      // 设置超时时间为 5 秒

try {
  // 发起网络请求，超时时间为 5 秒
  URL url = new URL("http://www.example.com/");
  HttpURLConnection connection = (HttpURLConnection)url.openConnection();
  connection.connect();

  // 从响应流中读取数据
  InputStream inputStream = connection.getInputStream();
  BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
  
  String line;
  while ((line = reader.readLine())!= null) {
    System.out.println(line);
  }

  // 关闭所有资源
  reader.close();
  inputStream.close();
  connection.disconnect();
  
} catch (IOException e) {
  if (e instanceof InterruptedByTimeoutException) {
    // 请求超时
    System.err.println("请求超时！");
  } else {
    // 其他网络异常
    e.printStackTrace();
  }
}
```

## 3.5 监听多个地址
Socket 支持同时监听多个地址，也就是说，允许绑定多个端口号，然后通过不同的地址向服务器发起请求。这种能力可以方便地实现负载均衡。

监听多个地址的 Java 示例如下：

```java
ServerSocket ss = new ServerSocket();        // 创建 ServerSocket 对象

ss.bind(new InetSocketAddress("localhost", 8080), 10);     // 绑定第一个地址

ss.bind(new InetSocketAddress(InetAddress.getByName("192.168.1.10"), 8080), 10);    // 绑定第二个地址

System.out.println("服务器已启动，监听两个地址...");

while (true) {
    Socket socket = ss.accept();       // 接受新连接
    System.out.println("有新的客户端连接：" + socket.getInetAddress().getHostAddress());

    // TODO: 对客户端请求做出响应
}
```

## 3.6 Socket 选项设置
Java 支持设置 Socket 选项，比如 SO_REUSEADDR 和 TCP_NODELAY，可以通过 setsockopt() 或底层 Java API 来完成。

例如，设置 SO_REUSEADDR 可以让程序在意外退出后，仍然可以使用相同的端口号来重新启动服务。

```java
serverSocket.setReuseAddress(true);         // 设置 SO_REUSEADDR
```

设置 TCP_NODELAY 可以禁止 Nagle 算法，提升性能。

```java
tcpSocket.setTcpNoDelay(true);              // 设置 TCP_NODELAY
```

还有其他诸如 SO_RCVBUF 和 SO_SNDBUF 之类的选项，可以根据实际情况调整。

# 4.具体代码实例和详细解释说明
下面，我将结合前面的介绍，展示几个 Socket 编程实例。希望能够帮助读者加深对 Socket 编程的理解。

## 4.1 文件上传下载实例

### 服务端

```java
import java.io.*;
import java.net.*;

public class FileUploadServer {
    public static void main(String[] args) throws Exception {

        // 定义服务器端绑定的本地端口号
        int port = 8080;

        // 创建服务器端Socket，指定绑定的端口号
        ServerSocket serverSocket = new ServerSocket(port);

        System.out.println("服务器已启动...");

        // 循环接收客户端连接
        while (true) {

            // 接收客户端连接请求，并生成对应的Socket对象
            Socket socket = serverSocket.accept();

            try {

                // 获取客户端上传文件的输入流
                BufferedInputStream bis = new BufferedInputStream(socket.getInputStream());

                // 生成上传文件的保存目录
                String fileDir = "D:\\tmp\\upload\\"+Thread.currentThread().getName();
                File file = new File(fileDir);
                if (!file.exists()){
                    file.mkdirs();
                }

                // 获取客户端上传文件的文件名
                String fileName = getFileName(bis);

                // 获取客户端上传文件的总大小
                long fileSize = Long.parseLong(getHeaderValue(bis,"Content-Length"));

                // 根据文件名构造上传文件保存路径
                String filePath = fileDir+"\\"+fileName;

                // 创建上传文件的输出流
                FileOutputStream fos = new FileOutputStream(filePath);


                // 定义缓冲数组，用于临时存储上传文件的内容
                byte buffer[] = new byte[(int)fileSize];

                // 初始化已经上传文件的大小
                long uploadedSize = 0L;

                // 读取客户端上传文件的内容，并保存到输出流中
                int readCount = -1;
                while((readCount = bis.read(buffer))!=-1 && uploadedSize<fileSize){

                    fos.write(buffer,0,readCount);

                    uploadedSize+=readCount;

                }

                // 如果整个上传过程没有出现异常，则表示上传完成
                if(uploadedSize==fileSize){

                    System.out.println("文件["+filePath+"]上传成功！");

                }else{

                    System.out.println("文件上传失败！");

                }

                // 关闭输出流
                fos.flush();
                fos.close();

                // 关闭输入流
                bis.close();

                // 通知客户端上传完成
                PrintWriter printWriter = new PrintWriter(socket.getOutputStream(), true);
                printWriter.println("文件上传完成！");

                // 关闭Socket
                socket.shutdownOutput();
                socket.close();

            }catch (Exception e){

                e.printStackTrace();

                // 通知客户端上传失败
                PrintWriter printWriter = new PrintWriter(socket.getOutputStream(), true);
                printWriter.println("文件上传失败！");

                // 关闭Socket
                socket.shutdownOutput();
                socket.close();

            }


        }


    }



    /**
     * 从输入流中读取HTTP头部信息，获取Content-Disposition字段中的文件名
     * @param bis
     * @return
     */
    private static String getFileName(BufferedInputStream bis){

        StringBuilder stringBuilder = new StringBuilder();

        boolean findFile = false;
        for (int i = 0 ;i < 10000; i++){

            int data = bis.read();

            if(data == '\r' || data == -1){

                break;

            }

            char c = (char) data;

            stringBuilder.append(c);

            if(!findFile){

                findFile = "-".equals(stringBuilder.toString());

            }

            if("-".equals(stringBuilder.substring(-5))){

                return getStringUntilColon(stringBuilder).trim();

            }

        }

        throw new RuntimeException("获取文件名失败！");

    }



    /**
     * 从字符串中获取除冒号(:)后的内容
     * @param str
     * @return
     */
    private static String getStringUntilColon(StringBuilder str){

        for(int i=str.length()-1;i>=0;i--){

            char ch = str.charAt(i);

            if(ch == ':'){

                return str.substring(i+1);

            }

        }

        return "";

    }



    /**
     * 从输入流中读取HTTP头部信息，获取指定的键的值
     * @param bis
     * @param key
     * @return
     * @throws IOException
     */
    private static String getHeaderValue(BufferedInputStream bis, String key) throws IOException {

        StringBuilder value = new StringBuilder();

        boolean findKey = false;

        for(int i=0;i<10000;i++){

            int data = bis.read();

            if(data == '\r' || data == -1){

                break;

            }

            char c = (char) data;

            if(!findKey){

                findKey = key.startsWith(value);

            }

            if(findKey){

                value.append(c);

                if(key.endsWith(value)){

                    break;

                }

            }else{

                continue;

            }

        }

        if(value.isEmpty()){

            throw new IllegalArgumentException("获取"+key+"值失败！");

        }

        return value.toString().trim();

    }


}
```

### 客户端

```java
import java.io.*;
import java.net.*;

public class FileDownloadClient {
    public static void main(String[] args) throws Exception {

        // 定义客户端连接的服务器地址和端口号
        String host = "localhost";
        int port = 8080;

        // 建立客户端Socket，指定连接的目标地址和端口号
        Socket socket = new Socket(host, port);

        System.out.println("客户端已连接到服务器：" + socket.getRemoteSocketAddress());

        // 获取客户端下载文件的输入流
        BufferedInputStream bis = new BufferedInputStream(socket.getInputStream());

        // 获取客户端下载文件的文件名
        String fileName = getFileName(bis);

        // 获取客户端下载文件的总大小
        long fileSize = Long.parseLong(getHeaderValue(bis,"Content-Length"));

        // 根据文件名构造下载文件保存路径
        String savePath = "D:\\tmp\\download\\"+fileName;

        // 创建下载文件的输出流
        FileOutputStream fos = new FileOutputStream(savePath);

        // 定义缓冲数组，用于临时存储下载文件的内容
        byte buffer[] = new byte[1024*1024];

        // 初始化已经下载文件的大小
        long downloadedSize = 0L;

        // 读取客户端下载文件的内容，并保存到输出流中
        int readCount = -1;
        while((readCount = bis.read(buffer))!=-1 && downloadedSize<fileSize){

            fos.write(buffer,0,readCount);

            downloadedSize+=readCount;

        }

        // 如果整个下载过程没有出现异常，则表示下载完成
        if(downloadedSize==fileSize){

            System.out.println("文件["+fileName+"]下载成功！");

        }else{

            System.out.println("文件下载失败！");

        }

        // 关闭输出流
        fos.flush();
        fos.close();

        // 关闭输入流
        bis.close();

        // 通知客户端下载完成
        PrintWriter printWriter = new PrintWriter(socket.getOutputStream(), true);
        printWriter.println("文件下载完成！");

        // 关闭Socket
        socket.shutdownOutput();
        socket.close();


    }


    /**
     * 从输入流中读取HTTP头部信息，获取Content-Disposition字段中的文件名
     * @param bis
     * @return
     */
    private static String getFileName(BufferedInputStream bis){

        StringBuilder stringBuilder = new StringBuilder();

        boolean findFile = false;
        for (int i = 0 ;i < 10000; i++){

            int data = bis.read();

            if(data == '\r' || data == -1){

                break;

            }

            char c = (char) data;

            stringBuilder.append(c);

            if(!findFile){

                findFile = "-".equals(stringBuilder.toString());

            }

            if("-".equals(stringBuilder.substring(-5))){

                return getStringUntilColon(stringBuilder).trim();

            }

        }

        throw new RuntimeException("获取文件名失败！");

    }



    /**
     * 从字符串中获取除冒号(:)后的内容
     * @param str
     * @return
     */
    private static String getStringUntilColon(StringBuilder str){

        for(int i=str.length()-1;i>=0;i--){

            char ch = str.charAt(i);

            if(ch == ':'){

                return str.substring(i+1);

            }

        }

        return "";

    }



    /**
     * 从输入流中读取HTTP头部信息，获取指定的键的值
     * @param bis
     * @param key
     * @return
     * @throws IOException
     */
    private static String getHeaderValue(BufferedInputStream bis, String key) throws IOException {

        StringBuilder value = new StringBuilder();

        boolean findKey = false;

        for(int i=0;i<10000;i++){

            int data = bis.read();

            if(data == '\r' || data == -1){

                break;

            }

            char c = (char) data;

            if(!findKey){

                findKey = key.startsWith(value);

            }

            if(findKey){

                value.append(c);

                if(key.endsWith(value)){

                    break;

                }

            }else{

                continue;

            }

        }

        if(value.isEmpty()){

            throw new IllegalArgumentException("获取"+key+"值失败！");

        }

        return value.toString().trim();

    }


}
```

## 4.2 Echo 服务器

```java
import java.io.*;
import java.net.*;

public class EchoServer {
    public static void main(String[] args) throws Exception {

        // 定义服务器端绑定的本地端口号
        int port = 8080;

        // 创建服务器端Socket，指定绑定的端口号
        ServerSocket serverSocket = new ServerSocket(port);

        System.out.println("服务器已启动...");

        // 循环接收客户端连接
        while (true) {

            // 接收客户端连接请求，并生成对应的Socket对象
            Socket socket = serverSocket.accept();

            try {

                // 获取客户端发送的数据输入流
                BufferedReader br = new BufferedReader(new InputStreamReader(socket.getInputStream()));

                // 获取客户端发送的数据输出流
                PrintWriter pw = new PrintWriter(socket.getOutputStream(), true);

                // 循环读取客户端发送的数据，并回显给客户端
                String data = null;
                while ((data = br.readLine())!= null) {
                    System.out.println("收到来自" + socket.getRemoteSocketAddress() + "的信息：" + data);
                    pw.println("收到信息：" + data);
                }

                // 关闭输入输出流
                br.close();
                pw.close();

                // 通知客户端连接断开
                PrintWriter printWriter = new PrintWriter(socket.getOutputStream(), true);
                printWriter.println("连接断开！");

                // 关闭Socket
                socket.shutdownOutput();
                socket.close();

            } catch (Exception e) {

                e.printStackTrace();

                // 通知客户端发生错误
                PrintWriter printWriter = new PrintWriter(socket.getOutputStream(), true);
                printWriter.println("发生错误：" + e.getMessage());

                // 关闭Socket
                socket.shutdownOutput();
                socket.close();

            }
        }

    }

}
```

## 4.3 聊天室

```java
import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.net.*;
import java.util.ArrayList;
import java.util.List;

public class ChatRoom extends JFrame implements ActionListener {

    private JTextField textField;
    private JButton sendButton;
    private JTextArea chatText;

    private List<User> userList;

    private User currentUser;
    private String currentUserName;

    private static final String DEFAULT_USERNAME = "defaultUsername";

    public ChatRoom() {
        super("Chat Room");

        this.userList = new ArrayList<>();
        this.currentUser = null;
        this.currentUserName = DEFAULT_USERNAME;

        this.initUI();
    }

    private void initUI() {
        JPanel panel = new JPanel();

        JLabel label = new JLabel("请输入用户名:");
        JTextField nameField = new JTextField(DEFAULT_USERNAME);

        ButtonGroup buttonGroup = new ButtonGroup();

        JRadioButton groupButton1 = new JRadioButton("游客");
        buttonGroup.add(groupButton1);

        JRadioButton groupButton2 = new JRadioButton("管理员");
        buttonGroup.add(groupButton2);

        JPanel radioPanel = new JPanel();
        radioPanel.add(groupButton1);
        radioPanel.add(groupButton2);

        this.textField = new JTextField("");
        this.sendButton = new JButton("发送");
        this.chatText = new JTextArea("", 10, 20);

        this.textField.addActionListener(this);
        this.sendButton.addActionListener(this);

        panel.add(label);
        panel.add(nameField);
        panel.add(radioPanel);
        panel.add(this.textField);
        panel.add(this.sendButton);
        panel.add(this.chatText);

        add(panel);

        pack();

        Dimension screensize = Toolkit.getDefaultToolkit().getScreenSize();
        setLocation((screensize.width - getWidth()) / 2, (screensize.height - getHeight()) / 2);

        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        setVisible(true);
    }

    @Override
    public void actionPerformed(ActionEvent event) {

        Object source = event.getSource();

        if (source == this.textField) {
            String userName = this.textField.getText();
            setUser(userName);
        } else if (source == this.sendButton) {
            sendMessage(this.textField.getText());
            this.textField.setText("");
        }

    }

    private void setUser(String userName) {

        if ("".equals(userName)) {
            JOptionPane.showMessageDialog(null, "用户名不能为空！", "提示", JOptionPane.WARNING_MESSAGE);
            return;
        }

        this.currentUserName = userName;

        if (currentUser!= null) {
            removeUserListListener(currentUser);
        }

        currentUser = getUser(currentUserName);

        if (currentUser!= null) {
            addUserListListener(currentUser);
        }

        this.textField.requestFocusInWindow();
    }

    private void addUserListListener(final User user) {
        user.addMessageListener(new MessageListener() {
            @Override
            public void onMessage(User fromUser, String message) {
                appendChatMessage(fromUser.getUserName() + "：" + message);
            }
        });
    }

    private void removeUserListListener(User user) {
        user.removeMessageListener(user);
    }

    private void sendMessage(String content) {

        if (content!= null &&!"".equals(content.trim())) {
            String msg = getCurrentUserName() + ":" + content;
            writeMessage(msg);
        }
    }

    private synchronized void writeMessage(String message) {

        if (currentUser!= null) {
            currentUser.sendMessage(message);
        }
    }

    private synchronized void appendChatMessage(String message) {
        this.chatText.append(message + "\n");
    }

    private synchronized User getUser(String userName) {
        for (User u : this.userList) {
            if (u.getUserName().equalsIgnoreCase(userName)) {
                return u;
            }
        }
        return null;
    }

    protected String getCurrentUserName() {
        return this.currentUserName;
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            new ChatRoom();
        });
    }

}

class User implements Runnable {

    private String userName;

    private Socket socket;

    private Thread receiveThread;

    private List<MessageListener> messageListeners;

    public User(String userName, Socket socket) {
        this.userName = userName;
        this.socket = socket;
        this.receiveThread = new Thread(this, "receiveThread-" + userName);
        this.receiveThread.start();
        this.messageListeners = new ArrayList<>();
    }

    public void sendMessage(String message) {

        if (socket!= null &&!socket.isClosed()) {

            try {
                PrintStream out = new PrintStream(socket.getOutputStream());
                out.println(message);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    @Override
    public void run() {
        try {
            BufferedReader in = new BufferedReader(new InputStreamReader(socket.getInputStream()));

            String message;
            while ((message = in.readLine())!= null) {
                System.out.println(userName + "：" + message);
                notifyMessageReceived(message);
            }

        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            closeSocket();
        }
    }

    public void addMessageListener(MessageListener listener) {
        this.messageListeners.add(listener);
    }

    public void removeMessageListener(MessageListener listener) {
        this.messageListeners.remove(listener);
    }

    private void notifyMessageReceived(String message) {
        for (MessageListener l : this.messageListeners) {
            l.onMessage(this, message);
        }
    }

    public void closeSocket() {
        try {
            this.socket.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public String getUserName() {
        return this.userName;
    }

}

interface MessageListener {

    void onMessage(User fromUser, String message);

}
```