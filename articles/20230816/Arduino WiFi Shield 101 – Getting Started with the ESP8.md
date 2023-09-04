
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 什么是Arduino？

## 什么是Arduino WiFi Shield?

## 如何使用Arduino WiFi Shield?
Arduino WiFi Shield是一块简单易用的Wi-Fi扩展板，用户只需简单配置好网络信息即可接入电脑，实现简单的 Wi-Fi 通信。

1.首先安装Arduino软件。下载安装后打开IDE并添加对应的硬件：



2.烧写程序到Arduino板子上。将示例程序复制到一个新工程文件中并保存。

   ```c++
   #include <ESP8266WiFi.h> //引用库
   
   const char* ssid     = "yourssid";    //你的WIFI名称
   const char* password = "yourpassword";//你的WIFI密码
   
   
   void setup() {
     Serial.begin(9600);          //设置串口波特率
     WiFi.mode(WIFI_STA);         //设置WIFI模式为STA
     WiFi.begin(ssid, password);  //开始WIFI连接
   
     while (WiFi.status()!= WL_CONNECTED) {
       delay(500);                 //等待WIFI连接成功
     }
     
     Serial.println("WIFI connected"); 
   }
   
   
   void loop() {}
   ```

   **注意**：上面的程序需要替换成自己的WIFI名称和密码。修改完毕保存，编译上传到Arduino。当程序运行时，会自动连接WIFI。如果连接失败，可以重新启动程序或检查网络是否正常。



3.接收和发送数据。将下列程序拷贝到另一个新的程序文件中。其中，`myServerIP`变量需要填写服务器IP地址；`myClientPort`变量需要填写服务器监听端口号；`myMessage`变量需要填写要发送的数据；`client`变量是一个TCP客户端对象。

   ```c++
   #include <ESP8266WiFi.h>      //引用库
   #include <SoftwareSerial.h>  //引用串行库
   
   SoftwareSerial mySerial(2, 3);  //定义外部中断（TX=D2, RX=D3）
   
   String myServerIP = "xxx.xxx.xxx.xxx";    //服务器IP地址
   int myServerPort = xxx;                    //服务器监听端口号
   
   const char * myMessage = "Hello World!";    //要发送的数据
   
   WiFiClient client;                        //创建一个TCP客户端对象
   
   
   void setup() {
     Serial.begin(9600);                     //设置串口波特率
     WiFi.mode(WIFI_STA);                    //设置WIFI模式为STA
     WiFi.begin("yourssid", "yourpassword");  //开始WIFI连接
     
     if (!mySerial) {                         //判断是否正确打开串口
       while (true) {
         ;
       }
     }
     
     Serial.print("Waiting for connection...");
     while (!client.connect(myServerIP, myServerPort)) { //尝试连接服务器
       delay(500);                                //等待连接成功
     }
     Serial.println("Connected!");
   }
   
   
   void loop() {
     static unsigned long lastMillis = 0;        //声明静态变量lastMillis
     if ((millis() - lastMillis) > 500) {       //每隔500毫秒发送一次数据
       lastMillis = millis();
       
       mySerial.println(myMessage);             //打印要发送的数据
       char c;                                  //定义一个字符变量
       
       while (mySerial.available()) {           //循环读取串口输入
         c = mySerial.read();                   //读取一个字节
         if (c == '\n') break;                  //遇到换行符则退出
         client.write((uint8_t)c);              //向服务器发送数据
       }
       
       while (client.available()) {             //循环读取服务器输出
         c = client.read();                     //读取一个字节
         mySerial.write(c);                     //打印到串口显示
       }
     }
   }
   ```

   **注意**：上面的程序需要根据实际情况替换成自己的WIFI名称和密码、服务器IP地址、服务器监听端口号等参数值。修改完毕保存，编译上传到Arduino。当程序运行时，首先等待串口连接，然后自动连接WIFI并尝试连接指定的服务器。如果连接成功，程序便开始周期性地发送数据和接收服务器的响应。

