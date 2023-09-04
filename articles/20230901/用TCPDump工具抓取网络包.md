
作者：禅与计算机程序设计艺术                    

# 1.简介
  

　　TCPDump是一个开源的数据包分析工具，它可以截获网络数据包，并将它们保存成日志文件或者直接显示在屏幕上。这个工具非常方便用于网络调试和分析，尤其是在分析传输层协议如TCP/IP时，它提供了一个不错的途径。
　　本文主要介绍如何安装、配置以及使用TCPDump工具抓取网络包。
# 2.基本概念
## 2.1 TCPDump命令参数
### 2.1.1 概述

　　TCPDump的命令行语法如下所示:
```bash
tcpdump [-c count] [-nn] [-i interface] [-s snaplen] [-v] [expression]
```
其中各个参数的含义如下表所示:
|选项|描述|
|---|---|
|-c count|指定要捕获的包数量。|
|-nn|将数字形式的地址转换成主机名或域名。|
|-i interface|指定从哪个接口捕获数据包，默认是第一个有效接口。|
|-s snaplen|设置捕获数据包最大长度，默认为65535字节，即不限制大小。|
|-v|详细模式输出，打印出每个包的数据。|
|expression|指定筛选表达式，根据表达式过滤需要捕获的包，例如`-s`指定的接收长度，`-w`指定的输出文件等。|

### 2.1.2 模糊匹配(pattern matching)
　　TCPDump支持模糊匹配，允许用户通过表达式来过滤需要捕获的包。通过指定表达式后，只有满足指定条件的包才会被捕获。表达式的格式遵循正则表达式的规则。下面是一些常用的表达式示例:

- `-s length`: 指定捕获的包长度，length是一个整数。
- `-w filename`: 将捕获到的包保存到文件filename中，filename是保存的文件名称。
- `-e|--encap <ether[ip]|fddi[et]>`: 解封装，将捕获到的数据包解开指定的封装层。ether表示以太网帧，ip表示IP数据报，fddi表示FDDI帧等。
- `host ipaddr or net ipaddr`: 根据指定的IP地址过滤，其中ipaddr是IPv4地址，可以带掩码。
- `net host ipaddr and not port number`: 根据指定的源IP地址和端口号过滤，排除目的地址和端口号都符合的包。

以上表达式均可以在命令行下用`-h`选项查看完整的帮助信息。

### 2.1.3 数据包内容解析(packet dissection)
　　TCPDump可以通过提供的参数进行数据包的解析和显示。比如，通过`-vv`参数可以显示详细的包头信息，包括Ethernet头部、IP头部、TCP头部等。而通过`-xx`参数可以显示每一个字段的值，便于分析。通过结合这些参数，就可以了解到数据包的整体结构以及各个字段的意义。

除了上面提到的参数外，还有几个比较实用的选项，如下表所示:
|选项|描述|
|---|---|
|-A|将MAC地址转换成友好的名字。|
|-B len|将数据包的每段内容以ASCII字符的方式显示，最多显示len个字节。|
|-C size|按指定的大小分割数据包内容，每次显示size个字节的内容。|
|-E ciphername|对捕获的数据进行加密处理，ciphername为加密算法的名称，例如"DES","AES-128"等。|


# 3. 技术实现
## 3.1 安装TCPDump
### 3.1.1 Linux系统安装方式
　　如果是Linux系统，通常可以直接从系统的包管理器里进行安装。比如，对于Ubuntu系统，可以使用以下命令安装tcpdump:
```bash
sudo apt install tcpdump
```
如果系统没有包管理器，也可以下载源码编译安装。

### 3.1.2 Windows系统安装方式

## 3.2 配置环境变量
### 3.2.1 Linux系统配置环境变量
　　为了使得命令行下可以直接运行tcpdump命令，需要配置环境变量。方法如下:
　　1.打开编辑器，输入gedit ~/.bashrc或nano ~/.bashrc，然后回车。
　　2.添加以下两行命令到文件的最后面，然后保存退出。
```bash
alias tcptop='sudo tcpdump -i any -n src host YOUR_HOST' # 显示目标主机的所有TCP包
alias tcpping='ping YOUR_HOST >/dev/null && echo "ping ok." || echo "ping fail."' # 检查目标主机是否可达
```
其中YOUR_HOST应该替换成你的目标主机IP地址或者域名。

保存后，使用source ~/.bashrc重新加载环境变量使之生效。

### 3.2.2 Windows系统配置环境变量
　　如果安装了WinPcap驱动，那么配置环境变量就比较简单。直接打开注册表编辑器，依次点击:
```
计算机\HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session Manager\Environment
```
创建新的字符串值，名称为Path，值为安装目录下的bin文件夹路径。注意，在路径末尾不能有空格。

# 4. 操作实例
## 4.1 捕获特定主机的TCP包
如果只想捕获某个特定的主机发送的TCP包，可以使用如下命令:
```bash
tcpdump -i eth0 host myserver.com -nn
```
这里的`-i`选项指定的是网卡接口名称，可以换成实际的网卡名称。`-nn`参数用来显示主机名而不是数字形式的地址。`-c`参数可以指定要捕获的包的数量，默认为无限。

另外，还可以使用`-l`参数将结果输出到标准输出，然后再重定向到文件或其他目的地。

## 4.2 在线跟踪流量走向
使用`-r`参数可以读取pcap格式的文件，并实时显示它的内容。这个选项适用于离线分析pcap文件，或者想要实时查看实时流量的情况。比如:
```bash
tcpdump -r trace.pcap -nn -tt
```
这里的`-nn`参数用来显示主机名而不是数字形式的地址。`-tt`参数用来显示时间戳。此外，`-c`参数也可以用来限制要显示的包的数量。

## 4.3 使用表达式过滤
可以使用表达式过滤需要捕获的包，指定`-s`选项来指定收包长度，`-w`选项来指定输出文件等。也可以结合不同的参数使用组合的方式来获得想要的结果。下面是几个例子:

- 只捕获目标主机到自己的通信：
  ```bash
  tcpdump dst MY_HOST and src MY_IP
  ```
- 只捕获目标主机发送的UDP包：
  ```bash
  tcpdump dst MY_HOST udp 
  ```
- 查看目标主机HTTP请求：
  ```bash
  tcpdump'src MY_HOST and (dst HTTP or dst HTTPS)'
  ```
- 查看目标主机之间所有包：
  ```bash
  tcpdump'src MY_HOST or dst MY_HOST'
  ```
- 合并多个pcap文件:
  ```bash
  mergecap file1.pcap file2.pcap > mergedfile.pcap
  ```