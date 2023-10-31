
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


## 什么是Socket?
Socket又称"套接字"，应用程序通常通过"套接字接口"向内核发出请求建立通信连接或监听端口请求。其间通信通常采用字节流或者数据包的形式。因此，Socket是一个术语，而Internet Socket则是一个协议族。它是支持TCP/IP协议的网络通信的基本方法。目前主要由TCP/IP协议簇支持。

## 为什么需要Socket?
应用层对传输层提供的端到端的逻辑通信通道进行抽象，从而可以屏蔽底层物理信道的各种差异，使上层应用能更方便地开发、移植和部署。

Socket可以看做是一种特殊的文件描述符，应用程序可以通过调用标准库函数socket()创建Socket，并通过connect()或listen()来请求建立TCP或UDP通信连接，或者绑定相应的服务端侦听端口。通过调用send()和recv()函数就可以实现数据的发送和接收。Socket提供了低级的通信机制，应用程序可以在Socket之上构建高级通信API，比如Berkeley sockets API和Winsock API等。所以，Socket是高性能、可靠、可伸缩的网络编程技术基础。

## Socket编程模型简介

1. Server-Client模式: 服务端先启动监听端口(bind+listen)，等待客户端的连接请求；连接成功后，双方完成数据交换，最后断开连接。
2. Pipe模式: 进程间的管道通信，也叫半双工通信，只允许单方向的数据流动。任何一方都可以往管道里写入数据，另一方也能从管道里读出来。进程也可以作为中转站，将一个管道的内容复制到另一个管道，甚至在两个不同机器上的两个进程之间通过网络互联。
3. Signal模式: 信号驱动I/O（SIGIO）机制，通过信号通知内核某个文件描述符是否准备好读取，避免轮询的方式影响效率，提升性能。
4. Stream模式: 流式Socket（SOCK_STREAM），可以实现双向通信，但每个链接只能一个进程使用。多线程模式（如pthreads）可以使用同一个Socket连接，实现多路复用。
5. Datagram模式: 数据报Socket（SOCK_DGRAM），没有连接状态，不保证数据顺序，可广播和组播。用于无连接或尽力而为的场景。Datagram模式可用于广播和组播，可以实现可靠性较差的消息传递，但是延迟时间相对较短。

# 2.核心概念与联系
## 结构体
### struct sockaddr
struct sockaddr定义了通用套接字地址结构。可以用来表示IPv4地址或IPv6地址。
```
struct sockaddr{
    unsigned short int sa_family; // address family
    char sa_data[14];            // up to 14 bytes of direct address
};
```
其中，`sa_family`字段指明了地址类型，AF_INET表示IPv4地址，AF_INET6表示IPv6地址。`sa_data`数组是个变长字段，长度根据不同类型的地址有所区别。对于IPv4地址，`sa_data`数组有16个字节，前面8个字节存储网络地址，后面8个字节存储主机地址；对于IPv6地址，`sa_data`数组有28个字节，前面16个字节存储网络地址，后面16个字节存储主机地址。

### socket地址转换函数
POSIX标准中提供了几种函数来处理套接字地址。
- `inet_aton()`函数用于把点分十进制的字符串形式的IP地址转换成二进制表示的IP地址。
- `inet_pton()`函数用于把任意字节串转换成特定类型的IP地址。
- `inet_addr()`函数是inet_aton()的一个简单封装，用于把点分十进制形式的IP地址转换成网络字节序（big-endian）的32位整数形式的IP地址。
- `gethostbyname()`函数用于根据域名查找IP地址和域名信息。
- `getaddrinfo()`函数是gethostbyname()的高级版本，可同时查询多个IP地址的信息。
- `getnameinfo()`函数用于把IP地址和端口号转换成可读形式的地址字符串。

## 函数
### 创建socket
`int socket(int domain, int type, int protocol)`
- 参数domain表示协议族，目前主要有两种协议族：PF_INET和PF_INET6。
- 参数type表示Socket类型，常用的有SOCK_STREAM(TCP)、SOCK_DGRAM(UDP)、SOCK_RAW(原始套接字)。
- 参数protocol表示要使用的协议，一般设置为0即可。
- 返回值：若成功，返回一个有效的Socket描述符，否则返回-1并设置errno的值。

示例代码：创建一个TCP socket。
```
#include <sys/types.h>
#include <sys/socket.h>
#include <stdio.h>

int main(){
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1){
        perror("socket error");
        return -1;
    }
    printf("create socket success\n");
    close(sockfd);
    return 0;
}
```

### 设置选项
#### 设置socket选项
`int setsockopt(int sockfd, int level, int optname, const void *optval, socklen_t optlen)`
- 参数sockfd表示要设置选项的Socket描述符。
- 参数level表示该选项所在层次，SOL_SOCKET表示最高层次，一般不用设置。
- 参数optname表示要设置的选项名。
- 参数optval表示要设置的选项值指针。
- 参数optlen表示选项值的长度。
- 返回值：若成功，返回0，否则返回-1并设置errno的值。

#### 获取socket选项
`int getsockopt(int sockfd, int level, int optname, void *optval, socklen_t *optlen)`
- 参数sockfd表示要获取选项的Socket描述符。
- 参数level表示该选项所在层次，SOL_SOCKET表示最高层次，一般不用设置。
- 参数optname表示要获取的选项名。
- 参数optval表示保存选项值的缓冲区指针。
- 参数optlen是指向socklen_t变量的指针，保存当前选项值的实际长度。
- 返回值：若成功，返回0，否则返回-1并设置errno的值。

#### SOL_SOCKET层次下的选项
- SO_ACCEPTCONN：监测接受连接状态的标志，返回值是布尔值，1表示处于监听状态，0表示已关闭。
- SO_BROADCAST：是否允许广播数据报。返回值是布尔值，1表示允许，0表示禁止。
- SO_ERROR：获取和清除错误状态。只有调用connect()或accept()时才有意义。
- SO_KEEPALIVE：保持活动状态。返回值是布尔值，1表示开启，0表示关闭。
- SO_LINGER：设置关闭选项。当仍有未处理的数据时，等待两秒钟再关闭Socket。返回值是struct linger结构体，其定义如下：
```
struct linger {
    int l_onoff;    /* 是否启用 */
    int l_linger;   /* 超时时间 */
};
```
- SO_RCVBUF：接收缓冲区大小。单位是字节，默认值为SO_RCVBUF_DEFAULT(512字节)，最大值为系统限制。
- SO_REUSEADDR：地址重用标志。设为1表示可以重复绑定。
- SO_SNDBUF：发送缓冲区大小。单位是字节，默认值为SO_SNDBUF_DEFAULT(512字节)，最大值为系统限制。
- SO_TYPE：获取Socket类型。

示例代码：设置SO_KEEPALIVE选项，表示开启保持活动状态。
```
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <stdio.h>

int main(){
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    int keepalive = 1;
    if (setsockopt(sockfd, SOL_SOCKET, SO_KEEPALIVE, &keepalive, sizeof(keepalive)) == -1){
        perror("setsockopt error");
        close(sockfd);
        return -1;
    }
    printf("set SO_KEEPALIVE option success\n");
    close(sockfd);
    return 0;
}
```

### 连接和关闭连接
#### 连接过程
`int connect(int sockfd, const struct sockaddr *addr, socklen_t addrlen)`
- 参数sockfd表示正在执行连接的Socket描述符。
- 参数addr表示服务器地址。
- 参数addrlen表示地址长度。
- 返回值：若成功，返回0，否则返回-1并设置errno的值。

#### 断开连接
`int shutdown(int sockfd, int how)`
- 参数sockfd表示要断开连接的Socket描述符。
- 参数how表示shutdown的类型，取值范围为SHUT_RD、SHUT_WR、SHUT_RDWR。
- 返回值：若成功，返回0，否则返回-1并设置errno的值。

示例代码：创建TCP socket，连接到远程主机，然后断开连接。
```
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(){
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);

    struct hostent* hp = gethostbyname("www.baidu.com");
    if (hp == NULL || hp->h_length!= sizeof(struct in_addr)){
        fprintf(stderr,"can't get remote ipaddress\n");
        exit(-1);
    }
    struct sockaddr_in serv_addr;
    memset(&serv_addr, 0, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    memcpy(&serv_addr.sin_addr, hp->h_addr, hp->h_length);
    serv_addr.sin_port = htons(80);

    if (connect(sockfd,(struct sockaddr*)&serv_addr,sizeof(serv_addr)) == -1){
        perror("connect error");
        exit(-1);
    }

    printf("connected to %s(%s:%d)\n",hp->h_name,inet_ntoa(*((struct in_addr*)hp->h_addr)),htons(serv_addr.sin_port));

    shutdown(sockfd, SHUT_WR);
    
    printf("shutdown write complete\n");

    close(sockfd);

    return 0;
}
```