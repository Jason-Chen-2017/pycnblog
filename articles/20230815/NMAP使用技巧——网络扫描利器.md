
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Nmap (Network Mapper) 是一款网络扫描工具，能够探测网络上主机、服务和版本信息。它可以扫描整个互联网，也可以指定IP地址或网段进行扫描。在 Linux 系统中，Nmap 的安装包可以在软件管理器或命令行中直接安装。

本文将从以下方面详细介绍 Nmap 使用技巧：

 - 安装配置 Nmap 
 - 使用方法及其参数选项
 - 演示实践案例：对某网站进行网络扫描 
 - 提高扫描效率的方法：速率限制、排除故障设备等
 - 用法提示：可疑设备处理、自定义脚本、与其他工具集成等
 - 安全防护措施
# 2.基本概念和术语
## 2.1 网络扫描（scanning）
网络扫描就是网络上的主机、服务和版本信息的收集，包括探测网络上主机是否存活、开放哪些端口、提供什么服务、运行了什么系统版本等。

网络扫描通常分为两步：

1. 侦察阶段：主要用来了解目标网络中的网络拓扑、存活主机数量、网络连接情况等。
2. 数据收集阶段：通过逐个主机发送特定请求，收集目标机器的网络数据，包括开放的端口、运行的服务、操作系统版本、计算机名等。

## 2.2 TCP/IP协议
TCP/IP 协议是 Internet 上最通用的协议，用于传输各种数据。其中，“TCP” 和 “IP” 分别代表 Transmission Control Protocol 和 Internet Protocol，它们一起构成了传输层协议栈。

- TCP 负责提供可靠、按序到达的数据流；
- IP 负责将数据包从源地址传送至目的地址。

## 2.3 端口（port）
端口是一个虚拟通信接口，每个端口都有唯一标识符号。一台计算机可以有多个网络接口，每个接口都可以通过不同的端口号与其他计算机通信。通常情况下，通信的双方都会事先知道对方使用的端口号。

例如，常用的 HTTP 协议默认端口号是 80，HTTPS 默认端口号是 443，SMTP 默认端口号是 25。

## 2.4 ICMP 报文
ICMP 报文是 Internet 控制报文协议的一部分，它是在 IP 数据报传输过程中产生错误和传播控制消息的协议。ICMP 报文用于网络诊断、错误通知、可用性通知、测量延迟时间、最大传输单元的发现等。

ICMP 报文可作为任何 IP 数据报的组成部分，但只有 TCP/IP 实现了其应用层协议。例如，当服务器无法响应某个用户的请求时，它会发送一个不可达报文给用户。而当路由器收到一个没有意义的分组时，它也会发送一个类型为“时间戳”的 ICMP 报文。

# 3.核心算法原理和具体操作步骤
## 3.1 命令行模式
Nmap 可以使用命令行模式或图形界面模式运行。命令行模式下，Nmap 采用命令行参数执行，通过交互的方式向 Nmap 输入要扫描的对象和参数，生成并显示扫描结果。

### 3.1.1 查看帮助信息
```
nmap --help
```
显示 Nmap 命令的帮助信息。

### 3.1.2 指定扫描类型
Nmap 可以对不同的网络服务进行扫描，包括常见的网络服务如 http、https、tcp、udp、smtp 等，以及不常见的网络服务如 domain、ipid、mdns、nbname、xmpp 等。

```
nmap –sS [host]
```
- `-sS`：指定 TCP SYN 扫描。这是一种全连接扫描，它建立 TCP 三次握手过程，并通过捕获 ACK 回复确定远程主机是否正在监听指定的端口。

```
nmap –sT [host]
```
- `-sT`：指定 TCP connect() 扫描。它建立 TCP 三次握手过程，但不接收任何回应，因此速度比 SYN 快。此外，由于它不是完全连接的，所以不能探测主机是否开启了反射式端口扫描。

```
nmap –sU [host]
```
- `-sU`：指定 UDP 扫描。这是一种无连接扫描，它通过发送 UDP 广播或单播数据报到目标主机来检查是否开启了指定端口。

```
nmap –sV [host]
```
- `-sV`：尝试识别出目标主机正在运行的服务版本。该选项需要额外的处理时间，因此可能会造成扫描变慢。

```
nmap –sA [host]
```
- `-sA`：尝试检测主机的所有活动服务，包括数据库、DNS、SNMP、LDAP、SMB 等。这类服务一般具有复杂的鉴权机制，因此如果开启了`-A`选项，扫描可能需要花费较长的时间才能完成。

```
nmap –sW [host]
```
- `-sW`：探测主机的操作系统类型及版本。

```
nmap –sM [host]
```
- `-sM`：尝试进行主动摘取的探测，此时 Nmap 将自己伪装成客户端访问服务端，并发送一个特殊的 SYN 报文到目标主机的特定端口。

```
nmap –sZ [host]
```
- `-sZ`：探测 SSL/TLS 协议支持。

```
nmap –sp [host]
```
- `-sp`：发送 ping 包到目标主机。

```
nmap –sr [host]
```
- `-sr`：进行 RPC 请求。

```
nmap –ss <port> [host]
```
- `--scanflags`:指定 TCP 扫描方式。

### 3.1.3 指定主机范围
Nmap 支持扫描一系列主机，或者扫描 IP 地址范围。

```
nmap host1[,host2[...]]
```
host 为目标域名或 IP 地址。

```
nmap 192.168.0.*
```
扫描 IP 地址 192.168.0.1~192.168.0.254。

### 3.1.4 指定扫描端口
Nmap 可以根据实际需求选择扫描端口。

```
nmap -p <port>[,<port>[,...]]] [-g<num>] [--top-ports <number>] [host]
```
- `-p`：指定要扫描的端口。
- `-g`：指定枚举策略，适用于目标主机启用了 `RPCINFO` 服务。
- `--top-ports`：扫描前几个端口的常用端口。

### 3.1.5 设置扫描速率
Nmap 可以设置扫描速率，以控制扫描时间。

```
nmap –min-rate <number> [host]
```
- `--min-rate`：设置最小扫描速率，单位 kbps。

```
nmap –max-rate <number> [host]
```
- `--max-rate`：设置最大扫描速率，单位 kbps。

```
nmap –delay <milliseconds> [host]
```
- `--delay`：设置每次连接之间的延迟，单位毫秒。

```
nmap –ttl <value> [host]
```
- `--ttl`：设置每跳的 TTL 值。

```
nmap –paranoid [host]
```
- `--paranoid`：启用精细化扫描模式，使得 Nmap 在收到异常响应时更积极地响应。

### 3.1.6 设置超时时间
Nmap 可以设置超时时间，以避免无响应的主机影响扫描速度。

```
nmap –initial-rtt-timeout <milliseconds> [host]
```
- `--initial-rtt-timeout`：设置初始 RTT 超时，单位毫秒。

```
nmap –min-rtt-timeout <milliseconds> [host]
```
- `--min-rtt-timeout`：设置最小 RTT 超时，单位毫秒。

```
nmap –max-rtt-timeout <milliseconds> [host]
```
- `--max-rtt-timeout`：设置最大 RTT 超时，单位毫秒。

```
nmap –max-retries <number> [host]
```
- `--max-retries`：设置最大重试次数。

### 3.1.7 设置报文类型
Nmap 可以设置报文类型，以过滤不必要的响应。

```
nmap –data-length <bytes> [host]
```
- `--data-length`：设置发送数据的长度，单位字节。

```
nmap –ip-options <options> [host]
```
- `--ip-options`：设置 IP 选项。

```
nmap –probe-args <probe_arguments> [host]
```
- `--probe-args`：设置传给 probe （探测）程序的参数。

```
nmap –scanflags <flags> [host]
```
- `--scanflags`：设置 TCP 扫描标志。

```
nmap –mtu <mtu> [host]
```
- `--mtu`：设置 MTU 大小。

```
nmap –mss <mss> [host]
```
- `--mss`：设置 MSS 大小。

```
nmap –echofilter <expression> [host]
```
- `--echofilter`：设置 echo filter 参数。

### 3.1.8 配置扫描顺序
Nmap 可以调整扫描顺序，优化网络资源利用率。

```
nmap –randomize-hosts [host]
```
- `--randomize-hosts`：随机化目标主机列表的顺序。

```
nmap –seq-start <val> [host]
```
- `--seq-start`：设置起始扫描序列值。

```
nmap –seq-stop <val> [host]
```
- `--seq-stop`：设置结束扫描序列值。

```
nmap –max-parallelism <number> [host]
```
- `--max-parallelism`：设置并行扫描的数量。

```
nmap –min-parallelism <number> [host]
```
- `--min-parallelism`：设置最少并行扫描的数量。

### 3.1.9 输出报告格式
Nmap 可以指定扫描报告的格式。

```
nmap –oN/-oS/-oX [<file>] [host]
```
- `-oN`：输出正常格式报告文件。
- `-oS`：输出缩略格式报告文件。
- `-oX`：输出 XML 格式报告文件。

```
nmap –oG=<file>[.<ext>] [host]
```
- `-oG`：输出 grepable 格式报告文件。

```
nmap –v/-d [host]
```
- `-v`：显示详细信息。
- `-d`：调试模式，输出更多日志信息。

```
nmap –vv/-dd [host]
```
- `-vv`：显示更加详细的信息。
- `-dd`：调试模式，输出所有日志信息。

### 3.1.10 浏览网页
Nmap 可以通过抓取网页来获取目标站点的网址，从而确认网站是否存在漏洞或提供恶意内容。

```
nmap –webscan [host]
```
- `--webscan`：进行网页扫描。

```
nmap –script=<lua_script> [host]
```
- `--script`：加载 Lua 脚本，对主机进行定制化扫描。

```
nmap –dns-server <address> [host]
```
- `--dns-server`：指定 DNS 服务器。

### 3.1.11 防火墙和代理设置
Nmap 可以通过防火墙或代理设置来绕过访问限制。

```
nmap –proxy <type>[://]<host>:<port> [host]
```
- `--proxy`：使用代理。

```
nmap –proxy-auth <username>:<password> [host]
```
- `--proxy-auth`：设置代理用户名密码。

```
nmap –ignore-bad-checks [host]
```
- `--ignore-bad-checks`：忽略失效的检查项。

```
nmap –append-output [host]
```
- `--append-output`：追加输出文件。

### 3.1.12 使用自定义扫描脚本
Nmap 可以使用自定义扫描脚本，对主机进行定制化扫描。

```
nmap –script-args=<args> [host]
```
- `--script-args`：设置脚本参数。

```
nmap –list-scripts
```
- `--list-scripts`：列出可用脚本。

```
nmap –iflist
```
- `--iflist`：显示可用网卡列表。

# 4.实践案例
## 4.1 对某网站进行网络扫描
### 4.1.1 目标网址
```
http://www.example.com
```
### 4.1.2 使用 nmap 命令对目标网站进行网络扫描
```
nmap www.example.com
```
### 4.1.3 命令输出分析
Nmap 会对目标网站进行全方位扫描，发现它的主机名、操作系统类型和版本、开放的端口以及各个服务运行状态。如下所示：
```
Starting Nmap 7.80 ( https://nmap.org ) at 2021-08-01 15:47 CST
Nmap scan report for www.example.com (192.168.3.11)
Host is up (0.037s latency).
Not shown: 997 filtered tcp ports (no-response), 1 closed port
PORT    STATE SERVICE VERSION
22/tcp  open  ssh     OpenSSH 7.4p1 Debian 10+deb9u7 (protocol 2.0)
| ssh-hostkey: 
|   2048 e1:e0:a6:4c:cd:9b:f4:e5:33:ed:cf:9f:dc:2f:aa:5a (RSA)
|   256 fdf8:f53e:61e4::18:4b:fb:16:da:ca:73:15:6a (ECDSA)
|_  256 c9:6f:af:f3:81:6c:0d:a6:fd:7f:d0:bb:db:d6:e7:cb (ED25519)
80/tcp  open  http    nginx/1.10.3 (Ubuntu)
| http-methods: 
|_  Supported Methods: GET HEAD POST OPTIONS
|_http-title: Site doesn't have a title (text/html; charset=UTF-8).
443/tcp open  ssl/http nginx/1.10.3 (Ubuntu)
|_http-server-header: nginx/1.10.3 (Ubuntu)
| tls-alpn: 
|_  h2
Service Info: OS: Unix

Service detection performed. Please report any incorrect results at https://nmap.org/submit/.
Nmap done: 1 IP address (1 host up) scanned in 52.91 seconds
```
从以上结果可以看出，目标网站的 IP 地址是 192.168.3.11，开放了 22、80、443 三个端口。其中 22 端口是 SSH 登录端口，80 端口是 HTTP 服务端口，443 端口是 HTTPS 服务端口。HTTP 服务的版本信息显示为 nginx/1.10.3 ，而 SSH 版本信息显示为 OpenSSH 7.4 。