
作者：禅与计算机程序设计艺术                    

# 1.简介
         

         ## 1.前言
         
         ### 为什么要写这篇文章？
         
         eBPF（extended Berkeley Packet Filter）是由<NAME>于2012年提出的一种虚拟机，可以对Linux内核中的网络数据包进行高级过滤、修改、收集等操作，并且是安全且免费的。BCC（Berkeley Cloud Computing Compiler），是由该团队开发的一套工具链，用于编译、加载并运行eBPF程序。同时，Rust编程语言也成为容器编排领域的主要选择，很多公司在使用Rust开发容器云平台时都会用到这个工具。相对于同类项目来说，eBPF/BCC和Rust是两个非常重要的新生力量，它们都可以极大地扩展容器云平台的功能。因此，编写一篇专门介绍如何利用eBPF和BCC框架开发Rust版监控工具具有重要意义。
         
         ### 本文目标读者
         本文面向计算机专业的本科及以上学历，具备一定的编程基础。但阅读本文并不要求完全理解作者所涉及的内容。主要内容为对eBPF和BCC的相关知识点做一个简单而系统的回顾，希望能够帮助读者建立起相关概念之间的联系，并通过实践掌握这些知识的应用技巧。
         
         ### 文章结构
         本文将分为以下几个部分：
         
         （一）介绍背景知识：本部分首先给出eBPF和BCC的简介，然后介绍其主要特性以及典型用途。
         
         （二）eBPF概述：本部分简要介绍了eBPF程序的结构、指令集和工作方式，并阐述了eBPF可用的操作类型。
         
         （三）BCC概述：本部分简要介绍了BCC工具链的组成，包括前端和后端，然后详细介绍了BCC提供的各项工具，如clang-bpf、tcptracer-ebpf和biosnoop-ebpf等。
         
         （四）Rust环境安装：本部分介绍了如何安装Rust开发环境，并配置好Cargo构建环境。
         
         （五）基于bcc-tools的实战案例：本部分通过实例介绍如何利用BCC工具编写Rust版监控工具。
         
         （六）总结和展望：本部分简要回顾了文章的主要内容和写作目的，并展望了后续文章的方向。
        
         为了更好的阅读体验，文章的内容将采用如下图示的顺序呈现。
         
         
         # 2.介绍背景知识
         
         ## 1.什么是eBPF？
         
         Extended Berkeley Packet Filter，即可扩展的贝尔康大学数据包过滤器（Packet Filtering System），是一个开源的虚拟机，它可以在内核中对网络数据包进行高级过滤、修改、收集等操作，并且是安全且免费的。由用户态的字节码指令组成，可以实现复杂的网络协议解析和过滤规则，可有效提升性能和可靠性。
         
         ## 2.什么是BCC？
         
         BCC（Berkeley Cloud Computing Compiler），由贝尔康大学研究团队于2017年开发的一套工具链，用于编译、加载并运行eBPF程序。包括以下几部分：
         
         - bcc/kernel: Linux内核源代码改动，支持eBPF。
         - bcc/libbpf: 提供BPF库，可以生成字节码、加载BPF程序、处理BPF事件。
         - bcc/tools: 支持eBPF的工具集合，例如tcpdump、tcptraceroute等。
         - bcc/python: 对bcc/tools中命令行工具的封装，可以使用Python脚本调用相应的工具。
         
         ## 3.eBPF的特征
         
         ### 1.高性能
         
         eBPF在设计上有三大特点：JIT（Just In Time）编译、异步执行和软中断模式。JIT编译使得内核只需要执行少量的BPF指令就能完成过滤操作，并且允许添加新的功能或修正已有的错误。异步执行则允许多个BPF程序同时运行，适合于处理大量流量。软中断模式则允许在不占用CPU资源的情况下处理BPF程序，比传统硬件中断更加高效。
         
         ### 2.安全性
         
         在用户空间运行的BPF程序没有系统调用权限，只能访问受限的资源，并且不能执行一些危险的操作，例如分配内存、打开文件、调用execve()等。另外，eBPF还提供了安全沙箱机制，限制BPF程序的运行时间和可用资源。
         
         ### 3.可移植性
         
         eBPF程序编译后可以直接加载到当前的运行环境中运行，无需任何特殊的配置和编译过程，甚至可以跨平台运行。同时，eBPF支持多种语言，包括C、C++、Go、Java、Python等。
         
         ## 4.eBPF的典型用途
         
         eBPF最典型的用途就是网络监控。由于其运行在内核态，可以截获所有进出网卡的数据包，并对传输层协议、网络层协议、应用层协议进行分析和过滤，从而监控整个网络活动。常见的用途包括检测流量异常、攻击行为分析、QoS管理、Web请求处理、内核模块开发等。
         
         # 3.eBPF概述
         
         ## 1.eBPF程序结构
         
         eBPF程序是由用户编写的字节码指令组成，指令以elf文件的格式保存在文件中，并由bcc-tools工具链进行编译、链接和加载。程序的主要流程可以划分为三个阶段：
         
         - **预处理阶段**：该阶段由bcc-tools的clang-bpf工具进行，负责将用户编写的代码转换成LLVM IR语言。
         - **编译阶段**：该阶段由LLVM把IR代码编译成可执行的机器代码。
         - **加载阶段**：该阶段由bcc-tools的tcptop、biotop等工具将生成的BPF程序加载到内核中。
         
         下图展示了eBPF程序的整体结构：
         
         
         上图中的“eBPF字节码”指的是由clang-bpf工具翻译得到的原始字节码文件。在编译和加载过程中，bcc-tools会进行以下操作：
         
         1. 将eBPF字节码编译成BPF指令；
         2. 用BPF linker链接多个BPF程序；
         3. 把BPF程序注入到指定位置；
         4. 设置BPF map表项；
         5. 执行BPF程序。
         
         概括一下，eBPF的运行流程如下：
         
         1. 用户编写BPF程序，包括头文件、尾部、全局变量声明、函数定义等；
         2. 通过clang-bpf工具将BPF程序转化为LLVM IR语言；
         3. LLVM编译器编译BPF程序，生成机器码；
         4. 用tcptop、biotop等工具将BPF程序加载到内核中；
         5. 内核根据BPF程序的指令执行相应的操作。
         
         ## 2.eBPF指令集
         
         eBPF使用汇编语言的语法定义了一系列指令集，包括加载、存储、算术运算、逻辑运算、控制转移、函数调用、地址偏移、跳转、帮助函数、报告错误等。目前，eBPF共有140个指令。其中，LOAD和STORE指令用于读取或写入寄存器或堆栈中的值，MATH和LOGIC指令用于执行基本的数学或逻辑运算，JMP和CALL指令用于实现条件和循环语句，MISC指令用于实现系统调用，返回结果等。
         
         每条eBPF指令都是固定大小的32位无符号整数，并以ELF文件的形式存储在BPF程序中。指令的第一个字节表示操作码，剩余的字节表示操作数，例如ADD r1, r2, r3表示将r2的值加到r3的值，并将结果放到r1。
         
         虽然eBPF提供了丰富的指令集，但是仍然有许多细节需要注意。例如，eBPF支持指针运算，可以通过取址或偏移来获取指针指向的内存地址。此外，eBPF还提供了一个帮助函数表，用来实现一些常用的功能，例如将十进制整数转换为ASCII字符串。
         
         ## 3.eBPF操作类型
         
         eBPF支持两种类型的操作：编程接口和事件通知。
         
         ### 1.编程接口
         
         eBPF支持两种类型的编程接口：核心函数（Kprobe）和跟踪点（Tracepoint）。
         
         #### Kprobe
         
         Kprobe是在内核源码中插入回调函数的一种方式，是一种比较底层的BPF编程方式。当用户调用Kprobe API时，bcc-tools工具链就会在内核源码的特定位置插入一个回调函数。这种方式可以用来监视内核函数的执行情况，包括参数传递、返回值、系统调用次数等。Kprobe API如下所示：
         
         ```c
         int kprobe_register(struct kprobe *kp);
         void kretprobe_unregister(struct kretprobe *kr);
         ```
         
         当用户注册Kprobe时，bcc-tools工具链就会在对应的内核源码位置插入一个回调函数，并保存它的地址。当内核中相应的函数被调用时，就会进入这个回调函数，可以读取或者修改函数的输入输出参数。
         
         #### Tracepoint
         
         Tracepoint是在内核源码中嵌入特殊的注释，bcc-tools工具链就可以捕获到这些注释，生成对应的BPF程序。这种方式类似于系统调用追踪，只不过不需要修改内核源码，而且可以捕获内核内部的各项事件，例如进程创建、上下文切换、网络收发包等。Tracepoint API如下所示：
         
         ```c
         static inline int tracepoint_probe_register(void (*func)(void *), const char *name,...);
         static inline void tracepoint_probe_unregister(int tp_id);
         ```
         
         用户可以调用tracepoint_probe_register() API来注册一个回调函数，并指定一个tracepoint名称。当内核中出现指定的tracepoint时，bcc-tools工具链就会自动生成相应的BPF程序，并加载到内核中执行。
          
         ### 2.事件通知
         
         eBPF除了提供编程接口之外，还可以接收系统事件通知，比如定时器超时、页面分配或释放、系统调用退出等。在这些事件发生时，eBPF可以触发一个BPF程序，执行一些操作。eBPF提供了四种事件通知接口，如下所示：
         
         ```c
         int bpf_open_perf_buffer(void (*cb)(void *, struct perf_event_header*,
                                           unsigned long, void*),
                                 struct bpf_map *map,
                                 uint32_t type);
         
         int bpf_attach_kprobe(enum bpf_probe_attach_type attach_type,
                             const char *symbol,
                             const char *fn_name,
                             uintptr_t addr,
                             int maxactive);
                         
         int bpf_attach_uprobe(enum bpf_probe_attach_type attach_type,
                             pid_t target_pid,
                             const char *library_path,
                             const char *symbol,
                             uintptr_t offset,
                             uintptr_t fn_address,
                             int maxactive);
                             
         int bpf_attach_tracepoint(const char *tp_category,
                                 const char *tp_event,
                                 void (*tp_fn)(void *),
                                 int tp_pid);
         ```
         
         用户可以调用bpf_open_perf_buffer() API来创建一个性能缓冲区，并设置回调函数来处理事件通知。bcc-tools工具链也可以捕获到其他类型的事件通知，并生成相应的BPF程序。
         
         # 4.BCC概述
         
         ## 1.BCC工具链
         
         BCC工具链包括前端和后端两部分。前端包括Clang-BPF、LLVM、Python，后端包括BCC/Kernel、bcc/tools等。
         
         Clang-BPF是一个集成开发环境（IDE），提供高级语言的编辑、语法检查、语法突出显示、编译、调试、优化等功能。Clang-BPF编译器将源代码编译成LLVM中间代码，再用LLVM编译器生成机器码，最后由bcc-tools工具链链接生成最终的eBPF程序。
         
         LLVM是开源的编译器技术，提供编译成不同平台代码的能力。bcc-tools提供了针对eBPF的各种工具，包括tcptrace、tcptop、biotop等。BCC/Kernel项目提供eBPF API，让BPF程序可以在用户态和内核态之间通信，并提供一些内核数据结构的访问接口。
         
         bcc/tools是用于开发和测试eBPF程序的工具集，包括bcc、ubpf等。bcc命令行工具可以作为交互式shell使用，也可以通过脚本的方式调用bcc/tools提供的各项工具。
         
         ## 2.BCC工具及示例
         
         ### 1.tcptracer-ebpf
         
         tcptracer-ebpf是一个监控TCP连接状态的工具。它会跟踪系统中所有TCP连接的新建、销毁等信息，并打印每个连接的完整信息，包括本地IP、本地端口、远端IP、远端端口、发送和接收字节数等。如果某台服务器有多个网卡，可以指定网卡的名字作为参数。tcptracer-ebpf示例如下：
         
         ```sh
         sudo./tcptracer-ebpf netdev c 192.168.1.1
         ```
         
         命令行参数：
         
         1. netdev：指定网卡设备名
         2. c：输出连接信息，而不是监听端口信息
         3. 192.168.1.1：指定要追踪的服务器IP地址
         
         输出示例：
         
         ```
         IP address       Local port     Remote port    PID    Protocol   Direction        Sent            Recv             Status
          192.168.1.1:46818          -> 192.168.1.2:www 31595  TCP        Outgoing          2872 bytes     0 bytes          ESTABLISHED
          192.168.1.2:www            -> 192.168.1.1:46818 43440  TCP        Incoming         0 bytes        2872 bytes      ESTABLISHED
          192.168.1.1:52654          -> 192.168.1.2:smtp 31455  TCP        Outgoing          841 bytes      703 bytes        TIME_WAIT
          192.168.1.2:smtp           -> 192.168.1.1:52654 43238  TCP        Incoming         703 bytes      841 bytes        FIN_WAIT2
          192.168.1.2:34212          -> 192.168.1.1:http 31607  TCP        Outgoing          1857 bytes     539 bytes        ESTABLISHED
          192.168.1.1:http           -> 192.168.1.2:34212 43447  TCP        Incoming         539 bytes      1857 bytes       ESTABLISHED
        [... ]
         ```
         
         ### 2.biotop-ebpf
         
         biotop-ebpf是一个监控磁盘IO的工具。它可以统计系统中所有块设备的I/O请求数量和字节数，包括读取和写入，以及读写的时间延迟。biotop-ebpf示例如下：
         
         ```sh
         sudo./biotop-ebpf -d sda,sdb
         ```
         
         命令行参数：
         
         1. -d sda,sdb：指定要监控的块设备名
         
         输出示例：
         
         ```
         Device:            disk
        Totaled operations: Read = 24617033, Write = 52044553
               Success:    Read = 24617033 (100.00%), Write = 52044553 (100.00%)
            Errors:      none
             Latency:
                           min    avg    max  stddev   MBit/s
         Device:         sda
          READ:    avg=0.10ms   med=0.10ms   max=1.66ms   0.00
                                          |    |||||
                                         _|    |   |_
                                  _____|_|_____|___|
                                 /______\
                                <________>
                              min       avg       max
                              15.70ns   104.91ns  884.27us
                                    ┬┴╶╮ ╭╯ ╭╮
                                      ║│  │  │ ║
                                    31.60    10.00 KB/s
         Device:         sdb
          WRITE:   avg=0.08ms   med=0.08ms   max=1.23ms   0.00
                                                  ▒▒▓╮
                                                    │ │
                                                   ░ │
                                                 ▒▒▒
                                                ▒▒▒▒
                                           42.10KB/s
         [ 66s CPU]
         ```
         
         ### 3.tcpconnect-ebpf
         
         tcpconnect-ebpf是一个监控TCP连接的工具。它会跟踪TCP连接的建立过程，并输出连接的初始包序列号、服务名称、远端主机IP和端口等信息。tcpconnect-ebpf示例如下：
         
         ```sh
         sudo./tcpconnect-ebpf -p www.example.com
         ```
         
         命令行参数：
         
         1. -p www.example.com：指定要跟踪的目标域名
         
         输出示例：
         
         ```
         IP version: IPv4
             Client MAC: 00:0c:29:b8:0f:d8
             Server MAC: 00:50:56:aa:ca:fd
                    IP: 192.168.3.11
                 PORT: 80
              Domain: www.example.com
           Connecting to: http://www.example.com:80
              Start time: Jul 26 15:46:13.154
              End time: Jul 26 15:46:13.453
                  TCP Seq: 2458728882
                   SACK ok: 0
                     ACK: 0
                   Window: 65535
                Timestamp: Jul 26 15:46:13.153111000 nsec
             Data length: 0
                       ---
     Transmission Control Protocol, Src Port: 59327, Dst Port: 80, Seq: 4762709863, Ack: 3115675666
     ECE CCSDS FP AE NO PAD I-BIT LEN FCS CHECKSUM
     
         Sequence Number: 4762709863    Acknowledgment Number: 3115675666
         Flags: 0x10 DF PA SYN
         Window size value: 65535
         Checksum: 0xa24a
         Urgent Pointer field Value: 0
         Options: (12 bytes), NOP, Maximum Segment Size, Window Scale: 128
         Padding: (0 bytes)

         Payload length: 0 byte
             [No payload]
                      ---
 
         IP version: IPv4
             Server MAC: 00:0c:29:37:fc:ed
                    IP: 192.168.3.11
                 PORT: 80
              Domain: example.com
               Header Length: 20 bytes
          Type of Service: BE
         Total Length: 28 bytes
       Identification: 0x0000
          Fragment Offset: 0 bytes
      Time to live: 64
    Differentiated Services Field: Not Set
                 Flags: No More Fragments
           Protocol: TCP
          Header Checksum: 0xaaaa
         Source Address: 192.168.3.11
         Destination Address: 172.16.17.32
                         DATA OFFSET: 5
                        Reserved: 0
                       Urgent pointer: 0
                    Options: (20 bytes), No Operation Padding
                          Padding: (1 bytes), End Of Option List
                            Body: GET / HTTP/1.1\r
Host: www.example.com\r
User-Agent: curl/7.64.1\r
Accept: */*\r
\r

                             ---- END OF HEADERS --
                             IP version: IPv4
                             Server MAC: 00:50:56:aa:ca:fd
                                    IP: 172.16.17.32
                                 PORT: 80
                               Header Length: 20 bytes
                           Type of Service: BE
                          Total Length: 358 bytes
                         Identification: 0x0000
                             Flags: 
                      Fragment Offset: 0 bytes
                     Time to Live: 64
                   Protocol: TCP
                     Header Checksum: 0x2de1
                    Source Address: 172.16.17.32
                    Destination Address: 192.168.3.11
                           DATA OFFSET: 5
                          Reserved: 0
                         Options: (24 bytes), Maximum segment size option, No Operation Padding, Padding
                          Padding: (1 bytes), End Of Option List
                       Urgent Pointer: 0
                     Next Header: HTTP
                  Hop Limit: 64
                         PAYLOAD LENGTH: 246 bytes
                           RESPONSE CODE: 200 OK
                          RESPONSE MESSAGE: OK
                        SERVER NAME: Apache/2.4.41 (Ubuntu)\r
Server powered by PHP/7.2.34\r
Last-Modified: Wed, 14 Feb 2021 08:38:26 GMT
                        CONTENT TYPE: text/html; charset=UTF-8
                            MIME VERSION: 1.0\r
Connection: close
                             HEADER SIZE: 266 bytes
                             HTML SIZE: 412 bytes
                             TEXT SIZE: 345 bytes
                             IMAGE SIZE: 0 bytes
                             TITLE SIZE: 78 bytes
                             JAVASCRIPT SIZE: 3544 bytes
                             CSS SIZE: 2470 bytes
                              TOTAL SIZE: 9701 bytes
                                    ---
                            TRANSFER SIZE: 336 bytes per second
                                    ---
                            REQUEST PACKETS: sent = 1, received = 1

              Serviced Transactions:
                                Requests:
                                        total = 1, success = 1 (100.00%)
                                    errors:
                                        reset connections = 0 (0.00%)
                                         bad requests = 0 (0.00%)
                                      overruns & drops = 0 (0.00%)
                                   timeouts & aborts = 0 (0.00%)
                                Responses:
                                        total = 1, success = 1 (100.00%)
                                    errors:
                                         no responses = 0 (0.00%)
                                        bad headers = 0 (0.00%)
                                         bad datagrams = 0 (0.00%)
                                         other failures = 0 (0.00%)
                                      retries exceeded = 0 (0.00%)
                                    timeout at start = 0 (0.00%)
                                    response too late = 0 (0.00%)
                                Transaction times:
                                    RTT min/avg/max = 0.000/0.000/0.000 ms
                                             pctile = 0.000%
                                Connection Times:
                                    connection establishment = 0.154 seconds
                                         first transaction = 0.302 seconds
                                        last transaction = 0.392 seconds
                                            duration = 0.190 seconds
                                 Average transfer rate:
                                        bits/second = 1472.000 bit/s
                                         bytes/second = 179.200 bytes/s
                                 Data transferred this session:
                                       packets sent = 11483, received = 11483
                                    payload length = 341891 bytes
                                      percentage of total = 0.11%\r
\r
<!-- Start Footer -->\r
...\r
<!-- End Footer -->
                                                                                 ----- END OF RESPONSE ------
         ```