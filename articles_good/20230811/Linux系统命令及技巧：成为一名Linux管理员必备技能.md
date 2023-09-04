
作者：禅与计算机程序设计艺术                    

# 1.简介
         

Linux操作系统是一个开源且免费的类Unix操作系统，可以说，它是当今最流行、功能最强大的服务器操作系统。基于Linux，众多知名公司和个人开发者都建立了自己的云服务、部署了应用系统等等。作为一个IT从业人员或企业，掌握Linux操作系统管理知识将会加快工作效率并提高工作能力，让你的工作更加顺利，也能对企业和自身的发展有着更好的贡献。因此，学习Linux系统管理命令和技巧，将成为成功的关键一步。
本文通过大量实践案例，全面讲述Linux系统管理命令的基础知识、高级用法和扩展知识，系统化地讲解如何高效地运用这些命令，创造出独具魅力的管理效果。文章既可作为初级到中级用户阅读，也可作为中高级Linux管理员参考学习。文章通过实际场景，生动地讲解各类Linux命令，使读者能快速学会使用命令解决实际问题。本文并不逐条罗列所有的命令选项和参数，只为大家呈现给您最常用的命令，帮助您快速了解并提升效率。同时，我们还会以实例为导向，指导读者掌握命令的使用方法和基本技巧，能够充分利用命令实现各种系统管理任务。
## 2.Linux系统管理概念
首先，我们需要先认识一些Linux系统管理的基本概念和术语。
### 2.1 用户管理
在Linux系统中，每个用户都被分配唯一的ID（User ID，UID）和组ID（Group ID，GID）。在Linux中，通过文件/etc/passwd和/etc/group可以查看当前所有用户及其相关信息。

- 添加用户: `useradd` 命令用于添加新的用户账号；
- 修改用户密码: 使用 `passwd` 命令可以修改指定用户的密码；
- 删除用户: 使用 `userdel` 命令删除指定的用户账户；
- 设置用户主目录: 使用 `usermod` 命令可以设置或修改用户主目录路径；
- 查看用户信息: 使用 `id`、`last` 和 `w` 命令可以查看指定用户的信息；
- 更改用户组: 使用 `usermod` 命令更改用户所属的用户组。

### 2.2 文件权限管理
Linux系统中，文件的访问权限由三部分组成——用户（u），群组（g），其他用户（o）。每一部分表示对应的读、写、执行权限。在命令行中，可以使用以下命令进行权限控制：

- chmod: 可以用来变更文件的权限模式；
- chown: 可以用来变更文件的拥有者和群组；
- chgrp: 可以用来变更文件所属的群组。

在命令行中使用 `ls -l` 可以看到详细的文件权限信息。

### 2.3 服务管理
Linux操作系统提供了丰富的服务管理工具，比如 systemctl、init、upstart等。其中，systemctl 是较新的服务管理工具，提供了一个高层的统一接口。一般来说，我们可以通过以下命令来管理服务：

- service: 可以用来控制运行中的服务，包括启动、停止和重启等操作；
- chkconfig: 可以用来启用或禁用系统服务的开机自启动项；
- systemctl: 可以用来管理系统的所有服务，包括启动、停止、重启、查看状态等。

### 2.4 进程管理
在Linux系统中，进程是系统资源分配和调度的基本单位，它是系统运行的最小单元。通过以下命令可以查看进程相关信息：

- ps: 可以查看当前正在运行的进程列表；
- top: 可以实时动态地查看系统中所有进程的资源占用情况；
- kill: 可以杀死指定的进程；
- nohup: 可以将进程放入后台执行；
- jobs: 可以显示后台任务的状态。

## 3.Linux系统管理命令
下面，我们将以实际案例的方式，详细介绍Linux系统管理命令。

### 3.1 目录管理
#### 3.1.1 创建目录
创建目录的命令是mkdir，其语法如下：

mkdir [OPTIONS] DIRECTORY...

其中，DIRECTORY 表示要创建的目录名称，如果多个名称之间存在空格，则需要使用引号包裹起来。另外，OPTIONS支持如下几种：

- -m, --mode=MODE 设置目录权限；
- -p, --parents 递归创建父目录直至最后一级；
- -v, --verbose 在创建每一个目录之后打印信息；
- -Z, --context[=CTX] 为新目录设置安全上下文标签。

例如，以下命令创建一个名为 test 的目录，并赋予它的权限为 777：

$ mkdir /test
$ ls -ld /test
drwxrwxrwx.  2 root     root          9 Sep 19 15:26 /test

此外，还可以用 -p 参数递归地创建目录：

$ mkdir -p /test/path/to/create
$ ls -ld /test/path/to/create
drwxr-xr-x. 2 root     root        6 Sep 19 15:31 /test/path/to/create

#### 3.1.2 修改目录属性
修改目录属性的命令是chmod，其语法如下：

chmod [OPTIONS] MODE DIRECTORY...

其中，MODE 表示权限模式，可能的值为 rwxr-xr-x 这样的组合，分别对应的是读、写、执行的用户、组和其它权限。当然，也可以使用符号表示法，比如 +x 可以授予用户执行权限，-x 可以取消用户执行权限。

例如，以下命令将上面的 /test 目录的权限模式设置为 755：

$ chmod 755 /test
$ ls -ld /test
drwxr-xr-x. 2 root     root         11 Sep 19 15:26 /test

#### 3.1.3 移动、复制、删除目录
移动、复制、删除目录的命令如下：

- mv: 将源文件或者目录移动到目标位置，若目标位置已存在文件，则覆盖掉它。
- cp: 将源文件或者目录复制到目标位置，若目标位置已存在文件，则覆盖掉它。
- rm: 删除文件或者目录。

例如，假设有一个目录 `/test`，想要把它移动到 `/tmp` 下面去，可以使用以下命令：

$ mv /test /tmp
$ ls -ld /tmp/test
drwxr-xr-x. 2 user    group           11 Sep 20 09:14 /tmp/test

注意，mv 命令也可以用作目录之间的相互移动：

$ mv dir_a dir_b
$ mv dir_c/* dir_d
$ mv file* dir_e

#### 3.1.4 列出目录内容
列出目录内容的命令是ls，其语法如下：

ls [OPTIONS] [FILE...]

其中，FILE 表示要显示信息的目录名称，默认值是当前目录。OPTIONS 支持如下几种：

- a, --all 显示隐藏文件（以. 开头的目录或文件）；
- l, --long 显示文件的详细信息；
- R, --recursive 递归列出子目录内容；
- A, --almost-all 只显示非. 或.. 开头的文件；
- h, --human-readable 以易读方式显示文件大小；
- d, --directory 仅列出目录本身；
- F, --classify 根据类型分类文件；
- f, --file-type 显示文件类型而非权限标志；
- o, --only-dir 不显示文件，只显示目录。

例如，用 ls 命令查看 /home 目录的内容，如下所示：

$ ls -lh /home
total 40K
dr-xr-xr-x 15 root root 4.0K Sep 20 09:35 john
dr-xr-xr-x  6 root root 4.0K Sep 20 09:35 sarah
drwxr-xr-x  2 root root 4.0K Aug 22  2016 tmp

其中，total 后面跟的数字表示目录下文件的总数量，drwxr-xr-x 是目录权限信息，后面跟的是用户名、组名、文件大小和日期信息。

### 3.2 文件管理
#### 3.2.1 创建文件
创建文件命令是 touch，其语法如下：

touch [OPTIONS] FILE...

其中，FILE 表示要创建的文件名称。OPTIONS 支持如下几种：

- -a, --access TIME 指定最后存取时间；
- -c, --no-create 不创建任何文件；
- -d, --date DATE 指定日期时间；
- -m, --modify TIME 指定最后修改时间；
- -t, --time TIME 指定日期时间，该选项根据指定的年月日生成时间戳。

例如，用 touch 命令创建一个名为 hello 的文件：

$ touch hello
$ ls hello
hello

#### 3.2.2 拷贝文件
拷贝文件命令是 cp，其语法如下：

cp [OPTIONS] SOURCE DESTINATION

其中，SOURCE 表示源文件名称，DESTINATION 表示目标文件名称。OPTIONS 支持如下几种：

- -a, --archive 对复制文件同时保持文件的属性，包括权限、所有权、时间戳等；
- -d, --dereference 如果源文件是一个链接文件，那么直接复制链接指向的文件而不是链接文件本身；
- -f, --force 强制覆盖已经存在的文件；
- -i, --interactive 询问是否覆盖文件；
- -P, --no-dereference 不要复制符号连接文件本身，复制其指向的文件；
- -R, --recursive 递归复制整个目录。

例如，将 hello 文件拷贝到 /tmp 目录：

$ cp hello /tmp
$ ls -lh /tmp/hello
-rw-r--r-- 1 root root 0 Sep 20 09:49 /tmp/hello

#### 3.2.3 移动文件
移动文件命令是 mv，其语法如下：

mv [OPTIONS] SOURCE DESTINATION

其中，SOURCE 表示源文件名称，DESTINATION 表示目标文件名称。OPTIONS 支持如下几种：

- -b, --backup 恢复备份文件；
- -f, --force 强制覆盖已存在的文件；
- -i, --interactive 询问是否覆盖文件；
- -T, --no-target-directory 当目标目录不是一个普通目录的时候不要尝试进入；
- -U, --update 只覆盖较新文件；
- -v, --verbose 显示执行过程。

例如，将 /tmp/hello 文件移动到 /root 目录：

$ mv /tmp/hello /root
$ ls -lh /root/hello
-rw-r--r-- 1 root root 0 Sep 20 09:49 /root/hello

#### 3.2.4 删除文件
删除文件命令是 rm，其语法如下：

rm [OPTIONS] FILE...

其中，FILE 表示要删除的文件名称。OPTIONS 支持如下几种：

- -i, --interactive 询问是否删除文件；
- -f, --force 强制删除文件或目录，无需确认；
- -r, --recursive 递归删除目录及目录内的文件；
- -d, --dirtarget DIR 作为前缀的参数指定目录而不是文件。

例如，删除 /root/hello 文件：

$ rm /root/hello
$ ls -lh /root/hello
ls: cannot access '/root/hello': No such file or directory

#### 3.2.5 文件搜索
文件搜索命令是 find，其语法如下：

find [PATH] [OPTIONS] [ACTIONS]

其中，PATH 表示搜索的起始路径，默认值是当前目录。OPTIONS 支持如下几种：

- -name PATTERN 指定搜索文件名的正则表达式匹配规则；
- -iname PATTERN 指定搜索文件名的忽略大小写的正则表达式匹配规则；
- -size N[k|M|G] 指定文件大小的单位；
- -amin N 指定查找文件的访问时间，单位为分钟；
- -anewer FILE 指定查找比 FILE 新（更新）的文件；
- -empty 查找为空的文件；
- -maxdepth NUM 指定最大递归深度；
- -mindepth NUM 指定最小递归深度。

ACTIONS 支持如下几种：

- -print 打印匹配的文件名；
- -ls 显示匹配的文件的详细信息；
- -delete 删除匹配的文件。

例如，用 find 命令查找 /root 目录下的所有 txt 文件：

$ find /root -name "*.txt"
/root/abc.txt
/root/def.txt

### 3.3 网络管理
#### 3.3.1 IP配置
IP地址配置命令是 ifconfig，其语法如下：

ifconfig [OPTIONS] DEVICE

其中，DEVICE 表示要配置的网络设备，通常是以 eth 开头的网卡设备。OPTIONS 支持如下几种：

- -s, --subnet SUBNET 设置子网掩码；
- -n, --netmask NETMASK 设置子网掩码；
- -b, --broadcast BROADCAST 设置广播地址；
- -a, --mtu MTU 设置MTU；
- -A, --append 追加路由表项；
- -d, --delete 从路由表中删除指定的路由项目；
- -i, --inet 显示IP地址信息；
- -I, --link 显示网卡信息；
- -m, --metric METRIC 指定路由的质量因子；
- -r, --route 显示路由信息；
- -t, --tunnel 建立隧道接口；
- -w, --ether 显示以太网地址；
- -W, --hw ether 设置以太网地址。

例如，用 ifconfig 配置 eth0 网卡的IP地址和子网掩码：

$ sudo ifconfig eth0 192.168.0.1 netmask 255.255.255.0
$ ifconfig eth0
eth0      Link encap:Ethernet  HWaddr fa:16:3e:ec:cb:ba  
inet addr:192.168.0.1  Bcast:192.168.0.255  Mask:255.255.255.0
UP BROADCAST RUNNING MULTICAST  MTU:1500  Metric:1

#### 3.3.2 域名解析
域名解析命令是 dig，其语法如下：

dig [OPTIONS] DOMAIN [TYPE] [CLASS]

其中，DOMAIN 表示需要解析的域名，TYPE 表示查询记录类型，默认为 A 。CLASS 表示查询记录的类别，默认为 IN 。OPTIONS 支持如下几种：

- -x, --hex 输出结果十六进制编码；
- -y, --yaml 输出结果 YAML 格式；
- -t, --tcp 使用 TCP 协议发送 DNS 请求；
- -c, --count COUNT 查询次数；
- -j, --json 使用 JSON 格式输出结果；
- -k KEYWORD ，--key KEYWORD 指定搜索关键字；
- -e,--edns 打开 EDNS 扩展，可以设置缓冲区大小；
- @SERVER 设置服务器地址；
- +search+version 设置“搜索”和“版本”标签；
- +noall+answer 只显示结果。

例如，用 dig 命令解析 www.baidu.com 域名的 A 记录：

$ dig www.baidu.com +short
192.168.3.11

#### 3.3.3 端口扫描
端口扫描命令是 nmap，其语法如下：

nmap [OPTIONS] [TARGET] [PORT]

其中，TARGET 表示要扫描的主机名或 IP 地址，可以指定多个主机，用空格隔开。PORT 表示要扫描的端口范围，格式如 “20-30” 。OPTIONS 支持如下几种：

- -sS, --syn 开启 SYN 扫描；
- -sU, --udp 开启 UDP 扫描；
- -sA, --arp 开启 ARP 扫描；
- -sN, --null 开启 NULL 扫描；
- -sF, --fin 开启 FIN 扫描；
- -sX, --xmas 开启 XMAS 扫描；
- -sO, --ip-protocol 开启 IP 协议扫描；
- -f, --fragment 允许 IP 分片；
- -D decoy1[,decoy2][,ME], --spoof-mac DECOY 设置欺骗 MAC 地址；
- -e,--edge-scan 探测入侵主机；
- -g GATEWAY, --source-port GATEWAY 指定数据包来源端口；
- -p PORT | -p FROM-TO 指定扫描端口；
- -PE, --ping-echo 通过 ICMP 回显检测主机存活；
- -PS, --ping-sweep 扫描网段内主机存活；
- -PA, --ping-ack 检测主机是否回应 ping 数据包；
- -PU, --ping-udp 使用 UDP 协议探测主机存活；
- -PY, --sniffer-off 关闭嗅探模式；
- -PR, --traceroute 追踪路径上的所有路由器；
- -PP, --sniff-probes 嗅探 ICMP 探测报文；
- -PM, --mobile-scan 扫描苹果设备；
- -PO, --rpc-ping 远程调用 ping 方法；
- -f, --fast 快速模式；
- -sL, --list Scan listing mode；
- -p-,--top-ports[=NUMBER] Scan top N most popular ports；
- -sV, --version-intensity INTensity 设置版本扫描强度；
- -O, --osscan-guess 使用 OSSCAN 自动探测操作系统；
- -v, --verbose 详细模式；
- -d, --debugging 调试模式；
- -oN, --output-file OUTPUT.gnmap 将结果保存到 gnmap 文件中；
- -oX, --xml-output OUTPUT.xml 将结果保存到 xml 文件中。

例如，用 nmap 命令扫描主机 192.168.1.1 上所有端口：

$ nmap 192.168.1.1

#### 3.3.4 网络监控
网络监控命令是 tcpdump，其语法如下：

tcpdump [OPTIONS] [PATTERN]

其中，PATTERN 表示过滤条件，可以指定流量类型、主机地址等。OPTIONS 支持如下几种：

- -i INTERFACE, --interface INTERFACE 指定监听的网络接口；
- -nn, --numeric-hosts 不进行域名解析；
- -c NUMBER, --number NUMBER 捕获的包的数量；
- -w FILENAME, --write FILENAME 写入文件；
- -B SIZE, --buffer-size SIZE 设置缓冲区大小；
- -A, --ascii 以 ASCII 打印数据包；
- -XX, --xxverbose 详细模式；
- -Z, --packet-trace 开启数据包跟踪。

例如，用 tcpdump 命令捕获主机 eth0 上的所有流量，并且写入文件 capture.pcap 中：

$ tcpdump -i eth0 -w capture.pcap