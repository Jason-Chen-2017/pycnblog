
作者：禅与计算机程序设计艺术                    

# 1.简介
  
  
虚拟局域网（Virtual Local Area Network）技术或称虚拟链路聚合（VxLAN）技术是在网络层协议族中的一种，它利用IEEE 802.1q VLAN的标签数据报封装技术和UDP协议在二层转发报文的方法，将许多VLAN的数据报封装成一个大的IP数据报，通过源目的地址映射的方式在二层转发网络报文。VxLAN是一种专门用于大规模私有云环境下的容器网络的解决方案。本文介绍了VxLAN技术的基本概念及其实现过程，并基于VxLAN技术实现了一套边界路由系统。  

# 2.背景介绍  
## 什么是VLAN？
VLAN（Virtual Local Area Network）即虚拟局域网，它是基于IEEE 802.1Q协议标准的一套广播电路交换技术，可以把许多计算机网络连接到同一物理网络上，这样就可以把广播信号转化为独占的通信线路，提高网络性能，提高系统安全性。  
在实际生产环境中，VLAN用于划分广播域，划分不同业务的区域，控制网络流量等功能。但是在私有云环境中由于环境隔离性的要求，VLAN就显得无力地发挥作用了，这时需要另一种网络虚拟化技术来实现网络资源的共享。

## 为什么要用VLAN?
在私有云环境中，由于不同的应用都需要不同的网络地址空间，所以传统的VPC（Virtual Private Cloud）模式就无法适应需求，因此VLAN技术就应运而生。使用VLAN技术进行网络划分可以把网络中的不同的应用和服务分开，比如：业务A需要区分出属于自己的网络，同时也不需要和业务B共享网络资源；业务C可能需要单独获得网络资源，但又不希望被其他业务所影响。这些都是用虚拟机隔离虚拟网络的理想状态，而用VLAN实现则不一定完全符合。 

## 什么是VxLAN？
VxLAN（Virtual eXtensible LAN）是一种基于IEEE 802.1Q协议的网络虚拟化技术。它的主要优点包括：
1. IP-over-IP数据包封装。VxLAN通过扩展原始IP头部信息实现了IP数据报的封装，从而实现IP到IP的数据报传输。
2. 灵活且无损。由于是二层的虚拟化技术，它的网络带宽占用比其他二层虚拟化技术低很多。而且由于原始报文通过封装技术打包后，再在二层发送，不存在任何性能损失。
3. 支持多播。VxLAN支持多播，即多个主机之间可直接进行广播。

# 3.核心算法原理和具体操作步骤以及数学公式讲解

## VxLAN基本原理

### 数据报封装过程

1. 原始IP数据报包含源IP地址和目的IP地址，如果需要建立VxLAN隧道，那么VxLAN的双方都需要知道对方的MAC地址，以确定是从哪个网口出去发出的包，否则就不能正确转发到目标主机。

2. 在IP数据报的首部增加一个新的字段“Protocol Type”，用来标识此数据报封装了什么协议类型。在这里，需要把原始IP数据报封装成“VXLAN”类型的数据报。  

3. 根据VXLAN的标准定义，VXLAN的数据报包含两部分，第一部分是VXLAN头部，第二部分是原始IP数据报。其中，VXLAN头部包含两个重要的参数，分别是“Flags”（标志位）和“vni”。“Flags”用于标记该数据报是否被加密、压缩等处理过，“vni”用于唯一标识一个VXLAN隧道。  

4. 把封装好的VxLAN数据报封装进一个UDP数据报中，并且设置UDP的源端口号为4789，目的是为了兼容IPv4/IPv6网络栈，让VxLAN数据报在IPv4/IPv6互通。  

### 路由过程

当一个经过封装的VxLAN数据报抵达目的节点时，首先根据目的IP地址查找到目的VTEP所在的主机。然后，主机会查看此数据报的源MAC地址，若此源MAC地址存在相应的接口，则直接将数据报发送给此接口，否则会根据ARP表进行地址转换，得到正确的源MAC地址。至此，数据报就完整了，只不过还要经过一次额外的转发过程才能最终到达目标主机。  

## VxLAN实现边界路由

实现边界路由，需要构建边界控制器（Border Gateway），它连接着内部网络与外部网络，由它来监听各种数据报，判断其是否应该通过某条路径到达目的地，并做相应的过滤和NAT操作。一般来说，边界路由器是放在核心路由器之后，工作在2-3跳距离内，能够充分利用内部网络的计算能力。 
VXLAN作为一种轻量级的虚拟链路聚合技术，能够有效地减少内部网络的路由负担。VxLAN和VLAN的组合使得边界路由成为可能，而实现边界路由需要在核心路由器（Cisco ASR设备）上安装VXLAN模块，通过对封装的数据报进行解析并通过本地策略进行转发，实现更细粒度的基于网络和应用的流量控制。

具体步骤如下：  
1. 安装VXLAN模块：下载、编译并安装VXLAN模块。
2. 配置VXLAN隧道：配置创建VXLAN隧道的命令，并加载到核心路由器中。
3. 配置策略路由：配置策略路由规则，使得在目的地址为相应网段的数据包通过VXLAN隧道传输。
4. 配置BGP动态路由：配置BGP邻居，使得本地路由器能与外部网络路由器建立BGP动态路由。
5. 配置边界控制器：配置边界控制器，连接着核心路由器和外部网络，并提供相应的管理界面。

## VxLAN优缺点

### 优点：
- VxLAN能够有效地利用二层的带宽资源，能够支持上万台服务器的集群部署；
- VxLAN采用IP-over-IP的封装方式，不改变底层协议，对运维人员透明；
- VxLAN能够快速传递数据报，避免了因网络拥塞导致的丢包问题；
- VxLAN的多播和动态路由特性能够满足复杂的业务需求。

### 缺点：
- VxLAN的隧道建立时间较长，延迟较高；
- VxLAN有自身的网络栈处理，无法支持所有类型的业务场景；
- 因为VxLAN是在二层封装，所以其最大传输单位为帧（Frame），无法承载大数据块。

# 4.具体代码实例和解释说明

## Cisco IOS配置VxLAN

### 安装VXLAN模块

下载、编译并安装VXLAN模块：  

```
#下载、编译并安装VXLAN模块
[root@VSRX-SRX1: ~]# cd /home/cisco/src
[root@VSRX-SRX1: src] # wget https://github.com/openvswitch/ovs/archive/v2.15.0.tar.gz
[root@VSRX-SRX1: src] # tar zxvf v2.15.0.tar.gz -C /opt/
[root@VSRX-SRX1: src] # mv /opt/ovs* ovs2.15
[root@VSRX-SRX1: src] # cd /opt/ovs2.15
[root@VSRX-SRX1: ovs2.15] #./boot.sh
[root@VSRX-SRX1: ovs2.15] #./configure --prefix=/usr --localstatedir=/var --sysconfdir=/etc && make && sudo make install
[root@VSRX-SRX1: ovs2.15] # git clone https://github.com/openvswitch/ofproto-vxlan.git
[root@VSRX-SRX1: ovs2.15] # cd ofproto-vxlan
[root@VSRX-SRX1: ofproto-vxlan] #./autogen.sh
[root@VSRX-SRX1: ofproto-vxlan] #./configure --with-linux=/lib/modules/$(uname -r)/build && make && sudo make install
[root@VSRX-SRX1: ofproto-vxlan] # modprobe vxlan && lsmod | grep vxlan
```

### 配置VXLAN隧道

配置创建VXLAN隧道的命令，并加载到核心路由器中：

```
#配置创建VXLAN隧道的命令，并加载到核心路由器中
[root@VSRX-SRX1: ~] # cat <<EOF > /tmp/vxlan_config.txt
feature vn-interconnect ;
router bgp 65001
 bgp router-id 1.1.1.1
 no bgp default ipv4-unicast
 neighbor 172.16.10.2 remote-as 65002
  address-family vxlan
   route-target export 65001:100
   route-target import 65002:100
   activate
  exit-address-family
!
!
interface Vxlan1
 description CORE-Srx-to-Wan
 ip address 192.168.10.1/24
 encapsulation vxlan source-interface Loopback0 vni 101
 shutdown
 vrf attach core
 ip helper-address 172.16.10.2
!
line console
 exec-timeout 0
 stopbits 1
!
end
EOF
[root@VSRX-SRX1: ~] # copy run start

#查看配置是否成功
[root@VSRX-SRX1: ~] # show run section nv overlay | in "vxlan" | i "vni\|encapsulation"
ip routing vn-segment-routing
vrf context core
 rd auto
 vni 101
  l3vni ingress mac-learning
  bridge-domain br1
    service instance isis core isis-instance
      interface Vxlan1 unit 0 family point2point
        ip address 192.168.10.1/24 primary
     !
   !
    routing protocol static
     route 0.0.0.0/0 next-hop 192.168.10.2 via vlan12
!
interface Vxlan1
 description CORE-Srx-to-Wan
 encapsulation vxlan vni 101 inner-vlan 0
!
```

### 配置策略路由

配置策略路由规则，使得在目的地址为相应网段的数据包通过VXLAN隧道传输：

```
#配置策略路由规则
[root@VSRX-SRX1: ~] # config t
Enter configuration commands, one per line.  End with CNTL/Z.
ios(config)# ip prefix-list NN permit 192.168.10.0/24 le 32
ios(config)# route-map NN permit 10
 match ip address NN
!
ios(config)# router bgp 65001
 bgp bestpath as-path multipath-relax
 network 192.168.10.0/24 backdoor
 redistribute connected route-map NN out
!
#查看配置是否成功
ios# show ip route
Codes: K - kernel route, C - connected, S - static, R - RIP,
       O - OSPF, I - IS-IS, B - BGP, E - EIGRP, N - NHRP,
       T - Table, v - VNC, V - VNC-Direct, A - Babel, D - SHARP,
       F - PBR, f - OpenFabric,
       > - selected route, * - FIB route

       [172.16.10.0/24]/32 is subnetted, 1 subnets
C        172.16.10.0 is directly connected, Ethernet0
i L3    172.16.10.1 dev Vxlan1 scope link table local proto vxlan


            Network          Next Hop            Metric LocPrf Weight Path
*>i [172.16.10.0/24]   0.0.0.0                           100      32768 i
                                                0         32768 i
O>*    0.0.0.0/0          172.16.10.1                      0            0 10 20 30
                                             Ethernet0
                                     0.0.0.0                            0 32768 i
                                      0.0.0.0                            0 32768 i
                                      ::/0                              0 32768 i
                                      ::/0                              0 32768 i
                                      FE80::/64                         0     100 20 30
                                        fd00:a516:7c1b:17cd:6d81:2137:bd2a:2c5b                             0     100 20 30
                                        2001:DB8::                          0            0 10 20 30
                                            ::                                 0 32768 i
                                            ::                                 0 32768 i
```

### 配置BGP动态路由

配置BGP邻居，使得本地路由器能与外部网络路由器建立BGP动态路由：

```
#配置BGP邻居
[root@VSRX-SRX1: ~] # config t
Enter configuration commands, one per line.  End with CNTL/Z.
ios(config)# router bgp 65001
 bgp log-neighbor-changes
 neighbor 172.16.10.2 remote-as 65002
  address-family ipv4 unicast
   update-source Loopback0
   send-community
   advertise-best-external
   allow 65001
   soft-reconfiguration inbound always
   filter-list export block
   distribute-list export LEAKING
   maximum-paths ibgp 2
   exit-address-family
!
#查看配置是否成功
ios# show ip bgp neighbors
BGP neighbor is 172.16.10.2,  remote AS 65002, internal link
  BGP version 4, remote router ID 172.16.10.2
  BGP state = Established, up for 01:13:22
  Last read 00:00:28, last write 00:00:11, hold time is 180, keepalive interval is 60 seconds
  Neighbor sessions:
    1 active, is not multisession capable (disabled)
  Neighbor capabilities:
    Route refresh: advertised and received(new)
    Four-octets ASN Capability:advertised and received
    Address families:
      IPv4 Unicast (was enabled during session startup)
    Graceful Restart Capability:received
      Remote Restart timer is 120 seconds
      Address families advertised by peer:
        IPv4 Unicast
  Message statistics:
    Inq depth is 0
    Outq depth is 0
                         Sent       Rcvd
    Opens:                  1          1
    Notifications:          0          0
    Updates:                5          5
    Keepalives:           170       170
    Route Refresh:          0          0
    Total:               177       177
  Minimum time between advertisement runs is 0 seconds

  For address family: IPv4 Unicast
  Community attribute sent to this neighbor(all)
  Connections established 1; dropped 0
  Last reset never
  External BGP neighbor configured for receive filtering
  Dynamic capability received from neighbor 172.16.10.2
```

### 配置边界控制器

配置边界控制器，连接着核心路由器和外部网络，并提供相应的管理界面：

```
#配置边界控制器
[root@core-Router:~]$ cat <<EOF > /usr/local/codenotify/bin/srx-notify.py
#!/usr/bin/python

import sys
from subprocess import call

def main():
    cmd = "/sbin/codenotify start systemctl restart conntrackd"
    rc = call([cmd], shell=True)
    if rc!= 0:
        print >>sys.stderr, "Failed to execute command: ", cmd

if __name__ == "__main__":
    main()
EOF

[root@core-Router:~]$ chmod +x /usr/local/codenotify/bin/srx-notify.py

[root@core-Router:~]$ cat <<EOF > /usr/local/codenotify/bin/srx-monitor.py
#!/usr/bin/env python

import os
import sys
from subprocess import check_output

def get_process_pid(proc):
    pids = []

    try:
        lines = open('/proc/{}/status'.format(os.getpid())).readlines()

        for line in lines:
            name, value = line.split(':')

            if 'Name' == name.strip():
                process_name = value.strip().lower()

                break

        else:
            raise RuntimeError("Could not determine current process name.")

        for pid in os.listdir('/proc'):
            if pid.isdigit():
                try:
                    exe = os.readlink('/proc/{}/exe'.format(pid))

                    if proc.lower() in exe.lower():
                        pids.append(int(pid))

                except OSError:
                    pass

    except IOError:
        return None

    return pids

def main():
    procs = ['conntrackd']

    while True:
        for proc in procs:
            pids = get_process_pid(proc)

            if not pids:
                message = "{} process not running.".format(proc)
                code = 0

            elif len(pids) > 1:
                message = "{} processes ({}) are running.".format(len(pids), ', '.join(str(pid) for pid in pids))
                code = 1

            else:
                message = "{} process ({}) is running".format(proc, pids[0])
                code = 0

            print message
            
            if code!= 0:
                os.system("/usr/local/codenotify/bin/srx-notify.py")
        
        time.sleep(60)

if __name__ == '__main__':
    main()
EOF

[root@core-Router:~]$ chmod +x /usr/local/codenotify/bin/srx-monitor.py

[root@core-Router:~]$ systemctl enable srx-monitor.service
Created symlink from /etc/systemd/system/multi-user.target.wants/srx-monitor.service to /usr/lib/systemd/system/srx-monitor.service.

[root@core-Router:~]$ systemctl daemon-reload
[root@core-Router:~]$ systemctl start srx-monitor.service
```