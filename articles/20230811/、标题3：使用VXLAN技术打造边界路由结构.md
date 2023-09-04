
作者：禅与计算机程序设计艺术                    

# 1.简介
         

随着云计算和容器技术的兴起,边界路由技术也逐渐得到重视,边界路由指的是将不同数据中心之间的通信连接起来并进行流量管理的功能模块。为了实现边界路由，需要解决两个关键问题:

① 数据包在物理网络上传输的过程中经过多个网关和路由器,可能会发生失败或延迟。
② 不同组织之间的数据互通性较差,需要对不同数据中心之间的通信进行流量控制。
基于以上两点原因,业界提出了使用虚拟局域网(VLAN)技术构建边界路由结构的想法。VLAN技术可以在数据中心内部通过划分子网的方式实现数据隔离,避免数据包跨越路由和交换机,从而有效防止数据包传输过程中的环路、拥塞和延迟等问题。但是,VLAN技术无法跨越数据中心的边界,使得各个数据中心之间无法直接互通。
另一种方式是使用隧道技术,通过VPN建立可穿越路由器的加密隧道,并使用隧道协议传输数据。但是,建立和维护VPN隧道成本高昂,难以满足快速变化的业务需求。因此,业界又开始寻求其他解决方案。一种新型的边界路由技术——VXLAN（Virtual eXtensible LAN）正逐渐成为解决方案之一。
VXLAN是一种隧道协议,它允许多个Overlay Virtual Local Area Networks(OvLans)在一个物理网络上同时运行。OvLans之间的通信通过IP协议进行封装,并在底层交换机上实现隧道传输。由于每个OvLan都与物理网络完全隔离,因此可以有效防止数据包传播过程中出现环路、拥塞和延迟的问题。通过在多个数据中心之间建立VXLAN tunnels,可以实现跨越边界的数据互通。
# 2.基本概念术语说明
## VLAN技术
VLAN (Virtual Local Area Network)即虚拟局域网技术。VLAN是一种网络设备用来实现数据分割的一种技术。VLAN把物理LAN分割成多个小的子网，每个子网内的数据包只在相邻的子网间转发。VLAN使得数据中心内部网络可以按照工作组或用户要求灵活配置，避免冲突和资源浪费。VLAN引入了虚拟化技术，使得网络由独立的局域网集合组成，每个局域网又可以看做是一个整体。

在VLAN中，每一个VLAN都有唯一标识ID号。标识范围是1~4094。数据包在到达某个VLAN后，会根据目的地址字段中的MAC地址决定是否要进行转发。如果目的地址不属于当前VLAN成员，那么该数据包就被丢弃。如果目的地址属于当前VLAN成员，那么该数据包就会被转发到同一VLAN内的所有主机。虽然VLAN允许划分子网，但实际应用中，一般还是采用VLAN的形式，而不是真正意义上的子网。

通常情况下，系统管理员会将同一个工作组（如HR部门）的数据放在同一个VLAN中，这样就可以利用VLAN提供的一些安全性和资源隔离优势，加强组内的通信。然而，VLAN并不能完全地保障数据包的安全性，仍存在一定的隐患。另外，VLAN的作用也是受限的，因为它只能限制在同一个工作组内部的通信，对于不同工作组的主机之间仍无法实现隔离。

## VXLAN
VXLAN (Virtual eXtensible LAN)，即虚拟可扩展局域网技术。它是一种隧道协议，它的目标就是解决VLAN技术存在的一些问题。VXLAN是一种面向overlay的虚拟网络技术，通过VXLAN可以实现同一个物理网络的不同VLAN之间的数据包传输。VXLAN的特点是：

① 使用IP协议进行封装，类似于VLAN技术，能够有效避免数据包的路由问题；
② 支持多个VXLAN tunnels同时运行，能够支持复杂的多租户网络环境；
③ 每个VXLAN tunnel都是单独的，所以不会导致原生VLAN存在的各种问题。

在实现了VXLAN tunnel之后，数据包就可以在不同的数据中心之间传输。无论源主机所在的数据中心和目的主机所在的数据中心相同或者不同，数据包都会被正确地传输。而且，VXLAN还支持不同网络的安全隔离，并且对数据包的处理速度很快，不受网速的影响。

## IP-in-IP Tunneling Protocol
IP-in-IP Tunneling Protocol(IPIP) 是一种隧道协议，用于在隧道中传输IPv4数据包。IPIP与VXLAN是两种不同的技术，它们都可以通过隧道的方式在IPv4和IPv6之间进行传输。IPIP可以与其他任何类型的IP协议搭配使用，包括TCP、UDP、ICMP等。IP-in-IP tunneling的一个优点就是不需要修改现有的IPv4网络结构，能够有效降低对IPv4网络的改动。

# 3.核心算法原理和具体操作步骤以及数学公式讲解
## VXLAN隧道
VXLAN隧道是建立在VXLAN协议之上的VXLAN隧道技术。Vxlan tunnels是由VTEP (VXLAN Tunnel Endpoints) VXLAN隧道端点和Multicast组播MAC地址所构成的。Vxlan tunnels在底层的交换机上实现隧道传输。Vxlan tunnels的数据包与普通的IP数据包一样，封装在IP数据报中。Vxlan tunnels的类型为二层 tunnel，由VTEP VXLAN隧道端点实现。VTEP端点既负责创建tunnel，也负责接受其他端点加入tunnel。

### 创建VXLAN隧道
创建一个新的Vxlan tunnel需要VTEPs之间的协商过程。当VTEP1想要创建一个新的Vxlan tunnel时，它需要发送一个封装了VXLAN头部的原始数据包给VTEP2。VTEP2收到这个数据包后，检查它的VNI（Vxlan Network Identifier），如果是接收到的第一个VNI，那么就允许其创建新的Vxlan tunnel。然后，VTEP2会生成一个新的Vxlan头部，修改它的源IP地址和目的IP地址，并设置VNI域和其他参数，再重新封装原始数据包，并通过源Mac地址发送给VTEP1。VTEP1收到这个数据包后，取出里面VNI域的值，检查是否为一个已知的Vxlan tunnel，如果是，则允许其加入这个tunnel。VTEP1会生成一个新的Vxlan头部，修改它的源IP地址和目的IP地址，并设置VNI域和其他参数，再重新封装原始数据包，并通过广播的方法发送给所有其他VTEPs。其他的VTEP会收到这个包，并加入相应的Vxlan tunnel。至此，新的Vxlan tunnel已经建立完成。

Vxlan tunnels也可以由多个VTEP节点实现。如果要创建一个由两个VTEPs实现的Vxlan tunnel，首先VTEP1会发送一个封装了VXLAN头部的原始数据包给VTEP2。VTEP2收到这个包后，生成一个新的Vxlan头部，修改它的源IP地址和目的IP地址，并设置VNI域和其他参数，再重新封装原始数据包，并通过源Mac地址发送给VTEP1。接下来，VTEP1将这个包转发给VTEP3。VTEP3接受到这个包后，检查自己的VNI域是否是这个Vxlan tunnel的一部分，如果不是，则会拒绝这个包，并要求它转发给其它VTEP。否则，VTEP3会将这个包加入对应的Vxlan tunnel。至此，新的Vxlan tunnel也已经建立完成。

Vxlan tunnels也可以由边缘路由器实现。如果要创建一个由边缘路由器实现的Vxlan tunnel，首先边缘路由器必须知道所有的VTEPs。然后，边缘路由器会在本地创建新的Vxlan tunnel，与所有的VTEPs实现VTEP mesh。也就是说，每个边缘路由器都可以与其他的边缘路由器实现Vxlan tunnel，形成一个分布式的VTEP mesh。通过这种方式，边缘路由器就能够将网络流量通过VXLAN tunnel转发到其他数据中心。

### 加入VXLAN隧道
VXLAN tunnels的参与者必须知道如何加入Vxlan tunnels。当一个VTEP想要加入一个Vxlan tunnel时，他需要发送一个封装了VXLAN GPE头部的原始数据包给任意的VTEP。如果这个VTEP是个新的VTEP，则会生成一个新的Vxlan tunnel，然后尝试加入它。如果这个VTEP已经加入了一个Vxlan tunnel，他会忽略这个请求。如果这个VTEP属于已知的Vxlan tunnel，则会接受这个VTEP，并尝试加入这个Vxlan tunnel。加入成功后，VTEP1就能够收发Vxlan tunnel中的数据包。

Vxlan tunnels的每个节点都可以主动加入一个Vxlan tunnel。如果VTEP1想要加入一个Vxlan tunnel，他会发送一个封装了VXLAN GPE头部的原始数据包给任意的VTEP。这个包的内容是指向它自己的数据包，其目的地为一个特殊的目的Mac地址。当收到这个包的时候，VTEP2判断它是否是属于已知的Vxlan tunnel的一部分，如果不是，则不会加入这个Vxlan tunnel。如果是，则会接受这个VTEP，并且加入Vxlan tunnel。至此，新的VTEP就加入到了Vxlan tunnel中。
### 转发Vxlan tunnel中的数据包
在Vxlan tunnel中，数据包被封装在IP数据报中，并通过广播方式传播到整个网络。当一个数据包被接收到，VTEP会解析这个IP数据报，并取出里面的VNI域。然后，VTEP会查找相应的Vxlan tunnel，并将数据包传递给对应的VTEPs。VTEPs会解析数据的包，并判断是否为一条能够正常处理的数据包。如果是，则会根据之前的封装协议重新封装数据包，并将其发送给下一跳。如果不是，则会丢弃这个包。

### 删除Vxlan tunnel
当一个Vxlan tunnel没有参与者的时候，它就会变空闲状态。如果希望删除一个空闲的Vxlan tunnel，需要让其中一个VTEP发送一个封装了VXLAN尾部的原始数据包给另外一个VTEP。这个包会告诉另一个VTEP，它不再需要这个Vxlan tunnel了。然后另一个VTEP会删除相应的Vxlan tunnel，并释放相应的资源。

# 4.具体代码实例和解释说明
## 安装VXLAN
由于Open vSwitch支持Vxlan功能，因此在安装Open vSwitch时就可以集成Vxlan支持。

sudo apt-get update
sudo apt-get install openvswitch-switch dpdk dpdk-dev librdmacm-dev

创建配置文件/etc/openvswitch/conf.db，添加如下配置：

ovs-vsctl set Open_vSwitch. other_config:dpdk-init=true
ovs-vsctl set Open_vSwitch. other_config:dpdk-lcore-mask=0x7

## 配置VTEPs
VTEP是指一个可以创建或加入Vxlan tunnel的实体。每个VTEP都有一个Vxlan ID，用作标识符。如下命令为VTEP1分配Vxlan ID为1000：

ovs-vsctl add-port br0 vxlan1 -- set interface vxlan1 type=vxlan options:remote_ip="172.20.0.2" options:key=flow options:dst_port=4789 options:local_ip="172.20.0.1" options:vni=1000

VTEP1表示名称为vxlan1的端口，type设置为vxlan，并且设置了一些选项。remote_ip表示远端VTEP的IP地址，key表示所使用的认证密钥，dst_port表示VTEP与其它VTEP建立隧道时的端口号，local_ip表示本地VTEP的IP地址，vni表示VNI域值。

## 流量验证
配置好VTEPs后，我们可以使用ping命令测试Vxlan tunnel的连通性。我们可以分别在两个数据中心中创建两个容器，然后在两个容器之间使用ping命令进行测试。如下命令创建两个容器，每个容器在不同数据中心中：

docker run -dti --name c1 --net=none busybox sleep infinity
docker exec -it c1 ifconfig eth0 10.10.1.1/24 up

docker run -dti --name c2 --net=none busybox sleep infinity
docker exec -it c2 ifconfig eth0 10.10.2.1/24 up

最后，我们可以使用ping命令测试两个容器的连通性：

docker exec -it c1 ping 10.10.2.1 -c 5