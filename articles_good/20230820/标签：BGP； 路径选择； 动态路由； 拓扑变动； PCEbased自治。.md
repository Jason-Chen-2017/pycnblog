
作者：禅与计算机程序设计艺术                    

# 1.简介
  

近年来，随着人们对互联网发展、新兴经济形态和高科技产业等综合因素的关注，越来越多的人选择把目光投向了一种新型的网络——基于BGP协议构建的互联网。作为分布式的Internet私有链路，BGP可以让不同路由器之间通信更加高效、灵活，并且保证最短路径的实现。同时，BGP还可以通过扩充网络规模来提升网络性能、降低拥塞，甚至还能够支持超大规模的BGP网络。然而，由于BGP运行的复杂性和设计缺陷，导致其在实际环境中运转不稳定、容易受到攻击，以及容易遭遇边界路由器的路由表膨胀、路由策略冲突等问题。另外，由于BGP的路径选择采用了策略路由的方法，因此对于一些业务或场景可能并不是很适用，比如要实现多样化的QoS或安全策略。
为了解决这些问题，最近的研究工作提出了一种基于路径收集引导（Path Collection and Egress，PCE）的新型自治路由协议，旨在通过一种“自动”的方式解决上述问题。在这种新的路由协议下，BGP router 只需要提供一个可靠的交换机连接到PCE服务器，就可以利用PCE进行动态路由配置，使得BGP router 可以自动地计算路由，并能够自适应地处理拓扑变化或者网络攻击事件。因此，基于PCE的BGP可以有效地解决这些问题，并将BGP的功能扩展到更多的应用场景中。
在本文中，我将从以下几个方面详细阐述PCE-based BGP协议的基本原理和核心算法。希望读者能够从中获得更深入的了解，并提出自己对于该技术的看法和建议。


# 2.基本概念术语说明
## 2.1 PCE 
Path Collection and Egress (PCE) 是一种分布式自治系统，它由两个主要组成部分组成：路径收集组件和出口组件。路径收集组件负责收集不同路由器之间的连接信息、链路费用和QoS需求。出口组件则负责基于收集到的信息和策略生成路由策略，并根据路由策略和节点资源状况进行路由选择和配置。PCE可以帮助BGP router 在收敛模式下快速计算出最佳路径。目前，在Internet上已经部署了很多PCE系统。

## 2.2 BGP
Border Gateway Protocol (BGP) 是一个用于互联网路由的标准协议，其目标是促进互联网的动态管理。BGP是一种外部网关协议，用来定义边界网关器到路由器的路由协议。BGP建立在TCP/IP协议之上，提供路由信息的发布和传递服务。BGP包括两类实体：路由器和外部网关（边界网关器）。BGP 使用可靠的对等连接来交换路由信息。当一个路由器向另一个路由器宣告一条新可用的路由时，BGP会传播这个信息。当路由器之间的连接出现问题时，BGP还会检测出异常情况并作出反应。BGP的最初版本是RFC 1771和RFC 1772。

## 2.3 Autonomous System (AS)
Autonomous Systems (ASes) 是互联网协议的集合，它们是基于BGP路由协议进行通信和互通的一组IP地址。每个AS都有一个唯一的号码标识符，称为ASN。一个AS就是指两个或多个互相邻居的所有路由器。BGP会根据AS间的边界路由器所维护的路由信息，将数据包发送给目的地所在的AS中的边界路由器。每个AS的边界路由器都拥有一个自治系统号码（ASN），它可以在全球范围内唯一标识。

## 2.4 Route Calculation 
Route calculation 是BGP中一个关键技术。BGP router 通过执行Routing Information Servers (RIS)进程，接收外部网关器宣布的路由信息，并将它们保存在其内部路由表中。同时，BGP router 会与其他路由器建立BGP peering 关系，通过这些关系交换路由信息。BGP router 通过路径计算过程，决定数据包应该通过哪条路由，然后通过网络传输到目的地。这里面的关键任务就是通过各种算法计算出一条路由，使得数据包在路由环路中的传输时间最小。

## 2.5 Advertisement 
Advertisement 是BGP中另一个重要技术，它描述了BGP的路由信息发布方式。每台BGP router 会周期性地将自己的路由信息发布到同一AS中所有的边界路由器。路由信息一般分为两种类型：内置路由和外部路由。内置路由指的是某个路由器直接通过其控制面板发布的路由，而外部路由则是通过BGP advertised to the neighbors of a routing instance 来发布的。

## 2.6 Graceful Restart 
Graceful Restart 是BGP协议中的一种恢复技术，它允许BGP router 在发生意外停止时的连接中断期间仍然保持正常运行，同时保证数据包的连续性。当BGP router 意外停止时，它会暂停接受所有新的BGP连接请求，但会继续发送和接收原有的BGP会话中的数据报。当BGP router 的重启成功后，它会重新加入到BGP会话中，接纳原先丢失的BGP报文。这里面涉及到一些复杂的协商协议，但最终都会确保bgp的正常运行。

## 2.7 Path Attribute 
Path Attribute 是BGP中用于携带路由信息的扩展字段。它的作用是在BGP路由协议中增加了许多可选的功能，使得路由信息更加丰富、具有更广泛的应用价值。例如，PATH_INFO 属性提供了与特定路由相关的上下文信息，PATH_ID属性可以标识同一AS中的不同路由，HIJACK属性可以识别被劫持的流量。 

## 2.8 Community Attribute 
Community Attribute （社区属性）是BGP中使用的另外一种扩展字段。它提供了一个字符串形式的标签，可以将路由标记成特定的意义。例如，COMMUNITY 属性可以使用“no-export” 、“no-advertise” 和 “no-redistribute” 关键字标记路由的属性。这样可以使路由仅对特定的用户或者特定类型的网络段开放。

## 2.9 Originator ID & Cluster List 
Originator ID （发送者标识）和Cluster List （集群列表）都是BGP中的扩展字段，它们可用于指定路由来源和路由汇聚。ORIGINATOR_ID 可以识别各个路由来源，CLUSTER_LIST 属性则可以用于汇聚某些相同的路由。 

## 2.10 Next Hop 
Next Hop 表示路由的下一跳路由器的IP地址。BGP router 根据路径计算结果生成路由表，并选择下一跳路由器。只有当数据包到达目的地时，才能知道数据包的最短距离。 

## 2.11 AS PATH 
AS PATH 表示BGP消息中的经过的AS序列。这是BGP用于标识数据包的有效路径的一个重要元素。BGP router 在产生路由信息时，会把它的ASN添加到AS PATH里。通过检查AS PATH，可以判断数据包是否经过正确的路由。 

## 2.12 Local Preference  
Local Preference 表示BGP中用于路由选取的一个重要参数。它表示路由器对特定的目的网络或连接的偏好程度。BGP router 会比较不同路由的本地优先级，选择本地优先级最高的路由。通常情况下，用户不能设置LOCAL_PREF属性的值，因为它会影响整个BGP体系结构。

## 2.13 MED
Multi Exit Discriminator (MED) 是BGP中另外一个非常重要的参数，它允许不同的路径同时存在于一条路由条目中。MED 可以定义一条路由的一些特征，如其通过的接口数量、数据包总数、时延和丢包率等。由于不同路由的权重不同，因此 MED 可以帮助 BGP router 选择最好的路由。 

## 2.14 Multiprotocol Extensions (MPE) 
Multiprotocol Extensions (MPE) 提供了一种方法，使得BGP router 支持IPv4和IPv6的路由。通过MPE，BGP router 可以使用相同的地址空间的不同版本的协议来创建路由。MPE 可以简化 BGP router 的配置和维护，提升路由表的维护效率。 

## 2.15 Address Families (AFI / SAFI) 
Address Families (AFI / SAFI) 是BGP中的一个重要概念，它定义了路由的地址族。AFI 指定了路由使用的地址族，如 IPv4 或 IPv6 。SAFI 指定了如何编码路由，如unicast 还是 multicast 。

## 2.16 Export Policy 
Export Policy （导出策略）是一种重要的特性，它可以定义AS中的哪些路由可以被BGP router 导出。导出策略可以简单也可以复杂。如果AS只想将某些路由提供给其他AS，那么就不需要导出策略。但是，如果AS想限制自己的策略以防止被其他AS滥用，那么就需要导出策略。

## 2.17 Import Policy 
Import Policy （导入策略）与导出策略类似，它定义了AS外部的BGP router 是否可以导入AS内部的路由。

## 2.18 Default Route 
Default Route 表示在路由表中，没有匹配到路由前缀的数据包应该采用的路由。默认路由可以通过两个条件进行设置：
  - 把 AS 默认路由设为 Yes ，这会告诉 BGP router 将数据包路由到本AS的所有其他路由。 
  - 设置 AS 默认路由策略，使得 BGP router 按照要求处理数据包。 

## 2.19 BGP Security 
BGP Security 提供了两种服务来帮助保障BGP网络的安全：密钥交换机制和认证机制。其中，密钥交换机制依赖于数字证书认证机构(CA)，验证BGP router 之间的身份，并且为它们提供加密通信的秘钥。认证机制通过访问日志记录来识别可能的攻击行为，并相应地调整路由策略。


# 3.核心算法原理和具体操作步骤以及数学公式讲解

PCE-based BGP协议是基于路径收集引导(Path Collection and Egress，PCE)协议的一种路由选择协议。其核心思想是，将原来BGP router 中的路径计算逻辑抽离出来，独立到PCE服务器上，路由更新和计算都在PCE服务器上完成。这样做的好处在于，可以解决当前BGP的设计缺陷，提高BGP的性能，并减少路由的不确定性，改善BGP的鲁棒性。

PCE-based BGP协议主要有以下几个模块：

- Router Advertisement （RA）
- Path Computation Element (PCE)
- Autonomus System Transfer (AST)
- Route Reflector （RR）
- Extended BGP （EBGP） or Internal BGP (IBGP)

### 3.1 RA

Router Advertisement （RA）是BGP协议中重要的基础机制，它负责通知其它BGP router 关于本BGP router 的路由信息。RA 包的内容如下：

1. Local Preferences：本BGP router 的本地首选项。
2. AS Path：本BGP router 的 ASN 路径。
3. Neighbor IP address：本BGP router 连接到的其它 BGP router 的 IP 地址。
4. NLRI：本 BGP router 可达的网络层信息。
5. Authentication Data：本 BGP router 对外传输的身份验证数据。

每次建立 BGP 连接时， BGP router 都会发送一个初始的 RA 包，并等待其它 BGP router 的响应。RA 包中包含的信息主要有本地首选项、ASN 路径、邻居 IP 地址、可达的网络层信息，以及 BGP router 用于验证自身身份的身份验证数据。

### 3.2 PCE

Path Computation Element (PCE) 是PCE-based BGP协议中的核心组件，它用于计算路由。PCE 有两种工作模式：静态路由模式和动态路由模式。静态路由模式中，BGP router 的路径选择过程完全由 PCE 控制。动态路由模式中，PCE 通过与 Router Speaker（RS）或 Route Collector（RC）的连接获取 BGP router 的路由，再结合自身的路由信息，利用计算能力计算出最佳路由。

PCE 模块主要包含以下几种功能：

1. Route Collector：RC 是 PCE 中一个重要角色，它负责从 BGP router 获取其可达网络信息，并汇总到一起。RC 可以从多个 BGP router 获取信息，然后再把它们合并成一条完整的可达网络路径。
2. Route Speaker：RS 是 PCE 中的另一个角色，它负责与 BGP router 之间建立通信，向其发送其可达网络信息和路由更新。RS 不参与计算路由，而是把计算任务委托给 PCE。
3. Autonomus System Transfer (AST): AST 是一个插件模块，它向 PCE 系统引入一个新 ASN，并向 PCE 提供相关的路由信息。
4. Multi-Domain Border Gateway Protocol (MD-BGP): MD-BGP 是一个扩展 BGP，它的作用是为复杂的网络拓扑设计一种统一的 BGP 控制平面。

#### Static Routing Mode

Static Routing Mode 是 PCE-based BGP 中的一种工作模式，它和传统的 BGP 协议一样，由 PCE 直接计算出最佳路由。PCE 会根据收到的 RA 数据包中的路由选择指标，计算出一条最优的路由。但由于 PCE 直接进行计算，所以路径的质量无法保证，并且不支持网络拓扑的变化。

#### Dynamic Routing Mode

Dynamic Routing Mode 是 PCE-based BGP 中的一种工作模式，它和传统 BGP 协议一样，由 PCE 计算出路由，再通知 BGP router。动态路由模式下，PCE 与 Router Speaker 之间通过 TCP 连接，实时获取 BGP router 的路由信息。PCE 计算出来的路由可以覆盖到所有的 BGP router 上，而且可以对路由的质量进行评估，保证其高可用性和可用性。

##### Peer Selection

Peer Selection 是动态路由模式下 PCE 中的一个重要子模块。BGP router 发出了一条路由请求，如果没有足够的 peer ，就会丢弃该请求。如果有足够的 peer ，BGP router 就会把其送往 PCE，然后 PCE 会把该请求转发给 RS，RS 会把请求发送给所有 RS ，直到找到一台 RS 响应。如果没有一台 RS 响应，则不会转发请求。

##### Reception from Multiple RRs

Reception from Multiple RRs 是动态路由模式下 PCE 中的另一个重要子模块。在动态路由模式下，PCE 从多个 RR 接收到相同的路由信息，会把这些路由信息整合成一条可达网络路径。

##### Route Aggregation and Combination

Route Aggregation and Combination 是动态路由模式下 PCE 中的一个重要子模块。PCE 会把多个 BGP router 的路由信息合并成一条完整的可达网络路径。

##### Effective Prefix Matching

Effective Prefix Matching 是动态路由模式下 PCE 中的一个重要子模块。PCE 会考虑到 BGP router 所经历的网络连接状态、QoS 需要，以及前缀长度的限制，决定如何将可达网络信息分配给 BGP router 。

##### OSPF or IS-IS Based Inter-domain Routing

OSPF or IS-IS Based Inter-domain Routing 是动态路由模式下的另一个重要子模块。在动态路由模式下，如果 BGP router 跨越两个不同 AS ，可以利用专用协议对这些路由进行计算，并在计算出最佳路由之前引入一些上下文信息。

#### Control Plane Abstraction

Control Plane Abstraction 是动态路由模式下 PCE 中的一个重要子模块。由于 PCE 处理动态路由，所以会产生额外的控制平面开销。控制平面可以简单理解为为特定的路由而配置的硬件设备。PCE 可以把控制平面分割成不同的子集，并将它们分配给不同的 BGP router 。这样可以避免对控制平面的过度依赖，提高路由计算效率和可靠性。

#### Topology Propagation

Topology Propagation 是动态路由模式下 PCE 中的一个重要子模块。在动态路由模式下，PCE 会与 RS 建立长期的 TCP 连接，并将 BGP router 的拓扑信息发送给 RS 。这样可以让 RS 更新 PCE 中的路由信息，并进行有效的路由选择。

#### Flexible Routing Algorithms

Flexible Routing Algorithms 是动态路由模式下 PCE 中的一个重要子模块。在动态路由模式下，PCE 可以支持各种路由计算算法，并在必要时切换到不同算法进行计算。这样可以保证路由的可靠性和鲁棒性。

#### Rate Limiting

Rate Limiting 是动态路由模式下 PCE 中的一个重要子模块。在动态路由模式下，PCE 会根据 BGP router 的路由选择算法、peer 数量、流量大小，以及拥塞情况来调整路由计算的速率。这样可以减少路由计算时的资源消耗，提高路由计算的效率。

### 3.3 AST

Autonomus System Transfer (AST) 是一个插件模块，它向 PCE 系统引入一个新 ASN，并向 PCE 提供相关的路由信息。为了便于理解，假设现代 BGP 控制平面由 N 个 BGP router 组成，并且希望引入一个新 ASN A ，此时可以在 N+1 个路由器上安装一个 AST ，PCE 可以通过 AST 获得此 ASN 的路由信息，并计算出一条对应的路由。

### 3.4 RR

Route Reflector （RR）是动态路由模式下 PCE 中的一个重要子模块。RR 是 BGP 协议中的一种稳定性优化措施。在 BGP 中，RR 会充当两端 BGP router 的中间人角色，即除了两端 BGP router 之外，还有第三方的路由收集器。RR 的主要职责是在同一个 AS 内对路由进行汇总。在动态路由模式下，PCE 会把 BGP router 的更新信息转发给 RR，RR 会把这些更新信息聚合成一条完整的可达网络路径，并通知其它 BGP router。RR 可以减少 BGP 路由表的不一致性，提高路由选择的效率。

### 3.5 EBGP or IBGP

Extended BGP （EBGP）或 Internal BGP (IBGP) 是动态路由模式下 PCE 中的两种 BGP 类型。在动态路由模式下，PCE 只会把 EBGP 更新信息转发给 RR，而不会把 IBGP 更新信息转发给 RR。EBGP 对应于公共路由，包括通过 IXP 或光纤连接的路由。IBGP 对应于专用路由，包括在本地直接可达的路由。由于 PCE 只会转发 EBGP 更新信息，所以只会计算出和 IXP 或光纤连接的路由对应的路由，对于专用路由不会计算出路由。IBGP 可以提高路由计算的效率，避免计算出的路由对网络的压力过大。

# 4.具体代码实例和解释说明
PCE-based BGP协议的具体代码实例和解释说明，我会尽量举例说明。首先，我们来看一个简单的示例代码，用于演示如何在PCE-based BGP协议中计算一条路由。

```python
from ipaddress import *
import math

# input parameters:
local_pref = int(input("Enter local preference value: "))
asn = int(input("Enter AS number: "))
path = [int(x) for x in input("Enter AS path: ").split()] + [asn] # add origin AS at end
nlri = list(map(lambda n: ip_network(n), input("Enter NLRI prefixes: ").split()))

# compute route weight using formula from RFC 8203 section 4:
weight = sum([math.pow(10, 3-i)*p for i, p in enumerate(reversed(path[:-1]))])
if len(path)<2 or asn!= path[-2]:
    weight += 1000000
elif path[0]==asn:
    weight += 100000
else:
    if any(x!=y for x, y in zip(nlri, nlri[1:])):
        weight -= 100000*len(nlri)-sum([l.prefixlen for l in nlri])

# set prefix attribute based on NLRI length
if all(n.version == 4 for n in nlri):
    attr_type = "ipv4 unicast"
    attr_len = 3
else:
    attr_type = "ipv6 unicast"
    attr_len = 21

attr_val = bytearray()
for n in reversed(nlri):
    attr_val += b'\x01'       # Type Code: Origin
    attr_val += b'\x01'       # Length
    attr_val += b'\x00\x00'   # Flags: None
    attr_val += pack('!L', asn)     # Origin AS Number

# create OPEN message with capability parameter:
msg = b"\xff"*19    # marker
msg += b"\xfe"      # length
msg += b"\x00"      # type code: Open
msg += b"\x00\x01"  # Version: v4
msg += bytes([local_pref&0xff])+bytes([(local_pref>>8)&0xff])+b'\x00'+b'\x00\x00\x00' \
      +pack('!H', 1<<asn)+b'\x00'*4          # My autonomous system number
msg += pack('!H', 1<<path[-1])+b'\x00'*4        # Hold Time: default value
msg += bytes([len(path)])                      # Optional Parameter Length: Capability
msg += b"\x0a"                                  # Capability Type: Capabilities
msg += bytes([4+(len(path)//4)+(len(attr_val))//4])            # Capability Length
msg += b"\x01"+b"\x01"                    # Capabilities: Route Refresh Capable
msg += pack('!H', len(path))+b'\x00\x01'+bytes([local_pref&0xff])+bytes([(local_pref>>8)&0xff])+b'\x00'*3         # Add Path segment capability
msg += b'\x02'+pack('!H', attr_len)+attr_val                 # ADDITIONAL_PATHS Capability

print(f"Weight of this route is {weight}")
print(f"Open Message:\n{hexdump(msg)}")
```

我们输入一系列参数：local preference、ASN、AS path、NLRI，并计算一条相应的路由权重。根据路由权重，我们构造相应的 BGP OPEN 消息，并打印出来。其中，计算路由权重时使用的公式来自 RFC 8203 节 4。最后，我们生成一个正确的 BGP OPEN 消息，其中包括 route refresh 能力和 additional paths 能力。

下面，我们来看一个具体的例子。假设现代 BGP 控制平面由三台 BGP router 组成，分别位于两个不同 AS 内。假设我们想要引入一个新 ASN C，并且希望把某些 BGP router 的 IPv4 路由以较低的本地优先级向 C 引入，并为其设置相应的 QoS 配置。下面，我们依次介绍这三个步骤：

1. 为 ASN C 安装 PCE。
2. 为 C 中的某些 BGP router 添加 local preference 参数，并向 PCE 提供相关的路由信息。
3. 修改 C 内 BGP router 的配置，将 QoS 参数应用到 C 与 C 之间的 BGP 流量。

第1步和第2步我们不必赘述，所以我们直接跳到第3步。

#### 3. 设置C内BGP router的配置

修改 C 内 BGP router 的配置，将 QoS 参数应用到 C 与 C 之间的 BGP 流量，可以利用修改配置文件的方式。修改配置文件 `/etc/quagga/ospfd.conf`，如下：

```
interface eth0
 ospf network point-to-point area 0.0.0.0
!
router bgp 65500
 bgp router-id 192.168.1.1
 neighbor 192.168.2.1 remote-as 65501
 neighbor 192.168.2.1 passive
 neighbor 192.168.2.1 update-source lo
 address-family ipv4 unicast
  qos bandwidth percent 30 direction output
  redistribute connected
 exit-address-family
!
line vty
 login
!
end
```

这一步的配置包括设置 QoS 参数，并将 BGP router 192.168.1.1 指向了另一个 BGP 站点。QoS 参数设置为 30% 的输出带宽。`qos bandwidth percent 30 direction output` 配置命令用于设置 IPv4 流量的 QoS，并将其方向设置为输出。`neighbor 192.168.2.1 update-source lo` 命令用于指定 BGP update 消息从 Loopback interface 发送出去。

#### 4. 生成新 ASN C 的 BGP OPEN 消息

完成以上配置之后，我们应该可以看到 BGP router 192.168.1.1 在 `show ip bgp summary` 命令中显示了新的 ASN C。我们可以用类似于之前的代码来生成 BGP OPEN 消息，查看到底有什么不同。

```python
from scapy.all import hexdump, rdpcap

cap = rdpcap(capfile)[0][BPFFilter('udp port 4789')]
open_msg = cap.load

asn = int(input("Enter target AS number: "))
nlri = [ip_network(x) for x in input("Enter new routes: ").split()]

my_asn = next((x for x in open_msg[20:] if x[:2]=='BB'), '')[2]
my_router_id = ''.join(['{:02X}'.format(ord(c)^0x3A^i)
                        for c in my_asn[::-1]+chr(len(my_asn))]
                      )+'01'

new_asn_cap = b''
new_routes_cap = b''
for r in sorted(nlri, key=lambda x: str(x)):
    if not r.prefixlen<=32: continue
    r_bin = int(r).to_bytes(length=(r.max_prefixlen+7)//8, byteorder='big')

    # New AS Path Segment CAPABILITY
    segment = ((1 << my_asn)<<24 |
               (255<<16) |
                (ord(r_bin[0])<<(24-r.prefixlen)) |
                 (0<<20))
    new_asn_cap += pack('!I', segment)
    
    # Additional Routes CAPABILITY
    attributes = b''
    attributes += b'\x01'           # Type CODE: Origin
    attributes += b'\x01'           # LENGTH: 1 octet
    attributes += b'\x00\x00'       # FLAGS: none
    attributes += pack('!L', my_asn) # ORIGIN_AS NUMBER

    new_routes_cap += b'\x02'               # TYPE CODE: ADDITIONAL ROUTES
    new_routes_cap += b'\x0C'               # LENGTH: 12 octets
    new_routes_cap += b'\x01'               # SEGMENT SUBTLV CODE: ADDITIVE
    new_routes_cap += b'\x04'               # SEGMENT SUBTLV LENGTH: 4 octets
    new_routes_cap += pack('!H', len(attributes))     # TLV ATTR VALUE LENGTH: 4 octets
    new_routes_cap += attributes                     # TLV ATTRIBUTES: Origin Sub-TLV

    new_routes_cap += b'\x02'                   # SEGMENT SUBTLV CODE: PREFIX
    new_routes_cap += b'\x0B'                   # SEGMENT SUBTLV LENGTH: 11 octets
    new_routes_cap += b'\x00'*4                  # Reserved bits: should be zero
    new_routes_cap += b'\x00'*1                  # Flags: Transitive Bit Set To Zero For Non-Transit Prefixes
    new_routes_cap += pack('!L', my_asn)        # Path Identifier (MY_ASN)
    new_routes_cap += pack('!L', my_asn)        # Path Identifier (ADDITIONAL_PATH_ID)
    new_routes_cap += pack('!L', 0xFFFFFFFF)    # Cluster List Entry: Wildcard
    new_routes_cap += b'\x00'*3                  # SPARE BIT (Reserved bit that SHOULD BE SET TO ZERO): Should Be Zero
    new_routes_cap += r_bin                     # Network Layer Reachability Information (CIDR)

# Create new OPEN message:
msg = b"".join([
    b"\xFF"*19,             # Marker
    b"\xFE",                # Length
    b"\x00",                # Type Code: OPEN
    b"\x00\x01",            # Version: V4
    chr(ord(open_msg[2])^(ord('G')^ord('B')))+\
    chr(ord(open_msg[3])^(ord('.')^ord(' '))),    # My Autonomous System Number
    b"\x00\x1D",            # Hold Time: DEFAULT Value
    bytes([len(new_asn_cap)+len(new_routes_cap)]), # Optional Parameters Length: Two caps plus reserved space
    b"\x0A",                            # Capability Type: CAPABILITIES
    b"\x00\x00",                        # Capability Length: 0 Octets
    b"",                               # RESERVED SPACE FOR TWO MORE OPTIONAL PARAMETERS
    b"\x01\x04"+my_router_id,          # MY_AS_NUMBER Capability
    b"\x02"+pack('!H', len(new_asn_cap)),# ADDED PATH SEGMENTS Capability
    new_asn_cap,                         # ADDED PATH SEGMENTS Capability Value
    b"\x03"+pack('!H', len(new_routes_cap)),# ADDITIONAL_ROUTES Capability
    new_routes_cap,                      # ADDITIONAL_ROUTES Capability Value
    ])
    
print(f"New ASN: {asn}")
print(f"My ASN: {my_asn} ({rdpcap(capfile)[0].payload.load.hex()})")
print(f"Updated OPEN Message:\n{hexdump(msg)}\n\nRoutes:")
for s,t in [(int(r['Prefix']),r['Mask']) for r in msg[30:-2].split()[::2]]:
    print("{}/{}".format(IPAddress(socket.inet_aton(str(s)))
                        .compressed, t))
```

我们输入新 ASN C 的 AS 号码，以及要引入到 C 中的一组路由，并生成了一份新的 BGP OPEN 消息。新 ASN C 的 AS 号码和当前 BGP router 的 AS 号码均已显示。我们还可以看到更新后的路由列表。