                 

# 1.背景介绍


## SDN简介
随着互联网业务的发展，网络设备日渐增多、规模化、复杂化，传统网络架构面临着越来越多的问题，例如网络拥塞、高延迟、安全问题等，而新一代的网络架构则通过采用软件定义的网络（SDN）技术来解决这些问题。

SDN（Software Defined Networking）即软件定义的网络，它允许网络管理员用编程的方式来配置网络设备。目前已有多个供应商推出了基于OpenFlow协议的SDN控制器，使得网络管理员可以灵活地管理网络资源和实现各种功能。通过SDN，网络可以自动地在可靠性、可用性、可扩展性、性能、节能效益、安全性等方面进行优化。

基于SDN的网络架构主要包括控制器、交换机、链路聚合器、访问控制和QoS模块等，如下图所示：

网络控制器负责网络拓扑的自动生成、流表的配置、QoS策略的实现、计费及其他网络管理功能的协调。交换机上运行SDN控制器，并与其它交换机相连接，从而形成一个完整的网络。控制器根据网络的需求生成路由表、防火墙规则、负载均衡算法等信息，并通过与其它模块的交互，将其转发至相应的交换机上。链路聚合器负责将多个物理端口连接到一个逻辑端口上，提高网络的带宽利用率。QoS模块根据业务的优先级对数据包进行分类，设置不同的队列长度、传输速率等限制条件。

## 为什么要学习SDN
网络工程师往往都没有接触过SDN技术，或者只知道一些相关名词如控制器、交换机等概念，但很少有人能够全面、系统地理解SDN架构、流程及其工作原理，因此如果想更好地应用SDN技术来提升网络的可靠性、可用性、可扩展性、性能、节能、安全等方面的性能，必须深入地了解SDN的工作原理、机制及其各个组件的功能和作用。掌握SDN后，才能更有效地运用其在实际网络中的优势。

## 如何学好SDN
本教程的内容适用于具有一定网络知识的人员，希望能够帮助读者快速了解SDN并学会应用其优势。如果你是刚开始接触SDN，可以先简单阅读一下下面的相关资料，然后再参阅本教程进行学习。
### 技术文档
1. Open vSwitch官方文档：https://docs.openvswitch.org/en/latest/intro/
2. OVSDB协议参考手册：https://tools.ietf.org/html/rfc7047
3. IEEE 802.1X协议参考手册：https://standards.ieee.org/findstds/standard/802.1X-2010.html
4. Ryu控制器开发指南：http://osrg.github.io/ryu-book/zh_cn/html/index.html
5. NFV综述：https://www.redhat.com/zh/topics/nfv
6. Data Center Networking Guide：https://wiki.dcloud.io/datacenter/network_overview/overview/
7. 5G云计算网络白皮书：https://arxiv.org/pdf/2111.02811.pdf
8. VMware NSX技术文档：https://docs.vmware.com/cn/VMware-NSX-T/3.1/com.vmware.nsxt.310.doc/GUID-C2CBEA65-EDDC-4C8B-AFD3-66FBA9DCA759.html?hWord=N4IghgNiBcIBIGYAGAUKGnSAOwFdAbkAzsgJYBuIA8gMYkA6ATwAdAYxTo0AKAcjAmYgGjKNJALwwz8RgDMIAJklFgAeMAFkRigLZou+AJhqQA
9. Calico项目官网：https://projectcalico.docs.tigera.io/about/about-calico
10. VXLAN和GRE协议介绍：https://segmentfault.com/a/1190000040447309