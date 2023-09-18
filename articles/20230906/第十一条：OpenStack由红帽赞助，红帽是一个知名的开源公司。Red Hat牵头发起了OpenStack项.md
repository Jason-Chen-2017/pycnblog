
作者：禅与计算机程序设计艺术                    

# 1.简介
  

OpenStack（Open Source Stack）是一个开源的云计算平台即服务软件，它提供了一整套基础设施管理功能，包括计算、网络、存储、数据库、消息队列、中间件等，旨在通过软件定义的云环境提供低延时、高可靠的资源共享服务。OpenStack最初起源于 Rackspace Private Cloud 项目。OpenStack的主要项目有Nova、Neutron、Swift、Keystone、Glance、Cinder、Horizon、Trove、Ceilometer、Ironic、Barbican、Octavia、Murano、Solum等。
Red Hat基于其专有的企业级私有云应用场景开发的OpenStack，已在全球范围内被广泛采用。Red Hat公司致力于开发世界领先的开源产品，并将开源产品贡献给全社区用户。此外，红帽集团旗下红帽企业linux（RHEL）、红帽OpenShift以及红帽APM解决方案也都基于OpenStack提供商业级服务。
2.基本概念与术语
云计算（Cloud Computing）是一种按需提供计算资源的网络基础设施的方式，称为“按需服务”或“按量付费”。云计算平台服务包括计算机硬件、服务器系统、网络连接、软件应用程序、数据存储等，这些资源可根据实际需求而快速布置、启动和停止，灵活伸缩、按需计费，是构建新的一代网络应用的基础。云计算平台可以提供多种类型的服务，包括数据中心虚拟化、弹性计算、软件即服务、平台即服务、网络即服务、存储即服务、数据库即服务等。
OpenStack属于云计算平台服务软件的一部分。它提供一系列框架和工具，让不同的云服务商、供应商和消费者能方便地将自己的服务提供到公有云或者私有云中，从而实现多租户架构。OpenStack通常使用RESTful API接口和Python编程语言进行编程，并使用Apache许可证授权。
OpenStack中的术语及概念包括：
项目（Project）：一个项目是一个逻辑上的容器，它包含一个或多个用户和一组权限。项目可以用来管理权限和资源的分配，并通过分配的角色来限制用户对项目资源的访问。
节点（Node）：节点是指执行OpenStack组件的物理主机或虚拟机。
实例（Instance）：一个虚拟机就是一个实例，它可以启动、停止、暂停、重启、迁移、删除。实例具有生命周期，在创建后便进入运行状态。实例由虚拟磁盘、网络接口卡、镜像和配置信息构成。
映像（Image）：存储在Glance的映像文件是一个通用模板，用于创建新虚拟机。映像文件可以包含操作系统、软件、数据文件等。
Flavor（Flavor）：一个云服务器的性能规格，包括处理器类型、内存大小、磁盘大小、网络带宽等属性。
虚拟网络（Virtual Network）：OpenStack支持不同虚拟网络技术，如VLAN、VXLAN、GRE和 Geneve等。每一个虚拟网络都有一个唯一标识符UUID，该标识符可以在实例间共享。
安全组（Security Group）：安全组是一种网络访问控制列表，它允许或拒绝对指定端口的访问。
密钥对（Key Pairs）：密钥对是一对匹配的公钥和私钥，用于SSH登录和加密通信。密钥对保存于Keystone中，用于身份验证和授权。
角色（Role）：角色定义了一组权限，可以赋予用户以完成特定的工作任务。
区域（Zone）：区域是OpenStack集群的逻辑划分，一个区域可以包含多个节点。区域通常对应于具体的城市或数据中心。
外部网络（External Network）：外部网络代表公网互联网，所有外部客户端都可以访问该网络。
内部网络（Internal Network）：内部网络代表专用网络，仅限内部主机之间访问。
负载均衡器（Load Balancer）：负载均衡器是一个分布式的设备，它根据流量的负载均衡分配请求到多个后端服务器。
VIP（Virtual IP）：VIP是云上虚拟IP地址，提供对外服务的真正IP地址。
浮动IP（Floating IP）：浮动IP是一种动态分配的公网IP，它可以映射到私有IP地址，而且可以随时变更。
边缘云（Edge Cloud）：边缘云指的是部署在边缘位置的云，通常用来承载一些不太重要但计算量巨大的业务。OpenStack Edge Cloud的目标是在本地安装OpenStack云平台，通过边缘路由设备对外提供服务。
OpenStack由多个子项目组成，如Nova、Neutron、Swift、Keystone、Glance、Cinder、Horizon、Trove、Ceilometer、Ironic、Barbican、Octavia、Murano、Solum等。这些子项目按照功能分成不同的项目组，并且具有独立的开发路线图、版本号以及发布时间表。