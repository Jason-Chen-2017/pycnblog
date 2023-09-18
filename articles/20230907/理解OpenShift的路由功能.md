
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 什么是OpenShift？
OpenShift是一个基于Red Hat Enterprise Linux的容器云平台，它包括了一整套用于开发、部署和管理应用程序的工具。通过OpenShift，开发者可以轻松地在私有或公有云上部署容器化应用。此外，还有其它的开源项目，例如Kubernetes，CoreOS，Apache Mesos等，它们也提供基于容器技术的分布式系统部署、编排和管理方案。因此，OpenShift提供了完整的解决方案，让开发者能够在其本地环境中部署、管理和监控容器化应用。另外，OpenShift还提供了丰富的服务，包括构建管道（Jenkins）、镜像仓库（DockerHub）、日志分析（ELK），以及其他相关工具（Prometheus）。

## OpenShift路由组件
作为OpenShift集群中的一部分，路由组件负责将外部网络流量路由到OpenShift中的各个服务，即使某些服务不可用也不会影响整个集群的运行。OpenShift路由有两种工作模式：一种是基于服务的路由模式，另一种是基于自定义域名的路由模式。下面，我们主要关注基于服务的路由模式。

# 2.核心概念与术语
## OpenShift的路由模式
在基于服务的路由模式下，OpenShift会根据请求的目标地址匹配相应的路由，并将流量定向到对应的服务。路由由以下几部分构成：
* Service：OpenShift集群内部运行的服务，可以通过Service对象来声明。每个Service都有一个唯一的名称，并且可以选择性地指定一组Label，用于标识属于该服务的Pod集合。
* Route：声明了从外部网络到内部Service的映射关系。每条Route都会绑定一个由Service名和端口号组合成的Endpoint，或者也可以直接指定外部URL。当某个外部客户端发送HTTP请求时，OpenShift会根据请求的目标地址匹配对应的路由，然后把请求转发到对应的Endpoint。
* Endpoint：记录了一个服务的IP地址和端口号。每个Endpoint对应一个Service，并且可以具有多个IP地址。当路由表中存在多个Endpoint时，OpenShift会按照配置的策略，选取一个最佳的端点。
* Router：OpenShift集群中的一个独立的进程，它负责处理和调度请求，并确保路由规则的正确执行。Router可以配置为使用不同的策略来实现请求的负载均衡。目前支持Round-Robin、Random、Weighted和Least-Connection四种策略。

## 服务发现与负载均衡
OpenShift的路由功能依赖于Kubernetes的服务发现机制。当某个Service被创建时，Kubernetes会自动创建一个Endpoint对象，包含了指向Service IP地址的多个路由记录。对于通过外部访问的路由，一般情况下，会同时创建两个Endpoint，分别指向两台服务器上的相同的Service。

当外部客户端发送请求时，OpenShift会解析客户端请求的目的地址，并尝试匹配一条相应的路由规则。如果没有匹配到的路由，则请求会被拒绝。如果匹配到了路由，则OpenShift会按照配置好的策略，选取一个最佳的Endpoint，然后将请求发送给这个Endpoint。最终，请求会被路由至目标Service上的一个可用Pod上。

## 配置路由规则
路由规则通常需要通过路由控制器来配置。路由控制器是一个独立的控制循环，周期性地检查路由的状态，并根据集群当前的状态和配置来调整路由行为。在OpenShift中，路由控制器运行在独立的路由器 Pod 中。默认情况下，路由器 Pod 会以轮询的方式将所有传入的流量分发到 Endpoints 上。但是，管理员可以在路由规则中设置其他的路由策略，例如 Round-Robin、Random 或 Weighted。管理员也可以禁用某条路由规则，或将其优先级提高，以便满足特定的业务需求。