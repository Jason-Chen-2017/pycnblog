
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


Netflix OSS 是 Netflix 公司开源技术栈，它是一套完整的微服务架构体系。通过开源组件和工具包，Netflix 的工程师们可以将其部署到私有云或公有云环境中运行服务。本文主要介绍 Netflix OSS 微服务技术栈，希望能为读者提供一个更全面的微服务技术栈视角。

Netflix OSS 在微服务架构中分层结构，包括客户端、API网关、服务注册发现、配置中心、负载均衡、消息代理、数据流处理、CQRS命令查询责任分离模式、RESTful API、Hystrix断路器模式、Eureka服务注册中心、Zuul网关、Turbine聚合服务器、Archaius动态配置中心、Ribbon客户端负载均衡、Feign客户端调用库、Ribbon负载均衡。图1展示了 Netflix OSS 微服务架构分层结构。


2.核心概念与联系
## 2.1.微服务架构

微服务架构（Microservices Architecture）是一种分布式系统架构设计理念，它将复杂的单体应用服务拆分成多个小型的独立模块，每个模块之间相互独立，可以独立部署、测试、迭代、并协同开发。微服务架构可以有效地解决单体应用难以管理的问题。它提供了软件应用的灵活性、弹性扩展性和可靠性。

微服务架构由六个层次组成，分别为基础设施层、业务逻辑层、API 网关层、服务注册发现层、服务编排层、数据访问层。下图演示了微服务架构中的不同层之间的关系：


### 2.2.客户端
客户端层一般由浏览器、移动应用、PC 客户端等用户终端设备组成，接收用户请求，并向后台服务发送 HTTP 或 RPC 请求，通常采用 RESTful 或者 GraphQL 来定义接口协议。客户端层可以使用缓存机制减少后端系统负载，提高响应速度。客户端层应该具有容错能力，并且在遇到故障时应快速失败，不能让整个系统陷入不可修复的状态。

### 2.3.API网关
API 网关层负责请求过滤和路由转发，保护后端服务不被外界恶意访问，同时也是一个安全防范层。它可以通过认证、授权、限流、熔断、日志和监控功能对服务进行安全控制。API 网关层还可以做请求速率限制、流量整形以及静态资源缓存等工作，能够提升性能和降低成本。

### 2.4.服务注册发现

服务注册发现层负责服务实例的注册与发现。服务实例通常以集群形式部署，通过主动刷新或被动推送的方式实时获取服务最新信息。当一个新实例上线时，服务注册发现层会通知所有客户端更新其本地服务列表，并自动切换到新的服务实例上。服务注册发现层还要具备容错能力，应当具有自动重试和失效转移的机制，保证服务可用性。

### 2.5.服务编排

服务编排层负责服务的管理。例如，它可以根据预先定义好的策略组合、自动伸缩和故障自愈机制，动态调整服务集群规模，实现高可用和弹性伸缩。服务编排层还需具有容错能力，在出现服务失效、网络波动、硬件故障等情况时，应能及时检测、剔除出错节点，避免造成系统瘫痪。

### 2.6.数据访问

数据访问层负责数据的持久化、检索、存储、修改等操作。数据访问层需要使用 NoSQL 或 SQL 数据库，具备高吞吐量、低延迟、高可靠、水平扩展等特点。数据访问层应对访问压力进行弹性扩容，同时提供数据缓存机制减轻数据库负担。数据访问层应对异常数据、脏数据进行检测、清理、补偿，确保系统数据完整性。

## 2.7.Apache Zookeeper

Apache Zookeeper 是 Apache Hadoop 项目中使用的分布式协调服务。它是一个开放源码的分布式应用程序协调服务框架，它是一个用于配置管理、名称服务、集群管理和命名空间的服务软件。Zookeeper 提供了一套分布式一致性方案，基于 Paxos 协议开发，用于维护分布式数据一致性，主要包括 Leader 选举、分布式锁、配置管理、集群管理和负载均衡等功能。

Zookeeper 本身是 CP(Consistency and Partition tolerance) 系统，即任意时刻集群中只能有唯一的Leader角色存在，而其他Server只提供支持，不能独享Leader角色。同时，它同时保障了CP的特性，即数据一致性和分区容忍性。

Zookeeper 使用以下四种方式保证集群数据一致性：

1. 数据版本：每次更新操作都会带上递增版本号，Zookeeper 会将版本号与数据关联起来，所有 watchers 将基于版本号监听变化，确保数据的正确性。
2. 子节点通知：Zookeeper 每次节点的数据发生变更，都会通知已订阅它的子节点。
3. 同步广播：Zookeeper 集群间同步数据，确保各 Server 上的相同数据副本。
4. 选举产生 leader：每个 Server 会在一个固定的时间间隔内竞选一个 leader 角色，确保集群中唯一的leader角色存在。

## 2.8.Netflix Eureka

Netflix Eureka 是 Netflix 公司开源的一个基于 REST 的服务治理及Registries的Client，为 Netflix 流行的微服务架构设计提供了云端中间层支持。Eureka通过提供服务注册和发现的中心支持 Spring Cloud、Amazon Web Services (AWS)、Microsoft Azure、Docker Container Platforms、Kubernetes、Consul 等平台的服务治理及Registries功能。

Eureka 分为两个角色：

1. 服务提供方（Service Provider）：提供具体的业务逻辑和服务，并把自身服务信息注册到Eureka服务器。
2. 服务消费方（Service Consumer）：通过向Eureka服务器获取服务提供方的实例信息并调用相应服务。

Eureka支持多种客户端 SDK 和 Server 端集成方式，如 Spring Cloud、Java 客户端、RESTful API、AWS Discovery Service、Azure Spring Cloud Integration等。

Eureka 通过心跳报告机制检测服务提供方是否正常运行，失效时从服务器中移除服务，服务消费方通过注册表可以获知当前可用的服务实例，实现了动态拉取服务的效果。

## 2.9.Google Consul

HashiCorp Consul 是 HashiCorp 公司开源的一款开源的服务网格(Service Mesh)产品，其主要功能是提供服务发现和配置，利用 sidecar 代理技术，支持微服务架构下的动态服务发现，负载均衡，流量控制，熔断和监测等。Consul 提供了五种服务发现机制：

1. DNS：Consul 支持DNS的服务发现机制，因此可以将服务注册到Consul中，并通过域名解析的方式找到对应的服务IP地址。
2. HTTP：Consul 可以作为HTTP接口服务，进行服务的注册和发现。
3. gRPC：Consul 支持gRPC的服务发现机制，因此可以将服务注册到Consul中，并通过gRPC接口的方式找到对应的服务。
4. 语音：Consul 提供了一个服务寻址工具，可以用声音的方式进行服务发现。
5. 第三方：Consul 支持与主流服务注册中心进行集成，包括Amazon的EC2、Google的GCE、Kubernetes。

Consul 使用 Gossip 协议构建分布式一致性协议，在复杂网络环境下保证最终一致性。

## 2.10.Netflix Ribbon

Netflix Ribbon 是 Netflix 公司开源的一款基于HTTP和TCP客户端的负载均衡器，主要用于 Java 环境下的服务调用。它提供了多种负载均衡策略，适用于 Netflix 内部多种环境，比如 Amazon AWS，Netflix ，eBay等。

Ribbon 有两种负载均衡策略：

1. RoundRobinRule：轮询策略，按顺序循环选择服务器。
2. BestAvailableRule：最少连接策略，优先选择负载最小的服务器。

Ribbon 提供了多种负载均衡算法，比如电梯算法和随机算法，以满足不同的场景需求。

## 2.11.Spring Cloud LoadBalancer

Spring Cloud LoadBalancer 是 Spring Cloud 中负载均衡器模块。它为 Spring Cloud 服务消费者提供了声明式的方法，用来配置负载均衡器，并实现基于不同的负载均衡策略。它提供了七种负载均衡策略：

1. Round Robin：简单的轮询策略。
2. Random：随机策略。
3. Least Connections：最少连接策略。
4. Round Robin based on Availability：根据可用性选择服务器。
5. IP Hash：根据 IP 哈希策略选择服务器。
6. Session Persistence：基于 session 的持久化策略。
7. Retry：重试机制。

## 2.12.Netflix Hystrix

Netflix Hystrix 是 Netflix 公司开源的一款容错工具，为基于异步或事件驱动架构的系统提供延迟和错误容错功能。它提供熔断、降级和恢复功能，帮助识别和隔离故障点，从而使系统更加健壮。

Hystrix 由两部分组成，包括执行层和仲裁层。

### 执行层（Execution Layer）

执行层通过拦截远程服务调用，统计调用失败比例，触发熔断机制，记录断路器打开的原因和时间，从而达到监控和控制的作用。

Hystrix 执行层包括以下几个重要组件：

- 命令组件：定义了访问远程服务的命令对象，包括方法名、参数、超时设置等。
- 请求上下文组件：保存了 Hystrix 命令执行过程中的相关信息，包括线程池、信号量、滑动窗口、请求缓存等。
- 线程隔离组件：隔离了不同线程的影响，每个线程都有自己的请求上下文信息。
- 缓存组件：缓存了请求结果，对于频繁访问的远程服务可以提高性能。
- Fallback 组件：容错机制，提供自定义的回退逻辑。
- Event Stream：提供近实时的监控指标，包括每秒的请求次数，错误比例，成功次数，平均响应时间，超时次数，短路次数等。

### 仲裁层（Collapser）

仲裁层将多个依赖请求合并成一个批处理请求，以节省网络资源和提高响应速度，减少服务依赖的数量，同时还能避免过度依赖。

Hystrix 仲裁层包括以下几个重要组件：

- Collapser：定义了批量任务请求，包括方法名、参数、超时设置等。
- Batch Request Context：保存了 Hystrix Collapser 执行过程中的相关信息，包括线程池、信号量、滑动窗口等。
- Request Cache：缓存了批处理任务的请求结果。
- Collapsed Request Timer：定时收集批处理任务的计时信息。
- Thread Pool Size Configuration：动态调整线程池大小，提前创建线程。

## 2.13.Netflix Zuul

Netflix Zuul 是 Netflix 公司开源的一款基于 JVM 的边缘服务网关，它旨在为 web 应用提供动态路由，接管服务请求并提供请求级的安全性、监控以及富化的静态响应。它可以很好地与现有的云平台如 Amazon Web Services (AWS)，Microsoft Azure 和 Google Cloud 结合使用，可以提供云环境下的 API Gateway 服务。Zuul 的设计目标是简单、快速和可扩展。

Zuul 架构分为前端和后端，前端主要负责请求的接收和过滤，后端主要负责实际的业务处理。Zuul 有三个主要组件：

- Router：路由器组件，根据请求路径匹配后端服务。
- Filters：过滤器组件，可以在请求和响应之间添加额外的行为。
- Spike Arrest Filter：峰值流控过滤器，基于请求数目和时间窗口来限制请求的流量。

Zuul 提供了动态路由和服务限流，另外还可以提供服务熔断和负载均衡。

## 2.14.Netflix Turbine

Netflix Turbine 是 Netflix 公司开源的一款用于聚合流数据和监控数据的组件，Turbine 可以将多个来源的 metrics 数据汇总到一起，提供统一视图查看和分析。它支持多种输入数据类型，包括 Prometheus 和 Graphite，可以与 Zuul 联动，将网关产生的数据发送给 Turbine。

Turbine 可以在内存里存储汇总数据，也可以将汇总数据写入 Kafka 主题或 Cassandra 表中，供其他监控系统使用。

## 2.15.Pivotal Cloud Foundry(PCF)

Pivotal Cloud Foundry (PCF) 是 VMware 开源的开源平台，可以部署、管理和运行应用程序。它提供容器虚拟化、动态服务发现、弹性扩展、自动水平伸缩、服务健康检查、日志记录、认证和授权功能。PCF 还支持 Open Source Buildpacks、Docker Images 和 Kubernetes 配置文件。

PCF 微服务架构由多个应用程序组成，这些应用程序共享底层计算资源和服务，通过 API Gateway 交互，允许用户从外部访问它们。

PCF 为应用和服务分配资源有如下规则：

1. 应用：默认情况下，每个 PCF 安装都会自动分配至少 1 个 vCPU 和 1GB RAM。应用程序可以通过分配更多的资源来增强计算性能。
2. 服务：每个服务都有一个或多个实例，每个实例都分配了一定数量的 CPU 和 Memory。服务的实例数可以通过分配更多的实例来增加计算资源。

PCF 拥有自己的 Service Broker 插件，它可以自动发现 PCF 中的服务，并提供统一的管理界面。开发人员不需要知道底层的细节，就可以方便地与 PCF 服务打交道。