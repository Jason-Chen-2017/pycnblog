
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Docker 在开源社区的火爆还带动了一波容器技术的普及，各大公司纷纷推出自己的容器技术解决方案如 Kubernetes、Mesos等。而 Docker 自身拥有一个容器编排工具--Docker Compose ，用于定义和运行多容器应用程序。为了能够在云端和本地环境下都可以使用 Docker 容器技术，Amazon Web Services (AWS) 推出了 Amazon EC2 Container Service （ECS）平台。该平台支持运行面向服务的工作负载（Service-Oriented Workloads，SOW），并通过 Docker 容器集群的方式提供资源隔离、弹性伸缩、批量部署等能力。另外 AWS 还推出了基于任务的工作流服务 Amazon Step Functions (SFN)，可以用来编排多步骤的任务流程。

如今，Amazon ECS Fargate 是 ECS 的一种新的运行时环境，可以在不用购买服务器的情况下快速部署和运行容器化应用，并且具有极高的可靠性。它帮助用户获得低成本的、灵活且可伸缩的容器运行环境，能够满足各种不同的业务需求。作为一个托管的容器运行时环境，Fargate 可以免除运维开支，使开发者从管理服务器或集群上手的时间变短，同时减少了对容器资源配置的要求。

因此，EFS Fargate 是今天构建高度可扩展、高可用、分布式系统的一种简单有效的方法。本文将主要介绍 Amazon ECS Fargate 相关的基础知识。

2.基本概念术语说明
## Amazon Elastic File System (Amazon EFS)
Amazon EFS 是一种文件存储卷，为运行中的 Amazon ECS 服务提供了共享的文件系统。它类似于网络文件共享，允许多个主机访问共享存储，而无需额外的设置。与其他共享文件系统不同的是，Amazon EFS 可以自动缩放，即使系统的容量需要扩大也会自动进行扩容。Amazon EFS 支持 NFS v4 和 SMB 文件系统协议，并且可以与 Amazon EC2 或 Amazon EKS 结合使用，以便轻松地部署到云中。

当创建一个 ECS 服务时，您可以选择使用 Amazon EFS 来存储其日志、数据或其他需要持久化的数据。这样可以减少镜像大小并提升启动时间，并在故障转移期间避免丢失数据。Amazon EFS 提供了两种模式：默认模式（generalPurpose）和最大吞吐量模式（maxIO）。前者适用于小文件，而后者则更加适合大文件的读写场景。

## Amazon Elastic Container Registry (Amazon ECR)
Amazon ECR 是 Docker 镜像仓库的托管服务，可以存储、管理、和分发您的容器镜像。借助 Amazon ECR，您可以安全、快速地分享容器镜像给其他开发人员、测试人员和ops团队。只要登录 Amazon ECR，即可直接从其中拉取或推送镜像，而不需要运行 Docker 命令。

当在 ECS 中运行任务或服务时，需要指定容器镜像。Amazon ECR 中的容器镜像可由其他开发人员、测试人员和ops团队共享。Amazon ECR 可以帮助您发布并更新应用程序，并提供长期保存和版本控制的功能，方便您的CI/CD过程。

## Amazon Elastic Container Service (Amazon ECS)
Amazon ECS 是 AWS 提供的一项托管服务，用于运行和管理容器化的应用程序，包括web服务、后台作业、实时游戏服务器等。与传统的虚拟机比起来，ECS 更加轻量级，而且价格也比较经济。它具有以下几个特点：

- 自动弹性伸缩：当应用请求增加或减少时，ECS 会自动调整相应的集群规模，使应用始终保持最佳性能状态。
- 按需付费：ECS 根据实际使用量付费，降低了云计算的使用成本。
- 服务发现：ECS 可让容器内的应用可以自动发现运行在同一集群上的其他服务。
- 可视化界面：ECS 提供了直观的图形化界面，可以直观显示集群状态，以及容器组、服务、任务、容器的运行情况。

## Amazon Elastic Compute Cloud (Amazon EC2)
Amazon EC2 是一种基于 web 服务的 IaaS 服务，为用户提供了一系列弹性计算资源，如 CPU、内存、磁盘空间、网络带宽等。ECS 是 Amazon EC2 上面的应用部署平台。所以，ECS 需要依赖于 EC2 来运行。

## Task Definition
Task Definition 是 ECS 中最重要的实体之一，它定义了 ECS 服务运行所需的配置，包括 CPU、内存、磁盘占用、镜像名称、挂载目录等信息。创建 ECS 服务时，需要指定 Task Definition。如果没有指定，ECS 将会根据默认的 Task Definition 创建新服务。

Task Definition 除了定义服务运行的配置之外，还可以定义环境变量、容器健康检查、联网策略等。这些配置项可以帮助 ECS 服务更好地响应外部世界的变化。

## Service Discovery
Service Discovery 是一种 Amazon ECS 服务，可以帮助容器内的应用自动发现其他服务的信息。Service Discovery 使用 DNS 查询，容器内的应用就可以发现其他服务的 IP 地址，并直接与之通信。这种方式可以实现服务之间的解耦、模块化和可扩展性。

## Amazon Elastic Load Balancing (Amazon ELB)
Amazon ELB 是一种负载均衡器，可以帮助 ECS 服务在多个可用区之间进行负载均衡。当某个区域出现故障时，ELB 会自动将流量切换至另一个区域，确保服务的高可用性。ELB 可以将 HTTP、HTTPS 和 TCP 协议的流量分配到多个目标，包括 EC2 实例、容器、Lambda 函数等。

## Amazon Simple Notification Service (Amazon SNS)
Amazon SNS 是一种消息通知服务，可以帮助 ECS 服务发送和接收通知。例如，当 ECS 服务有新的任务运行时，可以向 SQS 消息队列发送通知，告知其他系统进行处理。SNS 提供了完整的 API，可以通过 SDK 或命令行工具进行调用。

## Amazon Elastic Block Store (Amazon EBS)
Amazon EBS 是一种块存储设备，可以作为容器和 ECS 服务的持久化存储设备。EBS 允许用户创建独立于 EC2 实例生命周期的存储，并提供高效、可靠的块级别 IO。除了为 ECS 服务提供持久化存储之外，EBS 也可以用做数据备份、迁移、和容灾恢复等目的。