
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Hadoop YARN（Yet Another Resource Negotiator）是一个集群资源管理器（Cluster ResourceManager），用于统一管理Hadoop集群中的资源。YARN支持多种计算框架，包括MapReduce、Pig、Hive等。它采用模块化设计，可以动态扩展以支持更多的应用。
Apache Hadoop YARN提供了一个通用的资源管理框架，使得Hadoop应用程序能够请求运行在独立的容器中，而不是所有的任务都在同一个节点上。同时，YARN还提供了诸如容错处理、弹性伸缩、队列管理等功能，帮助管理员有效地管理Hadoop集群资源。
本文将主要探讨Hadoop YARN的组件结构及其作用。
# 2. 基本概念与术语
## 2.1 概念
YARN作为Hadoop的一个子项目，它是一个开源的集群资源管理器。
- **集群管理器（Cluster Manager）**：此角色管理整个Hadoop集群的资源分配。当一个客户端提交作业时，集群管理器会分配相应的资源给各个节点上的容器。YARN被设计成可插拔的，因此可以在不同的资源管理器之间切换，而无需重启Hadoop集群。
- **资源调度器（Resource Scheduler）**：此角色根据集群中所有可用资源和队列中的资源需求进行调度。资源调度器通过将资源分配到适当的队列和容器中，从而实现资源共享和隔离。
- **节点管理器（Node Manager）**：此角色负责启动并监视运行于每个结点上的容器。如果某个节点失败或耗尽资源，则节点管理器会自动释放资源，确保集群的稳定运行。
- **作业协调器（Job Coordinator）**：此角色协调各个任务之间的依赖关系，以确保它们按顺序执行。作业协调器也可用于容错处理和作业恢复。
## 2.2 术语
**集群（Cluster）**：指的是由一组工作节点（Nodes）和存储设备构成的计算机网络。通常来说，集群由一个中心主服务器和一个或多个辅助服务器组成。Hadoop集群由一个Namenode和一个Datanode组成。
**HDFS（Hadoop Distributed File System）**：基于Google文件系统（GFS）的分布式文件系统，它利用了廉价的 commodity hardware 提高计算性能。HDFS存储数据块（Block）而不是整个文件，这样就可以在集群中的任意两个节点间复制数据，以实现容错能力。HDFS的文件系统命名空间由路径名表示，路径名唯一标识文件系统中的一个文件或目录。
**MapReduce（Massive Parallel Processing）**：一种编程模型和运算框架，用于处理海量的数据集并生成结果。MapReduce程序由两个阶段组成：Map和Reduce。Map阶段将输入数据划分成较小的独立片段，并对这些片段进行映射处理，将其转换成中间键值对。Reduce阶段对这些键值对进行归约，汇总中间结果并生成最终结果。Hadoop MapReduce API支持许多语言，包括Java、C++、Python和Scala。
**Yarn（Yet Another Resource Negotiator）**：另一个集群资源管理器（RM）。它整合了MapReduce、Spark、HBase等众多计算框架，并且与HDFS、Hbase和其他外部系统集成。
**Container（容器）**：在yarn中，一个container是一个有固定资源（比如内存、CPU等）和单独隔离进程的集合。容器封装了可以并行执行的一系列程序。Yarn上运行的每个作业都会产生一个container，容器中运行着相关的任务。
**Application Master（AM）**：当client提交一个application的时候，它就会联系到Yarn集群，然后资源管理器就会创建对应的ApplicationMaster（AM）来管理这个application。AM主要做两件事情：一是向资源管理器申请container；二是和NM通信，监听container状态。
**NodeManager（NM）**：NodeManager是一个守护进程，每台机器上都会有一个。NodeManager主要负责执行container，它和NM建立心跳，告诉RM自己的状态。NM分配给它的container并启动container里面的进程。NM是一个计算密集型的过程，一般会启动很多个。
**ResourceManager（RM）**：ResourceManager管理整个集群的资源。当client提交一个job时，RM会根据job所需的资源量和队列配置，把资源分配给不同的node manager。ResourceManager对外提供RESTful API接口，供客户端和web ui调用。
**Queue（队列）**：一个队列就是一些具有共同特征的任务集合。队列可以用来控制访问权限，限流，以及优先级等属性。一般来说，不同的用户会被分配到不同的队列中，以达到资源的隔离和限制。
**Priority（优先级）**：任务的优先级决定了它获得的资源份额，任务越高级就获得更多的资源。
**Capacity scheduler（容量调度器）**：容量调度器是YARN中提供的一种队列管理策略。容量调度器会将集群的资源按一定比例分配给不同队列。它会监控队列的容量，并实时调整队列的资源分配以保证集群的资源利用率。
**Fair scheduler（公平调度器）**：公平调度器是一个新加入的队列管理策略，它会尝试为每个任务都划分出公平的资源配额。Fair scheduler不会一次将所有资源平均分配给所有的队列，而是根据任务的资源需求按比例分配资源。
**Container Launcher（启动器）**：启动器管理container生命周期。当AM向RM申请一个container时，RM会通知ContainerLauncher，然后它会向对应的NM发送container启动命令。
# 3.YARN架构概览
如图所示，YARN由ResourceManager，NodeManager，Client，HistoryServer四个主要组件组成。
- **ResourceManager (RM)**：资源管理器，是YARN最核心的组件之一，负责整个集群资源的分配、调度和治理，保证集群的运行安全。它是一个全局性的框架，包括两个主要功能：资源管理和作业调度。
- **NodeManager (NM)**：节点管理器，是YARN集群中每个节点上的守护进程，负责管理和维护各自节点上的容器。它向资源管理器报告其管理的容器的健康状况，并获取资源的使用情况。
- **Client**：客户端，用于提交应用程序到YARN集群。它可以是MapReduce或者Spark程序的驱动或者提交者，也可以是一个命令行界面。
- **HistoryServer**：历史服务器，用于存储历史作业信息。它包含了关于MapReduce作业的详细信息，例如启动时间，完成时间，使用的内存等。
# 4.YARN核心组件介绍
## 4.1 ResourceManager
ResourceManager 是 YARN 的核心组件，它负责整个集群资源的分配、调度和治理，包括资源管理和作业调度。
### 4.1.1 资源管理
ResourceManager 以全局的方式管理集群的所有资源，包括计算资源（CPU、内存、磁盘等）、网络资源（带宽等）以及持久存储资源（HDFS）。ResourceManager 会考虑到各个队列的资源使用情况，按照队列优先级为每个 ApplicationMaster 分配合理数量的资源。同时 ResourceManager 会检查各个 NodeManager 的健康状况，并为失效的 NodeManager 上的 container 撤销资源。
### 4.1.2 作业调度
ResourceManager 会接收 client 端提交的作业请求，它会选择一个可用的 NodeManager 来启动 AM。AM 将得到一定的资源（内存和 CPU）以便初始化自己。AM 和 RM 通过 RPC 协议通信，通过 heartbeat 和心跳机制保持通信连接。AM 根据作业的资源需求和队列配置，向 RM 申请相应的资源。AM 将这些资源提供给 tasks，tasks 根据自己的资源占用情况和优先级被调度到 NodeManagers 上运行。AM 通过 RPC 方式通知 NMs 有新的 tasks 需要启动。除此之外，AM 会向 YARN 中的其它组件提供各种服务，如日志收集、监控、安全等。
## 4.2 NodeManager
NodeManager 是 YARN 中最重要的组件之一，它管理和维护各自节点上的容器。它负责启动并监视运行于该结点上的容器。
### 4.2.1 资源管理
NodeManager 会获取 ResourceManager 的指令，启动和停止 container。NodeManager 会跟踪其管理的 container 的健康状态，并定时向 ResourceManager 报告其自身的资源使用情况。
### 4.2.2 监控
NodeManager 会定时向 ResourceManager 发起获取 Container 信息的请求，同时它也会向客户端返回当前正在运行的任务信息。它通过汇报 node manager、系统 metrics 和 application master 等信息来监测集群的运行状态。
### 4.2.3 健壮性
NodeManager 具备非常强大的容错能力，它会检测到硬件故障、崩溃、错误的状态变化等异常情况，并及时停止 container。当 NodeManager 检测到该容器发生故障时，它会立即终止该 container，同时通知对应的 Application Master。另外，当 NodeManager 发现某台机器出现故障时，它会主动地将其上的 container 转移到其它 NodeManager 上。
## 4.3 Client
客户端可以是提交应用程序到 YARN 集群的驱动或者提交者。它可以通过 REST API 或 CLI 命令来与 YARN 服务交互。YARN 支持两种类型的客户端，分别为命令行客户端和 Web UI 客户端。Web UI 客户端允许用户在浏览器中查看当前集群的状况，并提交作业。命令行客户端提供 shell 命令，方便用户提交作业。除此之外，还有第三方的集成开发环境（IDE）插件，如 Eclipse Plugin for YARN。
## 4.4 HistoryServer
历史服务器是一个可选组件，它提供关于 MapReduce 作业的信息。它会记录作业的启动时间、结束时间、运行时间、消耗的内存、使用的硬盘空间等信息。HistoryServer 可以通过 REST API 查看作业的详细信息。
# 5.YARN组件详解
## 5.1 Client
### 5.1.1 介绍
Client 组件用于提交应用程序到 YARN 集群。它可以是 MapReduce 或 Spark 程序的驱动或者提交者，也可以是一个命令行界面。YARN 支持两种类型的客户端，分别为命令行客户端和 Web UI 客户端。Web UI 客户端允许用户在浏览器中查看当前集群的状况，并提交作业。命令行客户端提供 shell 命令，方便用户提交作业。除此之外，还有第三方的集成开发环境（IDE）插件，如 Eclipse Plugin for YARN。
### 5.1.2 功能
- 应用程序提交：客户端需要向 ResourceManager 提交一个应用程序（Application），它包含了相关的元数据（如：运行所需的资源、命令、依赖关系等），并指定 ApplicationMaster 的位置。
- 请求资源：客户端向 ResourceManager 申请资源，资源包括（内存、CPU、磁盘、网络等）。
- 任务分配：ResourceManager 确定 ApplicationMaster 将要运行的任务，并分配它们到对应的 NodeManager。
- 任务执行：当 ApplicationMaster 获取到任务后，它会创建一个作业并启动一个线程来运行。
- 结果返回：当 ApplicationMaster 完成任务后，它会向客户端返回最终结果。
### 5.1.3 扩展
YARN 提供了许多扩展点，可以实现定制化的功能。
#### 5.1.3.1 定制化的资源类型
YARN 默认支持的资源类型包括内存、CPU、磁盘、网络。但是用户可以自定义资源类型。例如：用户可以定义自己的资源类型：GPU 资源、FPGA 资源等。
#### 5.1.3.2 定制化的调度算法
YARN 提供三种调度算法：FairScheduler、CapacityScheduler 和 ReservationSystem。用户可以选择适合自己的调度算法。
#### 5.1.3.3 定制化的日志收集和存储
YARN 使用 HDFS 来存储作业的日志。用户可以选择集成到 YARN 日志收集体系中。
#### 5.1.3.4 定制化的应用审核机制
YARN 提供了一个应用审核机制，可以审核待运行的应用程序。管理员可以设置规则来限制特定类型的应用程序的运行。
#### 5.1.3.5 定制化的资源管理界面
YARN 提供了一个基于 Web 的资源管理界面，供管理员查看集群的状态。用户可以登录到 Web 界面上，查看应用程序的进度、日志、监控等信息。
## 5.2 ApplicationMaster
### 5.2.1 介绍
ApplicationMaster（AM）是一个容错机制，它是 YARN 的主服务进程。当客户端提交一个 Application 时，它就会联系到 ResourceManager ，资源管理器就会为这个 Application 创建一个 ApplicationMaster 。AM 和 ResourceManager 通过 RPC 协议通信，通过心跳和 rpc 调用保持通信连接。AM 为任务和 container 分配资源、监控任务状态、处理应用程序故障。当 ApplicationMaster 失败时，ResourceManager 会重新启动一个新的 ApplicationMaster 。
### 5.2.2 功能
- 分配资源：AM 会根据应用的资源要求向资源管理器申请资源。
- 任务调度：AM 会将资源分配给任务，并监控它们的运行状态。
- 处理失败：当 AM 发现其所在的 NodeManager 出现问题时，会重新启动该 AM。
- 数据传输：AM 可以选择把数据上传到 HDFS 或本地文件系统。
- 容错处理：当 AM 失败时，ResourceManager 会重新启动一个新的 AM。
- 日志聚集：AM 会把任务的日志聚集到一起，然后提供给客户端。
- 监控：AM 可以采集节点的统计信息，以及容器的资源使用情况，并把它们传给 ResourceManager 。
### 5.2.3 扩展
YARN 提供了许多扩展点，可以实现定制化的功能。
#### 5.2.3.1 定制化的任务类型
YARN 默认支持 MapReduce 任务。用户可以使用其他类型的任务，如 PIG 任务、Spark 任务等。
#### 5.2.3.2 定制化的资源分配模型
YARN 默认使用 greedy 模型来分配资源。用户可以选择其他的资源分配模型，如 fair 模型。
#### 5.2.3.3 定制化的任务规划器
YARN 采用一个抢占式的任务规划器，它会在每个 NodeManager 上尝试运行多个任务。用户可以选择使用另一个任务规划器，如粘性任务规划器。
#### 5.2.3.4 定制化的任务重新规划
YARN 默认支持当一个任务失败时，重新规划任务。用户可以禁止这种行为。
#### 5.2.3.5 定制化的依赖关系处理
YARN 默认支持依赖关系的处理。用户可以禁止这种行为。
## 5.3 NodeManager
### 5.3.1 介绍
NodeManager （NM）是一个守护进程，运行在每个结点上。它管理和维护运行于其上的 container。它接受来自 ResourceManager 的指令，启动和停止 container。NM 和 ResourceManager 通过 RPC 协议通信，通过心跳和 rpc 调用保持通信连接。当某个 container 失败时，NM 会回收该 container 并重新启动另一个相同的 container 。
### 5.3.2 功能
- 资源管理：NM 会从资源管理器处获取容器资源。
- 容器启动：NM 在接受到 container 启动指令时，会启动 container。
- 监控和健康检查：NM 会周期性地向资源管理器发送汇报，汇报当前的负载，包括磁盘使用率、内存使用率等。NM 会在失败时自动重启 container。
- 数据传输：NM 会把数据下载到本地磁盘，或上传到 HDFS。
- 日志聚集：NM 会把日志聚集到一起，然后提供给 ApplicationMaster 。
### 5.3.3 扩展
YARN 提供了许多扩展点，可以实现定制化的功能。
#### 5.3.3.1 定制化的本地存储
YARN 默认支持 container 内部使用本地磁盘。用户可以选择使用更快的磁盘。
#### 5.3.3.2 定制化的镜像管理
YARN 可以创建镜像，在远程主机上运行任务。用户可以选择是否启用镜像管理。
#### 5.3.3.3 定制化的资源管理
YARN 可以将资源管理委托给外部的资源管理器，例如 Apache Mesos。用户可以选择是否使用外部资源管理器。
#### 5.3.3.4 定制化的垃圾回收策略
YARN 默认使用基于轻量级标记-清除垃圾回收策略。用户可以选择使用其他的垃圾回收策略。
#### 5.3.3.5 定制化的认证和授权机制
YARN 可以使用 Kerberos 或 SPNEGO 验证用户。用户可以选择使用其他的认证和授权机制。
## 5.4 ResourceManager
### 5.4.1 介绍
ResourceManager（RM）是一个全局性的框架，它负责整个集群资源的分配、调度和治理。它以全局的方式管理集群的所有资源，包括计算资源、网络资源以及持久存储资源（HDFS）。ResourceManager 会考虑到各个队列的资源使用情况，按照队列优先级为每个 ApplicationMaster 分配合理数量的资源。同时 ResourceManager 会检查各个 NodeManager 的健康状况，并为失效的 NodeManager 上的 container 撤销资源。ResourceManager 会接收 client 端提交的作业请求，它会选择一个可用的 NodeManager 来启动 AM。AM 将得到一定的资源（内存和 CPU）以便初始化自己。AM 和 RM 通过 RPC 协议通信，通过 heartbeat 和心跳机制保持通信连接。AM 根据作业的资源需求和队列配置，向 RM 申请相应的资源。AM 将这些资源提供给 tasks，tasks 根据自己的资源占用情况和优先级被调度到 NodeManagers 上运行。AM 通过 RPC 方式通知 NMs 有新的 tasks 需要启动。除此之外，AM 会向 YARN 中的其它组件提供各种服务，如日志收集、监控、安全等。
### 5.4.2 功能
- 集群资源管理：ResourceManager 会管理整个集群的资源，包括计算资源、网络资源、磁盘资源等。
- 资源分配：ResourceManager 会为各个 ApplicationMaster 分配合适的资源。
- 作业调度：ResourceManager 会为 ApplicationMaster 分配相应的资源，并监控它们的运行情况。
- 任务调度：ResourceManager 会为 task 分配资源，并监控它们的运行情况。
- 容错处理：ResourceManager 会处理失效的 NodeManager 和 container。
- 任务状态协调：ResourceManager 会协调各个任务之间的依赖关系，确保它们按序执行。
- 服务监控：ResourceManager 会监控系统的运行状况，提醒管理员存在的问题。
- 服务治理：ResourceManager 会实施安全策略，防止潜在的安全威胁。
### 5.4.3 扩展
YARN 提供了许多扩展点，可以实现定制化的功能。
#### 5.4.3.1 定制化的资源类型
YARN 默认支持 CPU、内存、磁盘、网络等资源类型。用户可以自定义资源类型。
#### 5.4.3.2 定制化的调度策略
YARN 提供两种调度策略：CapacityScheduler 和 FairScheduler。用户可以选择适合自己的调度策略。
#### 5.4.3.3 定制化的队列管理
YARN 默认支持普通队列和优先级队列。用户可以定义自定义队列。
#### 5.4.3.4 定制化的容错处理
YARN 默认支持容错处理。用户可以禁止容错处理。
#### 5.4.3.5 定制化的应用提交流程
YARN 提供一个默认的应用提交流程。用户可以定义自定义的流程。
# 6.总结
YARN 是 Hadoop 中的一个子项目，它是 Hadoop 的一个独立子系统，它不仅提供了资源管理、任务调度、容器化等功能，还提供了高可用性、可靠性、可扩展性、弹性、安全、性能等方面的功能。本文首先简要介绍了 YARN 的背景、概念及术语，接下来详细介绍了 YARN 的各个组件以及它们的作用，最后再总结了 YARN 的扩展性及开发人员可参考的扩展点。