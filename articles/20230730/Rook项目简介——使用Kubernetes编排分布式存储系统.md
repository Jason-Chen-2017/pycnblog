
作者：禅与计算机程序设计艺术                    

# 1.简介
         
2016年8月，Rook官方宣布开源分布式云原生存储管理系统Rook的诞生，该项目旨在通过提供一组kubernetes控制器来管理存储，包括Ceph、GlusterFS、AWS、Azure等。Rook可以帮助用户轻松地部署和管理各种类型的分布式存储集群，从而提升企业存储利用率，降低运营成本并满足企业对数据的安全和完整性要求。
         Kubernetes是最流行的容器编排工具，越来越多的人选择将其用于编排分布式存储集群。相比于其他方案，Rook在易用性和功能支持上都有非常突出优势，它基于Kubernetes框架构建，因此，可以在Kubernetes平台上部署、运行和扩展分布式存储集群。Rook官方提供了详细的安装指导文档，供用户快速上手。
         本文将以Rook官方文档作为基础，从分布式存储系统的角度出发，详细阐述Rook项目。
         ## Rook概述
         
         ### Rook项目由以下几个主要模块构成：

         - **Ceph Operator**: Ceph提供分布式存储的关键组件之一，Rook通过Ceph Operator实现了Ceph集群的自动化部署、管理和升级。Ceph Operator会监控CRD(Custom Resource Definition)，当用户创建或修改了CephCluster CR时，Ceph Operator就会根据配置部署相应的Ceph集群。同时，它还负责维护集群的健康状态，确保数据及元数据完整性。
         - **CSI Drivers for Storage:** Rook开发了多个不同的CSI驱动程序，可以使Kubernetes直接管理底层存储系统，无需本地挂载任何存储设备。目前，Rook支持CephFS、RBD、Ceph Block、NFS、AWS EBS、Azure Disk、GCP PersistentDisk和OpenStack Cinder等。
         - **Object Storage Services:** 对象存储服务模块主要用来提供对象存储接口，如S3 API或Swift API。对象存储服务是分布式存储中的一种特殊形式，它不像块存储那样由存储设备直接提供持久化存储，而是在远端服务器上运行着对象存储代理，来处理客户端的请求，如上传、下载和删除文件。Rook通过Object Service Module支持多个对象存储服务，包括Ceph Object Gateway、Minio、Amazon S3、Google GCS、Apache Swift和华为OBS等。
         - **EdgeFS:** EdgeFS是一个面向微服务和边缘计算环境设计的开源分布式数据网格。Rook通过EdgeFS Module支持部署、管理和扩展EdgeFS集群。

        ![](https://img.serverlesscloud.cn/202181/1629515907043-rook%E9%A1%B9%E7%9B%AE%E7%BB%84%E4%BB%B6%E5%9B%BE.png)

         ### 特色功能

         1. 自动化部署、管理和升级Ceph集群：Rook通过Ceph Operator模块实现了对Ceph集群的自动化部署、管理和升级，支持在线增加或者减少存储节点。
         2. 支持多种类型存储：Rook支持CephFS、RBD、Ceph Block、NFS、AWS EBS、Azure Disk、GCP PersistentDisk、OpenStack Cinder等多种类型的存储，支持在Kubernetes平台上部署存储集群。
         3. 提供对象存储服务：Rook通过Object Service Module提供各种类型的对象存储服务，支持在Kubernetes平台上部署对象存储集群，如S3 API或Swift API。
         4. 支持分布式存储网格：Rokcet通过EdgeFS Module支持部署、管理和扩展分布式存储网格，用于边缘计算和微服务架构中。
         5. 支持跨平台：Rook可以运行在任何支持Kubernetes的平台，如公有云、私有云、混合云等。


         # 2.基本概念术语说明

         ## 2.1. Kubernetes

         Kubernetes 是当前主流的容器编排调度系统之一，它由 Google、Facebook、CoreOS、IBM、Red Hat、微软和 Deis 等众多公司和个人开发者共同打造。它是一个开源的、可扩展的、支持自动化部署、扩展和自我healing 的容器集群管理系统。作为一个开源系统，Kubernetes 拥有一个庞大的生态系统和周边工具支持，其中包括各种第三方插件和解决方案，如 Prometheus、Weave Net 和 Heapster 。它的架构采用 master-slave 的模型，通过 master 节点的控制，slave 节点完成任务。由于 master-slave 模型的架构特性，Kubernetes 天生就具备高可用和可伸缩性，能够应付生产环境下的大规模集群。Kubernetes 最新版本为 v1.21.5 ，具有良好的兼容性，可以部署和运行在各种云环境中。

     
             架构图：
              
                   +------------------+               
                   |                  |               
                   |     Node         |<---------------+             
                   |                  |               
                   +--------+---------+                   
                            |                                   |
                  +--------v----+       +--------------+   |                           |
            +-------|    Master  |------>|      etcd    |---+                             |
      API call|        +----+-------+       +--------------+   |            controller      |  
            +--------| Kube-API |                   ^                     |               service
                     +----------+                   |                     |
                                         v                     |                     |
                                   +--------------------+          |                     |
                                   |      Pods         |----------|                     |
                                   |                   |                                      |
                                 +-----v-----+       +----------------+                 |
                         +-----| Container |---->|   Volumes      |----------------+         |
           +---------------| DaemonSet |       |                |                         | 
           |                +-----------+       +----------------+           kubelet             |
           |                                                            |                          |
           +------------------------------------------------------------+                       
                                                                                              

   Kubernetes 使用 YAML（YAML Ain't a Markup Language）配置文件定义对象的属性，这些配置文件被称为“资源”，资源的例子包括 Deployment、Service、ConfigMap、Secret 等。每个资源都有一个唯一的标识符，称作“名称”，可以通过 “kind” 和 “namespace” 属性组合起来唯一确定一个资源。
   
   Kubernetes 将资源分为三类：
   
   - **Pod：** 一个 Pod 就是一个或多个紧密相关容器组成的逻辑隔离单元，它包含一个或者多个容器，共享相同的网络命名空间、IPC 命名空间和uts 命名空间。Pod 中的容器之间可以通过 localhost 通信，也可以通过 IPC 机制进行通讯。
   - **Deployment：** Deployment 为管理 Pod 和 Replica Set 提供声明式 API。用户只需要描述 Deployment 描述期望的 Pod 副本数量、更新策略、滚动发布策略等信息，Deployment Controller 根据 Deployment 描述信息生成对应的 Replica Set 和 Pod。
   - **ReplicaSet：** ReplicaSet 是创建指定数量的 Pods 的集合。当用户通过 Deployment 创建 ReplicaSet 时，ReplicaSet 会保证实际运行的 Pod 数量始终保持指定的副本数量。
   - **Service：** Service 是一系列后台 Pod 的抽象，它通过 Label Selector 来匹配一组后台 Pod，并对外暴露统一的服务访问入口。
   
     除以上资源以外，Kubernetes 还有一些内置的资源对象，比如 Namespace、Node、PersistentVolumeClaim 等。
     
     总的来说，Kubernetes 是一个复杂而强大的系统，它在应用级和集群级别都提供了很多便利。
   
   ## 2.2. Ceph

    Ceph 是开源的分布式存储系统，它是一个高度可用的、可扩展的网络存储平台。它提供分布式文件系统、块存储、对象存储、消息队列、数据库和可扩展分析框架等功能。Ceph 已经成为世界领先的分布式存储技术，其架构既能够适应超大规模的数据处理场景，又可以提供高效的数据访问能力。
    
    Ceph 由两大部分组成：

    - **Ceph OSD**： Ceph OSD（Object Storage Device）模块是一个独立的守护进程，它负责存储集群中的数据块，并接收和响应客户端读写请求。OSD 以群组的形式进行分布部署，并且每个 OSD 可以横向扩展到数千个磁盘，以实现超高性能和可靠性。
    - **Ceph Monitor**： Ceph Monitor 是一个守护进程，它与其他 Ceph 服务组件如 MDS、RGW、RESTful 接口等进行交互，它负责元数据存储、集群状态检测、集群内各项服务之间的通信、故障转移和数据复制等工作。
   
    Ceph 集群通常由若干个有限资源的节点组成，这些节点可以是物理机、虚拟机甚至容器，它们彼此通过网络连接形成集群。集群中任意两个节点之间都可以互相通信，因此可以充分利用网络带宽，实现高速数据传输。Ceph 通过 CRUSH 算法（Controlled Replication Under Scalable Hashing），将数据分布到集群中不同的位置，因此可以有效地实现数据冗余，防止单点故障。同时，Ceph 还通过 erasure code（纠删码）的方式，通过引入冗余数据提高数据可用性，从而实现数据持久性。
    
    ## 2.3. CephCluster CRD

    在 Kubernetes 中，CRD（Custom Resource Definition）是一种 Kubernetes 扩展机制，允许用户创建自定义资源。Rook 项目提供了 CephCluster 自定义资源，可以通过定义这个资源，来声明 Ceph 集群的部署、管理和扩展。用户可以使用 kubectl 命令行工具来创建、编辑、删除 CephCluster 资源。下面的示例展示了一个典型的 CephCluster 配置：

        apiVersion: ceph.rook.io/v1
        kind: CephCluster
        metadata:
          name: my-cluster
          namespace: rook-ceph
        spec:
          dataDirHostPath: /var/lib/rook
          skipUpgradeChecks: false
          continueUpgradeAfterChecksEvenIfNotHealthy: false
          upgradeCheckInterval: 0
          monitoring:
            enabled: true
            rulesNamespace: rook-ceph
            rules:
              alarm_actions:
                - log
                - notify
                - email
              alerts:
                - rule_name: mon_port_down
                  expr: 'probe_success{job="rook-ceph-mgr", instance=~"$(CEPH_MGR).*", name="mon"} == 0'
                  severity: critical
                  annotations:
                    description: "Monitor daemon {{$labels.instance}} in $(CEPH_CLUSTER) is not responding to the MON health check."
                    summary: "MON down"
                  rearm_seconds: 180
                - rule_name: mon_quorum_status
                  expr:'sum(up{job="rook-ceph-mgr"}) by (job) < length($(MON_COUNTS))'
                  labels:
                      prometheus: operator
                  annotations:
                    description: >-
                      The monitor quorum status has degraded, either because too few mons are up or they can no longer communicate. This may result in data loss if you lose more than half of your monitors at once. Consider checking your network connectivity and trying again later.
                    summary: Monitor Quorum Status degraded
                - rule_name: osd_nearfull
                  expr: 'ceph_fs_state{endpoint="$SERVICEACCOUNT",instance!="",job="rook-ceph-manager",mounted="true",name=~".*?storage.*?",type="Filesystem"} == 1' and avg_over_time((ceph_osd_stat_bytes{endpoint="$SERVICEACCOUNT",instance!="",job="rook-ceph-manager",type="device",id=~".*?[a-z]$"}[1h])[(scalar(vector(count_over_time((ceph_osd_stat_bytes{endpoint="$SERVICEACCOUNT",instance!="",job="rook-ceph-manager",type="device",id=~".*?[a-z]$"})[1h]))>0)*-1]):avg{job="rook-ceph-mgr", endpoint="$SERVICEACCOUNT", instance!=""} by (host, storagepool) > ((ceph_fs_max_avail{endpoint="$SERVICEACCOUNT",instance!="",job="rook-ceph-mgr",name=~".*?storage.*?",type="Filesystem"} * 0.75)|0.9)
                  annotations:
                    description: >-
                      Available space on an OSD device is low (used >= $space_utilization_threshold%%), which could cause performance issues or data corruption. Please consider expanding the file system partition that contains this device or adding additional devices to the cluster.
                    summary: Low space utilization detected on OSD {{ $labels.instance }}
          mgr:
            count: 1
            modules:
              - name: pg_autoscaler
                enabled: true
          dashboard:
            enabled: true
          network:
            provider: host
            selectors: {}
          rbdMirroring:
            workers: 0
          disruptionManagement:
            manageMachineDisruptionBudgets: false
            machineDisruptionsAllowed: 1
            allowDrainingMultipleMachines: false
            controlledResources:
              - secrets
          removeOSDsIfOutAndSafeToRemove: false
    
    上面的示例定义了名为 `my-cluster` 的 Ceph 集群。它包括了以下几个重要参数：

    - `dataDirHostPath`: 指定 Rook 将 Ceph 数据目录映射到主机上的路径。
    - `skipUpgradeChecks`: 是否跳过 Rook 对现有集群的升级检查。
    - `continueUpgradeAfterChecksEvenIfNotHealthy`: 如果 Rook 检查发现集群处于危险状态，是否继续执行升级。
    - `upgradeCheckInterval`: 升级检查的间隔时间。
    - `monitoring`: 定义集群监控规则。
    - `rulesNamespace`: 设置监控告警使用的 Prometheus 实例所在的命名空间。
    - `rules`: 设置监控告警规则，包括：
        - `alarm_actions`: 当告警发生时要执行的操作。
        - `alerts`: 具体的告警条件和设置。
    - `mgr`: 定义 Manager 组件的配置，包括 Manager 的个数和模块。
    - `dashboard`: 定义 Dashboard 组件的配置，包括是否启用 Dashboard 服务。
    - `network`: 定义网络相关的配置，包括网络类型、Selectors。
    - `rbdMirroring`: 定义 RBD Mirroring 的配置，包括 Workers 的个数。
    - `disruptionManagement`: 定义机器停止维护时的配置，包括是否开启限制、允许的机器最大损失数。
    - `removeOSDsIfOutAndSafeToRemove`: 当 OSD 可移除且没有损坏时，是否自动清除 OSD。
    
    另外，Rook 还提供更多的参数，例如配置加密、Toleration、PriorityClass 等，详情参考官网文档。

# 3.核心算法原理和具体操作步骤以及数学公式讲解

## 3.1. Rook架构

Rook项目通过使用Kubernetes中的自定义资源Definition(CRD)机制，声明式地创建Ceph集群，通过Ceph Operator提供的管理框架，管理Ceph集群生命周期。整个流程如下图所示：

1. 安装Rook Operator，创建一个新的命名空间rook-ceph
2. 创建一个新的custom resource definitions: Rook根据自定义的资源定义，创建三个新的crd对象，分别是CephCluster、CephBlockPool、CephFileStore。
3. 用户提交一个CephCluster资源定义，描述所需的Ceph集群的大小，网络设置等信息。
4. Kubernetes通过该资源的spec来启动一个Operator的controller，通过调用Ceph Client library向Ceph Agent发送命令，创建一个新的Ceph集群。
5. 管理员可以检查集群的状态、创建或删除池、文件系统等，通过CephDashboard查看集群的运行情况。

![](https://img.serverlesscloud.cn/202181/1629515924393-%E5%AF%BC%E8%AE%BA%E6%96%87%E6%A1%A3%E5%9B%BE%E7%89%87.jpg)

## 3.2. 集群状态监控

Rook项目中提供了对集群状态监控的支持，通过Prometheus收集集群的监控指标，通过Grafana进行可视化展示。

Rook Operator会根据用户的配置，创建Prometheus和Grafana实例，Prometheus会定期采集Ceph Agent发送给Rook Operator的监控指标，然后存入监控存储。Grafana则会读取监控存储的数据，按照用户的配置绘制图表进行展示。

## 3.3. 存储池

Rook项目支持两种类型的存储池，分别是RBD Pool和CephFS Filesystem。

### 3.3.1. RBD Pool

Rados Block Device（RBD）是Ceph内部的块设备，它类似于传统的磁盘。Rook通过Ceph CSI driver提供基于RBD的块存储卷，用户可以很容易地创建和使用RBD Pool。

创建一个RBD pool，需要定义如下参数：

1. poolName: 自定义的pool名称。
2. failureDomain: 定义pool的可用区，通过CRUSH算法分配数据。
3. replicated: 表示数据是否复制到所有可用区，true表示复制，false表示不复制。
4. mirroring: 表示数据是否镜像到所有可用区，true表示镜像，false表示不镜像。
5. numOfMirrors: 表示镜像的副本数量，如果是2即表示每个池中的数据有2份副本，其它副本只保留一份，当出现故障时另一份副本可以补充数据。
6. PGNum: 表示pool中每个PG的数量，PG为Pool Group的缩写，一般默认为32。
7. PGPow: 表示pool中PG的复制因子，一般默认为2即32个副本，每一个副本都会包含一个PG。
8. maxSize: 表示pool中最大可使用的空间，默认是0，表示没有限制。
9. minSize: 表示pool中最小可使用的空间，默认是0，表示没有限制。

```yaml
apiVersion: ceph.rook.io/v1
kind: CephBlockPool
metadata:
  name: replicapool
spec:
  failureDomain: host
  replicated:
    size: 3
    requireSafeReplicaSize: true
  mirroring:
    mode: image
    enabled: true
    interval: 0
  compressionMode: none
  parameters:
    replica_size: "3"
    protection_level: "none"
```

### 3.3.2. CephFS 文件系统

Ceph File System（CephFS）是Ceph集群上的文件系统，它与Linux的文件系统不同，因为CephFS把文件系统数据分布到集群中不同的节点，并提供分布式读写操作。

创建一个CephFS文件系统，需要定义如下参数：

1. name: 自定义的CephFS文件系统名称。
2. metadataPool: 元数据池，用来存储文件的元数据。
3. dataPools: 数据池列表，用于存储文件的内容。
4. preservePoolsOnDelete: 删除CephFS文件系统后，是否保留元数据池和数据池。

```yaml
apiVersion: ceph.rook.io/v1
kind: CephFileSystem
metadata:
  name: myfs
spec:
  metadataPool:
    replicated:
      size: 3
  dataPools:
    - failureDomain: root
      replicated:
        size: 3
        requireSafeReplicaSize: true
        stretchThreshold: ""
  preservePoolsOnDelete: false
```

# 4.具体代码实例和解释说明

Rook的架构和功能实现主要依赖于以下几部分：

1. Custom Resources Definitions: Kubernetes基于Custom Resources定义自己的资源对象，Rook借助CRD可以声明式地创建和管理Ceph集群。
2. Ceph Cluster Controller: Rook基于Ceph的原生API和客户端库实现了一套Ceph Cluster Controller，Ceph Cluster Controller负责跟踪CRD中定义的Ceph集群的变化，包括集群成员变更、扩容、缩容等操作。
3. Ceph Cluster Agent: 每个Ceph节点都会运行一个Ceph Cluster Agent，它负责执行集群的管理任务。
4. Ceph Agent Proxies: 在每个节点上，Rook启动一个ceph-agent-x二进制文件，它是一个守护进程，运行在节点上，等待Ceph Cluster Agent发送命令。
5. Storage Provider Interface: Rook抽象出了一套标准的Storage Provider Interface，用户可以基于该接口实现自己的Ceph存储Provider，目前Rook支持CephFS、RBD和纠删码功能。
6. Flexible Placement Groups: 默认情况下，Rook会在部署时，根据硬件资源分配PG数量，使用户能够灵活调整部署方案，提供更好的容错能力。

## 操作步骤

1. 在Kubernetes集群上安装Rook Operator

```shell
$ git clone https://github.com/rook/rook.git
$ cd rook/cluster/examples/kubernetes/ceph
$ kubectl create -f common.yaml  // 先创建common.yaml，它包含Rook Operator的RBAC权限，镜像等资源定义；
$ kubectl create -f crds.yaml  // 再创建crds.yaml，它包含Ceph Cluster的CRD定义；
$ kubectl create -f operator.yaml  // 创建operator.yaml，它包含Rook Operator的Deployment和Service定义。
```

2. 创建一个CephCluster资源定义

```yaml
apiVersion: ceph.rook.io/v1
kind: CephCluster
metadata:
  name: rook-ceph
  namespace: rook-ceph
spec:
  dataDirHostPath: /var/lib/rook
  mon:
    count: 3
  dashboard:
    enabled: true
  network:
    provider: flannel
  storage:
    useAllNodes: true
    useAllDevices: true
    deviceFilter: "lvm-ssd"
```

3. 检查集群状态

```shell
$ kubectl get pod,svc -n rook-ceph --all-namespaces 
NAMESPACE        NAME                                                  READY   STATUS      RESTARTS   AGE
rook-ceph        rook-ceph-crashcollector-myhost-8486ccdf-rkvqv       1/1     Running     0          2m19s
rook-ceph        rook-ceph-crashcollector-myhost-8486ccdf-wvvqh       1/1     Running     0          2m19s
rook-ceph        rook-ceph-crashcollector-myhost-8486ccdf-zjjqw       1/1     Running     0          2m19s
rook-ceph        rook-ceph-detect-version-6ffbf5cb68-nhmkh            1/1     Running     0          3m2s
rook-ceph        rook-ceph-mgr-a-6b7f8fc6bc-blvkp                     1/1     Running     0          2m24s
rook-ceph        rook-ceph-mon-a-56d67854f-jntc6                      1/1     Running     0          2m21s
rook-ceph        rook-ceph-mon-b-5fd9f47dd8-fggs7                     1/1     Running     0          2m15s
rook-ceph        rook-ceph-mon-c-6cd7f8cfdc-tl6hg                     1/1     Running     0          2m9s
rook-ceph        rook-ceph-operator-688c7dc5bb-2fpdr                  1/1     Running     0          3m2s
rook-ceph        rook-discover-bnzh9                                   1/1     Running     0          3m2s

NAMESPACE   NAME                            TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)   AGE
default     kubernetes                      ClusterIP   10.96.0.1        <none>        443/TCP   3m41s
kube-system nginx-ingress-microk8s-service   LoadBalancer   10.152.183.77    192.168.1.11   80:31207/TCP,443:30709/TCP   2m20s
rook-ceph   rook-ceph-mgr                  ClusterIP   10.109.207.93    <none>        9283/TCP  2m24s
rook-ceph   rook-ceph-mgr-dashboard         NodePort    10.103.248.69    <none>        7000:31341/TCP                2m24s
rook-ceph   rook-ceph-nfs                  ClusterIP   10.107.78.190    <none>        2049/TCP  2m21s
rook-ceph   rook-ceph-rgw-myhost           ClusterIP   10.110.64.205    <none>        80/TCP    2m21s
```

4. 创建一个RBD Pool

```yaml
apiVersion: ceph.rook.io/v1
kind: CephBlockPool
metadata:
  name: replicapool
spec:
  failureDomain: host
  replicated:
    size: 3
    requireSafeReplicaSize: true
  mirroring:
    mode: image
    enabled: true
    interval: 0
  compressionMode: none
  parameters:
    replica_size: "3"
    protection_level: "none"
```

5. 检查RBD Pool状态

```shell
$ kubectl get CephBlockPool
NAME        AVAIL    USED     CAPACITY   STATUS
replicapool 35 GiB   10 KiB   37 GiB     HEALTH_OK
```

6. 创建一个CephFS文件系统

```yaml
apiVersion: ceph.rook.io/v1
kind: CephFileSystem
metadata:
  name: myfs
spec:
  metadataPool:
    replicated:
      size: 3
  dataPools:
    - failureDomain: root
      replicated:
        size: 3
        requireSafeReplicaSize: true
        stretchThreshold: ""
  preservePoolsOnDelete: false
```

7. 检查CephFS文件系统状态

```shell
$ kubectl get CephFileSystem
NAME   SIZE   AVAILABLENESSPCT   PHASE   MESSAGE
myfs   16Ti   99%                Ready  
```

# 5.未来发展趋势与挑战

## 5.1. 兼容性

Rook项目完全兼容Kubernetes平台，它可以部署在任何支持Kubernetes的环境中，包括公有云、私有云、混合云等。目前，Rook已成功部署在亚马逊的EKS、腾讯的TKE、微软的AKS、百度的CCE、OpenShift等环境中。

## 5.2. 功能完善

Rook目前已经覆盖了存储类的基本需求，如动态配置、存储池管理、数据持久化等，正在迭代和丰富新功能。例如，Rook希望实现以下功能：

1. 卷快照功能：目前Rook只支持块存储的卷快照功能，对于文件系统的快照功能尚未支持。
2. 全量迁移功能：Rook计划实现全量迁移功能，帮助用户完成数据迁移过程，并完成数据完整性验证。
3. 活跃数据识别：Rook在存储池中支持存储数据的活跃度测评，以识别出业务流量密集的数据，提升存储性能。
4. 细粒度监控：Rook收集和显示存储池和集群级别的详尽监控，包括集群性能指标、PG、OSD、对象存储和区域分布指标等。
5. 边缘计算友好：Rook希望兼顾边缘计算场景的需求，提供分布式云存储网格系统，为边缘应用提供快速可靠的数据存储能力。

## 5.3. 可靠性

Rook团队一直致力于确保Rook产品质量，包括高可用性、自动恢复能力和完备的文档、测试、集成和发布体系。Rook社区也在不断壮大，拥有活跃的开发者群体，欢迎大家加入Rook社区一起参与Rook的贡献！

# 6.附录常见问题与解答

## 6.1. Q：为什么要使用Rook？

**A:** Rook项目是由CoreOS、RedHat、SUSE等著名公司和组织推出的开源项目，目标是为Kubernetes提供一种简单、可靠、一致的存储编排方案。Kubernetes已经成为容器编排领域的领头羊，越来越多的企业和组织选择将Kubernetes用于分布式存储的管理，但Kubernetes本身的存储编排仍然较为复杂，使用传统存储插件无法满足需求。Rook使用CRD模式，简化了分布式存储的部署、管理和扩展，使得分布式存储管理变得十分便捷和直观。

