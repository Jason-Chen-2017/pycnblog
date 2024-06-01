
作者：禅与计算机程序设计艺术                    

# 1.简介
  


容器技术（Containerization）和容器编排工具（Orchestration Tools）正在成为IT界的热门话题。无论是在微服务架构、云计算领域还是在基于容器的应用部署上，都需要容器技术来实现自动化和弹性伸缩。容器编排工具则可以帮助容器集群管理者管理复杂的容器化应用程序的生命周期，例如动态分配资源、调度容器、监控健康状态等。Kubernetes 是一个开源的容器编排系统，它提供了方便的接口和工具来快速部署和管理容器化的应用，并提供强大的容错和自愈能力。

本文将以初级读者的角度，全面介绍Kubernetes以及相关的概念和术语，并通过具体的实例讲解其使用方法和原理。希望能够帮助到读者更好地理解和掌握 Kubernetes 及其周边技术，从而利用其丰富的特性帮助企业构建更加复杂的分布式系统。

本文适合作为阅读和学习Kubernetes技术的基础课程，也可以作为技术作者的博客文章发布出来。

# 2.基本概念术语说明

## 2.1 Kubernetes概述
Kubernetes 是由 Google 团队开发的开源容器编排系统，其主要功能包括：

1. 集群管理：Kubernetes 可以很容易地进行集群部署、扩容、缩容、升级、故障转移等。
2. 服务发现和负载均衡：Kubernetes 提供基于 DNS 的服务发现和负载均衡，使得容器化的应用可以方便地对外提供服务。
3. 存储编排：Kubernetes 提供了简洁易用的 API 来动态配置和管理存储卷，比如持久化存储、共享存储等。
4. 滚动更新和金丝雀发布：Kubernetes 可以让应用滚动发布，即逐步更新应用的多个实例，并且支持同时运行多个版本的应用，以提升应用的鲁棒性和安全性。
5. 密钥和配置管理：Kubernetes 可以集中管理应用的配置文件和密钥信息，并且提供 API 和界面来管理这些信息。
6. 资源监控和日志收集：Kubernetes 提供完善的资源监控和日志收集功能，帮助运维人员掌握集群的实时状态，并找出异常情况。
7. 自动扩展：Kubernetes 支持根据应用的实际负载自动扩展 Pod 数量，并且可以控制 Pod 在集群中的分布方式。

## 2.2 Kubernetes架构

Kubernetes 集群由以下几个主要组件组成：

- Master节点
  - kube-apiserver：暴露 RESTful API，接受各种请求并响应。
  - etcd：集群的数据存储和交互服务，用于保存集群数据的键值对。
  - kube-scheduler：Pod 的调度器，选择节点并为它们安排工作。
  - kube-controller-manager：执行控制器任务，比如副本控制器、Endpoints 控制器、Namespace controller等。
- Node节点
  - kubelet：每个节点上的代理，负责维护 Pod 及 Pod 中容器的生命周期。
  - kube-proxy：为 Service 提供外部访问的代理。
  - container runtime：用于运行 Docker 或 rkt 等容器引擎。


## 2.3 Kubernetes对象模型
Kubernetes 对象模型是 Kubernetes 用来描述集群中各项资源的抽象模型。Kubernetes 中定义了五种最重要的对象类型：

- Pod：一个或多个容器（可选），共享网络空间和文件系统。
- Deployment：声明式的部署，允许用户创建和管理多个相同 Pod 的实例。
- ReplicaSet：提供声明式的副本控制。
- Service：提供单个或多个Pod之间的负载均衡。
- Namespace：提供虚拟隔离，可以把集群分割成独立的命名空间。

其中，Pod 是 Kubernetes 中最小的工作单元，可以看作是一组具有相同 IP 地址和端口集合的容器，以及这些容器的存储卷。

下图展示了 Kubernetes 对象模型的层次结构：


除了以上介绍的资源对象之外，Kubernetes 还定义了一套 CRD（Custom Resource Definition）机制，允许用户创建自定义资源对象。

## 2.4 Kubernetes的术语

### 2.4.1 Kubelet
Kubelet 是 Kubernetes 中负责管理节点上所有容器的组件。kubelet 从 api-server 获取 podSpecs，根据podSpecs生成对应的容器，然后直接启动运行容器，且kubelet可以根据pod中指定的restartPolicy来决定容器何时重新启动。当某个容器不再处于 Running 或者 Stopped 状态时，kubelet 将会重启该容器。

### 2.4.2 Kube-Proxy
Kube-Proxy 是 Kubernetes 中负责Service负载均衡的组件。当创建Service时，kube-api会接收到Service的定义，然后根据service的spec的内容生成ClusterIP类型的service，这个ClusterIP的service将会注册到kube-proxy的监听列表中，kube-proxy会监视service和endpoint的变化，并对访问该service的客户端做负载均衡。

### 2.4.3 Kubernetes Controller
Kubernetes 中的控制器是一种运行在主服务器上的守护进程，通过调用 Kubernetes API 创建、修改、删除或其他方式处理 Kubernetes 对象的特殊化逻辑。控制器将为集群中当前状态和期望状态之间存在的差异进行自我修复，并确保集群始终保持稳定。Kubernetes 控制器可以分为两类，集群范围的控制器和名称空间范围的控制器。

### 2.4.4 ReplicationController / ReplicaSet
ReplicationController 和 ReplicaSet 是 Kubernetes 中用来保证应用高可用性的资源。ReplicationController 通过检查指定数量的 pod 是否健康，来保证应用的正常运行；ReplicaSet 则进一步封装 ReplicationController，提供更高级别的策略，如多副本的需求。

### 2.4.5 DaemonSet
DaemonSet 是 Kubernetes 中用来给所有Node上运行特定应用的资源。当有新节点加入集群时，DaemonSet 会自动将应用部署到新的节点上。DaemonSet 一般用于运行日志收集、监控等集群共用任务。

### 2.4.6 Job
Job 是 Kubernetes 中用来批处理任务的资源。Job 以批量的方式一次性完成一系列任务，并管理任务的执行，直到任务成功完成或者失败。

### 2.4.7 ServiceAccount
ServiceAccount 是 Kubernetes 中用来代表用户或服务账号的资源。当创建一个 Pod 时，可以通过 spec.serviceAccountName 指定使用的 ServiceAccount 。

### 2.4.8 ConfigMap/Secret
ConfigMap 和 Secret 分别是用来保存非敏感数据和敏感数据的资源。ConfigMap 可以保存少量的配置信息，而 Secret 可以保存证书、密码等敏感数据。

### 2.4.9 Label
Label 是 Kubernetes 中用来标记对象属性的键值对。通过标签可以为对象指派元数据标签，可以方便地实现对象管理、查询和分类。

### 2.4.10 Annotations
Annotations 也是用来保存额外信息的键值对。注解不会被直接处理，仅提供元数据信息。

### 2.4.11 Ingress
Ingress 是 Kubernetes 中用来给 Service 提供外部访问入口的资源。Ingress 根据访问路径，使用不同的方式（如轮询、权重、url路由）将流量导向后端的 Service 。

# 3.核心算法原理和具体操作步骤以及数学公式讲解

## 3.1 Kubernetes的基本架构

Kubernetes 是一款开源的容器编排软件，它基于谷歌公司基于Borg系统的生产环境容器化管理系统进行设计开发，目前已经被广泛应用于阿里巴巴、腾讯、美团、百度、网易、58同城等互联网企业内部的容器集群管理系统。Kubernetes 核心功能包括：

1. **集群管理**：通过 Master 节点对整个集群进行管理和调度。
2. **资源调度**：实现容器跨主机集群资源的有效调度和管理。
3. **应用部署**：支持多种方式来部署和管理容器化应用，如 ReplicationController、ReplicaSet、Deployment、StatefulSet、DaemonSet 等。
4. **服务发现与负载均衡**：支持内置 DNS 服务和客户端负载均衡，通过统一的服务发现和负载均衡管理集群内的所有服务。
5. **存储编排**：提供 PersistentVolume 资源来预留存储，提供 StorageClass 资源来实现动态供应。
6. **密钥和配置管理**：支持对敏感信息进行加密保护，提供 Secret 资源来保存和管理凭证。
7. **资源监控**：提供统一的集群资源监控，包括 CPU、内存、磁盘、网络等的监控。
8. **自动伸缩**：支持对应用的横向和纵向自动扩展，通过 HPA（Horizontal Pod Autoscaling）来实现 CPU 使用率的自动扩展。

## 3.2 Kubernetes的核心组件

Kubernetes 的架构分为三个主要组件，分别是：

1. **Master 组件**：由 Master 节点管理整个集群，包括 API Server、Scheduler、Controller Manager 等。
2. **Node 组件**：在每台机器上都要安装的组件，包括 Kubelet、Kube-Proxy、Container Runtime。
3. **Pod 组件**：组成 Kubernetes 集群的最小工作单位，包含一组紧密耦合的容器。

### Master 组件

**API Server**：API Server 是 Kubernetes 集群的中心节点，所有的操作请求都会首先经过 API Server，因此，它是整个 Kubernetes 系统的关键组件之一。API Server 负责对各类资源的增删改查，并提供统一的认证、授权、访问控制和准入控制等功能，保证整个系统的安全性和稳定性。

**Scheduler**：Scheduler 是 Kubernetes 的资源调度器，负责为新建的 Pod 分配合适的 Node 资源。Scheduler 对待调度的 Pod 有两种处理方式，第一种是普通调度（Normal Scheduling），第二种是优先调度（Preemptive Scheduling）。

普通调度就是为新创建的 Pod 分配资源，直到资源空闲，这也是默认的调度方式。而优先调度就是当资源紧张的时候，将部分最不活跃的 Pod 释放资源，为较为活跃的 Pod 分配资源，实现高效利用资源。

**Controller Manager**：Controller Manager 是 Kubernetes 集群中最核心的组件之一，它管理着 Kubernetes 集群的一些核心控制循环，比如 Node 控制器、Replication 控制器、Endpoint 控制器、Namespace 控制器等，确保集群中所有的资源按照预期的状态运行。

### Node 组件

**Kubelet**：每个节点都要安装的组件，负责运行属于自己的 Pod。每个 Node 上只能有一个 Kubelet 运行，它的作用是根据 Master 发来的指令将 PodSpecs 文件转换为真正的运行实例。

**Kube-Proxy**：它是一个网络代理，用来为 Service 提供 TCP/UDP 连接负载均衡，实现 Service 的内部连接和外部访问。

**Container Runtime**：它是 Kubernetes 集群中容器运行环境的一种，目前支持 Docker 和 Rocket 等多种容器运行时。Containerd、CRI-o、frakti 等项目正在跟 Kubernetes 社区合作，力争取代 Docker 成为容器标准运行时。

### Pod 组件

**容器组 (Pod)**：一个 Pod 可以包含多个容器，但通常情况下，它们应该是紧密耦合的——有关联的业务逻辑和共享存储，这样可以更好地利用资源。Pod 是 Kubernetes 中最小的可管理工作单元，由一个或者多个容器组成，共享网络空间和存储卷。一个 Pod 可以是短暂的，也可长期运行，Pod 中的容器共享 IPC、UTS、NET 命名空间，可以进行IPC通信。

## 3.3 集群初始化

kubernetes 安装后，需要首先对 kubernetes 集群进行初始化，包括 master 节点和 node 节点的初始化工作。master 初始化工作主要包括部署 kubernetes API server、kubernetes scheduler、kubernetes controller manager 三个组件，node 初始化工作主要包括部署 kubelet、kube-proxy 两个组件。为了简单起见，这里只简要介绍 master 节点的初始化过程。

首先，master 需要配置一个配置文件，它记录了 apiserver、etcd、scheduler 等组件的配置信息。这些信息一般存放在 `/etc/kubernetes/` 下的一个叫 `config.yaml` 的文件中。

然后，master 通过命令行工具 `kubeadm init` 来启动 kubernetes master 组件。这条命令会读取配置文件中的信息，完成如下几件事情：

1. 初始化 kubernetes 数据库 etcd，用来存储集群中各类资源的信息。
2. 生成 apiserver 的证书文件和私钥文件，用来对接外部客户端，比如 kubectl 命令行工具。
3. 生成 kube-scheduler、kube-controller-manager、kubelet 的证书文件和私钥文件，用来对各类组件之间的身份验证。
4. 用刚才生成的证书文件启动 kubernetes API server 和 scheduler 等组件。
5. 给予 kubelet root 用户权限，以便它可以直接修改宿主机的网络配置。

至此，master 节点初始化工作就结束了，集群的各类组件已经正常启动，可以使用 `kubectl get nodes` 命令查看集群中的 node 节点信息。

## 3.4 创建 pod

创建 pod 之前，首先需要有一个运行的 docker 容器镜像，可以使用 `docker pull nginx` 来拉取一个 nginx 镜像。然后，创建一个名为 `nginx-pod.yaml` 的 yaml 配置文件，内容如下所示：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-nginx
spec:
  containers:
  - name: nginx
    image: nginx:latest
    ports:
    - containerPort: 80
      hostPort: 80
    volumeMounts:
    - mountPath: "/usr/share/nginx/html"
      name: html
  volumes:
  - name: html
    emptyDir: {}
```

以上内容描述了一个名字为 "my-nginx" 的 pod，它包含一个名为 "nginx" 的容器，使用的镜像为 "nginx:latest"，容器的端口映射关系为本机的 80 端口映射到容器的 80 端口。

这个 YAML 文件定义了一个 pod，它包含一个 container，名为 "nginx", 使用的镜像为 "nginx:latest"。这个容器会在宿主机的 80 端口暴露一个服务。这个容器挂载了一个名为 "html" 的 emptyDir 卷，意味着这个卷为空，也就是说，这个容器的根文件系统上没有任何内容。

现在，可以使用 `kubectl create -f nginx-pod.yaml` 来创建这个 pod。如果创建成功，会得到一个类似下面的输出：

```bash
pod/my-nginx created
```

然后，可以使用 `kubectl describe pods my-nginx` 来获取 pod 的详细信息，包括 pod 的 IP 地址、所在节点等信息。

最后，可以使用 `minikube service my-nginx --url` 来打开浏览器访问这个 web 服务。

## 3.5 集群扩展

现在，我们已经有一个 pod 运行在集群中，但是这个集群只有一个节点，如果想让集群具备水平扩展的能力，那么就需要增加更多的节点。

为了增加集群的规模，可以采用手动的方式，每次新增一个节点之后，将其加入到现有的集群中即可，但是这种方式比较麻烦。更好的方式是使用自动化的集群管理工具，自动实现节点的增减。

kubernete 提供了一个名为 `kubectl scale` 的命令来实现集群扩展，它可以在不停止 pod 的情况下动态增加或减少 pod 数量。

例如，可以使用如下命令增加集群中的 worker 节点：

```bash
kubectl scale deployment my-nginx --replicas=2
```

这条命令会将 my-nginx 这个 deployment 的副本数设置为 2。

另外，使用 `kubectl get deployments` 可以看到 deployment 当前的状态：

```bash
NAME      DESIRED   CURRENT   UP-TO-DATE   AVAILABLE   AGE
my-nginx   2         2         2            2           1m
```

这里可以看到，deployment 中要求的副本数已经达到了 2。

## 3.6 服务发现和负载均衡

Kubernetes 提供了一个统一的服务发现和负载均衡解决方案——Service。在创建 Service 对象时，可以指定 Service 的类型，可以是 ClusterIP、NodePort 等。

- ClusterIP：通过 ClusterIP 来实现服务的内部访问。
- NodePort：通过 NodePort 来实现外部访问。
- LoadBalancer：通过 LoadBalancer 实现公网访问。

以下示例中，创建一个 ClusterIP 的服务，名字为 "my-nginx"，目标端口为 80：

```yaml
apiVersion: v1
kind: Service
metadata:
  name: my-nginx
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 80
  selector:
    app: my-nginx
```

在上面的例子中，设置的是目标端口为 80，这是因为 Kubernetes 默认会将容器暴露的端口号设置为目标端口。同时，设置了选择器，它可以匹配到该服务相关的 pod。

创建这个 Service 对象：

```bash
kubectl apply -f my-nginx-svc.yaml
```

创建成功后，可以使用 `minikube ip` 命令获取 minikube 集群的 IP 地址，然后在浏览器输入 `<minikube_ip>:<nodeport>` 就可以访问这个 web 服务了。

除此之外，Kubernetes 提供了更多的服务发现和负载均衡的选项，包括基于域名的 Ingress、Ingress Class、Ingress Controller 等。

# 4.具体代码实例和解释说明

通过前面的介绍和示例，读者应该对 Kubernetes 有了一定的了解。下面，我们来看一下 Kubernetes 的编程接口及其用法。

## 4.1 Java API

对于 Kubernetes 的编程接口，我们可以选择 Java API。目前，Kubernetes 官方提供了 Java Client，它基于 Swagger 规范生成的 Java 类，通过调用这些类的方法，可以实现对 Kubernetes 集群的各种操作，包括创建、删除、扩缩容 Pod、管理 Deployment 等。

下面，我们来看一下如何使用 Java API 来创建 pod、部署 pod、扩缩容 pod。

### 4.1.1 创建 pod

创建一个名为 "nginx-pod.yaml" 的 YAML 配置文件，内容如下所示：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-nginx
spec:
  containers:
  - name: nginx
    image: nginx:latest
    ports:
    - containerPort: 80
      hostPort: 80
    volumeMounts:
    - mountPath: "/usr/share/nginx/html"
      name: html
  volumes:
  - name: html
    emptyDir: {}
```

创建 pod 的代码如下：

```java
// 初始化 client
ApiClient apiClient = new ClientBuilder().setBasePath("http://localhost:8080").build();
Configuration.setDefaultApiClient(apiClient);
CoreV1Api coreV1Api = new CoreV1Api();

// 创建 pod
InputStream inputStream = Resources.asByteSource(new File("nginx-pod.yaml")).openBufferedStream();
coreV1Api.createNamespacedPod("default",
        Serialization.unmarshal(inputStream, V1Pod.class), null, null, null);
System.out.println("Create pod success");
```

上面代码先初始化了一个 CoreV1Api 类，然后加载了一个 nginx-pod.yaml 文件，并创建了一个名为 "my-nginx" 的 pod。由于没有指定 namespace，所以默认为 default。

### 4.1.2 部署 pod

创建一个名为 "nginx-deploy.yaml" 的 YAML 配置文件，内容如下所示：

```yaml
apiVersion: apps/v1beta2 # for versions before 1.9.0 use apps/v1beta1
kind: Deployment
metadata:
  name: nginx-deployment
  labels:
    app: nginx
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:latest
        ports:
        - containerPort: 80
          hostPort: 80
        volumeMounts:
        - mountPath: "/usr/share/nginx/html"
          name: html
      volumes:
      - name: html
        emptyDir: {}
```

上面是一个典型的 Deployment 配置文件。其中，spec.selector 和 spec.template 的 labels 需要匹配才能将 pod 绑定到这个 Deployment 上。

部署 pod 的代码如下：

```java
// 初始化 client
ApiClient apiClient = new ClientBuilder().setBasePath("http://localhost:8080").build();
Configuration.setDefaultApiClient(apiClient);
AppsV1beta2Api appsV1beta2Api = new AppsV1beta2Api();
ExtensionsV1beta1Api extensionsV1beta1Api = new ExtensionsV1beta1Api();
BatchV1Api batchV1Api = new BatchV1Api();

// 创建 deployment
InputStream inputStream = Resources.asByteSource(new File("nginx-deploy.yaml")).openBufferedStream();
appsV1beta2Api.createNamespacedDeployment("default",
        Serialization.unmarshal(inputStream, V1beta2Deployment.class), null, null, null);
System.out.println("Deploy pod success");
```

创建 Deployment 时，注意 apiVersion 要设置为正确的值，否则可能会导致创建失败。

### 4.1.3 扩缩容 pod

扩缩容 pod 的代码如下：

```java
String namespace = "default";
String deploymentName = "nginx-deployment";
Integer replicas = 2; // 设置新的副本数
extensionsV1beta1Api.patchNamespacedDeploymentScale(deploymentName, namespace,
        new Patch().addToObject("spec", new JSONObject().put("replicas", replicas)));
System.out.println("Scaling pod to " + replicas + " success");
```

上面代码先获得了命名空间、Deployment 名称和新的副本数，然后调用 patch 方法修改 Deployment 的副本数。

## 4.2 Python API

Kubernetes 的官方提供了 Python Client。对于 Python 程序员来说，Python Client 更加友好，可以使用更高级的语法特性，更方便地编写和维护程序。

下面，我们来看一下如何使用 Python API 来创建 pod、部署 pod、扩缩容 pod。

### 4.2.1 创建 pod

```python
from kubernetes import config, client
import os

# 初始化 client
config.load_incluster_config() # this loads the inside cluster configuration
v1 = client.CoreV1Api()
namespace = 'default'

# 创建 pod
with open('nginx-pod.yaml', 'r') as f:
    dep = client.ApiClient().deserialize(yaml.safe_load(f), client.V1Pod)
    v1.create_namespaced_pod(body=dep, namespace=namespace)
print('Create pod successfully.')
```

上面代码通过 Kubernetes API 访问集群，读取了本地目录下的 nginx-pod.yaml 文件，并反序列化为 V1Pod 对象，创建了一个名为 "my-nginx" 的 pod。由于没有指定 namespace，所以默认为 default。

### 4.2.2 部署 pod

```python
from kubernetes import config, client
import os

# 初始化 client
config.load_incluster_config() # this loads the inside cluster configuration
appsv1b2 = client.AppsV1beta2Api()
batchv1 = client.BatchV1Api()
extv1b1 = client.ExtensionsV1beta1Api()

# 创建 deployment
with open('nginx-deploy.yaml', 'r') as f:
    deploy = client.ApiClient().deserialize(yaml.safe_load(f), client.V1beta2Deployment)
    extv1b1.create_namespaced_deployment(body=deploy, namespace="default")
print('Deploy pod successfully.')
```

上面代码读取本地目录下的 nginx-deploy.yaml 文件，并反序列化为 V1beta2Deployment 对象，创建了一个名为 "nginx-deployment" 的 Deployment。

### 4.2.3 扩缩容 pod

```python
from kubernetes import config, client
import json

# 初始化 client
config.load_incluster_config() # this loads the inside cluster configuration
extensionsv1beta1 = client.ExtensionsV1beta1Api()

# 修改 Deployment 副本数
name = "nginx-deployment"
namespace = "default"
new_replicas = 2
result = extensionsv1beta1.patch_namespaced_deployment_scale(name=name, namespace=namespace, body={"spec": {"replicas": new_replicas}})
print('Scaling pod from %d to %d successfully.' % (len(result['status']['replicas']), new_replicas))
```

上面代码通过扩展 API 修改了 "nginx-deployment" 的副本数。

# 5.未来发展趋势与挑战

Kubernetes 一直是开源社区的一个热点话题，其热度源自以下几方面：

1. 业界一片红日，Kubernetes 是继 Hadoop 之后又一款颠覆性的技术革命。
2. 大量云服务厂商都加入到了 Kubernetes 的阵营，部署 Kubernetes 集群成为了大势所趋。
3. Kubernetes 的社区壮大，生态日渐成熟，各种解决方案也纷纷涌现。

不过，Kubernetes 仍然是一个比较年轻的系统，它还有很多潜在的增长空间。下面，我们列举几个可能的方向和挑战：

1. 多云支持：Kubernetes 目前只支持一家云厂商，它的架构天生就不是多云模式的架构。未来，Kubernetes 肯定会面临多云支持的挑战。
2. 深度监控：目前，Kubernetes 缺乏一个足够细致的监控系统，这将阻碍其监控深度和广度。未来，Kubernetes 一定会建设一个基于 Prometheus 的监控体系。
3. 弹性伸缩：Kubernetes 的弹性伸缩能力非常有限，只能进行自动扩缩容，无法做到精细化的弹性。未来，Kubernetes 肯定会努力打造一款能够满足复杂业务场景的弹性伸缩系统。
4. 服务网格：虽然 Kubernetes 最早的时候就引入了 Service Mesh 技术，但它的实施落地却遇到种种困难，最终变得僵死。未来，Kubernetes 将会建设一整套完整的服务网格解决方案，包括服务发现、负载均衡、流量控制、安全、可观测性等。
5. 云原生应用：由于 Kubernetes 自己定位为云管控平台，其运行环境都是 IaaS。未来，Kubernetes 将会推出云原生应用管理平台，其运行环境由 K8s 之外的其它平台提供。

总的来说，Kubernetes 的未来是一个充满挑战的一件事情，需要一场旷日持久的探索、研究和试验。