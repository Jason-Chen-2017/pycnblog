
作者：禅与计算机程序设计艺术                    

# 1.简介
  

容器技术由于其轻量级、快速部署和高度可移植性等特点，越来越受到各行各业的青睐。作为虚拟化技术的一种变体，容器技术可以将应用程序运行环境打包成一个独立的容器并通过标准的接口与其他容器共享资源。由此带来的好处之一便是可以同时运行多个相同或相似任务的容器，大幅提升了资源利用率和执行效率。近年来，基于容器技术的云计算平台也逐渐成为市场需求。本文以Kubernetes和Docker为代表的云平台工具进行介绍，并以面向开发者的视角，分享我在搭建和运用这些工具时所遇到的一些问题及解决方案。本文目标读者群体为对云平台有相关了解和兴趣的技术人员。
# 2.基础知识背景
## 什么是Kubernetes？
Kubernetes（K8s）是一个开源的系统用于管理云容器集群的软件，它提供一个分布式的、开放源码的平台，让容器ized应用可以自动部署、扩展和管理。其功能包括跨主机集群管理、自动调度以及自我修复，还提供基于声明式API的抽象层，帮助开发者和管理员快速建立自己的集群。Kubernetes由Google公司倡导，于2015年4月开源发布。
## 什么是Docker？
Docker是一个开源的应用容器引擎，基于Go语言实现。它允许用户打包他们的应用以及依赖包到一个轻量级、可移植的容器中，然后发布到任何流行的 Linux 或 Windows 机器上，也可以实现虚拟化。Docker 的 container 是完全使用沙箱机制，相互之间不会相互影响，所以更加安全可靠。Docker 的设计理念是一切皆为文件的初衷，促进了微服务架构模式。目前国内的许多大型互联网公司都已经开始采用 Docker 来进行内部的持续集成、持续交付工作。
## 为什么要使用Kubernetes+Docker？
Kubernetes+Docker的组合可以极大地提高云平台的效率、降低运营成本，为客户节省大量的时间和精力，同时增加业务敏捷度。以下是它的主要优点：

1. 可扩展性强：Kubernetes天生具有弹性伸缩能力，无需手工扩容；
2. 健壮性高：Docker提供了一套完整的容器技术栈，降低了集群维护成本；
3. 滚动更新：可以按需进行滚动升级，减少停机时间；
4. 服务发现：支持多种服务发现机制，使得应用可靠的找到依赖服务；
5. 网络模型简单：通过pod的网络隔离，可以实现应用之间的通信；
6. 混合云兼容性：可以在私有云、公有云、混合云环境下运行，最大限度的节省资源和时间。

# 3.项目实施过程
## 3.1 安装配置kubernetes集群
### 3.1.1 配置kubernetes master节点
首先需要配置kubernetes master节点。根据kubernetes安装包下载安装文件并解压。将`/etc/kubernetes/manifests`目录下的所有yaml配置文件删除或者备份。创建配置文件`/etc/kubernetes/config.yaml`，内容如下：
```yaml
apiVersion: kubeadm.k8s.io/v1alpha1
kind: MasterConfiguration
api:
  advertiseAddress: x.x.x.x # 指定API server监听地址
  bindPort: 6443 # 指定API server端口
controllerManager: {} # 不启动控制器
networking: # 禁用网络插件，使用flannel
  podSubnet: ""
  serviceSubnet: ""
etcd:
  endpoints: http://127.0.0.1:2379 # etcd的地址和端口
  dataDir: /var/lib/etcd # 数据存放目录
```
其中，`advertiseAddress`字段指定API server监听的IP地址，如果是单Master集群的话直接设置为master节点的IP即可。`bindPort`字段指定API server使用的端口。这里选择的默认端口是6443。

### 3.1.2 初始化kubernetes集群
执行初始化命令：
```bash
kubeadm init --config=/etc/kubernetes/config.yaml
```
命令执行完成后会显示`Join tokens`，记录一下。例如：
```bash
kubeadm join --token <PASSWORD> 192.168.0.10:6443 --discovery-token-unsafe-skip-ca-verification
```
### 3.1.3 添加node节点
添加节点的操作一般由Master节点负责。首先需要修改配置文件，添加node节点信息：
```yaml
apiVersion: kubeadm.k8s.io/v1alpha1
kind: NodeConfiguration
kubeletConfig:
  clusterDNS: [10.244.0.10] # 指定DNS服务器地址
  failSwapOn: false # 设置不检查swap分区

---

apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
authentication:
  anonymous:
    enabled: true # 设置匿名访问
authorization:
  mode: Webhook # 设置授权方式为Webhook
clusterDomain: "cluster.local" # 设置集群域名
failSwapOn: false # 设置不检查swap分区
```
其中，`kubeletConfig`字段设置kubelet的配置参数，这里指定了dns服务器地址。`KubeletConfiguration`字段设置kubelet的全局配置参数，这里启用匿名访问，设置授权方式为Webhook，设置集群域名，并且关闭swap检测。

执行加入命令：
```bash
kubeadm join --token <PASSWORD> 192.168.0.10:6443 --discovery-token-unsafe-skip-ca-verification
```
等待加入成功即可。

## 3.2 安装配置docker
在所有的节点上安装docker。可以从官方网站下载安装包安装。

## 3.3 使用kubectl管理集群
kubectl是kubernetes命令行工具，用来控制kubernetes集群。可以通过这个命令管理集群。

### 3.3.1 创建Pod
可以使用yaml文件定义Pod，然后提交给kubernetes执行创建操作。比如创建一个nginx的Pod：
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nginx-test
  labels:
    app: web
spec:
  containers:
  - name: nginx
    image: nginx:latest
    ports:
    - containerPort: 80
```
然后使用如下命令创建Pod：
```bash
kubectl create -f nginx-pod.yaml
```
创建成功之后，可以使用如下命令查看Pod状态：
```bash
kubectl get pods
NAME         READY     STATUS    RESTARTS   AGE
nginx-test   1/1       Running   0          1m
```

### 3.3.2 查看节点和pods的情况
可以通过如下命令查看节点的情况：
```bash
kubectl get nodes
NAME        STATUS                     ROLES     AGE       VERSION
worker      Ready                      <none>    2h        v1.9.3
master      NotReady,SchedulingDisabled   master    2h        v1.9.3
```
可以看到当前集群有两个节点，一个是`worker`，另一个是`master`。其中`master`是待加入的节点，还没有启动kubelet。

同样的，可以通过如下命令查看Pod的情况：
```bash
kubectl get pods --all-namespaces
NAMESPACE     NAME                           READY     STATUS    RESTARTS   AGE
default       nginx-test                     1/1       Running   0          2h
kube-system   kube-apiserver-master          1/1       Running   0          2h
kube-system   kube-controller-manager-master   1/1       Running   0          2h
kube-system   kube-proxy-gvwld               1/1       Running   0          2h
kube-system   kube-scheduler-master          1/1       Running   0          2h
```
可以看到当前集群有一个`nginx-test`的Pod正在运行。

### 3.3.3 扩充节点和pods
可以通过如下命令扩充节点：
```bash
kubectl scale --replicas=3 deployment/nginx-deployment
```
可以增加`nginx-deployment`的副本数量为3。

也可以使用`kubectl edit node worker`修改节点属性。比如可以把worker的内存增加为3G：
```yaml
apiVersion: v1
kind: Node
metadata:
  annotations:
    flannel.alpha.coreos.com/backend-data: '{"VtepMAC":"c2:4d:5e:aa:b3:a3"}'
    flannel.alpha.coreos.com/public-ip: 192.168.0.11
    node.alpha.kubernetes.io/ttl: "0"
  creationTimestamp: 2018-08-07T03:49:28Z
  labels:
    beta.kubernetes.io/arch: amd64
    beta.kubernetes.io/os: linux
    kubernetes.io/hostname: worker
  name: worker
  resourceVersion: "1487"
  selfLink: /api/v1/nodes/worker
  uid: a5ebcf10-c016-11e8-b3dd-000c29b48f8f
spec:
  externalID: i-uf6ctjsy
  podCIDR: 10.244.1.0/24
  taints:
  - effect: NoSchedule
    key: node-role.kubernetes.io/master
  unschedulable: false
  configSource: null
  capacity:
    cpu: "4"
    memory: 3072Mi
    pods: "110"
  allocatable:
    cpu: "4"
    memory: 3072Mi
    pods: "110"
  phase: Online
status:
  addresses:
  - address: 192.168.0.11
    type: InternalIP
  - address: worker
    type: Hostname
  allocatable:
    cpu: "4"
    memory: 3072Mi
    pods: "110"
  capacity:
    cpu: "4"
    memory: 3072Mi
    pods: "110"
  conditions:
  - lastHeartbeatTime: 2018-08-08T08:19:32Z
    lastTransitionTime: 2018-08-07T03:49:28Z
    message: Flannel is running on this node
    reason: FlannelIsUp
    status: "True"
    type: NetworkUnavailable
  daemonEndpoints:
    kubeletEndpoint:
      Port: 10250
  images:
  - names:
      - gcr.io/google_containers/hyperkube-amd64@sha256:61e0dc5c8dbbf32ce01f18a9863cc79547cd7d7cbba3b56616c1fc06ffec6e24
      - gcr.io/google_containers/kube-apiserver-amd64@sha256:a7d9de9cf7f2cb1dd1c1fb2abfa410d46bc468368b124fc98f8b4ee000d4122d
      - gcr.io/google_containers/kube-controller-manager-amd64@sha256:7e12d20ed6908fa516cf75e2fdaf2e073407b47b2608e4372c8d1a676d5d1704
      - gcr.io/google_containers/kube-scheduler-amd64@sha256:b3e19309057e19d9fc4be574a5b715671fa18a6fb961edae21269d6c51a7b5a5
  nodeInfo:
    architecture: amd64
    bootID: d3c01cf8-bf96-4a4b-a1da-900a200b352d
    containerRuntimeVersion: docker://1.12.6
    kernelVersion: 4.4.0-134-generic
    kubeProxyVersion: v1.9.3
    kubeletVersion: v1.9.3
    machineID: f855d1d92d1b4b9eb1a780dfdd5cf4af
    operatingSystem: linux
    osImage: Ubuntu 16.04.4 LTS
    systemUUID: EC22EBA5-DFF2-BFAF-63B0-4BEC17DADCA3
```
修改完后保存退出，然后执行`kubectl apply -f `worker.yaml``命令更新节点配置。更新完成后，可以通过`kubectl describe node worker`查看节点详情。

同样的方法，可以扩充Pod的数量。比如，可以把`nginx-test`的副本数量扩充为3：
```yaml
apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  name: nginx-deployment
spec:
  replicas: 3
  template:
    metadata:
      labels:
        app: web
    spec:
      containers:
      - name: nginx
        image: nginx:latest
        ports:
        - containerPort: 80
```
修改完后保存退出，然后执行`kubectl apply -f `nginx-deployment.yaml``命令更新Pod配置。更新完成后，可以通过`kubectl describe deployment nginx-deployment`查看Pod详情。