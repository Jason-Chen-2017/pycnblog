
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Kubernetes是一个开源容器集群管理系统。它能够自动化地部署、扩展和管理应用，并且提供自我修复能力，可以有效地实现服务的高可用性。在企业内部运行Kubernetes，可以大幅度降低运维成本，提升资源利用率，同时提升集群的稳定性。

Kubernetes master 负责对集群进行维护，包括服务发现和负载均衡、监控告警、弹性伸缩等。本文将详细介绍Kubernetes master的初始化过程。

# 2. 准备环境
1. 配置 Docker 加速器（可选）
Docker 加速器主要用于加快 Docker Hub 和 Google Container Registry 的拉取速度。如果你需要的话，可以在国内配置一个 Docker 加速器，然后再从该镜像仓库中拉取镜像。

2. 安装 kubeadm、kubelet 和 kubectl
为了安装 Kubernetes master，首先要安装 Kubernetes 命令行工具 kubectl。

```bash
sudo apt-get update && sudo apt-get install -y apt-transport-https curl
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb http://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee /etc/apt/sources.list.d/kubernetes.list
sudo apt-get update
sudo apt-get install -y kubelet=1.17.0-00 kubeadm=1.17.0-00 kubectl=1.17.0-00 --allow-downgrades
```

kubeadm 是用来建立 Kubernetes 集群的命令行工具。kubelet 是 Kubernetes 中用于维护节点运行状况的组件。通过 kubeadm 可以快速地生成并配置最简单的单节点 Kubernetes 集群。

# 3. 初始化 Kubernetes master

## 3.1 设置静态 Pods 目录

为了让 kubelet 从文件系统中获取 Pods，我们需要指定 static pods 所在的目录。执行以下命令创建 `manifests` 文件夹，并修改 `kubelet` 服务配置文件。

```bash
mkdir -p /etc/kubernetes/manifests
systemctl edit --full --force kubelet.service
```

在编辑器里找到 `--pod-manifest-path=` 参数，添加如下内容：

```bash
--pod-manifest-path=/etc/kubernetes/manifests
```

保存退出后，重启 `kubelet` 服务：

```bash
systemctl restart kubelet.service
```

此时，`kubelet` 会自动创建 `/etc/kubernetes/manifests` 目录，并加载这里面的 Pod 定义文件。

## 3.2 创建 Kubernetes CA

Kubernetes 使用 TLS 来加密集群间通讯，其中 kubelet 使用证书验证自己的身份。由于这种机制需要给每个 kubelet 发放一个证书，因此首先需要创建一个 Certificate Authority (CA) 。

执行下面的命令创建 Kubernetes CA：

```bash
sudo kubeadm init phase certs all --config=/tmp/master_config.yaml
```

上面的命令会在 `/etc/kubernetes/pki` 路径下生成证书和密钥文件。

## 3.3 配置 Kubelet 服务

`kubelet` 服务可以通过配置文件或命令行参数指定启动参数，但建议通过配置文件的方式设置，这样更加方便管理。执行如下命令创建 `kubelet` 服务配置文件：

```bash
sudo vi /var/lib/kubelet/config.yaml
```

添加以下内容：

```yaml
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
authentication:
  anonymous:
    enabled: false
  webhook:
    cacheTTL: 2m0s
    enabled: true
  x509:
    clientCAFile: "/etc/kubernetes/pki/ca.crt"
authorization:
  mode: Webhook
  webhook:
    cacheAuthorizedTTL: 5m0s
    cacheUnauthorizedTTL: 30s
clusterDomain: cluster.local
cpuManagerPolicy: none
enableControllerAttachDetach: true
healthzPort: 10248
hostnameOverride: master
kubeAPIQPS: 100
kubeAPIBurst: 200
makeIPTablesUtilChains: true
maxOpenFiles: 1000000
networkPluginMTU: 1460
nodeStatusReportFrequency: 1m0s
nodeLeaseDurationSeconds: 40
port: 10250
readOnlyPort: 0
serializeImagePulls: true
tlsCertFile: "/etc/kubernetes/pki/apiserver.crt"
tlsPrivateKeyFile: "/etc/kubernetes/pki/apiserver.key"
volumePluginDir: "/usr/libexec/kubernetes/kubelet-plugins/volume/exec/"
```

上面配置了 `kubelet` 的很多选项，如认证方式、授权方式、访问地址端口号等。这些配置项可以在 `kubelet` 服务的配置文件 `kubelet.service` 或 `kubelet` 命令的参数中进行设置，具体方法根据实际情况而定。

最后，保存并关闭配置文件，并重启 `kubelet` 服务：

```bash
sudo systemctl daemon-reload
sudo systemctl restart kubelet.service
```

## 3.4 在 master 上启用 Control Plane Components

执行下面的命令启用 master 上的 Control Plane Components：

```bash
sudo kubeadm init phase control-plane all --config=/tmp/master_config.yaml
```

上面命令会部署以下组件：

* kube-apiserver：提供 Kubernetes API 服务；
* etcd：分布式键值存储数据库；
* kube-scheduler：集群中的调度器，为新建的工作负载分配资源；
* kube-controller-manager：集群中的控制器，实现不同子系统之间的逻辑；
* cloud-controller-manager（可选）：集群外部依赖，比如云厂商提供的接口。

如果 Kubernetes 版本小于等于 v1.16，还会部署以下组件：

* kube-proxy：集群中的网络代理，实现 Service 和 Pod 之间流量路由；
* kube-dns：集群内 DNS 服务；
* Dashboard：Web 可视化控制台；
* Heapster：集群性能数据采集和汇聚工具。

## 3.5 查看集群信息

执行下面的命令查看集群信息：

```bash
kubectl get componentstatuses
```

输出结果类似：

```
NAME                 STATUS    MESSAGE             ERROR
controller-manager   Healthy   ok                  
scheduler            Healthy   ok                  
etcd-0               Healthy   {"health": "true"}   
```

如果输出状态都为健康，则说明集群已经正常运行。

# 4. 添加 node

集群的动态扩容和缩容需要手动完成，因此这里不赘述。