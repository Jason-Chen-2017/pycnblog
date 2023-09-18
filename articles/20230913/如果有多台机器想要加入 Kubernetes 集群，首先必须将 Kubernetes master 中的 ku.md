
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Kubernetes 是一个开源的、分布式的容器编排调度系统。它可以管理 Docker 和 Rocket 等容器运行时引擎，并提供一个统一的操作界面和 API 让用户方便地部署和管理容器化的应用。Kubernetes 中有三个角色——Master、Node 和 Pod。

Master 是 Kubernetes 的主控节点，主要负责控制和协调整个集群的工作，包括分配资源、调度 pod、维护集群状态等；

Node 是 Kubernetes 集群中的工作节点，承载着容器的生命周期，存储着 pod 的数据卷和日志文件；

Pod 是 Kubernetes 中最小的工作单元，它是一组一个或多个容器（以及它们所需要的资源）的集合。

因此，当有多台机器想要加入 Kubernetes 集群的时候，首先必须将 Kubernetes master 中的 kubelet 服务禁止自启动，然后再在每台 worker 节点上安装 kubelet 并注册到 Master 上。这是因为 kubelet 需要依赖于 Master 才能正常工作，而 Master 在启动kubelet 时会与之进行通信，所以如果不禁止自启动的话，kubelet 将无法连接到 Master 导致集群不可用。另外，为了避免对业务造成影响，可以先在测试环境中完成以上步骤，待验证无误后再推广到生产环境中。

下面我们就来详细介绍一下具体操作步骤：

1.在 Kubernetes Master 主机上编辑/etc/systemd/system/kubelet.service.d/10-no-auto-start.conf 文件，添加以下配置项：
```
[Service]
ExecStart=
ExecStart=/bin/sleep 9999999999
```
该配置表示禁止 kubelet 自动启动。重启系统后生效。

2.将各个 Kubernetes Worker 主机上的 /usr/local/bin/kubelet 文件拷贝至 Kubernetes Master 主机上。

3.在 Kubernetes Master 主机上编辑 ~/.kube/config 文件，添加如下内容：
```
apiVersion: v1
kind: Config
clusters:
- name: local
  cluster:
    server: https://<master_ip>:<port> # 根据实际情况填写 Master IP 和端口号
    certificate-authority: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
users:
- name: kubelet
  user:
    tokenFile: /var/run/secrets/kubernetes.io/serviceaccount/token
contexts:
- context:
    cluster: local
    namespace: default
    user: kubelet
  name: kubelet-context
current-context: kubelet-context
```
其中，server 参数的值根据实际情况填写 Kubernetes Master IP 地址和端口号，certificate-authority 指向 Kubernetes CA 证书文件的位置，tokenFile 则是用来授权 kubelet 访问 Kubernetes API 的令牌文件。

4.在各个 Kubernetes Worker 主机上执行以下命令安装 kubelet，注意替换掉 <node_name> 为 Kubernetes 节点名：
```
sudo cp /usr/local/bin/kubelet /usr/local/bin/<node_name>-kubelet && sudo mv /usr/local/bin/<node_name>-kubelet /usr/local/bin/kubelet && chmod +x /usr/local/bin/kubelet
```

5.在各个 Kubernetes Worker 主机上执行以下命令注册到 Kubernetes Master：
```
sudo /usr/local/bin/kubelet --address=<worker_ip> \
                            --hostname-override=<node_name> \
                            --pod-infra-container-image=gcr.io/google_containers/pause-amd64:3.0 \
                            --cgroup-driver=<driver_type> \
                            --cluster-dns=<dns_ip> \
                            --kubeconfig=/root/.kube/config
```
参数说明：
--address=<worker_ip>：指定当前节点的 IP 地址。
--hostname-override=<node_name>：指定当前节点的名称，一般设置为节点的主机名。
--pod-infra-container-image=gcr.io/google_containers/pause-amd64:3.0：指定 Pod 内部的基础容器镜像。
--cgroup-driver=<driver_type>：指定 Cgroup 驱动类型，可选值有 cgroupfs 或 systemd。
--cluster-dns=<dns_ip>：指定 DNS 服务器地址。
--kubeconfig=/root/.kube/config：指定 kubeconfig 配置文件路径。

注：本文仅用于技术交流与学习，文章中内容不代表阿里巴巴集团技术策略，不做任何商业用途。