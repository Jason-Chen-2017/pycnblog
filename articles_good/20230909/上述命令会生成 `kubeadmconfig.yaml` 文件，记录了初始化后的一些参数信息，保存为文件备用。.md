
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 一、为什么要写这个博客？
`kubeadm init` 命令能够帮助用户在集群中部署一个工作节点，但是其内部执行逻辑还是比较复杂的，如果我们想知道 `kubeadm init` 执行后最终生成了什么配置文件，并且如何配置这些参数，就需要阅读 Kubernetes 的源码并理解其中的机制。另外，作为一个 Kubernetes 的老用户，我自己也经常面临着一些疑惑或者痛点，通过写一系列的文章，能够帮助广大技术人员快速入门并掌握 Kubernetes 的核心知识。
## 二、怎么写这个博客？
既然是专业的技术博客，那么首先需要定义清楚文章的内容。一般来说，一个知识点相关的博客大概会包括以下几个部分：
1. 介绍：对于所涉及到的知识点，介绍一下它的历史沿革、基本概念等；
2. 术语说明：介绍一些本文涉及到的专业术语，并对它们的定义进行说明；
3. 原理解析：从计算机底层原理出发，理论联系实际，深刻剖析知识点背后的原理和机制；
4. 操作流程：具体地给出操作步骤和代码实现过程，更贴近实际应用场景；
5. 实践案例：针对某个具体的问题或场景，用实例阐述解决方案的妙处，让读者感受到前瞻性的气息；
6. 深度分析：综合多个角度，理论联系实际，进行详尽的分析；
7. 总结反思：最后再总结一下学习收获与技巧，以及对未来的展望。
除了以上内容外，还应该包含如下几点：
1. 推荐阅读：推荐一些相互关联的资料或博客；
2. 源码分析：通过开源的代码库来理解某些关键模块的实现原理和调用方式；
3. 提问解答：遇到疑问，及时回答，提升专业素养；
4. 学习路径：列出阅读完本文后所需要了解的其他相关知识，并提供对应的学习资源；

## 三、什么样的内容适合做这样的博客？
对于初级阶段的技术人员，比如刚接触 Kubernetes 的新手，就应该注重基础知识的讲解。而对熟练掌握 Kubernetes 技术的技术人员来说，可以深入浅出的讨论一些高级特性的实现原理。因此，编写的文章应当具有高度的专业性，达到“知其然，知其所以然”的境界。

在写作过程中，也建议严格遵循作者自己的观点，不允许抄袭他人的成果。这对我自己也是一种学习的奖励。
# 2.基本概念术语说明
Kubernetes (K8s) 是一款开源容器集群管理系统，由 Google、CoreOS、RedHat 等公司共同开发维护。Kubernetes 提供了一套完整的基础设施抽象和自动化机制，让用户可以方便地部署和管理容器化应用。Kubernetes 使用了容器技术，能够将多个容器封装起来，形成一个整体的服务，在分布式环境下自动分配和调度资源。

`kubeadm` 是一个用来快速部署 Kubernetes 集群的工具。它会自动检测系统的要求并安装相应的组件，然后利用 `kubelet` 和 `kube-proxy` 来启动集群节点上的 kubelet 服务和 kube-proxy 服务，这样集群就可以正常运行。

`kubeadm-config.yaml` 是 `kubeadm` 安装完成后生成的一个配置文件。它记录了 `kubeadm` 初始化的参数信息，包括 API server 的地址，etcd 的地址，证书签名请求（CSR）等。
# 3.核心算法原理和具体操作步骤以及数学公式讲解
## 1.初始化 Master 节点
首先，我们需要下载 Kubernetes 发行版包，这里我们下载最新版本的 `kubernetes-server-linux-amd64.tar.gz`。解压该包，进入 `cluster/images/hyperkube` 目录，将所有 yaml 文件放在一起，并按照以下顺序创建配置文件：

1. kubeadm-config.yaml: 用于指定初始化参数，如 kubernetes 版本号、Pod 网络配置、Service 网络配置等。
```yaml
apiVersion: kubeadm.k8s.io/v1beta1
kind: InitConfiguration
bootstrapTokens:
- groups:
  - system:bootstrappers:kubeadm:default-node-token
  ttl: "0"
localAPIEndpoint:
  advertiseAddress: <master_ip>
  bindPort: 6443
---
apiServer:
  certSANs:
  - <master_ip>
  extraArgs:
    anonymous-auth: false
    enable-admission-plugins: NodeRestriction
    encryption-provider-config: /etc/kubernetes/ssl/encryption-config.yaml
    tls-cert-file: /etc/kubernetes/pki/apiserver.crt
    tls-private-key-file: /etc/kubernetes/pki/apiserver.key
  timeoutForControlPlane: 4m0s
apiVersion: kubeproxy.config.k8s.io/v1alpha1
kind: KubeProxyConfiguration
clientConnection:
  kubeconfig: "/etc/kubernetes/admin.conf"
mode: "iptables"
---
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
authentication:
  anonymous:
    enabled: false
  webhook:
    cacheTTL: 2m0s
    configFile: /var/lib/kubelet/config.json
authorization:
  mode: Webhook
  webhook:
    cacheAuthorizedTTL: 5m0s
    cacheUnauthorizedTTL: 30s
cgroupDriver: cgroupfs
cgroupsPerQOS: true
clusterDomain: cluster.local
cpuManagerPolicy: static
evictionHard:
  imagefs.available: 25%
  memory.available: 100Mi
  nodefs.available: 25%
  nodefs.inodesFree: 5%
evictionPressureTransitionPeriod: 5m0s
failSwapOn: false
fileCheckFrequency: 20s
healthzBindAddress: 127.0.0.1
healthzPort: 10248
httpCheckFrequency: 20s
imageGCHighThresholdPercent: 100
imageGCLowThresholdPercent: 0
kind: KubeletConfiguration
logging: {}
nodeStatusReportFrequency: 10s
nodeLeaseDurationSeconds: 40
port: 10250
readOnlyPort: 0
rotateCertificates: true
runtimeRequestTimeout: 2m0s
serializeImagePulls: true
staticPodPath: /etc/kubernetes/manifests
streamingConnectionIdleTimeout: 4h0m0s
syncFrequency: 1m0s
volumeStatsAggPeriod: 1m0s
```

2. apiserver-pod.yaml: 指定 apiserver Pod 的相关属性。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: kube-apiserver
  namespace: kube-system
  labels:
    component: kube-apiserver
    tier: control-plane
spec:
  hostNetwork: true
  containers:
  - command:
    - kube-apiserver
    - --advertise-address=<master_ip>
    - --allow-privileged=true
    - --anonymous-auth=false
    - --authorization-mode=Node,RBAC
    - --client-ca-file=/etc/kubernetes/pki/front-proxy-ca.crt
    - --enable-admission-plugins=NamespaceLifecycle,LimitRanger,ServiceAccount,DefaultStorageClass,ResourceQuota,DefaultTolerationSeconds
    - --enable-swagger-ui=true
    - --etcd-cafile=/etc/kubernetes/pki/etcd/ca.crt
    - --etcd-certfile=/etc/kubernetes/pki/apiserver-etcd-client.crt
    - --etcd-keyfile=/etc/kubernetes/pki/apiserver-etcd-client.key
    - --etcd-servers=https://<master_ip>:2379
    - --insecure-port=0
    - --kubelet-client-certificate=/etc/kubernetes/pki/apiserver-kubelet-client.crt
    - --kubelet-client-key=/etc/kubernetes/pki/apiserver-kubelet-client.key
    - --kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname
    - --proxy-client-cert-file=/etc/kubernetes/pki/front-proxy-client.crt
    - --proxy-client-key-file=/etc/kubernetes/pki/front-proxy-client.key
    - --requestheader-allowed-names=front-proxy-client
    - --requestheader-client-ca-file=/etc/kubernetes/pki/front-proxy-ca.crt
    - --requestheader-extra-headers-prefix=X-Remote-Extra-
    - --requestheader-group-headers=X-Remote-Group
    - --requestheader-username-headers=X-Remote-User
    - --secure-port=6443
    - --service-account-key-file=/etc/kubernetes/pki/sa.pub
    - --tls-cert-file=/etc/kubernetes/pki/apiserver.crt
    - --tls-private-key-file=/etc/kubernetes/pki/apiserver.key
    image: k8s.gcr.io/kube-apiserver:<version>
    imagePullPolicy: IfNotPresent
    livenessProbe:
      failureThreshold: 8
      httpGet:
        host: <master_ip>
        path: /healthz
        port: 6443
        scheme: HTTPS
      initialDelaySeconds: 15
      periodSeconds: 10
      successThreshold: 1
      timeoutSeconds: 15
    name: kube-apiserver
    resources:
      requests:
        cpu: 250m
    securityContext:
      allowPrivilegeEscalation: true
      capabilities:
        drop:
        - ALL
      readOnlyRootFilesystem: true
    volumeMounts:
    - mountPath: /etc/ssl/certs
      name: ssl-certs-host
      readOnly: true
    - mountPath: /usr/share/ca-certificates
      name: ca-certs
      readOnly: true
    - mountPath: /etc/kubernetes/pki
      name: k8s-certs
      readOnly: true
    - mountPath: /etc/kubernetes/admission-controls
      name: admission-control
      readOnly: true
    - mountPath: /etc/kubernetes/configmaps
      name: etc-kubeconfig
      readOnly: true
    - mountPath: /etc/kubernetes/ssl
      name: etc-kubernetes-ssl
      readOnly: true
    - mountPath: /etc/machine-id
      name: etc-machine-id
      readOnly: true
  volumes:
  - hostPath:
      path: /usr/share/ca-certificates
    name: ca-certs
  - hostPath:
      path: /etc/kubernetes/pki
    name: k8s-certs
  - hostPath:
      path: /etc/kubernetes/admission-controls
    name: admission-control
  - hostPath:
      path: /etc/kubernetes/configmaps
    name: etc-kubeconfig
  - hostPath:
      path: /etc/kubernetes/ssl
    name: etc-kubernetes-ssl
  - hostPath:
      path: /etc/ssl/certs
    name: ssl-certs-host
  - hostPath:
      path: /etc/machine-id
    name: etc-machine-id
```

3. controller-manager-pod.yaml: 指定 controller manager 的相关属性。

```yaml
apiVersion: v1
kind: Pod
metadata:
  annotations:
    scheduler.alpha.kubernetes.io/critical-pod: ''
  creationTimestamp: null
  labels:
    component: kube-controller-manager
    tier: control-plane
  name: kube-controller-manager
  namespace: kube-system
spec:
  containers:
  - command:
    - /hyperkube
    - controller-manager
    - --allocate-node-cidrs=true
    - --bind-address=127.0.0.1
    - --cluster-cidr=10.244.0.0/16
    - --cluster-name=kubernetes
    - --configure-cloud-routes=false
    - --controllers=*,bootstrapsigner,tokencleaner
    - --kubeconfig=/etc/kubernetes/controller-manager.conf
    - --leader-elect=true
    - --root-ca-file=/etc/kubernetes/pki/ca.crt
    - --service-account-private-key-file=/etc/kubernetes/pki/sa.key
    - --use-service-account-credentials=true
    image: k8s.gcr.io/kube-controller-manager:<version>
    imagePullPolicy: IfNotPresent
    livenessProbe:
      failureThreshold: 8
      httpGet:
        host: 127.0.0.1
        path: /healthz
        port: 10257
        scheme: HTTP
      initialDelaySeconds: 15
      periodSeconds: 10
      successThreshold: 1
      timeoutSeconds: 15
    name: kube-controller-manager
    resources:
      requests:
        cpu: 200m
    terminationMessagePath: /dev/termination-log
    terminationMessagePolicy: File
    volumeMounts:
    - mountPath: /etc/ssl/certs
      name: ssl-certs-host
      readOnly: true
    - mountPath: /etc/kubernetes/pki
      name: k8s-certs
      readOnly: true
    - mountPath: /etc/kubernetes/config
      name: kubeconfig
      readOnly: true
    - mountPath: /etc/kubernetes/controller-manager.conf
      name: ctrl-manager-kubeconfig
      readOnly: true
    - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
      name: sa-token-secret
      readOnly: true
  dnsPolicy: ClusterFirst
  hostNetwork: true
  priorityClassName: system-cluster-critical
  restartPolicy: Always
  schedulerName: default-scheduler
  securityContext: {}
  serviceAccount: kube-controller-manager
  serviceAccountName: kube-controller-manager
  terminationGracePeriodSeconds: 10
  tolerations:
  - effect: NoExecute
    key: node.kubernetes.io/not-ready
    operator: Exists
    tolerationSeconds: 300
  - effect: NoExecute
    key: node.kubernetes.io/unreachable
    operator: Exists
    tolerationSeconds: 300
  volumes:
  - hostPath:
      path: /usr/share/ca-certificates
    name: ssl-certs-host
  - hostPath:
      path: /etc/kubernetes/pki
    name: k8s-certs
  - hostPath:
      path: /etc/kubernetes/config
    name: kubeconfig
  - hostPath:
      path: /var/run/secrets/kubernetes.io/serviceaccount
    name: sa-token-secret
status: {}
```

4. scheduler-pod.yaml: 指定 scheduler 的相关属性。

```yaml
apiVersion: v1
kind: Pod
metadata:
  creationTimestamp: null
  labels:
    component: kube-scheduler
    tier: control-plane
  name: kube-scheduler
  namespace: kube-system
spec:
  containers:
  - command:
    - /hyperkube
    - scheduler
    - --bind-address=127.0.0.1
    - --kubeconfig=/etc/kubernetes/scheduler.conf
    - --leader-elect=true
    image: k8s.gcr.io/kube-scheduler:<version>
    imagePullPolicy: IfNotPresent
    livenessProbe:
      failureThreshold: 8
      httpGet:
        host: 127.0.0.1
        path: /healthz
        port: 10259
        scheme: HTTP
      initialDelaySeconds: 15
      periodSeconds: 10
      successThreshold: 1
      timeoutSeconds: 15
    name: kube-scheduler
    resources:
      requests:
        cpu: 100m
    terminationMessagePath: /dev/termination-log
    terminationMessagePolicy: File
    volumeMounts:
    - mountPath: /etc/ssl/certs
      name: ssl-certs-host
      readOnly: true
    - mountPath: /etc/kubernetes/pki
      name: k8s-certs
      readOnly: true
    - mountPath: /etc/kubernetes/config
      name: kubeconfig
      readOnly: true
    - mountPath: /etc/kubernetes/scheduler.conf
      name: sched-kubeconfig
      readOnly: true
    - mountPath: /var/run/secrets/kubernetes.io/serviceaccount
      name: sa-token-secret
      readOnly: true
  dnsPolicy: ClusterFirst
  hostNetwork: true
  priorityClassName: system-cluster-critical
  restartPolicy: Always
  schedulerName: default-scheduler
  securityContext: {}
  serviceAccount: kube-scheduler
  serviceAccountName: kube-scheduler
  terminationGracePeriodSeconds: 10
  tolerations:
  - effect: NoExecute
    key: node.kubernetes.io/not-ready
    operator: Exists
    tolerationSeconds: 300
  - effect: NoExecute
    key: node.kubernetes.io/unreachable
    operator: Exists
    tolerationSeconds: 300
  volumes:
  - hostPath:
      path: /usr/share/ca-certificates
    name: ssl-certs-host
  - hostPath:
      path: /etc/kubernetes/pki
    name: k8s-certs
  - hostPath:
      path: /etc/kubernetes/config
    name: kubeconfig
  - hostPath:
      path: /var/run/secrets/kubernetes.io/serviceaccount
    name: sa-token-secret
status: {}
```

5. etcd-service.yaml: 创建 etcd 服务，使 master 可以访问到 etcd 数据库。

```yaml
apiVersion: v1
kind: Service
metadata:
  creationTimestamp: null
  labels:
    component: etcd
    tier: control-plane
  name: kube-etcd
  namespace: kube-system
spec:
  ports:
  - name: client
    port: 2379
    protocol: TCP
    targetPort: 2379
  - name: peer
    port: 2380
    protocol: TCP
    targetPort: 2380
  selector:
    component: etcd
    tier: control-plane
  type: ClusterIP
status:
  loadBalancer: {}
```

6. kube-proxy-ds.yaml: 配置 kube-proxy daemonset，启动 kube-proxy 服务。

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  creationTimestamp: null
  labels:
    k8s-app: kube-proxy
  name: kube-proxy
  namespace: kube-system
spec:
  revisionHistoryLimit: 10
  selector:
    matchLabels:
      k8s-app: kube-proxy
  strategy:
    rollingUpdate:
      maxUnavailable: 1
    type: RollingUpdate
  template:
    metadata:
      creationTimestamp: null
      labels:
        k8s-app: kube-proxy
    spec:
      containers:
      - args:
        - --hostname-override=$(NODE_NAME)
        - --v=2
        env:
        - name: NODE_NAME
          valueFrom:
            fieldRef:
              apiVersion: v1
              fieldPath: spec.nodeName
        image: gcr.io/google_containers/kube-proxy:<version>
        imagePullPolicy:IfNotPresent
        name: kube-proxy
        resources:
          requests:
            cpu: 100m
        securityContext:
          privileged: true
        terminationMessagePath: /dev/termination-log
        terminationMessagePolicy: File
        volumeMounts:
        - mountPath: /run/xtables.lock
          name: xtables-lock
          readOnly: false
        - mountPath: /etc/ssl/certs
          name: ssl-certs-host
          readOnly: true
        - mountPath: /etc/kubernetes/config
          name: kubeconfig
          readOnly: true
      dnsPolicy: ClusterFirst
      hostNetwork: true
      priorityClassName: system-node-critical
      restartPolicy: Always
      securityContext: {}
      serviceAccount: kube-proxy
      serviceAccountName: kube-proxy
      terminationGracePeriodSeconds: 30
      tolerations:
      - effect: NoSchedule
        key: node-role.kubernetes.io/master
        operator: Exists
      volumes:
      - hostPath:
          path: /run/xtables.lock
          type: FileOrCreate
        name: xtables-lock
      - hostPath:
          path: /usr/share/ca-certificates
        name: ssl-certs-host
      - hostPath:
          path: /etc/kubernetes/config
        name: kubeconfig
status:
  currentNumberScheduled: 0
  desiredNumberScheduled: 0
  numberMisscheduled: 0
  numberReady: 0
```

7. certificate-signing-request.yaml: 生成一份证书签名请求（CSR），等待 Kubernetes CA 对其签名并分发。

```yaml
apiVersion: certificates.k8s.io/v1beta1
kind: CertificateSigningRequest
metadata:
  name: kubernetes-master
  namespace: kube-system
spec:
  groups:
  - system:authenticated
  request: LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0t...
  usages:
  - digital signature
  - key encipherment
  username: system:node:<master_ip>
```

8. kubeadm-join-command.txt: 将 `kubeadm join` 命令写入文件。

```bash
kubeadm join <master>:<port> --token <token> \
    --discovery-token-ca-cert-hash sha256:<hash>
```

之后，按照步骤顺序依次执行每个文件，即可完成 `kubeadm init` 命令的执行。至此，初始化 Master 节点已经完成。

## 2.节点加入集群
`kubeadm` 支持多种类型的节点加入集群，但目前主要支持以下四种方法：
1. 手动加入：即以 root 或具有 sudo 权限的用户，直接在目标主机上运行 `kubeadm join` 命令加入集群；
2. 加入节点脚本：即将 `kubeadm join` 命令写入脚本文件，通过 ssh 或其它方式远程运行脚本，即可完成节点加入集群；
3. Join Token 方法：即获取 `kubeadm token create` 命令的输出结果，作为 `kubeadm join` 命令的参数，节点加入集群；
4. Bootstrap Token 方法：即在 `kubeadm init` 时指定 `--token`，不同于普通 Token，Bootstrap Token 有额外的功能限制。

这里我们只介绍手动加入的方法，其余两种方法与手动加入类似。在 `kubeadm` 下载包中，`cluster/addons` 目录提供了初始化成功后的各项功能，其中有一个名为 `flannel-rbac.yml` 的文件，内容如下：

```yaml
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: flannel
  namespace: kube-system
rules:
- apiGroups:
  - ""
  resources:
  - pods
  - namespaces
  verbs:
  - get
- apiGroups:
  - ""
  resources:
  - nodes
  verbs:
  - list
  - watch
- apiGroups:
  - extensions
  resources:
  - networkpolicies
  verbs:
  - get
  - list
  - watch
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: flannel
  namespace: kube-system
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: Role
  name: flannel
subjects:
- kind: ServiceAccount
  name: flannel
  namespace: kube-system
```

该文件通过创建一个名为 `flannel` 的 `Role` 和 `RoleBinding` 对象，授予了 `flannel` 服务帐户对 Pod、命名空间、节点和网络策略等资源的访问权限。

接着，我们先编辑 `kubelet-config.yaml` 文件，在 `volumes` 中添加以下条目：

```yaml
volumes:
- name: modprobe
  hostPath:
    path: /bin/modprobe
- name: lib-modules
  hostPath:
    path: /lib/modules
```

然后，编辑 `/etc/systemd/system/kubelet.service.d/10-kubeadm.conf` 文件，在 `[Service]` 下添加以下条目：

```ini
Environment="KUBELET_EXTRA_ARGS=--fail-swap-on=false"
ExecStartPre=-/sbin/modprobe br_netfilter
ExecStartPost=/usr/bin/kubectl apply -f https://raw.githubusercontent.com/coreos/flannel/bc79dd1505b0c8681ece4de4c0d86c5cd2643275/Documentation/kube-flannel.yml
```

这里我们开启 `br_netfilter` 模块和部署 `flannel` 插件。编辑 `/etc/kubernetes/manifests/kube-flannel.yml` 文件，将 `image` 字段值改为 `quay.io/coreos/flannel:v0.9.1`，然后重新启动 `kubelet` 服务：

```shell
sudo systemctl restart kubelet
```

这样，节点加入集群的准备工作已经完成。运行 `kubeadm reset` 清除现有的 Kubernetes 集群，再次执行 `kubeadm init` 命令，然后运行 `kubeadm join...` 命令加入集群即可。