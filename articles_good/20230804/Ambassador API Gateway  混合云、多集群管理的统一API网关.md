
作者：禅与计算机程序设计艺术                    

# 1.简介
         
Ambassador是一个开源的Kubernetes托管服务,它可以将多个Kubernetes集群中的微服务聚合在一起,通过单个的端点暴露出来，支持多种协议如HTTP、gRPC、WebSocket等,可以通过动态配置改变集群中服务的路由规则,而无需更改代码或者重新发布镜像。同时提供基于角色的访问控制(RBAC)、认证授权(OAuth2)等功能。而Ambassador还可以配合其他云平台或API网关(如AWS API Gateway)，实现混合云、多集群管理的统一API网关。因此，该技术方案具有以下优势:

1.易于使用：无论是从零开始还是转到Ambassador,都可以轻松实现服务网格功能。
2.高度可扩展性：Ambassador可以应用到各个 Kubernetes 集群上，使其能够处理更复杂的流量负载。
3.内置服务发现：Ambassador可以自动地发现并连接到 Kubernetes 中的服务。
4.API网关兼容：Ambassador可以和其他API网关集成，如 AWS API Gateway 或 Google Cloud Endpoints。
5.丰富的插件体系：Ambassador提供了丰富的插件体系，包括认证授权、速率限制、熔断器等，能够满足业务需要。
本文将首先对Ambassador做一个简单的介绍，然后对Ambassador API Gateway进行详细阐述。文章末尾还有一些常见问题的解答。

# 2. Ambassador介绍
## 2.1 什么是Ambassador？
Ambassador 是由 Datawire 公司开发的基于 Envoy Proxy 的 Kubernetes 上透明的 API 网关。它通过一种声明式的方法将 Kubernetes 服务暴露给消费者，而不需要对服务的代码进行任何改动。它完全符合 Istio 的设计理念，并且还增加了诸如基于角色的访问控制、高级请求处理、日志记录、弹性部署等功能。相比之下，Istio 更关注于服务间通信的可观察性、健康检查和流量控制。Ambassador 可以作为独立组件安装，也可以与 Prometheus 和 Grafana 这样的工具一起安装，形成全面的 API 网关解决方案。

## 2.2 Ambassador能做什么？
Ambassador 可以帮助 Kubernetes 用户实现以下目标：

1. **路由**：Ambassador 提供基于 HTTP 方法、路径、header 的路由功能。用户可以定义任意数量的路由规则，并根据需要调整它们的匹配顺序，以便 Ambassador 根据最佳匹配方式将请求转发到正确的服务端点。
2. **服务发现**：Ambassador 使用 Kubernetes 服务发现 API 将 Kubernetes 服务暴露给消费者。这让 Ambassador 不需要额外配置就能识别集群内部或外部的服务，并且可以通过多个路由规则将请求映射到不同的后端服务。
3. **TLS termination**：Ambassador 支持服务之间的 TLS 加密通信，并提供基于 SNI (server name indication) 的流量切换。
4. **灰度发布**：Ambassador 可以动态更新现有路由规则，而不会影响正在处理请求的服务。这使得可以在不停机的情况下部署新版本的服务，而无需影响生产环境。
5. **流量控制**：Ambassador 提供基于请求数量、超时、错误率的流控功能。这些功能可以帮助防止服务过载、减少性能损失。
6. **身份验证/授权**：Ambassador 提供基于 JWT token 或 OAuth2 的认证和授权功能，可以保护 Kubernetes 服务免受未经授权的访问。
7. **熔断器**：Ambassador 提供基于响应时间的熔断器功能，当某个服务响应时间超过设定的阈值时，Ambassador 会暂停向该服务发送请求，直到其恢复正常。
8. **重试/超时**：Ambassador 可以配置针对特定服务的重试策略，以及针对客户端请求的超时设置。

## 2.3 Ambassador架构图


Ambassador 由两大部分组成，包括运行在数据平面上的 Envoy Proxy 和运行在控制器面上的控制平面组件。Envoy 是由 Lyft 在 GitHub 上开源的开源边缘代理项目，可以直接运行在 Kubernetes 中，并作为 sidecar 容器加入到 Pod 中。这个组件负责监听 Kubernetes API Server，并通过 CRD（Custom Resource Definition）创建路由规则，将请求路由到正确的服务。而控制平面组件则根据路由配置生成 Envoy 配置文件，并通过 Kubernetes API Server 下发到各个工作节点上。

Ambassador 使用自定义资源定义（CRDs），通过声明式的方法描述集群内部的服务网格。这些规则会被控制器翻译为最终的 Envoy 配置，并下发到各个工作节点上。这种方式避免了修改应用代码的方式来配置网格，也不依赖特定的编程语言。

# 3. Ambassador API Gateway
## 3.1 Ambassador API Gateway产品介绍
Ambassador API Gateway 是 Datawire 为企业级客户提供的企业级服务网格（Service Mesh）解决方案，提供一种简单又安全的基于 HTTP 协议的流量路由及处理方式，通过简单的配置就能对外服务。它的主要功能如下：

1. 网关集群化：基于 Kubernetes 的无侵入性架构，简洁易用；
2. 全局负载均衡：基于一致的 Hash 分片算法，实现跨区域、跨云的分布式调度；
3. 动态路由：灵活的路由规则设置，细粒度的流量控制；
4. 流量加密：TLS 加密传输，保障敏感数据安全；
5. 可观测性：基于 Prometheus + Grafana，可监控每个 API 的访问情况；
6. 插件机制：提供多种流量控制及安全插件，满足不同场景需求；
7. 可视化界面：提供可视化页面，方便用户查看 API 状态。

## 3.2 Ambassador API Gateway架构

Ambassador API Gateway 通过 Envoy 代理实现流量调度，并采用七层模型。API Gateway 集群通过 Hash 分片算法实现多个 Envoy 节点之间的负载均衡。该架构有如下几个重要特征：

1. 逻辑分离：API Gateway 负责接收所有 HTTP 请求并进行相应的过滤、路由和处理；
2. 拆分网络：与业务服务隔离，提升整体网络安全性；
3. 数据加密：HTTPS 协议加密数据，保证敏感数据的完整性；
4. 可伸缩性：集群规模扩充时可提供高可用和低延迟；
5. 可观测性：可通过 Prometheus 和 Grafana 获取 API 的访问统计信息。

## 3.3 Ambassador API Gateway的基本概念
### 3.3.1 Mapping
Mapping 是指 Ambassador API Gateway 中的一项基础功能，用来定义 API 的路由规则。一条 Mapping 可以根据不同的条件选择不同的后端服务，并控制流量转发、限流、熔断、认证授权等参数。Mapping 可以配置如下属性：

1. Host：指定要绑定的域名；
2. Prefix：URI 前缀；
3. Method：HTTP 方法；
4. Service：目标服务名称；
5. Weight：流量分配权重；
6. Headers：添加 HTTP Header；
7. Authenticated：是否开启认证授权；
8. AllowedOrigins：允许跨域请求的源站地址；
9. RateLimit：请求速率限制；
10. FaultInjection：故障注入；
11. TimeoutPolicy：请求超时策略；
12. RetryPolicy：重试策略；
13. Mappings 之间可以组合，形成更复杂的路由策略。

### 3.3.2 Authentication
Authentication 是指用于身份认证、授权的模块，它支持包括 JWT 校验、HMAC 校验、OAuth2 校验等常用的认证方式。目前 Ambassador API Gateway 支持两种类型的认证方式：

1. BasicAuth：使用用户名密码进行认证；
2. BearerToken：使用 JWT Token 进行认证；

### 3.3.3 Rate Limiting
Rate Limiting 是指对 API 请求的流量限制，它可以使用固定窗口或令牌桶的方式进行流量控制。其中，令牌桶方式可以精确控制每秒的请求次数，适合具有突发流量的场景；固定窗口方式对流量总量进行限制，适合流量稳定但突发流量较多的场景。

### 3.3.4 Ingress Class
Ingress Class 是 Kubernetes 里用来标识 ingressClass 的对象，通过 IngressClassName 属性绑定多个 Ingress 对象。它的作用是在不同的 Kubernetes cluster 上安装不同的 Ingress Controller 时，用来区分对应的 controller。

### 3.3.5 OpenAPI Support
OpenAPI Support 是用来校验和转换 RESTful API 的规范。它提供与 Swagger 相同的接口，可以让 API 作者按照规范描述 API 文档。当 Ingress 收到 API 请求时，它可以根据文档的内容检查请求的参数类型、格式和值，并返回符合要求的结果。

### 3.3.6 Pluggable Security Module (PMSM)
PMSM 是 Ambassador API Gateway 自带的安全模块，支持包括 HMAC 签名、JWT 校验、IP Whitelist、API Key 等多种安全策略。用户可以根据自己的需求配置不同的安全策略，并通过 Policies 配置选项来启用。

## 3.4 Ambassador API Gateway如何与其他云平台集成？
Ambassador API Gateway 可以与 AWS API Gateway 或 Google Cloud Endpoints 集成，完成混合云、多集群管理的统一 API 网关。借助第三方服务网格，我们可以将服务网格的能力扩展到更多的 Kubernetes 集群上，为整个集群提供统一的 API Gateway 接入。AWS API Gateway 提供了强大的 Lambda@Edge 函数能力，让我们可以对服务网格的请求和响应进行各种处理，比如缓存、加速、屏蔽攻击行为等。Google Cloud Endpoints 则提供 RESTful API Gateway 服务，让我们可以快速部署和运行 RESTful 应用。

# 4. 演示
Ambassador API Gateway 有两种安装模式，分别是独立模式和共存模式。

## 4.1 安装 Ambassador API Gateway （独立模式）
在独立模式下，Ambassador API Gateway 需要单独安装到 Kubernetes 集群中，并且只能处理当前集群的流量。

使用 Helm Chart 来安装 Ambassador API Gateway。首先，需要创建一个命名空间 ambassador：

```bash
kubectl create namespace ambassador
```

然后，创建 Ambassador API Gateway 的 Deployment 和 Service：

```bash
helm install \
  --namespace ambassador \
  --name ambassador \
  datawire/ambassador
```

等待几分钟，Ambassador API Gateway 就可以启动成功，可以通过下面命令验证：

```bash
kubectl get pods -n ambassador 
NAME                       READY   STATUS    RESTARTS   AGE
ambassador-86bcfcc9bf-pqkps   1/1     Running   0          4m5s
```

然后，创建一个简单的 Mapping 来测试一下：

```yaml
---
apiVersion: getambassador.io/v2
kind: Mapping
metadata:
  name: qotm-mapping
  namespace: default
spec:
  prefix: /qotm/
  service: quote-of-the-day
```

该 Mapping 指定了一个前缀为 `/qotm/` 的路由规则，路由到的目标服务为 `quote-of-the-day`。

为了访问这个服务，我们可以用浏览器访问 http://localhost/qotm/。如果出现 Quote of the Day 随机的话，证明这个服务已经可以正常访问。

但是，Ambassador API Gateway 默认是只允许本地访问的，所以无法直接从集群外访问到它，不过我们可以通过 NodePort、LoadBalancer 等方式暴露出去，让集群外的服务访问到它。我们可以先删除刚才创建的简单 Mapping：

```bash
kubectl delete mapping qotm-mapping -n default
```

然后，编辑 Ambassador API Gateway 的 Service 对象，添加以下 annotations：

```yaml
---
apiVersion: v1
kind: Service
metadata:
  labels:
    app.kubernetes.io/component: "controller"
    app.kubernetes.io/instance: "ambassador"
    app.kubernetes.io/managed-by: "Helm"
    app.kubernetes.io/name: "ambassador"
    helm.sh/chart: "ambassador-6.6.1"
  name: ambassador-admin
  namespace: ambassador
  annotations:
    # use any other valid LoadBalancer annotation depending on your cloud provider
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb" 
    service.beta.kubernetes.io/gcp-load-balancer-type: "internal"
    service.beta.kubernetes.io/azure-load-balancer-sku: "Standard"
    
    # set this to enable external access
    getambassador.io/config: |
      ---
      apiVersion: ambassador/v1
      kind:  Module
      name:  ambassador
      config:
        admin_port: 8080

      ---
      apiVersion: ambassador/v1
      kind:  Mapping
      name:  qotm-mapping
      host: "*"
      prefix: "/qotm/"
      service: quote-of-the-day
spec:
  type: ClusterIP
  ports:
  - port: 80
    targetPort: 8080
    protocol: TCP
    name: http
  selector:
    service: ambassador
```

这里，我们添加了三个 annotations：

1. 添加 `service.beta.kubernetes.io/aws-load-balancer-type`、`service.beta.kubernetes.io/gcp-load-balancer-type`，以及 `service.beta.kubernetes.io/azure-load-balancer-sku` 以使用云提供商提供的负载均衡器。
2. 设置 `getambassador.io/config` 以启用外部访问，并添加了一个 Mapping 来指定 `/qotm/` 前缀的路由规则。

最后，保存退出，执行 `kubectl apply` 命令使得变更生效。

此时，外部的服务应该也可以通过 IP/域名访问到 Ambassador API Gateway，并且可以看到 Quote of the Day 随机。

## 4.2 安装 Ambassador API Gateway （共存模式）
在共存模式下，我们可以将 Ambassador API Gateway 安装到当前集群的任一命名空间中，并且它也可以处理其它命名空间中的流量。

Ambassador API Gateway 的 Helm Chart 支持将 Ambassador 作为系统组件安装到任意命名空间中，因此，我们只需要在其他命名空间中安装 Ambassador 的 ServiceAccount 和 CRD，即可与 Ambassador API Gateway 一起工作：

```bash
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Namespace
metadata:
  name: prod
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: prod-ambassador
  namespace: prod
subjects:
- kind: ServiceAccount
  name: default
  namespace: prod
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: ambassador-crd
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: prod-ambassador
  namespace: prod
subjects:
- kind: ServiceAccount
  name: default
  namespace: prod
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: ambassador-sa
EOF
```

这里，我们在 `prod` 命名空间中创建了一份 RoleBinding 文件，为 Ambassador API Gateway 的 ServiceAccount 和相关的 RBAC 设置提供了必要的权限。

然后，编辑 Ambassador API Gateway 的 CRD 对象：

```yaml
---
apiVersion: apiextensions.k8s.io/v1beta1
kind: CustomResourceDefinition
metadata:
  name: authservices.getambassador.io
spec:
  group: getambassador.io
  version: v2alpha1
  scope: Namespaced
  names:
    plural: authservices
    singular: authservice
    kind: AuthService
    shortNames: ["as"]
---
apiVersion: apiextensions.k8s.io/v1beta1
kind: CustomResourceDefinition
metadata:
  name: ratelimitservices.getambassador.io
spec:
  group: getambassador.io
  version: v2alpha1
  scope: Namespaced
  names:
    plural: ratelimitservices
    singular: ratelimitservice
    kind: RateLimitService
    shortNames: ["rls"]
---
apiVersion: apiextensions.k8s.io/v1beta1
kind: CustomResourceDefinition
metadata:
  name: tlscontexts.getambassador.io
spec:
  group: getambassador.io
  version: v2alpha1
  scope: Namespaced
  names:
    plural: tlscontexts
    singular: tlscontext
    kind: TLSContext
    shortNames: ["tlsctx"]
```

这里，我们创建了三个 CRD 对象，分别对应着 Ambassador API Gateway 的三种类型资源：AuthService、RateLimitService 和 TLSContext。

然后，编辑 Ambassador API Gateway 的 Deployment 对象：

```yaml
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ambassador-deployment
  namespace: prod
  labels:
    product: ambassador
spec:
  replicas: 1
  selector:
    matchLabels:
      service: ambassador
  template:
    metadata:
      labels:
        service: ambassador
    spec:
      containers:
      - name: ambassador
        image: quay.io/datawire/ambassador:1.7.0
        env:
        - name: POD_IP
          valueFrom:
            fieldRef:
              fieldPath: status.podIP
        ports:
        - containerPort: 8080
        livenessProbe:
          httpGet:
            path: /ambassador/v0/check_alive
            port: 8877
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ambassador/v0/check_ready
            port: 8877
          initialDelaySeconds: 30
          periodSeconds: 30
        resources:
          limits:
            cpu: 1000m
            memory: 2048Mi
          requests:
            cpu: 200m
            memory: 1024Mi
      serviceAccountName: default
```

这里，我们创建一个 Deployment 对象，以便于安装 Ambassador API Gateway。同样，由于我们需要在 `prod` 命名空间中安装 Ambassador API Gateway，因此我们也需要把 Ambassador 所在的命名空间设置为 `prod`。

保存并退出，执行 `kubectl apply` 命令使得变更生效。

如果一切顺利，应该就可以通过集群外的服务访问到 Ambassador API Gateway。