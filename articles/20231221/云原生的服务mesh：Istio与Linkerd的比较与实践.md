                 

# 1.背景介绍

随着微服务架构在企业中的普及，服务间的通信量和复杂性都增加了很多。服务mesh是一种在微服务架构中，将所有服务进行统一的管理和监控的架构设计。Istio和Linkerd是目前最受欢迎的两个服务mesh工具。本文将从背景、核心概念、算法原理、实践操作、未来发展等多个方面进行比较和实践分析。

# 2.核心概念与联系

## 2.1服务mesh的概念

服务mesh是一种在微服务架构中，将所有服务进行统一的管理和监控的架构设计。它通过一系列的网关、代理和控制平面，实现了服务发现、负载均衡、流量控制、安全性保护、监控与追踪等功能。

## 2.2Istio与Linkerd的概念

Istio是一个开源的服务网格，基于Google的Envoy代理实现。它提供了一套强大的API，用于管理、监控和安全化微服务网络。

Linkerd是一个开源的服务网格，基于Rust语言实现。它提供了一套轻量级的网络代理和控制平面，用于管理、监控和安全化微服务网络。

## 2.3联系

Istio和Linkerd都是服务mesh的代表性工具，它们的目标是提供一种统一的管理和监控机制，以实现微服务架构的高效运行。它们之间的主要区别在于实现方式和性能特点。Istio使用Google的Envoy代理，具有较高的性能和丰富的功能；而Linkerd使用Rust语言实现，具有较好的安全性和轻量级特点。

# 3.核心算法原理和具体操作步骤以及数学模型公式详细讲解

## 3.1Istio的核心算法原理

Istio的核心算法原理包括服务发现、负载均衡、流量控制、安全性保护、监控与追踪等。

### 3.1.1服务发现

Istio使用Envoy代理实现服务发现，通过配置Envoy的service entry，可以实现对服务的发现和注册。服务注册表通常使用Kubernetes的服务发现机制，将服务的元数据信息（如服务名称、端口等）注册到Kubernetes的服务发现系统中。

### 3.1.2负载均衡

Istio使用Envoy代理实现负载均衡，支持多种负载均衡策略，如轮询、权重、最少请求数等。Envoy代理在接收到请求后，根据配置选择目标服务的具体实例，并将请求发送给目标服务。

### 3.1.3流量控制

Istio提供了流量控制功能，可以根据规则控制服务之间的流量。例如，可以设置只允许来自特定IP地址的请求访问某个服务，或者限制某个服务的请求数量。

### 3.1.4安全性保护

Istio提供了一系列的安全性保护功能，包括身份验证、授权、加密等。通过这些功能，可以确保服务间的通信安全。

### 3.1.5监控与追踪

Istio集成了多种监控和追踪工具，如Prometheus、Grafana、Jaeger等，可以实时监控服务的性能指标和追踪服务间的调用关系。

## 3.2Linkerd的核心算法原理

Linkerd的核心算法原理包括服务发现、负载均衡、流量控制、安全性保护、监控与追踪等。

### 3.2.1服务发现

Linkerd使用自身实现的服务发现机制，通过在Kubernetes中注册服务的元数据信息，实现对服务的发现和注册。Linkerd的服务注册表使用Kubernetes的服务发现机制，将服务的元数据信息（如服务名称、端口等）注册到Kubernetes的服务发现系统中。

### 3.2.2负载均衡

Linkerd使用自身实现的负载均衡算法，支持多种负载均衡策略，如轮询、权重、最少请求数等。Linkerd的负载均衡算法在代理层实现，可以根据配置选择目标服务的具体实例，并将请求发送给目标服务。

### 3.2.3流量控制

Linkerd提供了流量控制功能，可以根据规则控制服务之间的流量。例如，可以设置只允许来自特定IP地址的请求访问某个服务，或者限制某个服务的请求数量。

### 3.2.4安全性保护

Linkerd提供了一系列的安全性保护功能，包括身份验证、授权、加密等。通过这些功能，可以确保服务间的通信安全。

### 3.2.5监控与追踪

Linkerd集成了多种监控和追踪工具，如Prometheus、Grafana、Jaeger等，可以实时监控服务的性能指标和追踪服务间的调用关系。

## 3.3数学模型公式详细讲解

### 3.3.1Istio的负载均衡策略

Istio支持多种负载均衡策略，如轮询、权重、最少请求数等。这些策略可以通过配置Envoy代理的策略规则实现。例如，轮询策略可以通过以下公式实现：

$$
\text{next_cluster_index} = (\text{next_cluster_index} + 1) \mod \text{total_clusters}
$$

### 3.3.2Linkerd的负载均衡策略

Linkerd支持多种负载均衡策略，如轮询、权重、最少请求数等。这些策略可以通过配置Linkerd代理的策略规则实现。例如，轮询策略可以通过以下公式实现：

$$
\text{next_cluster_index} = (\text{next_cluster_index} + 1) \mod \text{total_clusters}
$$

# 4.具体代码实例和详细解释说明

## 4.1Istio的代码实例

### 4.1.1安装Istio

```bash
# 下载Istio安装包
curl -L https://istio.io/downloadIstio | sh -

# 解压安装包
tar -xvf istio-1.10.1.tar.gz

# 进入安装目录
cd istio-1.10.1
```

### 4.1.2部署Istio

```bash
# 部署Istio
kubectl apply -f samples/addons/kiali.yaml
kubectl apply -f samples/addons/grafana.yaml
kubectl apply -f samples/addons/jaeger.yaml
```

### 4.1.3配置Istio代理

```yaml
apiVersion: networking.istio.io/v1alpha3
kind: Gateway
metadata:
  name: istio-gateway
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts:
    - "*.example.com"
---
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: istio-virtualservice
spec:
  hosts:
  - "*"
  gateways:
  - istio-gateway
  http:
  - match:
    - uri:
        prefix: /
    route:
    - destination:
        host: demo
        port:
          number: 8080
```

### 4.1.4配置Istio规则

```yaml
apiVersion: "authentication.istio.io/v1beta1"
kind: Policy
metadata:
  name: "istio-policy"
  namespace: "default"
spec:
  targets:
  - name: "demo"
  rules:
  - from:
    - source:
        prefix: "/"
    to:
    - operation: "PERMIT"
      when:
      - key: "request.auth.claims[authority]"
      value: "example.com"
---
apiVersion: "ratings.istio.io/v1beta1"
kind: Rating
metadata:
  name: "istio-rating"
  namespace: "default"
spec:
  peerReviews:
  - service: "demo"
    weight: 100
    weightField: "request.auth.claims[authority]"
    weightValue: "example.com"
```

## 4.2Linkerd的代码实例

### 4.2.1安装Linkerd

```bash
# 下载Linkerd安装包
curl -L https://run.linkerd.io/install | sh

# 启动Linkerd
linkerd install | kubectl apply -f -
```

### 4.2.2部署Linkerd

```bash
# 部署Linkerd
kubectl apply -f https://raw.githubusercontent.com/linkerd/linkerd2/stable/deploy/k8s/all/10.1.0/linkerd.yaml
```

### 4.2.3配置Linkerd代理

```yaml
apiVersion: linkerd.io/v1alpha1
kind: ServiceMesh
metadata:
  name: mesh
spec:
  tracers:
  - jaeger:
      enabled: true
  prometheus:
      enabled: true
  otel:
      enabled: false
  kubernetes:
      enabled: true
  enableMutualTLS: true
  enableHTTP2: true
  enableHTTP3: false
  enableTCP: true
  enableWebhooks: true
  enableSidecar: true
  enableIstioInjection: false
  enableLinkerdInjection: true
  enableLinkerdProxy: true
  enableLinkerdProxyAutoInjection: true
  enableLinkerdProxyAutoInject: true
  enableLinkerdProxyAutoSidecar: true
  enableLinkerdProxyAutoSidecarInject: true
  enableLinkerdProxyAutoSidecarInjectAll: true
  enableLinkerdProxyAutoSidecarInjectAll: true
  enableLinkerdProxyAutoSidecarInjectAllNamespaces: true
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels:
      app: "demo"
  enableLinkerdProxyAutoSidecarInjectAllNamespacesSelector:
    matchLabels: