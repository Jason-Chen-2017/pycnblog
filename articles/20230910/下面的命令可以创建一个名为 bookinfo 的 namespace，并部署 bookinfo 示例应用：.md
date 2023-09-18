
作者：禅与计算机程序设计艺术                    

# 1.简介
  
：本文以 Istio 为例，详细介绍了如何创建 Kubernetes namespace、部署 BookInfo 示例应用及访问该应用的流量路由规则等功能。
# 2.前置条件：您需要一个运行中的 Kubernetes 集群，本文所用到的所有命令都可以通过 kubectl 来执行。以下是使用 minikube 在本地创建单节点集群的步骤：
  $ minikube start --memory=4096
  $ kubectl create clusterrolebinding add-on-cluster-admin --clusterrole=cluster-admin --serviceaccount=kube-system:default
# 创建名为 bookinfo 的 namespace：
  $ kubectl create ns bookinfo
# 将 istio-injection 设置为 enabled 以便在这个命名空间中注入 sidecar proxy：
  $ kubectl label namespace bookinfo istio-injection=enabled
# 使用 Helm Chart 安装 BookInfo 示例应用：
  $ helm install --namespace bookinfo -f values-istio-demo.yaml install/kubernetes/helm/istio
# 您将看到如下输出，表示 Chart 已成功安装：
  NAME:   istio
  LAST DEPLOYED: Fri Aug 27 10:29:20 2018
  NAMESPACE: bookinfo
  STATUS: DEPLOYED
# 等待几分钟后，确认 BookInfo 服务及 pod 均处于 Running 状态：
  $ kubectl get svc -n bookinfo
  NAME              TYPE           CLUSTER-IP      EXTERNAL-IP     PORT(S)                                      AGE
  details           ClusterIP      10.98.62.238    <none>          9080/TCP                                     5m
  productpage       ClusterIP      10.105.154.37   <none>          9080/TCP                                     5m
  ratings           ClusterIP      10.97.164.59    <none>          9080/TCP                                     5m
  reviews           ClusterIP      10.101.192.80   <none>          9080/TCP                                     5m

  $ kubectl get pods -n bookinfo
  NAME                                        READY     STATUS    RESTARTS   AGE
  details-v1-6bcffc8fb-kbnmm                 2/2       Running   0          5m
  productpage-v1-7d8c9b68dd-z64zj            2/2       Running   0          5m
  ratings-v1-5f5dbfd8dc-hjxwh                2/2       Running   0          5m
  reviews-v1-ff6cb6bbcc-wctdf                2/2       Running   0          5m
  reviews-v2-7bf98c9ccb-drxsh                2/2       Running   0          5m
  reviews-v3-686ffb959f-bvsnq                2/2       Running   0          5m
# 通过浏览器或其他工具访问 BookInfo 示例应用：
  http://localhost:31380/productpage
# 查看 Istio 的 Dashboard：
  $ minikube service istio-pilot -n istio-system
# 添加一个新的 VirtualService 到 Kubernetes 中，把 /productpage 请求通过 reviews 版本 v3 路由到不同的 Pod：
  apiVersion: networking.istio.io/v1alpha3
  kind: VirtualService
  metadata:
    name: productpage-reviews-testversion
    namespace: bookinfo
  spec:
    hosts:
      - "bookinfo"
    gateways:
      - bookinfo-gateway
    http:
    - match:
        - uri:
            exact: /productpage
      route:
        - destination:
            host: productpage
            port:
              number: 9080
          weight: 100
        - destination:
            host: reviews
            subset: testversion
            port:
              number: 9080
          weight: 0
    subsets:
    - name: testversion
      labels:
        version: v3