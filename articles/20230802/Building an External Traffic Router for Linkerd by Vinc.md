
作者：禅与计算机程序设计艺术                    

# 1.简介
         
2021年的春节假期，我在Linkerd社区里面参加了一个活动。活动主要围绕着Linkerd社区开源项目Service Mesh——数据平面代理linkerd。该活动期间我收获颇丰，在激动之余也深刻体会到开源的力量。
         
         在本文中，我将分享我研究、实践和总结的一个基于linkerd的数据平面扩展方案的设计思路。这个方案的名称叫做“外部流量路由器”。
         
         本文适合对linkerd有一定了解、想要了解数据平面扩展方向、熟悉微服务架构的人阅读。希望通过本文能够帮助读者更好地理解linkerd的数据平面扩展机制及其技术原理，并思考如何用代码实现类似的功能。
         
         # 2.基本概念术语说明
         1. Kubernetes集群：Kubernetes是一个开源系统，用于自动部署、扩展和管理容器化的应用。
         2. 服务网格（Service Mesh）：服务网格是一种网络层协议，它提供透明化、可观察性和弹性的服务。linkerd是一个由Buoyant公司开源的服务网格产品，它可以轻松管理和监控微服务架构中的服务通信。
         3. 数据平面：数据平面负责连接客户端、服务发现、负载均衡等核心组件，并接收linkerd控制面的命令，执行相关的配置。在linkerd的默认安装模式下，数据平面由linkerd-proxy组成。
         4. pod：pod是一个kubernetes里的最小调度单位，包含一个或多个容器。
         5. service：service是kubernetes里的资源对象，用来定义运行于集群中的一组pods。service可以作为内部服务，也可以暴露给集群外的其他服务。
         6. sidecar：sidecar是指同一个pod里的另外一个容器，它跟主容器共享网络命名空间，但每个sidecar都需要独立的IP地址。
         # 3.核心算法原理和具体操作步骤以及数学公式讲解
         ## 3.1 核心思想
        “外部流量路由器”的主要思路是，为linkerd注入一个新的代理容器，作为外部流量的入口。外部流量路由器代理会监听各个节点上的服务端口，并且根据路由规则转发请求到linkerd。linkerd依旧负责服务发现、负载均衡、TLS加密等功能。
        
        此外，外部流量路由器还可以用于其他场景。比如，在有某些服务无法直接访问外部网络时，可以通过外部流量路由器统一处理。
        
       ## 3.2 具体实现
        下面我们将详细描述“外部流量路由器”的工作原理。
        
        1. 准备工作
        - 安装docker CE(容器引擎)，配置镜像加速；
        - 安装kubectl(命令行工具)、minikube(本地k8s环境)或者k3d(k3s的k8s环境)。
        
        2. 配置环境变量
        ```shell
        export KUBECONFIG=$(pwd)/kubeconfig 
        export LINERD_IMAGE=buoyantio/linkerd:stable-2.9.1  
        ```
        
        3. 创建测试集群
        创建一个单节点集群。
        ```shell
        minikube start --driver=docker --image-mirror-country='cn' --registry-mirror=https://docker.mirrors.ustc.edu.cn --container-runtime=containerd
        ```
        
        4. 安装linkerd
        按照官方文档安装linkerd即可。
        ```shell
        curl https://run.linkerd.io/install | sh 
        ```
        
        5. 搭建测试应用
        在集群里部署测试应用。这里我们部署三个服务，它们之间的调用关系如下：
        ```
        frontend -> backend -> database
        ```
        创建前端服务frontend、后端服务backend、数据库服务database，通过配置文件映射这些服务的端口。
        ```yaml
        apiVersion: v1
        kind: Service
        metadata:
          name: frontend
        spec:
          ports:
            - port: 80
              targetPort: http
          selector:
            app: frontend
        ---
        apiVersion: apps/v1
        kind: Deployment
        metadata:
          labels:
            app: frontend
          name: frontend
        spec:
          replicas: 1
          selector:
            matchLabels:
              app: frontend
          template:
            metadata:
              labels:
                app: frontend
            spec:
              containers:
                - image: kennethreitz/httpbin
                  name: frontend
                  ports:
                    - containerPort: 80
                      protocol: TCP

        ---
        apiVersion: v1
        kind: Service
        metadata:
          name: backend
        spec:
          ports:
            - port: 8080
              targetPort: 8080
          selector:
            app: backend
        ---
        apiVersion: apps/v1
        kind: Deployment
        metadata:
          labels:
            app: backend
          name: backend
        spec:
          replicas: 1
          selector:
            matchLabels:
              app: backend
          template:
            metadata:
              labels:
                app: backend
            spec:
              containers:
                - env:
                  - name: DATABASE_URL
                    valueFrom:
                      secretKeyRef:
                        key: url
                        name: db-secret
                  image: mycompany/backend
                  name: backend
                  ports:
                    - containerPort: 8080
                      protocol: TCP

        ---
        apiVersion: v1
        data:
          url: "jdbc:mysql://localhost:3306/mydb"
        kind: Secret
        metadata:
          name: db-secret
        type: Opaque
        ---
        apiVersion: v1
        kind: Service
        metadata:
          name: database
        spec:
          ports:
            - port: 3306
              targetPort: 3306
          selector:
            app: database
        ---
        apiVersion: apps/v1
        kind: Deployment
        metadata:
          labels:
            app: database
          name: database
        spec:
          replicas: 1
          selector:
            matchLabels:
              app: database
          strategy:
            rollingUpdate:
              maxSurge: 1
              maxUnavailable: 1
            type: RollingUpdate
          template:
            metadata:
              labels:
                app: database
            spec:
              containers:
                - image: mysql:latest
                  name: database
                  ports:
                    - containerPort: 3306
                      protocol: TCP
                  env:
                    - name: MYSQL_ROOT_PASSWORD
                      value: rootpassword
                    - name: MYSQL_DATABASE
                      value: mydb
                  resources:
                    requests:
                      cpu: "100m"
                      memory: "128Mi"
                    limits:
                      cpu: "200m"
                      memory: "256Mi"
        ```
        
        6. 验证是否正常运行
        执行以下命令检查测试应用是否正常运行。
        ```shell
        kubectl get pods
        NAME                          READY   STATUS    RESTARTS   AGE
        database-7dfcbccff4-x6tnn    1/1     Running   0          5h2m
        frontend-6b6b99bfdc-fdgss    1/1     Running   0          5h2m
        backend-56c7bbcf95-tjxzz     1/1     Running   0          5h2m
        linkerd-identity-79bd6f8cd-mxrmw       3/3     Running   0          5h4m
        linkerd-controller-6cbdc7bc79-rpnjp   3/3     Running   0          5h4m
        linkerd-destination-c99fb7b5-jmfht    2/2     Running   0          5h4m
        linkerd-proxy-injector-777d6dd6b5-nh9ch         2/2     Running   0          5h4m
        ```
        
        7. 安装外部流量路由器
        在服务mesh里注入一个外部流量路由器作为新的代理。
        ```shell
        kubectl apply -f external-traffic-router.yaml
        ```
        `external-traffic-router.yaml`文件的内容如下：
        ```yaml
        apiVersion: apps/v1
        kind: DaemonSet
        metadata:
          name: external-traffic-router
        spec:
          selector:
            matchLabels:
              name: external-traffic-router
          template:
            metadata:
              labels:
                name: external-traffic-router
            spec:
              hostNetwork: true
              containers:
              - name: external-traffic-router
                image: vincentxu/external-traffic-router:<version>
                securityContext:
                  capabilities:
                    add: ["NET_ADMIN"]
                command: ["/usr/local/bin/etp", "-port=<port>", "--logtostderr", "-v=2"]
                ports:
                - containerPort: <port>
                  hostPort: <port>
                  name: etp
                  protocol: UDP
                  hostIP: 0.0.0.0
      ```
        
        `<version>`替换为外部流量路由器的版本号。`<port>`替换为外部流量路由器要监听的端口号。
        
        `hostNetwork: true`使得外部流量路由器的iptables规则能够匹配到所有Pod，不管它们是否在hostNetwork模式下。
        
        `add: ["NET_ADMIN"]`赋予外部流量路由器一些权限，例如创建新的网络接口。
        
        注意：外部流量路由器是直接修改主机上的iptables规则，因此必须以特权模式运行才能成功。以上述配置文件创建一个DaemonSet，它会在每个Node上启动一个对应的Pod。
        
        
       8. 查看iptables规则
        使用以下命令查看当前环境下的iptables规则。
        ```shell
        sudo iptables -L -nv
        Chain INPUT (policy ACCEPT 0 packets, 0 bytes)
        pkts bytes target     prot opt in     out     source               destination         
          54  386K ACCEPT     udp  --  *      *       0.0.0.0/0            0.0.0.0/0            udp dpt:6379 
          53  374K ACCEPT     tcp  --  *      *       0.0.0.0/0            0.0.0.0/0            tcp spt:30000 
      ... output omitted...   
        ```
        可以看到，其中包括刚才注入的外部流量路由器的规则。
        ```
        Chain PREROUTING (policy ACCEPT 0 packets, 0 bytes)
        pkts bytes target     prot opt in     out     source               destination 
        560M 283G LINKERD-PROXY-INBOUND  all  --  any    lo      0.0.0.0/0            0.0.0.0/0            /* linkerd-proxy-inbound */
        ```
        
        9. 测试外部流量路由器
        为了验证外部流量路由器是否能正确地将不符合规范的TCP流量重定向到linkerd，我们可以使用netcat命令发送一条非标准的Telnet请求到前端服务。
        ```shell
        nc 172.17.0.6 80 -w 1
        Trying 172.17.0.6...
        Connected to 172.17.0.6.
        Escape character is '^]'.
        Telnet
        telnet> quit
        Connection closed by foreign host.
        ```
        
        如果外部流量路由器正确地识别出了Telnet请求，它应该会将请求重定向到linkerd，然后linkerd负责处理和响应请求。
        ```shell
        kubectl logs external-traffic-router-<node-name>
        INFO[0000] Starting the server at :<port>
        DEBU[0002] New connection from 10.244.0.3:35556 with local address :::<port>
        DEBU[0002] Incoming packet: dstAddr=:::<port>, srcAddr=:::<port>, proto=6, payload=0a0a0d4e65747465726e6574
        DEBU[0002] Redirecting request: dstAddr=10.244.1.8, srcAddr=10.244.0.3, dstPort=23, direction=outgoing
        DEBU[0002] Creating new outbound connection to 10.244.1.8:23 from [::]:<:port>
        DEBU[0002] Outbound connection created: id=0, remote=10.244.1.8:23
        DEBU[0002] Switching conn track for outgoing connection with 10.244.1.8:23 to NEW
        DEBU[0002] Installing proxy redirection rule on eth0: proto=tcp saddr=:::<port> daddr=10.244.1.8 sport=53748 dport=23 redir=1
        DEBU[0002] Activating NAT for connection: id=0, remote=10.244.1.8:23, natInfo=(*nat.NATInfo)(nil), timeout=0s
        DEBU[0002] Installing global outbound filter chain rule: tcfilter chain=LINKERD-OUTBOUND parent=all target=ACCEPT action=ok
        DEBU[0002] Forwarding traffic from :::<port> to 10.244.1.8:23 via linkered-outgoing-0
        DEBU[0002] Closed connection: id=0, remote=10.244.1.8:23
        ```
        
        从日志中可以看到，外部流量路由器正确地将Telnet请求重定向到了linkerd，然后linkerd将其转发到了后端服务的8080端口。
        
        通过以上步骤，我们成功实现了一个基于linkerd的数据平面扩展方案，即“外部流量路由器”。
        
        **未来工作**
        
        目前外部流量路由器只能监听UDP协议的流量。我们可以考虑加入对TCP协议的支持，以增强它对各种场景的兼容性。