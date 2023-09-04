
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## （一）什么是Netflix？
Netflix 是美国互联网公司，创立于 1997 年，在全球拥有超过 90% 的流量。Netflix 是一个视频分享网站、音乐流媒体平台、网络游戏发布商、在线电影租赁平台、购物网站等。截止目前（2020年7月），Netflix 在美国有 1.4 亿注册用户和近 1.1 亿独立影院观众。Netflix 作为互联网企业的领军者之一，占据了当下最热门的 IT 行业之首。然而，Netflix 在开源社区中的影响力却远不及其传统竞争对手。
Netflix 提供了丰富的免费视频、电影、电视节目等产品和服务，同时还推出了一系列付费内容。因此，Netflix 本身也受到广泛的关注和青睐，它通过技术驱动，希望为用户提供更加优质、便捷、舒适的网络服务。Netflix 非常注重开源，开源项目包括多个孵化器、软件开发工具包以及多个开源项目。
随着开源技术的日渐成熟、云计算的蓬勃发展以及容器技术的普及，越来越多的公司开始将自己的业务转移到云端。因此，Netflix 在过去几年间开始关注并采用微服务架构，逐步将内部系统迁移到基于容器技术的架构上。
## （二）Eureka是什么？
Netflix Eureka 是 Netflix 开源的一个服务注册与发现组件。它是一个 RESTful 服务，用于定位分布式应用中的各个服务，以实现动态集群管理、负载均衡和故障转移。Eureka 通过“注册”和“查询”两个阶段提供服务。服务启动后，会首先向 Eureka 发送自身的元数据信息（如 IP 地址、端口号、主机名等）。其他服务可以通过 Eureka 查询获取当前可用服务列表，然后基于软负载均衡策略选择一个合适的节点进行调用。Eureka 使用 REST API 来提供服务注册和查询功能，客户端可以从中获取可用服务列表，并根据需要负载均衡地访问这些服务。另外，Eureka 可以检测服务的健康状况，并自动移除不可用的服务节点。Eureka 的主要特性包括：

 - 简单易用：使用 HTTP(S) 或者 DNS 查找服务；客户端只需知道服务端地址即可；支持多种负载均衡策略，包括轮询、随机、基于响应时间加权的负载均衡；服务注册中心是一个独立部署的组件，没有中心节点也可正常工作。

 - 健壮性：Eureka 具有高可用性，即使在部分节点失效的情况下依然可以提供服务。而且它还具备自我保护机制，能够防止因网络分区导致部分节点之间出现网络分裂的情况。另外，Eureka 提供了容错机制，当遇到任何异常情况时仍然可以保证服务可用。

 - 分布式环境：Eureka 支持跨区域的异构环境，甚至可以连接不同的数据中心的 Eureka Server。由于使用的是 RESTful API，客户端无需依赖任何特定框架或编程语言，即可实现与 Eureka 的交互。

 - 定制化控制：Eureka 有多套自有的接口，满足不同场景下的需求。例如，对于特定类型的服务，可以自定义不同的负载均衡策略。另外，Eureka 提供了一整套完整的统计信息，帮助运维人员监控集群运行状态。
## （三）Eureka组件结构
Eureka组件的主要结构包括以下几个部分：

 - Eureka Server：它是一个独立的服务器，用来存储服务注册表信息，并且对外提供服务注册与查询的 RESTful API。集群中的任意节点都可以充当 Eureka Server。

 - Eureka Client：该客户端可以集成在应用程序中，用来向 Eureka Server 注册自己，并接收服务的变化通知。

 - Service Provider：这是向 Eureka Server 报告其自身状态的组件，通常是一个 Spring Boot 应用程序。

 - Service Consumer：这是消费者组件，用来向 Service Provider 获取服务的调用地址。

 - Discovery Client：这是 Spring Cloud 的子模块，用来实现 Spring Cloud 中服务的注册发现功能。它封装了 Service Discovery 的细节，使得 Spring Cloud 的工程师只需关心如何使用服务而不需要考虑底层的复杂实现。
## （四）Eureka工作流程
Eureka的工作流程如下图所示：
1. 当某个服务启动的时候，向 Eureka 发送自身的元数据信息，比如 ip 地址、端口号、主机名称等。

2. Eureka 收到注册请求后，会把该服务的信息存储到自己的服务注册表中，同时也会把这个信息通过消息通道发送给所有订阅了 Eureka 的服务。如果同一时刻有多个服务注册到同一台 Eureka Server 上面，Eureka 会在其中选择其中一个作为注册入口。

3. 如果服务的主节点宕机，Eureka 会把失效节点上的服务信息从注册表中摘除，并且通知所有的订阅了 Eureka 的服务，这样做的目的是为了避免单点故障。

4. 当服务调用方从 Eureka 获取服务列表之后，会轮询选择其中一个服务节点进行调用。

5. 如果调用失败或者超时，Eureka 会再次从服务注册表中拉取服务信息，并选择其中一个重新尝试调用。

6. Eureka 默认每隔 30 秒向服务提供方发送一次心跳消息，用于表明本节点仍然存活。

7. 如果某个服务超过一定时间没有发送心跳，Eureka 将认为此服务节点已经离线，并从注册表中摘除相应信息。

## （五）Eureka组件安装及配置
### （1）下载源码
从 github 上下载 eureka 源码并编译：
```bash
git clone https://github.com/Netflix/eureka.git
cd eureka
mvn clean install -DskipTests=true
```
### （2）配置环境变量
新建配置文件 `eureka.env` ，写入以下内容：
```bash
JAVA_OPTS="$JAVA_OPTS -Deureka.environment=production"
```
设置 JAVA_OPTS 为以上参数。
### （3）创建服务脚本文件
新建服务脚本文件 `start-service.sh`，并写入以下内容：
```bash
#!/bin/bash
echo "Starting service..."
java $JAVA_OPTS \
  -jar myapp.jar \
   --spring.profiles.active=prod & echo $! > app.pid
echo "Service started."
```
该脚本文件用来启动服务。
### （4）创建启动脚本文件
新建启动脚本文件 `startup.sh`，并写入以下内容：
```bash
#!/bin/bash
echo "Starting eureka server..."
nohup java $JAVA_OPTS -Deureka.client.registerWithEureka=false \
                     -Deureka.client.fetchRegistry=false \
                     -Xmx200m -Xms200m \
                     -jar target/eureka-server-2.2.1.RELEASE.jar >> logs/output.log &
sleep 10 # Wait for server to start up before continuing
echo "Eureka server started."
```
该脚本文件用来启动 Eureka Server 。
### （5）启动 Eureka Server 和服务
分别执行以下命令启动 Eureka Server 和服务：
```bash
./startup.sh   # 启动 Eureka Server
./start-service.sh    # 启动服务
```