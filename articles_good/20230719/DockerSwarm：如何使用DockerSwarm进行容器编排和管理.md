
作者：禅与计算机程序设计艺术                    
                
                
Docker Swarm 是 Docker 官方推出的集群管理工具，可以用来自动化部署和管理 Docker 服务。其最主要的功能包括：服务的编排、负载均衡、集群的伸缩、滚动升级等。

在容器化应用程序的开发和部署过程中，越来越多的人们转向了基于微服务的架构模式，在这种模式下，一个复杂的应用程序会被拆分成多个小型的模块，这些模块之间通过 RESTful API 或消息队列通信。

使用传统的基于容器的架构模式时，一个应用程序通常由多个独立的容器组成，并且需要人为地去实现相关的任务（如动态伸缩、服务发现、负载均衡等）。然而，随着微服务架构的发展，越来越多的人采用了基于微服务的架构模式，这意味着应用程序被拆分成许多的微服务，每个微服务都是一个独立的容器，因此也需要自动化地部署、管理和扩展它们。

Docker Swarm 提供了一套自动化的管理系统，它利用 Docker 的原生 API 和编排能力，通过调度器 (scheduler) 将应用部署到集群中，并提供面向服务的接口。借助 Docker Swarm，用户只需要定义好服务的配置，就可以让 Swarm 负责服务的创建、启动、停止、扩容等生命周期管理工作。

本文将从以下几个方面介绍 Docker Swarm 的使用方法：

1. Docker Swarm 集群的搭建与安装；

2. Docker Swarm 中的基本概念、术语和功能；

3. 通过 Docker Stacks 文件编排容器；

4. 在 Docker Swarm 上部署和管理服务；

5. 使用 Docker Swarm 的安全机制；

6. Docker Swarm 的高可用架构。 

希望通过阅读本文，读者能够学习到：

- Docker Swarm 集群的安装和使用；
- Docker Swarm 集群的架构和原理；
- Docker Swarm 中常用的命令行操作及其用法；
- 用 Docker Stacks 文件编排容器并管理 Docker Swarm 服务；
- 为 Docker Swarm 设置安全访问策略；
- Docker Swarm 集群的高可用架构设计。

# 2. 基本概念、术语和功能
## 2.1 Docker Swarm 集群
首先，我们先了解一下 Docker Swarm 集群的架构。Docker Swarm 集群是由多个 Docker 主机 (manager node) 构成的集群，这些节点被称作 manager。每台 Manager 节点都要运行 Docker daemon，然后通过 Swarm 模块与其他 Manager 或者 Worker 节点建立连接，以便执行集群内的各种操作。Worker 节点则是 Docker Swarm 的工作机器，主要负责处理分配给它的任务。

如下图所示，一台 Manager 节点和两台 Worker 节点组成了一个 Docker Swarm 集群。

![Alt text](https://img-blog.csdnimg.cn/2019071814474733.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg1MjYyNw==,size_16,color_FFFFFF,t_70)

**Manager 节点**：每台 Manager 节点上都要运行 Docker daemon，用于管理集群中的所有容器和服务，包括启停、伸缩、备份等。

**Worker 节点**：每台 Worker 节点上只能运行容器，不参与集群管理，用于接收任务并完成。Worker 节点主要负责处理分配给它的任务，包括拉取镜像、创建容器、启动容器、监控容器等。

## 2.2 SwarmKit 组件架构
Docker Swarm 中的各个组件是由 SwarmKit 构建的，该组件包含如下四个主要角色：

**Scheduler**: 根据资源利用率、任务需求、集群状态等信息选择出适合的 Worker 节点，并指派相应的任务到各个节点上。Scheduler 会监听集群中发生的变化并根据当前集群的状态来调整任务的分布情况。

**Raft Consensus Algorithm**: Raft 是一个分布式一致性算法，SwarmKit 使用它作为集群管理的基本算法，确保集群管理数据的一致性。Raft 可以保证整个集群的数据中心级可靠性。

**Dispatcher**: Dispatcher 是 SwarmKit 的网络代理模块，它接受客户端发来的请求，然后根据路由表将请求转发给对应的 SwarmKit 对象。Dispatcher 负责实现外部客户端与集群之间的通信。

**Object Store**: Object Store 是一个存储组件，它负责集群中各个对象的持久化存储。对象存储用于保存集群中所有的元数据，比如节点、服务、任务、网络等信息。当集群发生故障切换后，Object Store 可以帮助 SwarmKit 恢复之前的状态。

如下图所示，SwarmKit 的架构如此清晰易懂。

![Alt text](https://img-blog.csdnimg.cn/20190718150001931.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80Mzg1MjYyNw==,size_16,color_FFFFFF,t_70)

## 2.3 Service 编排与管理
Service 是 Docker Swarm 中最重要的对象之一，它代表着集群中的一个逻辑集合，其中的容器可以被动态管理、自动调度和弹性伸缩。

每个 Service 都有一个固定的名称，可以通过 DNS 查询到该 Service 的虚拟 IP 地址，也可以通过容器标签的方式来标识这个 Service。Service 可以通过 Service 模板来定义，模板中包含了 Service 需要的一系列参数。Service 模板定义了 Service 的镜像、环境变量、端口映射、卷绑定等属性。

当创建一个 Service 时，SwarmKit 会自动在集群中选择合适的 Worker 节点来运行容器，并通过 Scheduler 来决定如何将容器分布到不同的节点上。如果某个节点出现故障，SwarmKit 会自动重启对应节点上的容器，确保 Service 的可用性。

Service 还支持滚动更新、发布策略等高级特性，可以有效地管理应用程序的部署和版本变更。

## 2.4 Stacks 文件编排容器
Stack 是 Docker Swarm 中的一种编排方式，它允许用户创建和管理多个服务的集合，而无需单独操作每个服务。

Stack 文件是一个 YAML 文件，定义了一组 Service 模板，可以直接导入或导出到另一个 Docker Swarm 集群。通过 Stacks 文件，可以轻松地批量部署和管理一组相互依赖的容器。栈文件中可以定义 Service 模板，甚至可以跨多个 Compose 文件组合，形成更复杂的应用程序。

## 2.5 Job 计划任务
Job 是一种 Docker Swarm 中的异步操作，类似于 Kubernetes 中的 Job，可以用来处理批量任务。Job 中的多个容器会按照顺序串行执行，除非其中一个失败，否则 Job 操作才算成功。

例如，如果需要运行一个批处理任务，要求容器必须按顺序依次执行，那么可以使用 Job 来解决。通过设置依赖关系，Job 也可以定义两个容器的执行顺序，进一步提升灵活性。

## 2.6 Volume 共享与持久化存储
Volume 是 Docker Swarm 中的一种持久化存储机制，可以用来分享容器间的数据和文件。当容器重新启动后，Volume 中的数据仍然存在。Volume 可以在不同容器间共享或进行数据传递。

## 2.7 Secret 加密存储
Secret 是 Docker Swarm 中的敏感数据存储机制，它可以用来保存加密的密码、密钥、证书等敏感数据。Secret 数据在整个集群内不可见，只有经过授权的用户才能访问。

## 2.8 Config 配置管理
Config 是 Docker Swarm 中的配置数据管理机制，它提供了一种集中式的、高度安全的配置数据管理手段。管理员可以将配置文件上传到 Swarm，其他用户可以在运行容器时挂载 Config 并读取相应的文件。

## 2.9 Network 网络管理
Network 是 Docker Swarm 中的网络管理机制，它可以用来在不同容器间进行通信。Docker Swarm 默认提供三种类型的网络：

- **默认网络**: 这是 Docker Swarm 中的默认网络类型，提供了对外界的网络通信功能，但无法实现内部服务的通信。

- **桥接网络**: 这是一种简单的网络类型，容器直接通过物理网络链接起来。

- ** Overlay 网络**: 这是一种分布式的网络类型，通过 VPN 技术，容器间可以互相通信。Overlay 网络可以实现容器的动态伸缩、快速部署、零宕机部署等特性。

# 3. 安装与使用
## 3.1 安装 Docker CE
安装 Docker CE 可以参照 Docker 的文档进行。安装完毕后，测试一下是否安装正确。

```bash
sudo docker version
```

输出结果应该如下：

```
Client:
 Version:           18.06.1-ce
 API version:       1.38
 Go version:        go1.10.3
 Git commit:        e68fc7a
 Built:             Tue Aug 21 17:24:56 2018
 OS/Arch:           linux/amd64
 Experimental:      false

Server:
 Engine:
  Version:          18.06.1-ce
  API version:      1.38 (minimum version 1.12)
  Go version:       go1.10.3
  Git commit:       e68fc7a
  Built:            Thu Aug 22 01:17:44 2018
  OS/Arch:          linux/amd64
  Experimental:     false
```

## 3.2 创建 Swarm 集群
创建 Swarm 集群非常简单，只需在主节点上执行 `docker swarm init` 命令即可。

```bash
sudo docker swarm init --advertise-addr <主节点IP>
```

指定 `--advertise-addr` 参数后，Swarm 集群就会初始化。集群初始化后，Swarm 管理节点会返回一个 token。该 token 必须保存好，后续使用 Swarm 命令行必须提供该 token。

```bash
Swarm initialized: current node (eo6vlcwcdvjmlfnkoifaoqgpz) is now a manager.

To add a worker to this swarm, run the following command:

    docker swarm join --token <KEY>  192.168.0.10:2377

To add a manager to this swarm, run 'docker swarm join-token manager' and follow the instructions.
```

复制上述命令中的 `join --token...` 命令，在其它任意节点上执行，添加 Worker 节点到 Swarm 集群。其它节点也可以重复执行上述命令添加 Manager 节点。

```bash
sudo docker swarm join \
    --token <KEY> \
    192.168.0.10:2377
```

## 3.3 添加节点到集群
在某些情况下，可能需要增加一些节点到 Swarm 集群中，以满足业务的需求。可以使用以下命令增加新的节点。

```bash
sudo docker swarm join-token worker | sudo tee /tmp/worker-token.txt
```

以上命令会生成一个 Worker Token，可以通过该 Token 添加新节点到 Swarm 集群。假设新节点的 IP 为 192.168.0.11，那么可以将 Worker Token 发送给新节点。

```bash
sudo docker swarm join \
    --token $(cat /tmp/worker-token.txt) \
    192.168.0.10:2377
```

## 3.4 更新集群配置
在 Swarm 集群中，可以修改集群的一些配置参数。比如，可以修改最大连接数 (`--default-addr-pool`)，或者调整 Swarm 集群的 TLS 证书 (`--cert-expiry`)。

```bash
sudo docker swarm update \
    --default-addr-pool 128.0.0.0/8 \
    --cert-expiry 90days
```

上面命令会修改 Swarm 集群的默认 IP 池 (`--default-addr-pool`) 为 `128.0.0.0/8`，TLS 证书的有效期为 90 天 (`--cert-expiry`)。

## 3.5 查看集群信息
查看集群的详细信息可以使用 `docker info` 命令。

```bash
$ sudo docker info
Containers: 50
 Running: 24
 Paused: 0
 Stopped: 26
Images: 58
Server Version: 18.06.1-ce
Storage Driver: overlay2
 Backing Filesystem: extfs
 Supports d_type: true
 Native Overlay Diff: true
Logging Driver: json-file
Cgroup Driver: cgroupfs
Plugins:
 Volume: local
 Network: bridge host macvlan null overlay
 Log: awslogs fluentd gcplogs gelf journald json-file logentries splunk syslog
Swarm: active
 NodeID: ognwjrhuivzrluotlylpxpucy
 Is Manager: true
 ClusterID: jvy6lzrnqhxckiugsldhjuqvj
 Managers: 1
 Nodes: 3
 Orchestration:
  Task History Retention Limit: 5
 Raft:
  Snapshot Interval: 10000
  Number of Old Snapshots to Retain: 0
  Heartbeat Tick: 1
  Election Tick: 3
 Dispatcher:
  Heartbeat Period: 5 seconds
 CA Configuration:
  Expiry Duration: 3 months
 Node Address: 192.168.0.10
Manager Addresses:
 192.168.0.10:2377
Runtimes: runc
Default Runtime: runc
Init Binary: docker-init
containerd version: 468a545b9edcd5932818eb9de8e72413e616e86e
runc version: 69663f0bd4b60df09991c08812a60108003fa340
init version: fec3683
Security Options:
 seccomp
  Profile: default
Kernel Version: 4.18.0-10-generic
Operating System: Ubuntu 18.04.1 LTS
OSType: linux
Architecture: x86_64
CPUs: 8
Total Memory: 15.53GiB
Name: yinghao
ID: LFHD:EZOR:LVMP:YRUQ:JQTW:EHXU:OXCA:KYYT:ZYQT:BQWF:S3FP:RRIB
Docker Root Dir: /var/lib/docker
Debug Mode (client): false
Debug Mode (server): false
Registry: https://index.docker.io/v1/
Labels:
 provider=digitalocean
Experimental: false
Insecure Registries:
 127.0.0.0/8
Live Restore Enabled: false
```

通过 `docker ps` 命令可以看到当前所在节点的容器列表。

```bash
$ sudo docker ps
CONTAINER ID        IMAGE               COMMAND                  CREATED              STATUS              PORTS               NAMES
52a80cf39d3f        5b0bcabd39af        "/bin/sh -c '/usr/sb…"   15 minutes ago       Up 15 minutes                           brave_banach
dcddaa676aa7        7a2cb3ec9c9b        "traefik"                15 minutes ago       Up 15 minutes                           traefik
5ff73c0c415f        83373d5bf1bb        "portainer"              15 minutes ago       Up 15 minutes                           portainer
f19fd3ca246c        redis               "docker-entrypoint.s…"   15 minutes ago       Up 15 minutes                           redis
```

## 3.6 列出节点
可以使用 `docker node ls` 命令列出集群中的所有节点。

```bash
$ sudo docker node ls
ID                            HOSTNAME            STATUS              AVAILABILITY        MANAGER STATUS      ENGINE VERSION
eo6vlcwcdvjmlfnkoifaoqgpz *   yinghao             Ready               Active              Leader              18.06.1-ce
lnfjqqvabhdzfetlrkuzhkjqw     registry.example.com Ready               Active                                  18.06.1-ce
mlji1erwhqlwpbo6hbikue0lb     worker-node         Ready               Active                                  18.06.1-ce
```

## 3.7 退出 Swarm 集群
退出 Swarm 集群可以使用 `docker swarm leave --force` 命令。

```bash
$ sudo docker swarm leave --force
Node left the swarm.
```

# 4. 服务编排和管理
## 4.1 服务模板
首先，需要定义服务的模板，也就是 Service 模板。定义好的服务模板可以以 `.yaml` 文件的形式保存在本地磁盘，然后通过 `docker stack deploy` 命令部署到 Docker Swarm 集群。

示例服务模板：

```yaml
version: "3"
services:
  web:
    image: nginx:alpine
    ports:
      - "80:80"
    networks:
      - frontend

  db:
    image: postgres:latest
    environment:
      POSTGRES_PASSWORD: example
    volumes:
      - myapp-data:/var/lib/postgresql/data
    networks:
      - backend

volumes:
  myapp-data: {}

networks:
  frontend:
  backend:
```

这是一个简单的服务模板，包括两个服务：web 和 db。web 服务暴露了 HTTP 服务的 80 端口，db 服务使用 PostgreSQL 数据库。

注意事项：

- 指定好镜像名称 (`image`) 和端口映射 (`ports`)；
- 如果服务需要绑定到外部网络，则需要声明相应的网络 (`networks`)；
- 服务需要挂载外部目录或文件，则需要声明相应的卷 (`volumes`)；
- 服务使用的环境变量 (`environment`)；
- 如果没有特殊需求，应尽量使用较小的镜像大小；
- 不要在生产环境中使用 `latest` 标签；
- 请不要在 `volumes` 中定义与 `Dockerfile` 中的 `VOLUME` 指令冲突的卷名；
- 在 `network` 中定义的名称必须唯一。

## 4.2 服务部署
在准备好服务模板之后，可以使用 `docker stack deploy` 命令部署服务。

```bash
$ sudo docker stack deploy -c myapp.yaml mystack
Creating network mystack_backend
Creating network mystack_frontend
Creating service mystack_db
Creating service mystack_web
```

`-c` 参数指定了服务模板文件的路径，`mystack` 表示 Stack 名称。

部署过程结束后，可以使用 `docker stack services` 命令查看正在运行的服务。

```bash
$ sudo docker stack services mystack
ID                  NAME                MODE                REPLICAS            IMAGE                   PORTS
ozxpvpuheo6km        mystack_web         replicated          1/1                 nginx:alpine            *:80->80/tcp
qfxsb0irfvrrk        mystack_db          replicated          1/1                 postgres:latest
```

可以看到，mystack 有两个服务：web 和 db。每个服务都有一个固定且唯一的 ID，MODE 表示服务的类型，目前 Docker Swarm 支持两种服务类型：replicated 和 global。

## 4.3 服务管理
### 4.3.1 服务缩放
在运行中，可以对服务进行缩放操作。

```bash
$ sudo docker service scale mystack_web=3
mystack_web scaled to 3
overall progress: 3 out of 3 tasks
1/3: running   [==================================================>]
2/3: running   [==================================================>]
3/3: running   [==================================================>]
verify: Service converged
```

### 4.3.2 服务重新部署
如果服务发生错误或需要更新，可以使用 `docker service update` 命令重新部署服务。

```bash
$ sudo docker service update --image nginx:alpine mystack_web
Updating service mystack_web (id: qxgjxtxhkmklh6twuramle7v8)
images updated: 1
tasks: starting 1
tasks: updating 1
update completed
```

使用 `--image` 参数可以指定更新后的镜像名称。

### 4.3.3 服务暂停和恢复
可以使用 `docker service pause|unpause` 命令暂停或恢复服务。

```bash
$ sudo docker service pause mystack_web
mystack_web paused
```

暂停服务后，它将不会再接收新的任务，但仍保留当前的副本。

```bash
$ sudo docker service unpause mystack_web
mystack_web unpaused
```

恢复服务后，它将继续接收新的任务。

### 4.3.4 服务删除
可以用 `docker service rm` 命令删除服务。

```bash
$ sudo docker service rm mystack_web
mystack_web
```

删除服务后，其下的容器也会同时被删除。

### 4.3.5 服务日志查询
可以使用 `docker service logs` 命令查询服务的日志。

```bash
$ sudo docker service logs mystack_web
myweb.1.of8a8ctvbrtp@yinghao    | 172.20.0.1 - - [09/Jul/2019:08:28:11 +0000] "GET / HTTP/1.1" 200 612 "-" "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36" "-
```

## 4.4 服务健康检查
可以通过 Healthcheck 来监测容器的健康状态。Healthcheck 是一个 HTTP 请求或者 TCP 检查，可以用来确定容器是否处于健康状态。

可以通过 Dockerfile 中的 `HEALTHCHECK` 指令来定义 Healthcheck。

```Dockerfile
FROM busybox
RUN touch /healthcheck.txt
HEALTHCHECK --interval=5s --timeout=3s CMD cat /healthcheck.txt || exit 1
CMD ["tail", "-f", "/dev/null"]
```

示例 Dockerfile，定义了一个 Healthcheck，每隔 5 秒检测一次 `/healthcheck.txt` 是否存在，超时时间为 3 秒。如果超过 3 秒没有检测到 `/healthcheck.txt`，则认为容器不健康。

如果 Healthcheck 检测到异常，则会自动重启容器。

也可以在 `docker service create` 命令中指定 Healthcheck。

```bash
$ sudo docker service create \
    --name healthcheck \
    --health-cmd="curl -f http://localhost/" \
    --health-interval=5s \
    --health-retries=3 \
    --health-timeout=3s \
    nginx:alpine
```

创建的服务会定期发送 GET 请求到 localhost，超时时间为 3 秒，连续失败 3 次则认为服务不健康。

# 5. Secrets 管理
## 5.1 Secrets 简介
Secrets 是 Docker Swarm 中的敏感数据管理机制，可以用来保存加密的密码、密钥、证书等敏感数据。通过 Swarmkit 提供的密钥编解码器，可以将加密的 secrets 解码为明文形式。

## 5.2 加密 Secrets
为了加密 Secrets，可以将原始数据使用 Docker Swarm 集群中的 manager 节点上已有的 KMS （Key Management Services）服务进行加密。加密后的 secrets 就可以安全地存储在 Docker Swarm 集群中。

```bash
$ echo -n "<secret>" | docker secret create mysecret -
$ docker service create \
   --name myservice \
   --secret source=mysecret,target=/run/secrets/mysecret \
   alpine ls /run/secrets/mysecret
```

上面的命令使用 `echo` 命令创建 `<secret>` 字符串，然后使用 `docker secret create` 命令加密该字符串为 `mysecret`。

由于 `ls` 命令不能显示加密后的 Secrets，所以使用了一个 dummy 服务来将加密后的 secrets 从容器内解密出来。

## 5.3 分配 Secrets
可以使用 `docker service update` 命令分配 Secrets 给容器。

```bash
$ docker service update \
    --secret-add source=<source>,target=<target> \
    <servicename>
```

使用 `--secret-add` 参数可以指定要添加的源和目标。

```bash
$ sudo docker service update \
    --secret-add source=mysecret,target=/run/secrets/mysecret \
    mystack_web
updating secret mysecret for mystack_web
$ sudo docker service inspect --pretty mystack_web
ID:		2ruwbzbaarfoahtosj9nhnbcr
Name:		mystack_web
Image:		nginx:alpine@sha256:f902cfc1c6fa70b0f51d80befb7f24c7a0f40d56a525d312e1e7e9dc53c6a618
Node:		yinghao
Task Template:
 ContainerSpec:
  Image:		nginx:alpine@sha256:f902cfc1c6fa70b0f51d80befb7f24c7a0f40d56a525d312e1e7e9dc53c6a618
  Labels:
   com.docker.stack.namespace=mystack
  StopGracePeriod:	10000000000
  Env:
   HELLO:WORLD
  Mounts:
   Target:	/run/secrets
   Source:	mysecret
   Type:	volume
 Placement:
  Constraints:
   - Node.Role == manager
  Preferences:
   - spread: node.labels.region
 UpdateConfig:
  Parallelism:	1
  Delay:		10000000000
  Order:	stop-first
 RollbackConfig:
  Parallelism:	1
  Delay:		10000000000
  FailureAction:	pause
 Networks:	[bridge]
 Endpoint Mode:	vip
 Resources:
  Limits:
   CPUs:		0
   MemoryBytes:	0
  Reservations:
   CPUs:		0
   MemoryBytes:	0
 RestartPolicy:
  Condition:	any
  Delay:		5000000000
  MaxAttempts:	0
 Termination Grace Period:	30000000000
 Images:		2
  test:         latest
Secrets:		1
  SecretName:	mysecret
  Filenames:	[]
  Secrets:		map[/run/secrets/mysecret:REDACTED]]
Status:
 State:			running
 Started At:		2019-07-18T07:23:05.1244165Z
 Updated At:		2019-07-18T07:32:17.6779528Z
 Previous Spec:	2ruwbzbaarfoahtosj9nhnbcr:
    Name:		mystack_web
    Image:		nginx:alpine@sha256:f902cfc1c6fa70b0f51d80befb7f24c7a0f40d56a525d312e1e7e9dc53c6a618
    Node:		yinghao
    UpdateStatus:
       State:	completed
       StartedAt:	2019-07-18T07:23:05.1244165Z
       CompletedAt:	2019-07-18T07:32:17.6779528Z
    Annotations:
       com.docker.stack.namespace: mystack
       
Previous Status:
	1 instances running
	1 instances failed
	0 instances complete
	1 instances paused
Restarts:	0

