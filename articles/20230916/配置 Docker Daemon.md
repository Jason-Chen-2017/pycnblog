
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Docker 是一种容器技术，是一个轻量级、高效的虚拟化技术，通过容器隔离进程，解决了环境依赖问题。从使用上来说，它像是一个便携式虚拟机一样，可以快速部署应用。

Docker 作为容器技术，其原生命令行工具 docker 提供了很强大的功能。但是默认情况下，docker 的服务端运行在本地，也就是说，所有 docker 命令只能在当前主机进行执行，如果想要远程管理或扩展，则需要配置 docker daemon 。本文主要介绍如何配置 docker daemon 以实现远程管理及扩展。

# 2. 安装 Docker CE

首先需要安装 Docker CE（Community Edition）并启动 Docker 服务：

```shell
$ sudo yum install -y docker-ce
$ sudo systemctl start docker
```

接下来设置开机自动启动 Docker 服务：

```shell
$ sudo systemctl enable docker
```

# 3. 配置 Docker Daemon

为了允许远程客户端访问 docker daemon ，需要修改配置文件 `/lib/systemd/system/docker.service`，添加 `--host=tcp://<ip>:<port>` 参数。如此，当系统启动后，docker daemon 将监听指定端口上的请求。

编辑配置文件 `/lib/systemd/system/docker.service` 文件，在其中找到 `ExecStart` 一项，并添加 `--host=tcp://<ip>:<port>` 参数：

```ini
[Unit]
Description=Docker Application Container Engine
Documentation=https://docs.docker.com
After=network.target firewalld.service
Wants=containerd.io

[Service]
Type=notify
# the default is not to use systemd for cgroups because the delegate issues still
# exists and systemd currently does not support the cgroup feature set required
# for containers run by docker
EnvironmentFile=-/etc/sysconfig/docker
ExecStart=/usr/bin/dockerd --host=tcp://0.0.0.0:2375 $OPTIONS \
    $DOCKER_STORAGE_OPTIONS $DOCKER_NETWORK_OPTIONS $ADD_REGISTRY \
    $BLOCK_REGISTRY $INSECURE_REGISTRY
ExecReload=/bin/kill -s HUP $MAINPID
TimeoutSec=0
RestartSec=2
Restart=on-failure
WorkingDirectory=/var/lib/docker
User=root
# Note that StartLimit* options were moved from "Service" to "Unit" in systemd 229,
# so try to avoid using them here, but keep them in other sections if you rely on
# the old version.
LimitNOFILE=infinity
LimitNPROC=infinity
LimitCORE=infinity
TasksMax=infinity

[Install]
WantedBy=multi-user.target
```

其中 `<ip>` 为服务器外网 IP 地址， `<port>` 为 docker daemon 运行的端口号。修改完毕之后，保存文件并重新加载 systemd 配置：

```shell
$ sudo systemctl daemon-reload
```

重启 docker 服务使修改生效：

```shell
$ sudo systemctl restart docker
```

至此，docker daemon 可以被远程客户端连接到指定的端口，并提供 docker 服务。

# 4. 测试 Docker Remote API

可以使用 `curl` 或其它 HTTP 请求工具测试远程 docker api 是否可用：

```shell
$ curl http://<server_ip>:<port>/version
{
  "Platform": {
    "Name": "Engineering Testing Environment"
  },
  "Components": [
    {
      "Name": "Engine",
      "Version": "18.09.1",
      "Details": {}
    }
  ],
  "Version": "18.09.1-ce",
  "ApiVersion": "1.39",
  "MinAPIVersion": "1.12",
  "GitCommit": "",
  "Os": "linux",
  "Arch": "amd64",
  "KernelVersion": "3.10.0-957.el7.x86_64",
  "Experimental": false,
  "BuildTime": "2019-02-07T12:14:03.54915141Z"
}
```

以上输出表示远程 docker 服务已正常响应版本信息请求。