
作者：禅与计算机程序设计艺术                    

# 1.简介
         
1.1 Docker 是什么？
             Docker 是一个开源的应用容器引擎，让开发者可以打包、测试和分发他们的应用或者服务，将其运行在任意的沙盒环境中，真正实现“一次构建，到处运行”。基于容器的虚拟化技术实现了应用程序的快速部署、跨平台移植等特性。

             从基础概念上来说，Docker 可以帮助开发者搭建出属于自己的开发环境，例如本地主机上的 IDE 编辑器、编译工具、数据库服务器、Web 服务器以及其他所需组件。通过 Docker 的镜像机制，开发者可以方便地制作定制的环境供他人使用，避免了配置复杂的环境导致的环境差异性问题。同时，Docker 提供了容器管理和集群编排功能，方便开发者进行自动化部署。

         1.2 本书结构
             Ⅰ Docker 安装
             Ⅱ Docker 命令详解
                1-1 配置镜像加速器（可选）
                1-2 拉取镜像
                1-3 查看镜像
                1-4 创建并启动容器
                1-5 进入容器
                1-6 停止/删除容器
                1-7 删除镜像
             Ⅲ Dockerfile 编写
                 2-1 Dockerfile 概述
                 2-2 使用 ADD 和 COPY 添加文件
                 2-3 RUN 执行命令
                 2-4 CMD 设置启动命令
                 2-5 ENTRYPOINT 设置入口点
                 2-6 ENV 设置环境变量
                 2-7 EXPOSE 暴露端口
                 2-8 VOLUME 创建卷
             Ⅳ 数据卷
                 3-1 数据卷概述
                 3-2 在Dockerfile 中创建数据卷
                 3-3 启动容器时绑定数据卷
             Ⅴ Docker Compose
                 4-1 Compose 概述
                 4-2 安装 Compose
                 4-3 使用 Compose 文件快速构建环境
                 4-4 Compose 命令详解
                     4-4-1 up
                     4-4-2 down
                     4-4-3 ps
                     4-4-4 logs
                     4-4-5 exec
                     4-4-6 config
                 4-5 多容器间通信（可选）
             Ⅵ Docker Swarm（可选）
                5-1 Swarm 介绍
                5-2 安装 Swarm
                5-3 Swarm 模式
                5-4 服务的发布与更新
                5-5 存储卷与网络
             Ⅶ 安全防护（可选）
             Ⅷ 参考文献

         # 2.基本概念术语说明
         2.1 镜像（Image）
            Docker 镜像就好比是一个轻量级、独立的文件系统，其中包含了软件依赖及配置信息。镜像通常包括一个操作系统环境和一些特定应用的二进制文件。每当启动一个容器时，就会用镜像作为它的模板。

            每个镜像都由 Build 时生成的一组文件构成。这些文件一起告诉 Docker 底层文件系统应该如何设置，以及该怎么运行才能使得这个容器可以正常工作。另外，镜像还包含了元数据，比如作者、说明、标签、镜像大小等。通过元数据中的标签，用户可以指定要使用的哪个版本的镜像。

         2.2 容器（Container）
            Docker 容器是在镜像的基础上启动的一个可写层。它并没有变成 Virtual Machine (VM) 的形式，因为容器不包含完整的内核，只是一个进程隔离环境。因此，相对于 VM 来说，容器更加轻量级和迅速。

            当 Docker 需要扩展容量的时候，会使用复制而不是新建容器，从而节省资源。与传统虚拟机不同的是，Docker 的容器可以在任何地方运行，并且在不同的机器上可以互通。所以，如果某个服务需要多台机器提供服务，那么使用 Docker 可以很好的满足需求。

            通过查看正在运行的容器，可以通过命令 `docker ps` 或在 Docker Dashboard 上看到所有正在运行的容器。

         2.3 仓库（Repository）
            Docker 仓库用来保存、分发 Docker 镜像。每个仓库中可以包含多个命名空间（namespace）。当使用 `docker push` 或 `docker pull` 命令时，实际上就是对仓库中的镜像进行拉取或推送。默认情况下，DockeHub 是 Docker 官方维护的公共仓库，其他第三方团队或个人也可以创建私有仓库用于内部发布。

          2.4 Dockerfile
            Dockerfile 是一个文本文件，其中包含了一系列命令，用来在一个镜像里创建一个新的容器。dockerfile 中的命令会被 Docker 按照顺序执行，从基础镜像开始构建、安装软件，最终创建一个新的镜像。

            dockerfile 中主要有以下五类命令：

              FROM: 指定基础镜像。
              RUN: 在当前镜像基础上执行一条指令。
              COPY: 将宿主机目录下的文件拷贝进镜像。
              ADD: 将宿主机压缩文件或文件夹拷贝至镜像内。
              WORKDIR: 为后续的 RUN、CMD、ENTRYPOINT、COPY、ADD 命令指定工作目录。

            使用 `docker build` 命令可以根据 dockerfile 来构建镜像。

        # 3.核心算法原理和具体操作步骤以及数学公式讲解
         3.1 Docker 镜像的分层结构
             Docker 的镜像采用分层存储的方案。镜像构建时，会一层一层构建，前一层是后一层的基础。每个层都可以称之为一个镜像层，只有最后一层是可读写的，之前的所有层都是不可读写的。这样设计的目的是为了高效率利用存储空间和提高性能，最大限度的减少磁盘 IO 操作。

             下面是 Dockerfile 示例：

                FROM busybox
                MAINTAINER keke <EMAIL>
                RUN mkdir /data
                VOLUME ["/data"]
                CMD ["sleep", "infinity"]

             这个 Dockerfile 使用 busybox 镜像作为基础镜像，然后添加了一个新的层，在 `/data` 目录下创建一个新的目录并设为挂载点。启动此镜像时，`/data` 目录会自动映射到宿主机的某一目录下。

             如果将 Dockerfile 文件保存在 helloworld.Dockerfile，可以使用如下命令构建镜像：

                 docker build -t helloworld:v1. --no-cache

             `--no-cache` 参数表示不需要使用缓存，每次都重新构建整个镜像。`-t` 参数表示给新镜像定义一个名称和标签（tag），这里用 `helloworld:v1`。`.` 表示指明 Dockerfile 文件所在的路径。

             构建成功之后，可以通过 `docker images` 命令查看已经创建的镜像。

         3.2 Dockerfile 的基本语法
             Dockerfile 中的每条指令都必须以小写字母开头，且为大写字母。所有的指令都支持参数，但是大多数指令并不需要指定参数。如果需要传递参数，则使用双引号。

             注释： `#`，后面跟随的行会被忽略。

             下面是 Dockerfile 的基本语法：

                # comment
                FROM image
                LABEL key="value"
                RUN command param1 param2...
               ...
                [ONBUILD trigger_instruction]
                [MAINTAINER name]

         3.3 Dockerfile 的指令详解
             本章节重点介绍 Dockerfile 的八大常用指令，以及相关命令参数的使用方法。

              Ⅰ FROM
                  1. syntax: FROM <image>[:<tag>] [AS <name>]
                  2. description: 该命令指定基础镜像，并可选择性增加一个别名（name）。如果不指定 tag，则默认使用 latest 标签。如果指定了 tag，则应该始终使用显式的标签。
               Ⅱ LABEL
                  1. syntax: LABEL <key>=<value> [<key>=<value>...]
                  2. description: 为镜像设置标签。
               Ⅲ RUN
                  1. syntax: RUN <command>
                  2. description: 在当前镜像基础上，运行指定的命令，命令必须以单引号括起来，或换行符结尾。
               Ⅳ VOLUME
                  1. syntax: VOLUME ["/data"]
                  2. description: 标记一个挂载点，供运行容器使用。挂载到此处的任何东西都会持久化到主机，即便容器被删除也不会丢失。
               Ⅴ CMD
                  1. syntax: CMD ["executable","param1","param2"]
                  2. description: 在容器启动时，执行命令。命令既可以直接执行，也可以借助 shell 执行。除非用户覆盖掉这个命令，否则 Docker 会提供一个默认命令，一般是 ENTRYPOINT 指定的命令。
               Ⅵ ENTRYPOINT
                  1. syntax: ENTRYPOINT ["executable","param1","param2"]
                  2. description: 设置容器启动时执行的命令。容器启动时会检查是否存在 override 参数，若不存在则执行 entrypoint 命令；若存在则调用 override 命令替代entrypoint命令。如果 override 参数存在，则在 ENTRYPOINT 命令的后面追加参数。
               Ⅶ USER
                  1. syntax: USER <user>[:<group>]
                  2. description: 以特定的用户名或 UID 运行后续命令，以及切换工作目录。
               Ⅷ WORDIR
                  1. syntax: WORDIR <path>
                  2. description: 改变工作目录。
               Ⅸ ARG
                  1. syntax: ARG <name>[=<default value>]
                  2. description: 声明一个参数，用户可以在 docker build 命令中使用 --build-arg <varname>=<value> 指定值。
               Ⅹ ONBUILD
                  1. syntax: ONBUILD [INSTRUCTION]
                  2. description: 为当前镜像增加触发器，当其作为其它镜像的基础镜像时，会在触发时执行相应的命令。