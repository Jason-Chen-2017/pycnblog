
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


## 概念阐述
Docker容器(Container)是一个轻量级、可移植、自给自足的应用容器。它可以包括运行时环境、软件依赖、配置及文件系统。 Docker基于cgroup，namespace，以及联合挂载技术（UnionFS），其内核技术保证了轻量级和高效的隔离。虽然 Docker 提供简单的接口，但其功能远不止这些。Docker能够实现跨平台部署，使得应用程序可以在任何地方运行。
docker run命令用于创建一个新的容器，并在其中执行指定的命令。默认情况下，docker run将容器中进程的标准输入、输出和错误连接到主机的标准输入、输出和错误。也可以通过添加选项-i -t 可让容器的标准输入被重定向为一个控制台，从而实现交互式命令行界面。
在docker中运行应用最简单的方式就是用docker run 命令启动镜像。比如要运行一个Flask应用，可以使用下面的命令：
```bash
$ docker run --name flask_container -d -p 80:80 your_username/your_image_name /usr/bin/python app.py
```
上面这个命令会下载或者本地已有的Flask镜像，并启动名为flask_container的容器，映射容器的80端口到主机的80端口，然后运行/usr/bin/python app.py命令来启动Flask应用。这条命令里的--name参数指定了容器的名字，-d参数表示后台模式，-p参数表示端口映射，最后的两个参数分别是镜像名称和启动命令。注意，docker run的各个参数都不是必须的，可以根据自己的需求进行选择。如果你的Dockerfile中已经设置了CMD或ENTRYPOINT指令，则可以省略掉启动命令，直接运行：
```bash
$ docker run --name flask_container -d -p 80:80 your_username/your_image_name
```
## 相关链接
https://www.runoob.com/docker/docker-command-manual.html
https://www.cnblogs.com/shengulong/articles/9078206.html


# 2.核心概念与联系
## Docker镜像与容器的关系？
Docker 镜像是一个特殊的文件系统，其中包含了创建 Docker 容器时所需的一切：运行中的应用、语言运行时、依赖库等。你可以把 Docker 镜像看成是一个预打包好的，带有运行环境的脚本。要运行一个 Docker 容器，你只需要运行这个脚本就可以了。当你需要修改、更新或扩展你的应用时，你只需要重新构建一个新的 Docker 镜像即可。
如上图所示，Docker包括三个基本概念：镜像（Image）、仓库（Registry）、容器（Container）。镜像是一个面向Docker用户的可读、不可变的模板，用来创建Docker容器；仓库用于保存、分发镜像；容器是在镜像的基础上，运行一个可变的应用进程。镜像和仓库都是Docker的主要组件，容器是镜像运行时的实体。两者间的关系类似于生产者和消费者模型，镜像是一种原料，仓库是种东西的仓库，容器是一种产品。

Docker镜像和容器的关系如下图所示：




**总结**：

- Docker镜像是一个静态的文件系统，里面包含软件运行所需的所有资源。
- 每次运行容器的时候，都是先找到镜像文件，利用它创建一个容器，容器中包括了一个完整的软件运行环境。
- 可以把镜像看作是一个项目配置项、软件安装包及环境变量的集合。
- 可以把容器看作是一个轻量级的虚拟机，它提供了完整的运行环境，可以用来运行多个不同应用。