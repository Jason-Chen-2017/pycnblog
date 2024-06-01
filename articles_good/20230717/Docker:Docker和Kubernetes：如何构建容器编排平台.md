
作者：禅与计算机程序设计艺术                    
                
                
## 概述
Docker是一个开源的应用容器引擎，它可以让开发者打包他们的应用以及依赖包到一个轻量级、可移植的镜像，然后发布到任何流行的Linux或Windows机器上，也可以实现虚拟化。基于Docker的容器集群管理工具如Kubernetes等可以快速地部署和管理容器化的应用，简化容器的创建、运行和维护。本文主要探讨Docker和Kubernetes在企业级容器云平台中的应用。

## 为什么要用容器？
### 降低开发环境搭建难度
开发者不再需要关心底层硬件环境，只需要安装Docker并启动容器就可以开发项目了，可以解决频繁变更的系统环境导致开发环境搭建复杂的问题。
### 更快速的交付和部署
开发完成后，只需将应用打包为Docker镜像并上传到Docker镜像仓库即可，通过自动化部署工具部署至目标服务器，整个过程几乎不需要人为干预，可以大幅缩短开发-测试-部署周期，实现更快的交付和部署。
### 提升资源利用率
由于Docker利用资源隔离和限制，可以有效保证容器独占资源，提高资源利用率，减少因资源竞争带来的性能问题。
### 服务迁移更方便
可以将Docker镜像部署到任意环境中运行，跨平台的兼容性使得应用服务更容易迁移到其他环境中运行。

## 使用Docker的最佳实践
对于开发者来说，以下三个方面最为重要：
1.Dockerfile的编写:编写符合自己需求的Dockerfile文件，包括基础镜像、镜像标签、命令等信息。
2.镜像仓库的使用:一般选择官方或私有镜像仓库作为镜像源，进一步优化镜像构建效率。
3.CI/CD工具的集成:可以集成相关CI/CD工具，例如Jenkins、GitLab等，实现持续集成及自动化部署。

对于运维人员来说，以下几个方面也很重要：
1.镜像和容器的安全管理:Docker提供了一些机制来保障容器的安全性，例如基于角色的访问控制（RBAC）、命名空间和签名认证等。
2.监控与报警:容器监控系统可以对容器的资源、网络、应用等进行实时监控，做出相应的响应。
3.弹性伸缩策略的设计:可以通过Kubernets或其它编排工具来实现应用的弹性伸缩策略，根据实际负载动态调整资源分配。

总之，用好Docker可以极大的提高应用的敏捷性、部署效率和运维效率。

# 2.基本概念术语说明
## 1.容器
容器是一个轻量级、可执行的独立单元，它包括一个应用及其所有的依赖包、配置、环境变量、存储信息以及元数据。容器就是将应用及其所需的所有内容打包成一个可部署的单元，能够被分离，独立地运行在一个隔离的环境中，提供应用程序之间的互相隔离和资源共享。

## 2.镜像
镜像是一个只读的模板，用来创建容器。镜像包含了一系列描述该容器应当具备的内容：
* 基础镜像:基础镜像是指容器运行的最基础的系统环境，通常基于某个linux发行版，例如CentOS或Ubuntu等。
* 操作系统、语言运行时、框架、工具链等:镜像还会包含运行容器应当具有的一切东西，比如语言运行时、框架、工具链、web服务器、数据库、消息队列等等。
* 程序和软件包:镜像还会包含开发者编写的程序和软件包。

## 3.Docker Hub
Docker Hub是一个公共的镜像仓库，它是Docker官方推荐的镜像托管服务。用户可以在Docker Hub注册账号并创建自己的镜像仓库，从而分享、保存和使用他人的镜像。

## 4.Dockerfile
Dockerfile是一个用于定义镜像的文件，是由一系列指令和参数构成的脚本，用来告诉Docker如何构建镜像。Dockerfile包含了一个完整的标准化的指令集，使得开发者可以很容易地创建自己的镜像。

## 5.联合文件系统（Union File System）
联合文件系统（Union FS）一种分层、轻量级并且高性能的文件系统，它支持文件的重用、版本控制、权限管理和回滚功能。最早由 Linux 创始人 Linus Torvalds 和他的同事们在20世纪90年代设计实现。联合文件系统是Docker的基础。

## 6.docker-compose
docker-compose是一个用于定义和运行多容器 Docker 应用程序的工具。用户可以使用 Compose 来定义一组相关的容器为一个应用程序服务，通过一条命令就可以实现一键启动和停止所有服务。Compose 文件是 YAML 格式或者 JSON 格式，定义了各个容器的配置，并且可以设置一些依赖关系、外部链接等。

## 7.Kubernetes
Kubernetes 是 Google 在 2015 年为了更好地管理容器集群而推出的开源平台。它是一个开源的自动化部署、缩放和管理容器化应用的平台，可以轻松地让部署更新应用、扩大或缩小规模。Kubernetes 的核心组件包括 API Server、调度器、控制器管理器、etcd 等。

# 3.核心算法原理和具体操作步骤以及数学公式讲解

## 1.Docker的安装
下载地址：https://www.docker.com/get-started 

具体操作步骤如下：

1. 安装包：从官网下载对应系统的安装包进行安装，当前使用的 Windows 系统，则下载 windows 版的安装包 docker_for_windows.exe；
2. 配置镜像加速器：国内的网络环境，可能会存在下载过慢的问题，此时可以考虑配置镜像加速器，加速拉取镜像，提高速度；
3. 设置环境变量：安装成功后，需要将 docker 的环境变量添加到系统 PATH 中，否则终端无法识别 docker 命令；

## 2.HelloWorld案例演示
具体操作步骤如下：

1. 创建 Dockerfile

   ```dockerfile
   # 指定基础镜像
   FROM alpine
   
   # 设置作者
   LABEL maintainer="jianfeng"
   
   # 将本地目录下的 test.sh 复制到镜像中的 /data 目录下
   COPY./test.sh /data
   
   # 执行 /data 目录下的 test.sh 文件
   CMD ["/bin/sh", "-c", "chmod +x /data/test.sh && /data/test.sh"]
   ```

2. 创建 test.sh 文件

   ```bash
   #!/bin/sh
   
   echo "Hello World!"
   ```

3. 根据 Dockerfile 构建镜像

   ```powershell
   PS C:\Users\Administrator> cd D:\gitCode\github.com\jianfeng-Parker\learn-k8s\docker\helloworld
   PS D:\gitCode\github.com\jianfeng-Parker\learn-k8s\docker\helloworld> ls
   
   CONTAINERFILE           test.sh           
   PS D:\gitCode\github.com\jianfeng-Parker\learn-k8s\docker\helloworld> docker build -t helloworld.
   Sending build context to Docker daemon  1.539MB
   Step 1/5 : FROM alpine
    ---> e435ea43d061
   Step 2/5 : LABEL maintainer=jianfeng
    ---> Running in cdc4bf754b5a
   Removing intermediate container cdc4bf754b5a
    ---> d9de82035aa1
   Step 3/5 : COPY test.sh /data
    ---> 87e56bbcfbc5
   Step 4/5 : RUN chmod +x /data/test.sh
    ---> Running in bcb42ccfafe5
   Removing intermediate container bcb42ccfafe5
    ---> afc5c194120c
   Step 5/5 : CMD ["/bin/sh", "-c", "chmod +x /data/test.sh && /data/test.sh"]
    ---> Running in f1067866baac
   Removing intermediate container f1067866baac
    ---> 7b47ad890ec6
   Successfully built 7b47ad890ec6
   Successfully tagged helloworld:latest
   ```

4. 启动容器

   ```powershell
   PS D:\gitCode\github.com\jianfeng-Parker\learn-k8s\docker\helloworld> docker run --name hello world
   
   Hello World!
   ```


## 3.Dockerfile文件详解

Dockerfile 是一个用来构建 Docker 镜像的文件，通过这个文件，你可以创建新的镜像，也可以给已有的镜像添加新的配置或软件。一个简单的 Dockerfile 文件示例如下：

```dockerfile
FROM centos

MAINTAINER jianfeng <<EMAIL>>

RUN yum install -y nginx

ADD index.html /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
```

Dockerfile 包含四部分内容：

1. `FROM`：指定基础镜像，可以是任何有效的基础镜像，如 ubuntu、centos、alpine 等。
2. `LABEL`：为镜像设置标签，如添加维护者、版本号等。
3. `RUN`：用于执行后续命令行，安装新软件包、创建文件等。
4. `COPY`、`ADD`：用于复制新文件到镜像里。
5. `CMD`：用于指定容器启动时默认运行的命令，启动容器后也可以通过命令行输入新的命令来运行。

## 4.Dockerfile最佳实践

Dockerfile 编写指南：

1. 使用小写字符，使用多个单词连接 `-`。
2. 每条指令都必须有 `\` 换行。
3. 每条指令均建议使用一个空行进行分割。
4. 不要在行首增加空格。
5. 参数值使用双引号 `" "` 或单引号 `' '` 括起来。
6. 如果参数的值中含有 `$`，则使用反斜线转义 `$`。
7. 大部分情况下不要使用最新版本，除非确定有必要。
8. 一定要有描述性注释，便于维护。

Dockerfile 编写范例：

```dockerfile
# 这里是作者信息
# MAINTAINER author email
# 使用指定的基础镜像，这里是 Ubuntu
FROM ubuntu:18.04

# 更新软件源并安装 curl
RUN apt update \
    && apt upgrade -y \
    && apt install -y curl

# 添加运行时需要的文件或目录
WORKDIR /app
COPY app.py requirements.txt entrypoint.sh./

# 安装 Python 依赖库
RUN pip3 install --no-cache-dir -r requirements.txt

# 环境变量
ENV MY_ENV_VAR=/path/to/myfile

# 端口映射
EXPOSE 5000

# 启动命令
CMD [ "/bin/bash", "./entrypoint.sh" ]
```

## 5.Dockerfile中的常用命令

Dockerfile 中的常用命令有：

1. `FROM`：指定基础镜像，必选。
2. `LABEL`：设置镜像标签，可选。
3. `RUN`：运行指定命令，必选。
4. `COPY`：复制本地文件到镜像中，可选。
5. `ADD`：复制远程文件或目录到镜像中，可选。
6. `CMD`：设置容器启动命令，可选。
7. `ENTRYPOINT`：覆盖默认的 ENTRYPOINT，可选。
8. `USER`：指定运行用户名或 UID，可选。
9. `WORKDIR`：设置工作目录，可选。
10. `ARG`：定义传递给 `Dockerfile` 的参数，可选。
11. `ONBUILD`：当构建镜像时，自动触发后续命令。
12. `STOPSIGNAL`：设置停止信号，可选。
13. `HEALTHCHECK`：健康检查，可选。

下面对这些命令进行详细的说明。

#### 1.FROM

指定基础镜像，语法为：

```
FROM <image>:<tag|@digest>
```

- `<image>`：镜像名称。
- `<tag>`：镜像标签，默认为 latest。
- `@digest`：镜像哈希值。

例如：

```dockerfile
FROM nginx
```

#### 2.LABEL

为镜像设置标签，语法为：

```
LABEL <key>=<value> <key>=<value>...
```

- `<key>`：标签名。
- `<value>`：标签值。

例如：

```dockerfile
LABEL version="1.0.0" description="This is my web service."
```

#### 3.RUN

运行指定命令，语法为：

```
RUN <command>
```

- `<command>`：命令字符串。

例如：

```dockerfile
RUN apt-get update \
  && apt-get install -y nginx
```

如果命令较长，可以使用 `\` 进行换行，例如：

```dockerfile
RUN apt-get update \
    && apt-get install -y nginx
```

#### 4.COPY

复制本地文件到镜像中，语法为：

```
COPY <src>... <dest>
```

- `<src>`：源文件路径。
- `<dest>`：目的路径。

例如：

```dockerfile
COPY package*.json./
```

将当前目录下的所有 `package*.json` 文件复制到镜像的 `/` 根目录下。

#### 5.ADD

复制远程文件或目录到镜像中，语法为：

```
ADD <src>... <dest>
```

- `<src>`：源文件路径。
- `<dest>`：目的路径。

例如：

```dockerfile
ADD https://example.com/context.tar.gz /
```

从 example.com 下载 `context.tar.gz` 文件并解压到镜像的 `/` 根目录下。

#### 6.CMD

设置容器启动命令，语法为：

```
CMD <command>
```

- `<command>`：命令字符串。

例如：

```dockerfile
CMD ["echo", "hello world"]
```

设置容器启动时执行 `echo` 命令输出 `hello world`。

#### 7.ENTRYPOINT

覆盖默认的 ENTRYPOINT，语法为：

```
ENTRYPOINT <command>
```

- `<command>`：命令字符串。

例如：

```dockerfile
ENTRYPOINT ["python"]
```

设置容器启动时默认执行 `python` 命令。

#### 8.USER

指定运行用户名或 UID，语法为：

```
USER <user>[:<group>]
```

- `<user>`：用户名或 UID。
- `<group>`：用户组名或 GID。

例如：

```dockerfile
USER root
```

#### 9.WORKDIR

设置工作目录，语法为：

```
WORKDIR </path/to/workdir>
```

- `</path/to/workdir>`：工作目录路径。

例如：

```dockerfile
WORKDIR /root
```

设置工作目录为 `/root`。

#### 10.ARG

定义传递给 `Dockerfile` 的参数，语法为：

```
ARG <name>[=<default value>]
```

- `<name>`：参数名。
- `[=<default value>]`：参数默认值。

例如：

```dockerfile
ARG user
```

声明参数 `user`。

#### 11.ONBUILD

当构建镜像时，自动触发后续命令。

#### 12.STOPSIGNAL

设置停止信号，语法为：

```
STOPSIGNAL <signal>
```

- `<signal>`：信号名称或编号。

例如：

```dockerfile
STOPSIGNAL SIGTERM
```

设置停止信号为 `SIGTERM`。

#### 13.HEALTHCHECK

健康检查，语法为：

```
HEALTHCHECK [<options>] CMD <command>
```

- `<options>`：健康检查选项。
- `CMD <command>`：健康检查命令。

例如：

```dockerfile
HEALTHCHECK --interval=5m --timeout=3s \
  CMD curl -fs http://localhost || exit 1
```

每 5 分钟检查一次 localhost 是否能够访问，超时时间为 3 秒。

# 4.具体代码实例和解释说明

通过以上对 Docker 和 Kubernetes 有基本的了解之后，我们现在来看一下 Docker 和 Kubernetes 的用法以及具体的代码实例。

## 1.使用 Docker 部署 SpringBoot 应用

本节我们首先使用 Docker 部署一个 SpringBoot 应用。

### 准备工作

新建文件夹 `springboot` ，并进入该文件夹，创建一个新的 Maven 工程：

```shell
mkdir springboot
cd springboot
mvn archetype:generate -DgroupId=com.example -DartifactId=demo-service -DarchetypeArtifactId=maven-archetype-quickstart -DinteractiveMode=false
```

切换到 `pom.xml` 文件中修改 pom.xml 配置：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>com.example</groupId>
    <artifactId>demo-service</artifactId>
    <version>1.0-SNAPSHOT</version>

    <!-- 引入 Spring Boot parent -->
    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>2.4.4</version>
        <relativePath/> <!-- lookup parent from repository -->
    </parent>

    <dependencies>
        <!-- Spring Boot Web 支持 -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>

        <!-- Lombok 注解处理器，使用更方便 -->
        <dependency>
            <groupId>org.projectlombok</groupId>
            <artifactId>lombok</artifactId>
            <optional>true</optional>
        </dependency>

        <!-- Spring Boot Test 支持 -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-test</artifactId>
            <scope>test</scope>
        </dependency>
    </dependencies>

    <build>
        <plugins>
            <!-- lombok 插件 -->
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-compiler-plugin</artifactId>
                <configuration>
                    <source>${java.version}</source>
                    <target>${java.version}</target>
                    <annotationProcessorPaths>
                        <path>
                            <groupId>org.projectlombok</groupId>
                            <artifactId>lombok</artifactId>
                            <version>1.18.16</version>
                        </path>
                    </annotationProcessorPaths>
                </configuration>
            </plugin>

            <!-- jar 包插件 -->
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
            </plugin>
        </plugins>
    </build>

</project>
```

### 创建 SpringBoot 主程序

新建 Java 类 `DemoServiceApplication`，内容如下：

```java
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class DemoServiceApplication {
    public static void main(String[] args) {
        SpringApplication.run(DemoServiceApplication.class, args);
    }
}
```

这里直接标注了 `@SpringBootApplication` 注解，这是一个 SpringBoot 特有的注解，它会帮我们自动配置 Spring 环境。

### 添加 Controller

在 `src/main/java/com/example/` 下创建子包 `controller`，并在其中新建 Java 类 `IndexController`，内容如下：

```java
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class IndexController {

    @GetMapping("/")
    public String index() {
        return "Welcome!";
    }

}
```

这里定义了一个名为 `index()` 方法，返回值为 `"Welcome!"`。

### 编写配置文件

新建配置文件 `application.yml`，内容如下：

```yaml
server:
  port: 8080
```

这里配置了服务器的端口号为 `8080`。

### 编译打包镜像

在命令行中运行如下命令编译并打包镜像：

```shell
mvn clean package
```

编译成功后，在 `target` 目录下找到生成的 `jar` 文件。

### 使用 Docker 部署

在命令行中运行如下命令启动 Docker 容器：

```shell
docker run -p 8080:8080 demo-service:1.0-SNAPSHOT
```

`-p` 参数表示将主机的 `8080` 端口绑定到 Docker 容器的 `8080` 端口。

启动成功后，在浏览器访问 `http://localhost:8080`，看到页面显示 `"Welcome!"` 表示 Docker 部署成功。

## 2.使用 Kubernetes 部署 SpringBoot 应用

本节我们首先使用 Kubernetes 部署一个 SpringBoot 应用。

### 准备工作

同样，先新建文件夹 `springboot` ，并进入该文件夹，创建一个新的 Maven 工程：

```shell
mkdir springboot
cd springboot
mvn archetype:generate -DgroupId=com.example -DartifactId=demo-service -DarchetypeArtifactId=maven-archetype-quickstart -DinteractiveMode=false
```

切换到 `pom.xml` 文件中修改 pom.xml 配置：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
    <modelVersion>4.0.0</modelVersion>

    <groupId>com.example</groupId>
    <artifactId>demo-service</artifactId>
    <version>1.0-SNAPSHOT</version>

    <!-- 引入 Spring Boot parent -->
    <parent>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-parent</artifactId>
        <version>2.4.4</version>
        <relativePath/> <!-- lookup parent from repository -->
    </parent>

    <dependencies>
        <!-- Spring Boot Web 支持 -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-web</artifactId>
        </dependency>

        <!-- Lombok 注解处理器，使用更方便 -->
        <dependency>
            <groupId>org.projectlombok</groupId>
            <artifactId>lombok</artifactId>
            <optional>true</optional>
        </dependency>

        <!-- Spring Boot Test 支持 -->
        <dependency>
            <groupId>org.springframework.boot</groupId>
            <artifactId>spring-boot-starter-test</artifactId>
            <scope>test</scope>
        </dependency>
    </dependencies>

    <build>
        <plugins>
            <!-- lombok 插件 -->
            <plugin>
                <groupId>org.apache.maven.plugins</groupId>
                <artifactId>maven-compiler-plugin</artifactId>
                <configuration>
                    <source>${java.version}</source>
                    <target>${java.version}</target>
                    <annotationProcessorPaths>
                        <path>
                            <groupId>org.projectlombok</groupId>
                            <artifactId>lombok</artifactId>
                            <version>1.18.16</version>
                        </path>
                    </annotationProcessorPaths>
                </configuration>
            </plugin>

            <!-- jar 包插件 -->
            <plugin>
                <groupId>org.springframework.boot</groupId>
                <artifactId>spring-boot-maven-plugin</artifactId>
            </plugin>
        </plugins>
    </build>

</project>
```

### 创建 SpringBoot 主程序

新建 Java 类 `DemoServiceApplication`，内容如下：

```java
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
public class DemoServiceApplication {
    public static void main(String[] args) {
        SpringApplication.run(DemoServiceApplication.class, args);
    }
}
```

这里直接标注了 `@SpringBootApplication` 注解，这是一个 SpringBoot 特有的注解，它会帮我们自动配置 Spring 环境。

### 添加 Controller

在 `src/main/java/com/example/` 下创建子包 `controller`，并在其中新建 Java 类 `IndexController`，内容如下：

```java
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class IndexController {

    @GetMapping("/")
    public String index() {
        return "Welcome!";
    }

}
```

这里定义了一个名为 `index()` 方法，返回值为 `"Welcome!"`。

### 编写配置文件

新建配置文件 `application.yml`，内容如下：

```yaml
server:
  port: 8080
```

这里配置了服务器的端口号为 `8080`。

### 编译打包镜像

在命令行中运行如下命令编译并打包镜像：

```shell
mvn clean package
```

编译成功后，在 `target` 目录下找到生成的 `jar` 文件。

### 用 Kubernetes 部署

我们需要安装 `kubectl` 工具，它是 Kubernetes 命令行接口，帮助我们与 Kubernetes 集群交互。

安装步骤如下：

1. 前往 https://kubernetes.io/docs/tasks/tools/#kubectl 下载对应平台的 kubectl 二进制文件，并移动到环境变量 PATH 中；
2. 配置 kubeconfig 文件。

确认 kubectl 已经安装成功，可以使用如下命令查看版本信息：

```shell
kubectl version
```

如果你遇到了以下错误：

```shell
Unable to connect to the server: dial tcp <ip>:<port>: i/o timeout
```

可能是因为没有配置 kubeconfig 文件导致的，解决办法是在 `~/.kube` 目录下创建一个 config 文件，内容如下：

```yaml
apiVersion: v1
clusters:
- cluster:
    certificate-authority-data: LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0t...
    server: https://<master node ip>:<api server port>
  name: kubernetes
contexts:
- context:
    cluster: kubernetes
    user: kubernetes-admin
  name: kubernetes-admin@kubernetes
current-context: kubernetes-admin@kubernetes
kind: Config
preferences: {}
users:
- name: kubernetes-admin
  user:
    client-certificate-data: LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0t...
    client-key-data: LS0tLS1CRUdJTiBSU0EgUFVCTElDIEtFTk...
```

`<master node ip>` 替换成 Kubernetes master 节点 IP 地址；`<api server port>` 替换成 Kubernetes api server 默认端口，默认应该是 6443。

这样，kubectl 就能正常工作了。

接下来，我们可以用 Kubernetes 部署我们的 SpringBoot 应用。

首先，创建一个 Deployment 对象，内容如下：

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: demo-service
  name: demo-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: demo-service
  template:
    metadata:
      labels:
        app: demo-service
    spec:
      containers:
      - image: your-dockerhub-id/demo-service:latest
        name: demo-service
        ports:
        - containerPort: 8080
          protocol: TCP
```

这里创建了一个 Deployment 对象，名为 `demo-service`，使用 `your-dockerhub-id/demo-service:latest` 镜像启动一个 `replicas` 个数为 `3` 的 pod，每个 pod 包含一个容器，容器的端口号为 `8080`。

然后，创建一个 Service 对象，内容如下：

```yaml
apiVersion: v1
kind: Service
metadata:
  labels:
    app: demo-service
  name: demo-service
spec:
  ports:
  - name: http
    port: 8080
    targetPort: 8080
  selector:
    app: demo-service
  type: ClusterIP
```

这里创建了一个 Service 对象，名为 `demo-service`，监听端口号为 `8080`，pod 的 `selector` 为 `app: demo-service`。

最后，让我们来测试一下部署是否成功。在命令行中运行如下命令：

```shell
kubectl apply -f deployment.yaml
```

等待 Deployment 创建完毕后，再运行如下命令：

```shell
kubectl get pods
```

确认 Pod 处于运行状态，然后运行如下命令：

```shell
minikube service demo-service --url
```

打开浏览器访问获取到的 URL，看到页面显示 `"Welcome!"` 表示 Kubernetes 部署成功。

至此，我们已经成功使用 Kubernetes 部署了一个 SpringBoot 应用。

