
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


## Docker简介
Docker是一个开源的应用容器引擎，让开发者可以打包他们的应用以及依赖项到一个轻量级、可移植的容器中，然后发布到任何流行的 Linux或Windows机器上，也可以实现虚拟化。由于docker的轻量级特性和更高效率的虚拟化技术，使其在云计算领域掀起了一场容器革命。
## 为什么要用Docker？
容器技术的出现与发展引起了软件开发人员对虚拟机技术、分布式系统等新型计算资源的需求。传统的虚拟机技术较为昂贵，容器技术可以在相同硬件配置下运行多个相互隔离的应用，因此在降低资源开销的同时提升性能。另一方面，容器技术可以自动化地管理和部署应用，减少了重复性工作，实现快速迭代和交付，并节省了测试、部署和运维的时间成本。当然，Docker还提供了许多高级特性，如镜像管理、网络管理、数据管理、安全性等等。正因为这些优点，Docker成为容器技术发展的又一重要推手。
## Docker环境搭建
### 拉取镜像
拉取官方Java镜像
```bash
docker pull openjdk:8u171-jdk-alpine
```
通过运行以下命令验证是否成功拉取到了镜像：
```bash
docker images
```
### 创建Docker容器
创建名称为my-springboot-app的容器：
```bash
docker run -it --name my-springboot-app openjdk:8u171-jdk-alpine /bin/sh
```
其中，`-it`参数表示进入容器后进入交互模式，`--name`参数用于指定容器的名称，`openjdk:8u171-jdk-alpine`为拉取的Java镜像，`/bin/sh`表示启动容器后执行的shell命令。
### 进入容器内部
```bash
docker exec -it my-springboot-app /bin/sh
```
### 从Docker Hub拉取Spring Boot项目模板
```bash
git clone https://github.com/spring-projects/spring-petclinic.git
cd spring-petclinic
ls
```
### 安装Maven
```bash
apk add maven
mvn clean package
```
### 在容器外部开启端口映射
为了能够从外部访问到容器内的应用服务，我们需要将容器内部的端口映射到宿主机上的某个端口，这样才能通过浏览器访问到应用。
```bash
docker ps # 获取正在运行的容器ID
docker port <container ID> # 查看映射的端口
docker stop <container ID> # 停止容器
docker rm <container ID> # 删除容器
```
创建一个docker-compose.yml文件，添加如下内容：
```yaml
version: '2'
services:
  petclinic:
    build:.
    ports:
      - "9999:8080"
    environment:
      SPRING_PROFILES_ACTIVE: prod
    depends_on:
      - mysql
  mysql:
    image: mysql:latest
    restart: always
    environment:
      MYSQL_ROOT_PASSWORD: root
      MYSQL_DATABASE: petclinicdb
    volumes:
      -./init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "3306:3306"
```
这里面的build指的是用当前目录下的Dockerfile编译镜像，ports是将容器内的8080端口映射到主机的9999端口；environment是设置环境变量，SPRING_PROFILES_ACTIVE用来激活生产环境配置文件；depends_on表示依赖mysql服务，volumes用来将本地的init.sql文件复制到MySQL的初始化脚本目录下；ports是将MySQL的3306端口映射到主机的3306端口。

在项目根目录新建一个Dockerfile文件，添加如下内容：
```dockerfile
FROM openjdk:8u171-jdk-alpine
VOLUME /tmp
ADD target/*.jar app.jar
RUN sh -c 'touch /app.jar'
EXPOSE 8080
ENV JAVA_OPTS=""
ENTRYPOINT ["java","$JAVA_OPTS","-Djava.security.egd=file:///dev/urandom","-jar","/app.jar"]
```
这里面的VOLUME和ADD指令是在拷贝jar包时使用，RUN和EXPOSE分别是设置生成镜像时的缓存、暴露端口号；ENV和ENTRYPOINT用于设置环境变量和启动命令。

然后，使用命令`docker-compose up`，启动整个环境。

至此，我们的Spring Boot应用已经成功的迁移到Docker容器中运行，可以通过`http://localhost:9999/`访问到。