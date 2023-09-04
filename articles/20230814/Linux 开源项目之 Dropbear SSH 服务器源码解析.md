
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Dropbear SSH 是一款自由、开源的SSH协议版本。它支持包括RSA、DSS、ECDSA、Ed25519密钥、压缩及模糊传输等功能，并能够通过PAM模块进行身份验证，具有高性能，可靠性，安全性和易用性等优点。2017年6月1日，Dropbear SSH正式发布了第五个版本，相比于上一个版本增加了SFTP协议、内置的sftp-server工具，对OpenSSL库的依赖升级到1.1.1f版本。此次发布改善了安全性、性能、易用性以及稳定性等方面的问题。

对于新手而言，了解一下Dropbear SSH是如何工作的可以帮助理解其工作方式和原理。本文将会从以下几个方面介绍Dropbear SSH的工作原理：

# 1.背景介绍
## 1.1 概述
### 1.1.1 Dropbear SSH 是什么？

Dropbear SSH 是一款由 OpenBSD 发起开发的 SSH 服务端软件。它最初被设计用于在嵌入式系统上运行，并作为 OpenSSH 的替代品，后者是由 OpenBSD 社区维护的免费软件。Dropbear SSH 与 OpenSSH 有很多相同之处，但也有一些差异。例如，Dropbear SSH 支持不限数量的用户认证，而 OpenSSH 只允许单个 root 用户。另外，Dropbear SSH 比 OpenSSH 更加轻量级，安装包大小仅为 40k。

### 1.1.2 Dropbear SSH 版本
目前，Dropbear SSH 有两个主要版本：

1. Dropbear SSH v0.x（0.47-beta）：它最早于 2003 年发布，该版本基于原始的 SSLeay 库进行加密。

2. Dropbear SSH v2.x（2017.74）：它于 2017 年 6 月份发布，改进了实现，并且引入了新的加密库 libtomcrypt 作为默认加密库。

3. Dropbear SSH v3.x （2020.81）：它于 2020 年 8 月份发布，该版本在性能和安全性方面有重大提升。

这些版本之间的差异主要体现在加密和压缩方面。其中，libtomcrypt 加密库的引入使得 Dropbear SSH 可以更快地处理数据流，并提供更好的加密性能。此外，Dropbear SSH 还支持多种认证方式，如 RSA 或 DSS 密钥，以及压缩和模糊传输功能。

## 1.2 Dropbear SSH 安装配置
### 1.2.1 安装依赖软件包

为了编译 Dropbear SSH 最新版本，需要先安装依赖软件包，具体如下所示：

```shell
sudo apt update && sudo apt install build-essential libgmp-dev ncurses-dev libssl-dev zlib1g-dev cmake
```

编译过程中如果提示缺少某些依赖包，则可以通过以下命令安装：

```shell
sudo apt install 包名
```

这里，我们安装 `build-essential`、`libgmp-dev`、`ncurses-dev`、`libssl-dev` 和 `zlib1g-dev`。

### 1.2.2 配置环境变量

编译完成后，如果想运行或调试 Dropbear SSH，需要配置环境变量，使得系统可以找到 `dropbearmulti` 命令。具体做法如下：

```shell
export PATH=$PATH:/usr/local/sbin
```

上面这条命令将 `/usr/local/sbin` 添加到系统环境变量 `$PATH` 中。这样，就可以直接在终端中输入 `dropbearmulti` 来启动 Dropbear SSH 服务。

### 1.2.3 创建 dropbear ssh 用户组

为了让 Dropbear SSH 服务进程在后台正常运行，需要创建一个名为 `dropbear` 的用户组，并将当前用户加入该用户组。具体做法如下：

```shell
sudo groupadd dropbear
sudo usermod -a -G dropbear $USER
```

上面这两条命令分别创建了一个名为 `dropbear` 的用户组，并将当前用户加入该用户组。

### 1.2.4 安装下载的 Dropbear SSH

登录到 Linux 服务器之后，进入下载的 Dropbear SSH 目录，执行编译命令：

```shell
./configure --enable-syslog --with-pam --prefix=/usr/local
make && make install
```

上面这两条命令分别配置安装 Dropbear SSH 前缀路径，启用 syslog 日志记录，以及加载 PAM 模块。最后，执行 `make` 和 `make install` 命令编译安装 Dropbear SSH。

编译和安装成功之后，会出现以下信息：

```shell
mkdir -p /usr/local/etc/default
cp contrib/dropbear_default /usr/local/etc/default/dropbear
chmod go-w /usr/local/bin/*
install -m 0644 dropbear.init /etc/init.d/dropbear
ln -fs /usr/local/sbin/dropbear /usr/sbin/dropbear
update-rc.d dropbear defaults
```

上面这段输出表示已经完成 Dropbear SSH 的安装。其中，`/usr/local/sbin` 中的程序可供所有用户调用；`/usr/local/bin` 中的程序只能由超级管理员调用；`/usr/sbin` 中的程序只供 root 用户调用。

### 1.2.5 启动 Dropbear SSH 服务

为了启动 Dropbear SSH 服务，需要进入 `/etc/init.d/` 目录，找到 `dropbear` 文件夹，然后执行命令：

```shell
service dropbear start
```

上面这条命令启动 Dropbear SSH 服务，并进入服务状态。此时，可以使用 `ps aux|grep dropbear` 查看 Dropbear SSH 服务是否正在运行。

如果需要停止 Dropbear SSH 服务，可以使用命令：

```shell
service dropbear stop
```

上面这条命令停止 Dropbear SSH 服务。