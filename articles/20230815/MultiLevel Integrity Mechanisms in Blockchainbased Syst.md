
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 1.1 什么是区块链？
“区块链”最早起源于比特币(Bitcoin)，它是一个点对点的基于分布式网络的去中心化数字货币系统。该系统由一系列节点通过密码学技术实现共享账本的信任存储。不同节点之间通过生成新区块并在此基础上达成共识的方式来保证数据准确无误、真实可靠。由于区块链具有独创性和非对称加密技术的保护，使得其能够防止诈骗、虚假交易等恶意行为。除此之外，“区块链”还提供数字身份验证、记账、合约执行等一系列公共服务。
## 1.2 为什么要研究“区块链”中的多级完整性机制？
随着信息技术的飞速发展和传播，越来越多的人们开始采用区块链技术来构建互联网应用、记录商业活动或进行金融交易。相对于单纯的金融或电子支付来说，区块链更加注重的是安全、透明以及透明度。相比于现有的互联网应用，区块链赋予了其不可篡改的特性，让用户可以在不知情的情况下完成交易，从而降低了风险。虽然区块链技术提供了不可伪造的担保，但是如何在保证数据的真实性、有效性和完整性的同时，保障区块链系统中数据的隐私、数据主体的自主权以及数据的机密性是个难题。因此，“区块链”中的多级完整性机制是研究的热点。
## 2.多级完整性机制概述
“多级完整性机制”（MLIM）是指在区块链系统中，存在多个级别的完整性保护措施，即数据完整性、访问控制和事务原子性。这些措施可以帮助确保数据真实性、有效性以及数据的隐私性和机密性。MLIM是区块链系统的关键所在，因为它可以提高系统的安全性、可靠性以及用户的利益。MLIM的主要目的是为了建立一个数据交换的环境，使各方的数据传输更加安全可靠。
### 数据完整性
数据完整性是指数据被完整地写入到数据库中。“区块链”作为一种去中心化的数据库系统，在写入数据时需要遵循一些规则和要求，比如只允许特定节点来写入数据、每个写入操作都需要获得许可证和签名。这样，即便某个数据被篡改或者删除，也无法通过篡改前后的数据之间的比较来找到问题的根源。另外，“区块链”系统还提供了丰富的数据分析工具，可以对数据进行快速查询和检索，这对数据完整性的保障非常重要。
### 访问控制
访问控制是MLIM的一个重要组成部分。在“区块链”系统中，访问控制是依据所谓的授权策略来进行控制的。授权策略定义了一个参与者可以对某些数据项执行哪些操作，并且规定了何时对这些数据项进行更新。授权策略可以包括授权管理模块、加密签名模块、访问控制列表（ACLs）和其他相关模块。授权管理模块负责管理所有授权策略，如创建、更新、删除授权策略；加密签名模块用于对授权策略进行签名和认证；访问控制列表用于指定哪些实体有权访问数据项，以及其允许访问的权限。访问控制可以帮助确保数据只有受限的实体才能访问，并且访问控制变更只能通过得到足够授权的实体来进行。
### 事务原子性
事务原子性是指事务要么完全成功，要么完全失败，不能只执行一半。在“区块链”系统中，事务原子性就是指整个交易过程，包括数据发送、确认和写入数据库这一整套流程必须是完整且正确的，如果其中任何一步出现错误，就必须回滚至交易之前的状态。这样，数据完整性和访问控制都会得到保障。另外，“区cket链”系统还提供事务溯源功能，可以追踪数据在整个区块链系统中的流动路径，这对数据的完整性、有效性和审计提供了很大的帮助。
## 3.实验平台设置
本文将基于EOS智能合约和Hyperledger Fabric框架构建一个区块链平台，并进行如下实验：

1. 智能合约层面实验：
    - 在区块链平台上部署智能合约
    - 执行基本的增删查改合约操作
    - 测试数据的真实性、有效性和完整性
2. Hyperledger Fabric实验
    - 设置Fabric网络
    - 将数据存入Fabric网络
    - 查询Fabric网络上的数据
    - 使用Fabric网络上的数据做交易（链码调用）
    - 测试Fabric网络的可靠性、可用性和容错能力

整个实验平台分为两部分，第一部分为智能合约的开发实验，第二部分为Fabric实验。

## 4.实验环境准备
实验环境配置如下：
* 一台Ubuntu服务器作为区块链平台
* Docker Engine安装
* 安装Nodejs、npm包管理器
* 安装EOSIO客户端
    * 配置config文件
* 安装Hyperledger Fabric客户端
    * 下载Fabric网络配置模板文件（通道配置文件、docker-compose模板）
## 5.实验平台搭建
### 搭建EOS区块链平台
#### Ubuntu部署区块链平台
首先，下载Ubuntu Server 16.04 LTS iso镜像，把镜像烧写到U盘，然后利用虚拟机软件创建Ubuntu虚拟机，启动之后，根据提示安装Ubuntu Server系统。

接下来，配置U盘引导启动，用U盘引导启动Ubuntu虚拟机，选择英文语言，设置系统时间为东八区（上海）。

安装好Ubuntu之后，先更新软件源，然后安装NTP时间同步服务，开启防火墙，配置SSH远程登录。
```bash
sudo apt update && sudo apt install ntpdate
sudo systemctl enable ntpd.service
sudo ufw allow ssh
sudo reboot now
```

接下来，安装Docker Engine，下载Ubuntu对应版本的docker安装包。
```bash
curl -fsSL https://get.docker.com/ | sh
```

安装完毕后，运行以下命令启动Docker服务：
```bash
sudo systemctl start docker
```

最后，设置Docker镜像加速器，这样就可以加快拉取Docker镜像的速度，提升效率。
```bash
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<-'EOF'
{
  "registry-mirrors": ["http://hub-mirror.c.163.com"]
}
EOF
sudo systemctl daemon-reload
sudo systemctl restart docker
```

至此，Ubuntu服务器已经部署完毕。接下来，安装EOSIO客户端，这里以v1.8.4版本为例。
```bash
wget https://github.com/eosio/eos/releases/download/v1.8.4/eosio_1.8.4-1-ubuntu-18.04_amd64.deb
sudo dpkg -i eosio_1.8.4-1-ubuntu-18.04_amd64.deb
```

配置EOSIO客户端，修改`~/.eosio/config.ini`，添加如下内容：
```
# EOSIO Public Key of this node (e.g.: EOS6MRyAjQq8ud7hVNYcfnVPJqcVpscN5So8BhtHuGYqET5GDW5CV)
producer-name = produceraccount
# Can be public key or private key for a given key pair used to sign transactions by a particular actor account. The corresponding signature provider should also be added as one of the bios bps command line options.
signature-provider = MYKEY=KEYPAIR:PRIVATE_KEY
http-server-address = 0.0.0.0:8888
p2p-listen-endpoint = 0.0.0.0:9876
enable-stale-production = true # Required if you want to test partial block production
max-transaction-time = 3000   # Prevent unintended waiting for expired transactions
contracts-console = false    # Disable console RPC calls
plugin = eosio::history_api_plugin --filter-on *   # Enable history API plugin so we can track action traces
access-control-allow-origin = *  # Allow CORS from any domain
# listen-host = your host IP address
# enable-plugin = your desired plugin library name here
```

其中，`MYKEY`字段表示需要签名的密钥名，`KEYPAIR`字段表示该密钥对的公钥，`PRIVATE_KEY`字段表示私钥。

#### EOSIO区块链平台接口测试

登陆EOSIO区块链平台后，尝试执行基本的增删查改合约操作。

首先，新建一个账户：
```bash
./cleos create account eosio mycontractuser pubkey pubkey
```

然后，编译合约文件，合约文件名为`hello.cpp`，内容如下：
```cpp
#include <eosiolib/eosio.hpp>

using namespace eosio;

class hello : public contract {
   public:
      using contract::contract;

      /// @abi action
      void hi( const name& user ) {
         print( "Hello, ", user );
      }
};

EOSIO_ABI( hello )
```

合约中定义了一个名为`hi`的方法，该方法接收一个参数——`user`，打印`Hello, `字符串以及`user`参数的值。

编译合约文件：
```bash
eosio-cpp hello.cpp -o hello.wasm
```

将合约部署到区块链上：
```bash
./cleos set contract hello YOURACCOUNTHERE ~/workplace/hello/hello.wasm
```

其中，`YOURACCOUNTHERE`为你的EOSIO帐号名称。

使用合约：
```bash
./cleos push action hello hi '["mycontractuser"]' -p mycontractuser@active
```

运行结果显示，`Hello, `字符串以及`mycontractuser`参数的值被打印出来。

至此，EOSIO区块链平台的接口测试就结束了。

### 搭建 Hyperledger Fabric 区块链平台
#### 安装Hyperledger Fabric 网络组件
##### 下載Hyperledger Fabric v1.4.4
```bash
wget https://nexus.hyperledger.org/content/repositories/releases/org/hyperledger/fabric/hyperledger-fabric/1.4.4/hyperledger-fabric-1.4.4-linux-amd64.tar.gz
```

解压下载好的压缩包，得到如下文件夹结构：
```
├── bin
│   ├── configtxgen
│   ├── configtxlator
│   └── peer
├── bootstrap.sh
├── chaincode
└── examples
    ├── chaincode
    │   ├── abac
    │   ├── auction
    │   ├── ercc
    │   ├── fabcar
    │   ├── finance
    │   ├── golang
    │   ├── interest
    │   ├── jal
       ...省略其它示例
    ├── e2e_cli
    │   ├── abac
    │   ├── auction
    │   ├── etc.
    ├── first-network
    ├── hyperledger-versions.md
    └── scripts
```

将`bin`目录下的可执行文件放到环境变量PATH中：
```bash
export PATH=/path/to/your/fabric/binaries:$PATH
```

##### 配置 Hyperledger Fabric 网络

进入`examples/first-network`目录，编辑`docker-compose.yaml`文件，在`peer0.`开头的行里加入以下内容：
```yaml
  ports:
    - "9051:9051"
    - "7051:7051"
```

这样，`peer0.`开头的容器就会暴露两个端口：`9051`和`7051`。注意，不要覆盖掉`genesis.block`文件！

初始化一个网络，创建创世区块，以及初始的通道：
```bash
./byfn.sh up -a -n
```

`-a`参数代表创建CA（Certificate Authority），`-n`参数代表创建一个通道（Channel）。第一次执行这个命令时会花费几分钟时间，等待网络启动。

查看通道状态：
```bash
./byfn.sh channels
```
输出类似：
```
Channels peer0.org1.example.com:<|im_sep|> -> peer0.org2.example.com:<|im_sep|> on channel 'businesschannel':
	Height: 0, current block hash: 7ed1b25be6ff2debf268fb761bbabdbfc0ee2c09a4809b737f82b770662dc6a8
	OK
```

等待所有的容器都启动起来后，就可以启动CLI容器：
```bash
docker exec -it cli bash
```

在CLI容器里，我们可以使用命令行工具HLF（hyperledger fabric language）与Fabric网络交互。输入以下命令：
```bash
peer channel list
```

输出应该如下：
```
Channels peers has joined:
businesschannel
```

至此，Hyperledger Fabric网络就部署完毕。

#### Fabric 网络接口测试

首先，创建组织（Organization）和节点（Peer）：
```bash
docker exec -it cli bash
mkdir -p./organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com
```

编辑`./organizations/peerOrganizations/org1.example.com/peers/peer0.org1.example.com/msp/config.yaml`，设置节点的MSP（Membership Service Provider）：
```yaml
---
NodeName: peer0.org1.example.com
Identifier: ID.peer0.org1.example.com

...省略其它设置

-----BEGIN CERTIFICATE-----
<KEY>END CERTIFICATE-----
-----BEGIN PRIVATE KEY-----
MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQDumOgOrRcpTwArlEB
N+qLQcztYSPnkqAdJVWSjPRKcvAqbrsv2fgz5M0tmSlEzXWbVxQKUIvNXhjXnVjQkz
zhZwtuTdpTKFJ8tKxhsJUpyAYrxPZmDhUKBBDPwiCDHDMh8RgeKfYwSwCZRvj8phzGw
rzQHw5mIhHn90GrTXFckSyyD2NuNUT5IddxKJadVWua4X4zg2DFjRtMfBP6DgPYtykO
rbtzjToXh6sJBRwbIgPrtIJNGhwBLyARlHy3RpQvwhxFzbVC+QPjqu+RsKtdMklImSy
GhIpPbCKk7MOEof04CPHjhJZoy1nAJBNExAl+LNIrNf5BIPWTnpE6LyMKyOUomvgWGK
ohKXWlByyrOhfKwRQgJlTOGIth/DQydJrJaXtb4hmpTtCl6JvbzJsSMqWIHhEuKL5G
XpxWpUzFL1UwzokcNRMcTld7hqfDgjqBmxDugvxufqjUq8Weou20srRMEJCwwVhOOc
0jfTeJTbaKWw+kpUGt5dFgqCqruJXDHpQ89XJ
-----END PRIVATE KEY-----

AdminCert: |
  -----BEGIN CERTIFICATE-----
  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