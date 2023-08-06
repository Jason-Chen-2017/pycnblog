
作者：禅与计算机程序设计艺术                    

# 1.简介
         
16号开头写“Creating”一词，这似乎是个错误的表述，应该是“Setting up”，因为主要内容是在本地搭建一条测试网，而在服务器上部署也属于实施阶段的工作，并非创建阶段。这一点也是作者提出的疑问之一。因此，下面对文章内容做出修改：
         “Creating a testnet locally using docker-compose and deploy smart contracts on the actual blockchain network.”

         16号中“with docker-compose”这一关键词非常重要。docker是一个非常强大的容器技术，docker compose也是基于docker提供的编排工具，使得部署和管理容器变得十分方便。这正好可以用来快速搭建一条测试网。本文将着重介绍如何用docker-compose搭建一个测试链，并通过几个简单的例子来部署智能合约到该链上。

         # 2.基本概念术语说明
         本章节介绍一些与文章主题相关的概念、术语等。
         2.1 docker
         2.1.1 docker是什么？
         docker是一个开源的平台，提供了轻量级的虚拟化容器技术，可以把应用打包成镜像文件，然后在任何地方运行，真正实现了应用的“一次装配随处运行”。
         2.1.2 docker的优势？
         - 版本一致性:docker让开发者、测试人员、运维人员都可以获得完全一致的开发环境，消除了环境差异带来的错误；
         - 高效率:docker通过虚拟化技术，提升了资源利用率，使得多任务处理更加高效；
         - 可移植性:docker镜像可以在不同系统之间共享，无论是物理机还是虚拟机都可以运行docker容器；
         - 可复用性:docker镜像可以很容易地制作、发布、分享；
         - 松耦合:docker可以帮助开发人员构建松散耦合的应用程序，服务层之间通过API进行通信，这样就降低了模块间的依赖关系；

         更多关于docker的介绍可参考：https://www.runoob.com/docker/docker-intro.html
         2.2 docker-compose
         docker-compose是一个用于定义和运行复杂docker应用的工具，它可以通过yaml配置文件来自动化地完成容器的构建、链接等操作。官方文档地址为：https://docs.docker.com/compose/overview/.
         2.3 以太坊网络
         2.3.1 以太坊介绍
         以太坊（Ethereum）是一个分布式的去中心化平台，其代码即平台，提供了世界上最先进的区块链基础设施，可以安全地存储数据、执行计算和转账交易。可以说，以太坊是目前最火的区块链项目之一。
         2.3.2 以太坊测试网
         以太坊的主网(Main Net)和测试网(Test Net)都是共用的，但它们有不同的功能。其中，主网功能完整、实时，性能较强，但是资金比较宝贵；而测试网功能可能不全面，但是运行速度快、资金比较便宜。
         2.4 智能合约
         2.4.1 概念及特点
         智能合约（Smart Contract）是一种计算机协议，旨在实现数字化的契约或合同，按照合同约定的数据处理事务。智能合约由两部分组成，分别是合约文本和合约环境。
         2.4.2 Ethereum Solidity语言
         以太坊智能合约程序设计语言Solidity（简称为sol）是一种类JavaScript语言，被编译成EVM字节码后运行在以太坊区块链上。sol是一个高级语言，允许在程序内嵌入表达式、条件语句、循环控制等。Solidity是Ethereum生态系统的基石，支持众多框架、库和工具，为广大开发者提供了构建应用的基础。
         2.5 私钥、地址、公钥、钱包
         2.5.1 密钥对
          私钥（private key）是用户身份的重要标识符，它唯一对应一个账户，只能由用户拥有。公钥（public key）是私钥的公开形式，任何人都可以使用公钥来验证签名是否有效，从而确认某个消息的发送方就是拥有这个私钥对应的账户。
         2.5.2 地址（address）
          以太坊中的地址表示账户的唯一标识符。每一个以太坊账号都有一个独一无二的地址，它由四个256位随机数生成，使用SHA3-256哈希函数加密。
         2.5.3 钱包
          钱包（Wallet）是一个可以存储密钥对的软件，它可以用来保存用户的私钥，同时也可以用来接收、查看、发送以太币。目前，很多钱包都支持以太坊，例如MetaMask、MyEtherWallet等。

         # 3.核心算法原理和具体操作步骤
         3.1 安装Docker
         如果你还没有安装过docker，需要先安装docker才能继续下面的操作。
          - Windows：下载docker desktop（官方下载地址 https://hub.docker.com/editions/community/docker-ce-desktop-windows/）并安装即可。
          - Mac OS X：根据系统版本，可以选择直接安装dmg安装包或者手动安装命令行工具。
          - Linux：根据系统版本安装docker即可，安装方法可参考https://docs.docker.com/install/。
         3.2 获取测试网镜像
         在linux或mac上，输入以下命令获取测试网镜像：
         ```shell script
         sudo docker pull ethereum/client-go:latest
         ```
         windows系统则直接在docker desktop中搜索ethereum/client-go拉取最新版的镜像。
         3.3 配置并启动测试网节点
          在linux或mac上，进入/root目录并创建一个名为"geth_config"的文件夹，在该文件夹下创建一个名为"genesis.json"的文件，用于配置创世区块的参数。接着，打开终端，切换到geth_config文件夹，输入以下命令启动节点：
         ```shell script
         sudo docker run -d --name geth-node \
            -v /root/geth_config:/root/.ethereum \
            -p 30303:30303 \
            -p 8545:8545 \
            ethereum/client-go:latest \
                --syncmode full \
                --rpc \
                --cache=1024 \
                --maxpeers=25 \
                console
         ```
         命令参数的详细描述如下：
         - "-d":后台模式
         - "--name geth-node":指定容器名称
         - "-v /root/geth_config:/root/.ethereum":将主机的"/root/geth_config"目录挂载到容器的"/root/.ethereum"目录，实现配置文件的持久化存储。
         - "-p 30303:30303 -p 8545:8545":端口映射，将主机的30303端口映射到容器的30303端口，将主机的8545端口映射到容器的8545端口。
         - "ethereum/client-go:latest":指定使用的测试网镜像。
         - "--syncmode full":同步模式，设置为"full"代表全节点同步模式。
         - "--rpc":开启RPC接口。
         - "--cache=1024":缓存区大小为1024MB。
         - "--maxpeers=25":最大连接数量为25。
         - "console":启动Geth控制台。
         3.4 配置并连接钱包
          下载并安装相应的钱包客户端软件，创建或导入一个钱包账号，复制其地址。在浏览器访问http://localhost:8545/ 进入Geth控制台，执行以下命令进行配置：
         ```shell script
         eth.accounts[0] // 查看默认账户地址
         personal.newAccount("123456") // 创建新账户
         web3.personal.sendTransaction({from:"0x9aA5fAeBdc27eD1AEb7CEd6F00042Fd52c3BA3Dc", to:"0xFAdC3BdDfe7b6958ACa4deBC5F5DAB3Af8bcCd7C", value:web3.toWei('1', 'ether')}) // 向新账户转账ETH
         ```
         以上命令演示了用Geth控制台对钱包进行配置和操作。
         3.5 编写智能合约
          编写智能合约的方法有很多种，这里我们展示一种比较简单的方式——使用Remix IDE。
         3.5.1 安装Remix IDE
         Remix IDE是一个开源的Web IDE，可以用来编写和调试智能合约，并可以部署到区块链网络。
         下载地址：https://remix.ethereum.org/#optimize=false&version=solidity%200.8.11。
         安装完成后，在浏览器打开http://localhost:8080/#/ ，即可进入Remix IDE界面。
         3.5.2 编写智能合约
         使用Remix IDE编写智能合约很简单，只需在左侧窗口的contracts文件夹中新建文件，然后编写合约的代码。如此一来，Remix会自动进行语法检查、编译、部署到区块链网络。
         下面以一个简单的加法合约为例，演示一下Remix IDE的使用过程。
         首先，点击左侧菜单栏中的"File"->"New File..."，弹窗中输入文件名"Add"，然后点击确定按钮，在右侧编辑器中编写以下代码：
         ```solidity
         pragma solidity ^0.8.1;

         contract Add {
             uint public x = 0;

             function addNumber(uint _num) public returns (bool success){
                 require(_num > 0 && _num <= 10);
                 x += _num;
                 return true;
             }
         }
         ```
         此合约定义了一个变量x，并定义了一个addNumber函数，该函数接受一个uint类型的参数_num，返回值为布尔值success。函数要求_num的值必须在1~10范围内，否则合约调用失败。当合约部署到区块链网络后，就可以调用该函数来进行数字的加法运算。
         3.5.3 部署合约
         当合约编写完毕后，点击左侧菜单栏中的"Deploy"->"Deploy & Run Transactions..."，弹窗中选择"Injected Web3"，再点击"At Address"标签页，输入默认账户地址并单击"Load"按钮，然后点击"Deploy"按钮，等待合约部署成功。
         3.5.4 调用合约
         当合约部署成功后，可以在控制台窗口中看到合约地址。可以打开另一个Geth控制台窗口，输入以下命令调用合约的addNumber函数：
         ```javascript
         const addContract = new web3.eth.Contract(JSON.parse('[合约ABI]'); // 从Remix IDE导出合约ABI
         const address = '[合约地址]' // 从Remix IDE导出合约地址
         const account = '0x9aA5fAeBdc27eD1AEb7CEd6F00042Fd52c3BA3Dc';// 默认账户
         addContract.options.address = address; 
         let result = await addContract.methods.addNumber(5).call(); // 调用函数并读取返回值
         if (!result) {
           alert(`Failed to call addNumber`);
         } else {
           alert(`The sum is ${result}`);
         }
         ```
         上面的命令先解析Remix导出的合约ABI和合约地址，设置合约的地址，再调用addNumber函数并读取返回结果。
         执行命令后，如果函数调用成功，控制台窗口会显示`The sum is 5`，表示5 + x = 5 + 0 = 5。
         当然，你也可以尝试其他合约的调用方式，包括读取变量值、写入变量值、修改变量值等。