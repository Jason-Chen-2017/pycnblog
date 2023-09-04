
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 背景介绍
今年初Coinbase宣布将筹集$6.5 billion用于完成私密数字货币交易所Chainlink旗下的基础设施建设项目，成为该公司首家筹集上亿美元投资的A轮融资。本次上市还将带动整个加密货币行业的发展。

Chainlink是一个构建可靠、安全、透明的区块链技术的平台，其核心功能包括链接多个区块链的互操作性，提供一个统一的API接口，实现跨链资产流通、合约执行等。

此外，作为世界顶级的金融科技公司，Coinbase目前已经建立起了庞大的客户群体，并且在全球范围内拥有超过100个办事处，覆盖了全球各地的用户。

## 概念、术语及定义
- Chainlink: 一个构建可靠、安全、透明的区块链技术的平台。
- DeFi: Decentralized Finance 的缩写，即去中心化金融。
- NFT(Non-Fungible Token): 一种无法进行代币兑换或替代的数字资产。
- DAO(Decentralized Autonomous Organization): 分散自治组织。
- DEX(Decentralized EXchange): 分散式交易所。
- Gas: 是支付计算机运行过程中需要的成本费用的电汇金额，与计算能力无关。Gas是交易所链下协议的“油价”。
- Oracle: 外部数据源。Oracle服务可以让交易所链下协议获取到关于智能合约的状态信息。例如，Oracle服务可以帮助交易所链下协议通过加密价格数据源获取当前价格。
- Smart Contracts: 基于区块链技术的分布式应用，在区块链网络中存储并自动执行合同中的代码。

## 核心算法原理和具体操作步骤
### 第一步: 注册地址

2. 在注册页面填写邮箱、用户名、密码，点击Create Account（创建账号）。

### 第二步: 配置钱包地址

1. 创建或导入一个硬件钱包地址。
   - 使用Ledger或者Trezor等硬件钱包可以确保私钥的安全。
   - 如果没有硬件钱包，也可以在线生成一个由软件生成的随机钱包地址。
2. 选择一个支持BTC、ETH、DAI等多种数字货币的交易所。
   - 可以选择FTX，它提供较好的交易手续费优惠，适合于新用户。
   - FTX上有超过7000多只主流数字货币的交易对。
   - 其他交易所也有类似的交易对。
3. 在交易所上开立交易账户，将硬件钱包中的比特币转入交易所账户中。
4. 查看并复制交易所的充值地址。
5. 在Coinbase上的Dashboard左侧菜单栏点击Onchain Deposits（链上存款），创建一个新的Deposit（存款）。
6. 输入充值地址，选择您刚才生成的硬件钱包地址，勾选保存助记词。
7. 提交后，Coins将会从您的交易所账户转入Coinbase的存款账户，并存入硬件钱包。

### 第三步: 配置Chainlink节点

如果您不熟悉链上开发，建议先阅读相关的官方文档和教程。

1. 安装Go语言环境。
2. 下载并安装Golang。
3. 配置GOROOT和GOPATH变量。
4. 配置go.mod文件并设置依赖。

   ```
   go mod init example/module
   
   require (
       github.com/smartcontractkit/chainlink v0.9.7
   )
   ```

5. 初始化项目目录。

   ```
   mkdir example && cd example
   touch main.go
   ```

6. 编写代码。

   ```go
   package main
   
   import "github.com/smartcontractkit/chainlink/core/services"
   
   func main() {
       // Create and start a new chainlink node using the default config values
       n := services.NewNode()
       defer n.Close()
       
       // Start all core chainlink services
       err := n.Start()
       if err!= nil {
           panic(err)
       }
   }
   ```

7. 编译项目。

   ```
   go build
   ```

8. 执行项目。

   ```
  ./example
   ```

### 第四步: 安装并启动Chainlink节点

1. 将编译好的二进制文件上传至服务器，并将路径添加至系统PATH环境变量中。
2. 设置环境变量。

   ```bash
   export ETH_URL=<eth rpc url>
   export LINK_CONTRACT_ADDRESS=<link contract address>
   ```

3. 生成新的mnemonic（助记词）。
4. 复制mnemonic并保存好，这是重要的私钥备份。
5. 修改配置文件config.toml。

   ```toml
   minimum-gas-price = <minimum gas price in gwei> # 设置最低gas价格
   
   [database]
   type="postgresql"
   host="<db hostname>"
   port=5432
   user="<db username>"
   password="<<PASSWORD>>"
   name="chainlink"
   
   [blockchain]
   eth-url=$ETH_URL
   default-registry-contract-address=$LINK_CONTRACT_ADDRESS
   [chainlink]
   feature-external-initiators=true # 支持外部触发器
   operator-private-key=""    # 通过mnemonic生成的私钥
   operator-password=""       # 操作员密码
  ```

6. 执行命令启动Chainlink节点。

   ```bash
   chainlink local n
   ```

7. 观察日志输出，直到看到“Bootstrap complete!”（引导成功）字样。

### 第五步: 配置Chainlink预言机

1. 登录Coinbase Dashboard，选择之前创建的Deposit。
2. 选择Create Preprocessor（创建预处理器）。
3. 选择预言机类型为External Adapter（外部适配器）。
4. 选择External Initiator Endpoint URL（外部初始化器端点URL）。

   ```
   http://<your server ip>/run
   ```

5. 填写以下配置参数。

   ```json
   {
     "jobRunID": "${JOB_RUN_ID}",
     "data": {
       "string": "<any string to pass into adapter>",
     },
     "remoteAdapter": true,
     "maxMemoryUsage":"300Mi",
     "expectedStatusCode":200,
     "timeout":"1m0s",
     "pollInterval":"10s",
     "webhookUrl":""
   }
   ```

6. 点击Next（下一步）创建预处理器。
7. 返回Deposit页面，选择Add job（添加任务）。
8. 选择适合您的用例的模板，或者自定义模板。
9. 点击Configure Job（配置任务）。
10. 在Job Specifications中填入以下配置。

    ```yaml
    initiators:
      - external/http
    tasks:
      - externalInitiator:
          url: 'http://localhost:<port>/<path>'
          path: /run
    triggers:
      - blockheight:
          threshold: 1
    ```

11. 点击Save and Run（保存并运行）。

### 第六步: 测试Chainlink节点

您可以使用postman或其他api测试工具来模拟各种请求。下面以HTTP POST请求为例。

1. 安装Postman插件。
2. 打开Postman，切换到POST方法。
3. 添加URL。

   ```
   http://localhost:<node port>/v2/jobs/<your job id>/runs
   ```

4. 发送请求Body JSON数据。

   ```json
   {"data":{"string":"my data"}}
   ```

5. 发送请求。

如果响应状态码为200 OK，则表明Chainlink节点已正确运行。