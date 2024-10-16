
作者：禅与计算机程序设计艺术                    

# 1.简介
  

随着数字货币市场的快速发展，越来越多的人开始使用数字货币进行交易。如何让用户能够更好地管理数字货币资产，是一个值得探索的问题。
数字货币钱包应用(Digital Wallet App)就是为了解决这个问题而设计的。本文将以“Coinbase Pro”作为案例，阐述如何开发一个适合中国用户使用的数字货币钱包APP。如果你不是数字货币专家或对此不太了解，请不要担心，文章中的内容并不会涉及到太高级的技术。只需要有一点编程基础、兴趣、以及耐心，即可完成本文的阅读。

# 2.基本概念术语
## 什么是数字货币？
数字货币（Digital Currency）是一种通过计算机计算的，用密码方式存储的具有一定价值的加密货币。其单位通常为比特币（BTC），通过在区块链上记录交易信息实现价值流通，并通过网络传输给所有参与者。

## 什么是数字货币钱包？
数字货币钱包是指用来存储和管理数字货币资产的应用程序。用户可以通过数字货币钱包来进行数字货币的发送和接收，也可以进行数字货币的交易、查询等操作。目前市面上的主流数字货币钱包有Bitgo、BlockChain、MyCrypto、Coinbase Pro等。

## 什么是Coinbase Pro？
Coinbase Pro是美国的一家数字货币交易平台公司，提供比特币、莱特币、以太坊等加密货币的交易服务。2018年9月10日，Coinbase宣布完成IPO，估值达到了70亿美元。Coinbase Pro是一个基于AWS云端服务器的全球领先的数字货币钱包应用。

# 3.核心算法原理
首先，我们需要了解一些关于数字货币的基本术语。

## 公钥私钥对
要理解数字货币的交易，就需要理解公钥私钥对这一概念。公钥私钥对是生成公钥和私钥的方法。公钥可以被他人发布，但是私钥则只有自己知道。

公钥和私钥可以对应到不同的密钥对。生成公钥和私钥的方法如下：

1. 生成一对非对称加密密钥：
首先，选取两个质数 p 和 q 。然后，计算 n = p * q ，计算欧拉函数 φ(n) = (p-1)(q-1)，选择 e 满足 gcd(e,φ(n))=1 ，计算 d 满足 ed ≡ 1 mod φ(n)。

2. 根据 n （模数）和 e （公钥），计算出公钥 h = (n,e) 。

3. 根据 n （模数）和 d （私钥），计算出私钥 k = (n,d) 。

公钥和私钥都是二进制字符串。公钥是指公开的，任何人都可以看到；私钥是保密的，只有拥有者才知道。一般情况下，私钥只能由持有者掌控，公钥则可以自由散播。

## 比特币地址
比特币地址（Bitcoin Address）是每个比特币的唯一标识符。它类似于身份证号码或者银行账户号码，用户可以通过该地址来收款或者支付。比特币地址由公钥经过哈希运算生成，得到比特币地址。

公钥的哈希运算过程如下：

1. 将公钥转换成字节数组形式，也就是长度为 65 的数组。
2. 在第0个元素添加 0x04 。
3. 使用 SHA256 哈希函数对前面的数组进行加密。
4. 对加密结果再次进行 SHA256 哈希运算。
5. 取加密结果的前 20 个字节（40 字符）作为比特币地址。

例如，假设我们有一个公钥为 “02 FACB6DF7B2B4F5D1A9EFEC51CE20ADFF995DA9EBD1D5ED1BBCA0D10FEE1554EBA”，其对应的比特币地址为 “1GKoHbTQZjDjfbmfdNqAGfHhCUnNjdJwbz”。

## 钱包种类
Coinbase Pro 钱包分为两大类——汇总账户和子账户。

汇总账户（Summary Account）主要用于管理整个 Coinbase Pro 账户中各个币种的余额。汇总账户不能单独存在，它必须至少包含一个子账户才能正常工作。汇总账户的余额取自各个子账户的余额汇总。

子账户（SubAccount）可以理解为普通的比特币账号，可以存储多个不同币种的币。每个子账户都有自己的 ID 和名称。子账户之间是相互独立的。

对于个人用户来说，通常只会创建一个汇总账户，一个子账户就够了。当然，对于企业级用户或团队内部使用，可以创建多个子账户，共同管理整个团队的比特币资产。

## 交易所
交易所（Exchange）是指接受比特币用户转账的商家或机构。交易所通常会向用户提供一个网页或APP，用户可以在其中填写转账信息、查看账户余额、选择支付方式、确认支付。交易所一般都会接受多种币种的交易。

# 4.具体操作步骤
## 注册 Coinbase Pro 账号
打开 www.coinbase.com/signup 页面，点击右侧的“Get Started for Free”按钮。


填入邮箱、用户名和密码后，点击“Create an account”按钮，进入注册页面。


按照提示，完成验证，点击“Next: Profile Information”按钮。


输入真实姓名、国家、城市和信用卡信息后，点击“Next: Verify Identity”按钮。


这里会要求你上传身份认证材料，如个人护照或驾驶执照等，根据页面提示填写相关信息。


验证成功后，点击“Complete Verification”按钮，进入 Coinbase Pro 首页。

## 创建子账户
登陆 Coinbase Pro 账户后，点击左侧导航栏的“Dashboard”选项，进入 Dashboard 页面。


点击顶部的“Add an Account”按钮，弹出窗口选择“Subaccount”，进入创建子账户页面。


输入子账户名称、描述信息（可选）、备注信息（可选）后，点击“Create Subaccount”按钮。


子账户创建成功后，页面会显示新创建的子账户的信息。

## 充值资金
登陆 Coinbase Pro 账户后，点击左侧导航栏的“Accounts”选项，进入 Accounts 页面。


找到想要充值的子账户，点击“Deposit”按钮，进入 Deposit 页面。


选择充值的币种、数量，输入支付密码（会发短信或邮件通知）后，点击“Buy”按钮，进入购买页面。


确认支付订单信息无误后，点击“Place Order”按钮，完成充值。

## 发起交易
登陆 Coinbase Pro 账户后，点击左侧导航栏的“Market”选项，进入 Market 页面。


选择想要交易的币种，点击搜索框，开始输入对方的比特币地址或用户ID。


搜索结果列表中显示的所有用户和地址都会显示对应的推荐价格，用户可以点击相应币种的图标选择交易。


进入交易界面后，输入交易金额，确认无误后，点击“Buy”按钮，执行购买操作。


## 查询余额
登陆 Coinbase Pro 账户后，点击左侧导航栏的“Accounts”选项，进入 Accounts 页面。


点击左侧导航栏中的某个子账户，进入子账户详情页面。


点击右上角的“View on blockexplorer”链接，进入区块浏览器，查看地址详情，包括所有交易记录、余额变动历史等。


# 5.附录
## 常见问题

1. 为什么需要身份认证？
身份认证是为了保障数字货币交易平台的安全性，防止恶意交易者盗取您的个人信息。Coinbase Pro 提供身份认证功能，使得您可以使用其服务时需先进行个人身份验证。

2. 支付密码的作用是什么？
支付密码是支付系统的一个重要安全机制，只有拥有正确的支付密码才能进行支付。支付密码必须由字母、数字组成，并且长度最少 8 位。支付密码保存在您的交易所账户内，对您在 Coinbase Pro 服务中的所有操作均需依赖于支付密码进行授权。

3. Coinbase Pro 子账户之间的资金划转是否受到限制？
Coinbase Pro 不允许子账户之间资金划转，只能进行个人之间或子账户之间资金划转。即，您只能把资金从一个子账户转移到另一个子账户，而不能直接从一个子账户转移到汇总账户或其他子账户。

4. 交易所为什么需要审核？
为了确保 Coinbase Pro 用户的资金安全，Coinbase Pro 会对所有交易所申请的用户提交的交易行为进行审核。审核通过的交易所账户才会进入激活状态。因此，交易所需要提供真实有效的付款账户、支付密码等信息。

5. 我想关闭 Coinbase Pro 账户，应该怎么做？
如果您决定暂时停止使用 Coinbase Pro 平台，请联系您的交易所取消您的账户，并通知他们处理您的资金冻结、退回等事宜。如需永久关闭您的 Coinbase Pro 账户，请联系 Coinbase Pro 客服人员。