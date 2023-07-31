
作者：禅与计算机程序设计艺术                    

# 1.简介
         
PCI DSS(Payment Card Industry Data Security Standards)支付卡行业数据安全标准是美国网络支付公司VISA、MasterCard、American Express等政府组织为了保障支付卡机密信息的安全而制定的安全标准。PCI DSS旨在为所有参与网络支付业务的实体提供一套有效且可信赖的解决方案。本文介绍了PCI DSS的相关术语，并给出不同章节下的一些案例。
# 2.相关术语
## 2.1.PCI DSS
PCI DSS全称“Payment Card Industry Data Security Standards”，是由PCI组织提出的用于保护支付卡机密信息的安全标准。该标准适用于所有向网络支付机构(例如Visa、Mastercard、American Express等)提供服务的商户、接受支付处理的第三方支付机构、网关、代理商、金融机构等。
## 2.2.PCI
PCI(Payment Card Industry)是美国网络支付公司(Visa、MasterCard、American Express)与其他支付机构联合推出的非营利组织，其主要任务是制定网络支付安全标准。PCI可以说是网络支付领域的ISO，它负责监督其他网络支付机构遵循PCI标准。
## 2.3.DSS
DSS即Data Security Standards，即保护数据的安全标准。
## 2.4.SEIMAS
SEIMAS（Security Engineering Information Management Assistance System）是一个管理安全工程的工具包，是由国际标准化组织ISO/IEC JTC1 SC27 (ISO-DIS 291)制定的一组管理安全人员、流程、资料和设备的信息系统。包括安全计划、安全审计、风险分析、信息安全管理体系建设、应急响应、数据安全防护、漏洞检测与预防、事件响应、持续性安全培训、信息资源管理、计算机安全意识教育等方面。
## 2.5.HIPPA
HIPAA(Health Insurance Portability and Accountability Act of 1996)是美国医疗保健行业数据安全的法律。规定了个人信息的保护和使用规范。作为行业标准，HIPAA对支付卡数据隐私和安全非常关注。PCI组织将HIPPA中的安全控制点直接纳入PCI DSS中。
## 2.6.PCI 要求
PCI DSS共有五个要求，分别是：

1. 风险识别与评估；
2. 数据安全计划；
3. 数据安全培训；
4. 密码管理；
5. 漏洞管理及报告。

其中，第一条一般情况下是最重要的，也是最容易忽略的一条，也是PCI DSS试图减少或消除的风险。其他四条则是指导其他三个需求的基础，确保PCI DSS得到正确的落实。
## 2.7.常见病毒和攻击方式
常见的PCI DSS风险之一就是恶意攻击。常见的攻击方式有三种：

1. Denial of Service (DoS): 攻击者通过压力测试、拒绝服务攻击等手段让受害者无法正常访问网站或者服务器；
2. Brute Force Attack: 攻击者通过暴力破解的方式获取账户密码，进一步获取受害者的个人信息；
3. Social Engineering Attacks: 通过虚假的联系方式诱骗受害者点击链接或下载文件，然后通过木马窃取受害者的个人信息。
# 3. PCI DSS与HIPPA之间的关系
PCI DSS与HIPPA之间存在密切联系。HIPPA是美国医疗保健行业的数据安全准则。PCI DSS的第一个要求中就强调了要遵守HIPPA，这两个标准都包含了对个人信息的保护和使用规范。因此，PCI DSS实际上是在更高层次上对HIPPA进行了一定的补充。同时，在保证个人信息安全的前提下，PCI DSS还可以加强与个人身份信息相关的安全控制措施，比如密码管理和授权。所以，在考虑到数据泄露的影响和各种攻击手段后，PCI DSS最终会成为保护支付卡数据安全的最佳选择。

