
作者：禅与计算机程序设计艺术                    

# 1.简介
  

欢迎来到AWS云计算平台系列教程！从零到英雄，无需编程经验，只需要跟着六周的计划，一步步完成本书的内容，您将会掌握Amazon Web Services（AWS）平台的基础知识、应用、最佳实践等内容，建立自己的云计算能力。本课程适合所有想学习AWS云计算平台的技术人士。首先，让我们回顾一下什么是AWS？
## AWS简介
亚马逊网络服务（Amazon Web Services，简称AWS），是一个遍及全球的数据中心托管服务平台，它提供各种计算、存储、数据库、网络以及其他服务，包括云计算平台、服务器管理、应用程序开发、移动应用开发、分析处理、机器学习、物联网（IoT）设备、游戏开发、区块链等领域。通过提供完整的产品组合和解决方案集合，AWS帮助企业进行资源高度有效的利用，缩短交付时间，提高业务竞争力，降低成本。这些优点使得客户能够在必要时迅速响应需求变化，满足客户对可靠性、安全、可扩展性、可用性的要求。同时，AWS还提供广泛的工具、资源和支持，为客户节省了许多时间和金钱，尤其是在规模化部署上。AWS已经成为企业IT运营的领先选择之一，2017年被纽约证券交易所收购，目前，AWS共拥有超过27,000个数据中心，超过5万个商业应用、服务和API。截止2021年9月，AWS的云计算服务已覆盖各行各业，如零售、电子商务、医疗保健、金融、贸易、制造、运输、媒体和能源等多个领域。
图1. 亚马逊网络服务（AWS）架构图
## 本教程目标读者
本教程旨在帮助具有基础知识的技术人员快速入门AWS，具备至少一年以上相关工作经验，能够独立编写技术文档。如果您的学习方向偏重数据分析、机器学习、图像识别、深度学习、自然语言处理、金融科技、物联网（IoT）、区块链等方面，那么可以跳过第一章。这些知识是您应当熟练掌握的，而且不必担心对AWS平台的理解。本教程建议您按如下流程进行学习：

1.了解AWS的基本概念和价值。
2.注册并登录到AWS管理控制台。
3.尝试创建第一个虚拟机（VM）。
4.配置并使用安全组。
5.构建自动伸缩的VM集群。
6.配置负载均衡器和高可用性。
7.了解Amazon Elastic File System（EFS）和 Amazon Simple Storage Service（S3）。
8.构建无服务器架构（Serverless）应用。
9.深入探索AWS Lambda函数、Amazon Rekognition 和Amazon Polly。
10.熟悉容器技术。
11.进行服务之间的交互。
12.结合深度学习与AWS进行模型训练。
13.掌握Amazon Machine Learning、Amazon SageMaker 和Amazon Textract。
14.获得更多资料，研习心得和指导。

第2章 了解AWS的基本概念和价值
前言：“云”这个词汇近年来受到了越来越多人的关注。其真正意义是什么？如何实现？何时使用？AWS是当今最热门的云服务提供商之一，正在成为许多大型公司的首选云计算服务，2017年被称为全球第二大云计算供应商。AWS提供了云计算领域的所有功能和服务，从计算、存储、数据库、网络和应用开发等多个领域，都涉及其中。AWS提供了基础设施即服务（IaaS）、平台即服务（PaaS）和软件即服务（SaaS）三种形态的云服务，这三个形态依次增强、提升、升级客户的使用体验。
了解AWS的基本概念和价值对于掌握本教程至关重要。阅读完本章后，您将了解到以下几点：

1. AWS是什么？
2. AWS价格体系。
3. AWS价值主张。
4. AWS服务类型。
5. AWS主要区域分布。
6. AWS核心产品线。

## 1.1 AWS是什么？
Amazon Web Services（AWS）是亚马逊（Amazon）的一个系列产品，由软件工具、服务和产品组成，用于连接、管理和运营在线应用和产品，帮助客户在云端构建和运行业务应用程序。AWS提供了一个完全可管理的云端平台，使用户能够快速构建、测试和部署应用程序，而无需管理底层基础设施。AWS允许客户使用其基于Web的管理界面、命令行接口或应用程序编程接口（API）来访问其服务。其核心产品线包括Amazon EC2、Amazon S3、Amazon CloudFront、Amazon DynamoDB、Amazon ElastiCache、Amazon RDS、Amazon Redshift、Amazon Elasticsearch Service、Amazon Kinesis、Amazon Machine Learning、Amazon Neptune、AWS Lambda、Amazon API Gateway、Amazon CloudWatch、Amazon CloudTrail、AWS Secrets Manager、Amazon Cognito、AWS Batch、AWS Step Functions、AWS Glue、AWS Elemental MediaConvert、AWS Elemental MediaLive、AWS Elemental MediaPackage、AWS Elemental MediaTailor、Amazon Transcribe、Amazon Translate、Amazon Polly、Amazon Lex、Amazon Comprehend、Amazon Pinpoint、Amazon Personalize、AWS WAF和AWS Config等。截止2021年，AWS共拥有超过27,000个数据中心、超过5万个服务和API。
图2. AWS核心服务架构图
## 1.2 AWS价格体系
AWS提供了多种付费选项，不同的付费方式对应不同的价格体系，具体如下表所示：

图3. AWS价格体系示意图
根据不同用例的规模、使用频率和预算情况，AWS提供了多种付费选项，客户可以灵活选择：
- 消费券：用户可以使用优惠券享受折扣。
- 年包、季包、财年包：价格较低的一次性付款。
- 按需计费：每小时或每月按量计费。
- 预留实例：相比于按需付费，预留实例可降低费用，提高效率。
根据服务类型、用例、可用资源的数量、性能等因素，AWS向客户提供不同的套餐。例如：
- EC2实例：按vCPU和内存粒度计费。
- RDS实例：按容量大小、实例类型和地域计费。
- ElastiCache实例：按容量大小、实例类型、地域、多AZ选项等计费。
- AWS Lambda函数：按运行时间、内存占用等计费。
- 对象存储服务S3：按存储空间、请求次数、数据传输量计费。
- 云数据库服务RDS：按容量大小、实例类型、数据库类型和地域计费。
- 弹性文件系统服务EFS：按容量大小、冷存和网络带宽等计费。
- 流式数据服务Kinesis：按吞吐量、数据保留期、调用次数、集群实例数量计费。
- 批处理服务Batch：按运行任务量、执行时间和机器类型计费。
- 服务目录Service Discovery：按服务实例个数、名称长度和DNS查询次数计费。
- Amazon Athena：按查询次数、EBS IOPS和数据扫描量计费。
- AWS Quicksight：按用户、容量、加载数据的大小计费。
## 1.3 AWS价值主张
云计算是新一代IT技术，意味着大幅降低IT支出。基于这一新技术，AWS设立了一系列价值主张：
- 降低成本：通过采用云计算模型和服务，AWS可以通过高度优化的方式降低基础设施的投资和运营成本。
- 提高服务质量：AWS通过提供全面的基础设施和软件工具，帮助客户轻松应对复杂的业务环境。
- 高效运营：AWS通过提供一系列产品和服务，帮助客户管理复杂的基础设施，提高了业务的响应速度和效率。
- 创新能力：AWS通过提供一系列新产品和服务，帮助客户洞察和发现新的市场机遇，推动创新。
为了实现这些价值主张，AWS提供了多项服务，包括计算机、网络、存储、数据库、开发工具、安全工具等。每个服务都有不同的特性和功能，通过组合这些服务，客户可以构建出符合自己需求的系统架构。另外，AWS的生态系统也非常丰富，包括第三方应用和服务、工具和服务、解决方案以及培训、支持和社区支持等。总之，AWS就是通过简单、可靠、透明的方式，帮助客户降低成本、提高服务质量、创造价值。
## 1.4 AWS服务类型
AWS的服务分为四大类，分别是基础设施即服务（Infrastructure as a Service，IaaS）、平台即服务（Platform as a Service，PaaS）、软件即服务（Software as a Service，SaaS）和混合云。具体分类如下：
### （一）基础设施即服务
基础设施即服务（IaaS）包括计算、网络和存储服务，包括Amazon EC2、Amazon VPC、Amazon Route 53、Amazon S3、Amazon EBS、Amazon CloudFront、Amazon ElastiCache、Amazon Redshift、Amazon RDS、Amazon DynamoDB、AWS Lambda等。IaaS是一种提供基础设施服务的方法，客户可以在其自己的控制下，按照需求快速设置虚拟机、网络和存储等环境，不需要购买、维护和管理硬件。云计算平台经过IaaS层级，将底层基础设施的运维工作转移给云服务商，客户可以快速部署和扩展应用程序。比如，EC2可以提供弹性伸缩的虚拟服务器；Amazon ElastiCache可以缓存应用程序的数据；Amazon S3可以提供静态网站托管服务；AWS Lambda可以提供事件驱动的、自动执行的代码片段。
### （二）平台即服务
平台即服务（PaaS）是一种云服务模式，它包括应用程序开发框架、开发工具和服务，包括Amazon Elastic Beanstalk、Amazon CodeDeploy、Amazon Lightsail、AWS Elastic MapReduce、AWS CloudFormation、AWS OpsWorks等。PaaS通过封装底层的基础设施，让客户可以快速部署、测试、扩展和管理应用程序。Amazon Elastic Beanstalk为Java、.NET、PHP、Node.js和 Ruby等开发语言提供了支持，同时还集成了许多流行的软件库，包括关系型数据库MySQL、MongoDB、PostgreSQL、Redis、Memcached等。Amazon CodeDeploy可以帮助部署代码更新；Amazon Lightsail提供在线虚拟主机、负载均衡、DNS解析、SSL证书等服务。AWS Elastic MapReduce为批处理和分析提供了支持，它可以快速进行大数据分析和处理。AWS CloudFormation可以快速部署、更新和管理堆栈，简化了应用程序的部署、管理和扩展。AWS OpsWorks提供了一个编排工具，可以实现自动化、可重复使用的DevOps流程，帮助客户实现更快的发布周期。
### （三）软件即服务
软件即服务（SaaS）是一种按需付费的服务模式，客户只需要订阅并安装软件，就可以使用该软件。SaaS包括商业软件和云软件，如Microsoft Office 365、Salesforce、Zendesk、GitHub、Slack等。SaaS的最大优势在于用户不需要购买、安装和管理软件，使用起来非常便捷。对于那些基于云端的业务软件来说，AWS提供了许多服务，包括：
- Amazon WorkDocs：文档协作服务，适合团队合作；
- Amazon WorkMail：邮件服务，包括收件箱、日历、联系人、任务和通信；
- Amazon Chime：视频会议服务；
- Amazon Connect：语音服务；
- Amazon AppStream：远程桌面服务，适合虚拟现实、AR/VR等场景；
- Amazon QuickSight：BI和分析服务，帮助用户快速构建数据报告和仪表板；
- Amazon Alexa：语音助手，助力用户更便捷地访问服务和信息；
- AWS Marketplace：第三方应用市场，提供云计算服务的软件和应用；
除了商业软件外，还有一些基于云端的开源软件，如亚马逊的ECR、Amplify Console、CloudFormation、Data Pipeline、CodePipeline等。这些软件的免费版本往往具有有限的功能限制，但可以在需要时购买付费功能。
### （四）混合云
混合云是一种把本地数据中心、私有云和公有云结合起来的服务模式，包括Amazon Web Services（AWS）、Google Cloud Platform（GCP）、微软Azure、阿里云、腾讯云、IBM Cloud等。混合云将各家云服务商的优势互补，为客户提供一个统一的、综合性的云平台。不同类型的云服务之间通过网络连接，可以访问到同样的数据、应用和服务。这种服务模式可以让客户灵活地选择适合自己的服务，以便更好地满足业务需求。
## 1.5 AWS主要区域分布
AWS目前拥有27个国家和地区的数据中心。其中，美国东部、北美、西南亚、欧洲、日本、韩国、印度、中国、德国、法国、英国等地都拥有数据中心。除此之外，AWS还拥有亚太地区（东京、新加坡、首尔、迪拜、孟买、曼谷等地）和澳大利亚、俄罗斯、中国香港、中国台湾、马来西亚、泰国等国家的数据中心。
图4. AWS主要区域分布
## 1.6 AWS核心产品线
AWS拥有多种核心产品，它们包括Amazon EC2、Amazon S3、Amazon CloudFront、Amazon DynamoDB、Amazon ElastiCache、Amazon RDS、Amazon Redshift、Amazon Elasticsearch Service、Amazon Kinesis、Amazon Machine Learning、Amazon Neptune、AWS Lambda、Amazon API Gateway、Amazon CloudWatch、Amazon CloudTrail、AWS Secrets Manager、Amazon Cognito、AWS Batch、AWS Step Functions、AWS Glue、AWS Elemental MediaConvert、AWS Elemental MediaLive、AWS Elemental MediaPackage、AWS Elemental MediaTailor、Amazon Transcribe、Amazon Translate、Amazon Polly、Amazon Lex、Amazon Comprehend、Amazon Pinpoint、Amazon Personalize、AWS WAF和AWS Config等。这些产品的功能和特色可以实现超越传统IT环境的协同办公、移动办公、数字营销、人工智能、机器学习、物联网、区块链等领域的应用。例如，Amazon EC2提供了按需计算、弹性伸缩、云中镜像、IPv6等功能。Amazon S3提供了一个对象存储服务，适用于各种类型的应用，包括静态网站托管、企业文件共享、媒体文件存储、大数据分析等。Amazon CloudFront提供一个全球内容分发网络，为客户提供高速访问内容。Amazon DynamoDB提供了一个 NoSQL 数据库，可以快速、低延迟地处理海量数据。Amazon ElastiCache提供了一个内存缓存服务，可快速存储应用程序的数据，减少数据库的访问延迟。
## 小结
本章详细介绍了AWS的概念、价值主张、服务类型和主要区域分布等方面。之后，我们进入第2章，一起学习一些关于AWS基础知识的知识。