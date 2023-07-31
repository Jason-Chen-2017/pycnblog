
作者：禅与计算机程序设计艺术                    

# 1.简介
         
2020年已经过去了半个多世纪，人工智能领域取得了巨大的进步，伴随着技术革新带来的应用的爆炸式增长，数据驱动的业务模式正在改变着社会经济的运作方式。而作为云计算服务商 Hasura 的产品，则是在云端提供数据库即服务、API 网关、事件/消息总线及其它基础设施服务，帮助客户实现快速部署和实时同步。
         
         在过去的十几年里，Hasura 的产品和服务一直在向客户展示如何通过云端实现数据持久化、API 网关、微服务架构等功能，同时也成为大型企业客户的首选。在全球范围内拥有众多用户的云平台，是 Hasura 提供其服务的基础。近期，Hasura 宣布将推出其云平台服务 Hasura Cloud 。而本文将为大家带来 Hasura Cloud 的简单介绍，包括产品特性、核心优势、应用场景等方面。
         
         # 2.产品特性
         ## 功能强大
         Hasura Cloud 为客户提供了完整的管理环境，包含数据库即服务、API 网关、事件/消息总线及其它基础设施服务。除了基础设施服务之外，还提供了丰富的工具和组件，能够帮助客户实现数据分析、监控告警、测试和迭代，确保服务质量。
         
         此外，还提供高度可扩展性，可以满足各种规模的客户的需求。Hasura Cloud 支持部署和管理多个集群，并允许客户通过 API 或管理界面进行访问权限控制。此外，它还支持多种付费方案，包括免费试用版、个人开发者、中小型团队、大型公司和合作伙伴。
         
        ![](https://graphql-engine-cdn.hasura.io/learn-hasura/assets/cloud-vs-self-hosted/hs_pricing.png)
         
       ![](https://graphql-engine-cdn.hasura.io/learn-hasura/assets/cloud-vs-self-hosted/chart.png)
         
         ## 安全可靠
         为了确保服务的高可用性和稳定性，Hasura Cloud 使用高度安全的容器化技术，并采用了一系列安全措施，例如双向 TLS 加密、容器编排层隔离、主机基准检测、运行时威胁防护和自动更新机制。此外，还有针对事件处理和其它功能的访问控制机制，确保只有授权的用户才能访问相关信息。

         
       ![](https://graphql-engine-cdn.hasura.io/learn-hasura/assets/cloud-vs-self-hosted/secure.jpg)

         ## 操作友好
         Hasura Cloud 是基于一个开源项目构建的，因此无论是对用户还是企业管理员来说都容易上手。它的管理界面提供了完整的配置和设置选项，包括数据库和资源管理、订阅、认证和授权、服务升级等功能，用户可以根据自己的需求轻松地自定义配置。

         
        ![](https://graphql-engine-cdn.hasura.io/learn-hasura/assets/cloud-vs-self-hosted/user-interface.png)


     # 3.核心优势
     ## 性能优化

     云端服务在架构、存储和网络层面都有极大的优化空间。云端部署在 Azure 数据中心的多个机架之间建立了 99.9% 的容错率，并且通过区域冗余服务来实现高可用性。基于 Kubernetes 框架，Hasura Cloud 可有效利用每个节点上的计算资源来加速查询处理速度，而且可以按需调整计算资源规格，从而满足客户不同的业务需求。
     
    ![](https://graphql-engine-cdn.hasura.io/learn-hasura/assets/cloud-vs-self-hosted/speedup.png)

     ## 易于管理
     在 Hasura Cloud 中，管理员只需要登录到管理界面即可快速部署数据库服务和基础设施服务。无论是部署数据库、创建 GraphQL 接口、配置网络、授权或集成第三方系统，都可以通过单个控制台完成。这一切都是通过灵活的模板化引擎完成的，使得部署变得尽可能的快捷。Hasura Cloud 还集成了 SaaS 和 PaaS 服务，让客户可以使用最流行的开源软件包来实现自己的应用程序。
     
    ![](https://graphql-engine-cdn.hasura.io/learn-hasura/assets/cloud-vs-self-hosted/ui.png)

     
    ## 全面的工具和组件

    Hasura Cloud 提供了多项工具和组件，旨在帮助客户实现数据分析、监控告警、测试和迭代，确保服务质量。其中包括日志分析器、指标仪表盘、事件查看器、SQL 查询编辑器、服务器调试器、API 测试客户端、CI/CD 流程、文档生成器、应用模板库和插件市场。这些工具都具有独特的功能，能够帮助客户更快、更有效地进行数据分析。
     
   ![](https://graphql-engine-cdn.hasura.io/learn-hasura/assets/cloud-vs-self-hosted/components.png)
 
   # 4.应用场景

   目前，Hasura Cloud 已经在生产环境中部署了超过 700 个客户，涉及金融、零售、政府、制造、电信、媒体、医疗卫生等各行各业。在不同行业的应用场景下，Hasura Cloud 提供了完备的解决方案：

   1. **电子商务、外卖送货**

  ![](https://graphql-engine-cdn.hasura.io/learn-hasura/assets/ecommerce.png)

   电商网站、电商App、商城网站、外卖平台、物流管理系统都可以托管在 Hasura Cloud 上，实现统一的数据存储、事件驱动、订阅通信和支付结算等功能。
   
   2. **网络安全**

  ![](https://graphql-engine-cdn.hasura.io/learn-hasura/assets/network-security.png)

   有些组织希望将其关键网络设备和服务放在云端，同时，又需要确保这些设备始终保持最新状态。云计算服务可以帮助客户降低维护成本，因为它们能够自动更新和替换设备。网络安全解决方案还可以向客户提供状态监测和审计功能，帮助检测恶意攻击或入侵行为。
   
   3. **IoT 设备和传感器**

  ![](https://graphql-engine-cdn.hasura.io/learn-hasura/assets/iot-and-sensors.png)

   IoT 设备和传感器的数目不断增加，越来越多的设备会产生大量数据。数据的处理和分析对于提升工作效率、改善生产力和减少风险非常重要。使用 Hasura Cloud 可以轻松地收集和分析这些数据。同时，它也可以为客户提供简单的管理界面，方便实时监控数据。
   
   4. **移动应用开发**

  ![](https://graphql-engine-cdn.hasura.io/learn-hasura/assets/mobile-app-development.png)

   许多应用都会依赖于远程后端服务。使用 Hasura Cloud 可以为客户提供一个全面的服务，该服务将为移动应用开发人员提供 GraphQL 接口，并提供一个管理界面来部署和管理数据库、GraphQL 配置、身份验证和授权策略等。

