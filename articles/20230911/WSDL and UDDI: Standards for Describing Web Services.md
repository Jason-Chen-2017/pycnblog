
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 1.1 概述
Web服务已经成为当今互联网应用的基础设施，而其描述标准也正在不断地推进。过去十几年间，SOAP（Simple Object Access Protocol）、WADL（Web Application Description Language）、UDDI（Universal Description Discovery and Integration）等几种协议陆续成熟。在这其中，WSDL和UDDI都是用来描述Web服务的标准。本文将详细阐述WSDL和UDDI的历史发展和作用，并对两种协议进行比较分析，探讨其优劣及区别。

## 1.2 传统模式
### 1.2.1 SOAP
SOAP(Simple Object Access Protocol)由IBM、Microsoft、Oracle等公司共同开发，是一种通过XML格式定义接口的方式来访问网络服务的消息协议。目前已逐渐被WSDL所取代，但其仍然是WebService中使用的一种古老的通信方式。

1998年，为了解决分布式计算环境下的数据交换问题，IBM提出了分布式计算环境下的远程过程调用模型DRPC。这是一种用于跨网络环境中的远程过程调用的新型方法。

2000年，微软发布了Windows Communication Foundation (WCF)，提供了一个面向对象的编程模型和运行时环境，用于构建各种分布式应用程序。随后，Apache又基于此构建了Axis的WebService组件，作为Java平台的Web服务规范。

2001年10月，IBM宣布把SOAP技术引入商用标准组织OASIS，并发布了一系列相关的文档，包括Web Services Interoperability Guidelines (WS-I)、Web Services Addressing 1.0、Web Services Binding 1.0、SOAP with Attachments、SOAP Version 1.2。

2001年底，W3C批准了WebService标准的第二个版本——Web Services Description Language (WSDL)。WSDL旨在为开发人员提供一套全面的描述Web服务的方法，包括服务的类型、位置、端口、操作、消息、绑定等方面。

2003年6月，OASIS发布了第一个正式的UDDI版本。

### 1.2.2 RESTful API
RESTful API(Representational State Transfer)设计风格的出现改善了Web服务的可伸缩性和可用性。它通常基于HTTP协议，使用资源的表示形式而不是命令来交换数据，允许客户端改变服务器上的资源状态或资源的表现形式。这种新的Web服务形式使得服务器可以将状态信息的变化以多种形式传播给客户端，同时允许客户端自由选择需要的信息。例如，客户端可以指定自己的URL地址、请求方法（GET/POST/PUT/DELETE等）、提交的数据格式（XML/JSON）、请求参数、返回数据的过滤条件、分页控制等。目前，越来越多的公司和组织采用这种新的Web服务形态。

## 1.3 WSDL概览
WSDL(Web Service Description Language)是一种XML语言，它是Web服务交流的接口定义语言。它包括一个服务条目、一组端口类型、一组操作、一组消息、一组服务端点的引用及元数据。WSDL描述了一个服务的功能和属性，包括服务的名称、目标命名空间、支持的消息编码样式、操作列表、消息结构、端点地址、安全设置、绑定、操作期望的响应时间、错误处理机制等。

### 1.3.1 WSDL的历史

WSDL于1997年5月开始着手编写，1999年完成初稿，但是很快就发现有很多的问题要修订，WSDL还没有成为行业通用的标准。2001年底，OASIS发布了第一个正式的WSDL版本1.0。在此之后，OASIS发布了WSDL 2.0的草案，建议大家尽快把这个版本的草案纳入到标准化进程中。2003年，WSDL 2.0正式成为行业标准。

WSDL的第一版定义了五大元素：

* Types：定义了数据类型的语法和语义，用于指定服务的输入输出消息
* Messages：描述消息的结构、顺序、消息头、确认和错误条件
* PortType：定义了服务的端口，用于定义操作、消息的类型、操作期望的响应时间和错误处理机制
* Binding：定义了消息的编码方式、传输协议、安全措施、网络地址等
* Service：定义了服务的端点地址、服务的策略、服务依赖的其他服务等。

### 1.3.2 WSDL的作用

WSDL作为Web服务的描述语言，其重要作用之一就是方便服务的使用者了解如何调用服务，从而促进互联网的开放。当今的网络服务已经复杂到足够难以管理，如果没有对服务的描述，那将是无法利用这些服务的。对于开发者来说，WSDL也可以作为实现服务发现的工具。

## 1.4 UDDI概览
UDDI(Universal Description, Discovery, and Integration)是Web服务注册中心。它是一个根据通用描述、发现和集成技术开发的开放系统，能够存储、检索、索引和管理Web服务的信息。它的目标是在一个集中的数据库中保存所有的Web服务注册信息。用户可以通过浏览、搜索、查询等方式获取Web服务的相关信息。UDDI也是W3C组织推荐的元数据中心，被认为是部署Web服务的最佳实践。

### 1.4.1 UDDI的历史

UDDI于1998年由OASIS正式推出，至今已有20年历史。第一阶段的UDDI由IBM和Microsoft共同负责开发，UDDI主要提供服务注册、发现和浏览。1998年7月，OASIS批准发布UDI规范草案。

20世纪90年代初，随着互联网服务的广泛使用，网站的数量急剧增加，如何有效地管理和共享这么多网站的信息成为一项长久而复杂的任务。在此背景下，由于域名的单一性和管理困难，人们希望找到一种全面的服务注册中心，可以让用户更轻松地查询和获得网站的服务信息。因此，1998年9月，OASIS批准UDDI 1.0版正式发布。

到2000年，UDDI已经成为一个由50多个国家和地区的团体维护的Web服务注册中心。截止到2015年，UDDI依然是业界最知名的Web服务注册中心，包含超过20亿个服务条目。

### 1.4.2 UDDI的作用

UDDI可以帮助Web服务使用者快速发现、查找和利用Web服务。它提供统一且权威的服务发现机制，可以在线上和线下提供统一的服务目录，使得Web服务的使用者可以通过它来查找、评价、购买、部署和管理服务。目前，UDDI被广泛应用于各类政府部门、企业、金融机构、教育机构和公共组织等。