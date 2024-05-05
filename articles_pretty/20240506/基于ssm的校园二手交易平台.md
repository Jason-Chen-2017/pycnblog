# 基于ssm的校园二手交易平台

作者：禅与计算机程序设计艺术

## 1. 背景介绍

### 1.1 校园二手交易平台的需求分析

#### 1.1.1 校园二手交易现状
在大学校园中,学生们经常需要买卖二手物品,如教材、电子产品、生活用品等。然而,目前大多数学生还是通过线下方式进行二手交易,存在信息不对称、交易不便等问题。

#### 1.1.2 建设校园二手交易平台的意义
建设一个线上的校园二手交易平台,可以方便学生发布和浏览商品信息,提高交易效率,节约交易成本,同时也可以促进资源循环利用,减少浪费。

### 1.2 系统架构选型

#### 1.2.1 SSM框架简介
SSM框架是指Spring + Spring MVC + MyBatis的组合,是目前主流的Java Web开发框架。其中:
- Spring是一个轻量级的控制反转(IoC)和面向切面(AOP)的容器框架
- Spring MVC分离了控制器、模型对象、分派器以及处理程序对象的角色,是一个基于MVC架构的web框架
- MyBatis是一个支持定制化SQL、存储过程以及高级映射的持久层框架

#### 1.2.2 SSM框架的优势
SSM框架具有如下优势:
- 低耦合,各个层之间耦合度低,方便单元测试
- 易扩展,框架的各个层都可以用不同的技术实现,灵活性高
- 开发快速,框架提供了大量现成的组件,开发效率高
- 社区活跃,学习资料丰富,遇到问题时容易找到解决方案

### 1.3 系统功能需求

校园二手交易平台的主要功能需求如下:
- 用户注册与登录
- 用户角色管理(买家、卖家、管理员)
- 商品发布、编辑、删除
- 商品分类、搜索、筛选、排序
- 商品详情展示,支持图文并茂
- 在线聊天与消息通知
- 交易评价与用户信用体系
- 支付功能对接

## 2. 核心概念与关系

### 2.1 领域模型设计

#### 2.1.1 用户 User
属性:
- id
- username
- password
- email
- phone
- avatar
- role
- credit

关系:
- 一个用户可以发布多个商品
- 一个用户可以购买多个商品
- 一个用户可以评价多个其他用户

#### 2.1.2 商品 Item
属性:  
- id
- name
- description 
- price
- category
- status
- postTime
- images

关系:
- 一个商品属于一个分类
- 一个商品只能由一个用户发布
- 一个商品可以被多个用户收藏

#### 2.1.3 订单 Order  
属性:
- id
- itemId
- buyerId
- sellerId 
- totalPrice
- status
- createTime

关系:  
- 一个订单对应一个商品
- 一个订单包含一个买家和卖家

#### 2.1.4 评价 Comment
属性:  
- id
- userId
- itemId
- orderId
- content
- rating
- createTime

关系:
- 一个评价对应一个订单
- 一个评价对应一个用户

### 2.2 数据库设计

根据领域模型,设计数据库表结构如下:

```sql
CREATE TABLE `user` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `username` varchar(50) NOT NULL,
  `password` varchar(100) NOT NULL,
  `email` varchar(100) DEFAULT NULL,
  `phone` varchar(20) DEFAULT NULL,
  `avatar` varchar(200) DEFAULT NULL,
  `role` tinyint(4) NOT NULL COMMENT '1-buyer, 2-seller, 3-admin',
  `credit` int(11) NOT NULL DEFAULT '100',
  PRIMARY KEY (`id`),
  UNIQUE KEY `username` (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE `item` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `name` varchar(100) NOT NULL,
  `description` text,
  `price` decimal(10,2) NOT NULL,
  `category` varchar(50) DEFAULT NULL,
  `status` tinyint(4) NOT NULL DEFAULT '1' COMMENT '1-on sale, 2-sold, 3-removed',
  `post_time` datetime NOT NULL,
  `user_id` int(11) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `user_id` (`user_id`),
  CONSTRAINT `item_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE `item_image` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `item_id` int(11) NOT NULL,
  `image` varchar(200) NOT NULL,
  PRIMARY KEY (`id`),
  KEY `item_id` (`item_id`),
  CONSTRAINT `item_image_ibfk_1` FOREIGN KEY (`item_id`) REFERENCES `item` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE `order` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `item_id` int(11) NOT NULL,
  `buyer_id` int(11) NOT NULL,
  `seller_id` int(11) NOT NULL,
  `total_price` decimal(10,2) NOT NULL,
  `status` tinyint(4) NOT NULL DEFAULT '1' COMMENT '1-created, 2-paid, 3-delivered, 4-confirmed',
  `create_time` datetime NOT NULL,
  PRIMARY KEY (`id`),
  KEY `item_id` (`item_id`),
  KEY `buyer_id` (`buyer_id`),
  KEY `seller_id` (`seller_id`),
  CONSTRAINT `order_ibfk_1` FOREIGN KEY (`item_id`) REFERENCES `item` (`id`),
  CONSTRAINT `order_ibfk_2` FOREIGN KEY (`buyer_id`) REFERENCES `user` (`id`),
  CONSTRAINT `order_ibfk_3` FOREIGN KEY (`seller_id`) REFERENCES `user` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

CREATE TABLE `comment` (
  `id` int(11) NOT NULL AUTO_INCREMENT,
  `user_id` int(11) NOT NULL,
  `item_id` int(11) NOT NULL,
  `order_id` int(11) NOT NULL,
  `content` text NOT NULL,
  `rating` int(11) NOT NULL,
  `create_time` datetime NOT NULL,
  PRIMARY KEY (`id`),
  KEY `user_id` (`user_id`),
  KEY `item_id` (`item_id`),
  KEY `order_id` (`order_id`),
  CONSTRAINT `comment_ibfk_1` FOREIGN KEY (`user_id`) REFERENCES `user` (`id`),
  CONSTRAINT `comment_ibfk_2` FOREIGN KEY (`item_id`) REFERENCES `item` (`id`),
  CONSTRAINT `comment_ibfk_3` FOREIGN KEY (`order_id`) REFERENCES `order` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;
```

## 3. 核心算法原理与具体操作步骤

### 3.1 用户登录认证

#### 3.1.1 密码加密存储
用户密码不能明文存储在数据库中,需要经过加密处理。常用的加密算法有MD5、SHA等。

以SHA256为例,加密过程如下:
1. 前端使用js获取用户输入的密码明文
2. 对密码明文进行SHA256计算,得到密码密文
3. 将用户名和密码密文传给后端接口
4. 后端从数据库中查询该用户名对应的密码密文
5. 将数据库中的密码密文与前端传来的密码密文进行比对
6. 如果两个密文一致,则认证通过;否则认证失败

#### 3.1.2 登录状态保持
用户登录成功后,需要保持登录状态,以便在后续的请求中能识别该用户的身份。

常见的方案有:
- Session方案:服务端存储用户信息
  1. 用户登录成功后,服务端将用户信息存入Session中
  2. 服务端生成一个唯一的SessionId,通过响应头Set-Cookie返回给浏览器
  3. 后续请求时浏览器将SessionId放在请求头Cookie中发送,服务端根据SessionId找到对应的用户信息
- Token方案:客户端存储用户信息
  1. 用户登录成功后,服务端根据用户信息生成一个Token字符串(可以是随机字符串或JWT)
  2. 服务端将Token返回给浏览器
  3. 浏览器将Token保存在本地(如LocalStorage)
  4. 后续请求时浏览器将Token放在请求头Authorization中发送,服务端根据Token找到对应的用户信息

### 3.2 商品搜索与推荐

#### 3.2.1 商品分词
搜索引擎需要对商品标题和描述进行分词,提取关键词并建立倒排索引,以便快速检索。

常用的分词算法有:
- 基于字典的正向最大匹配算法
- 基于字典的逆向最大匹配算法
- 基于统计的N-最短路径算法
- 基于隐马尔可夫模型的Viterbi算法

以正向最大匹配算法为例,分词过程如下:
1. 从待分词文本的首字符开始,取指定大小的子串(如5)
2. 判断该子串是否在词典中
3. 如果在词典中,则将该子串作为一个词切分出来
4. 如果不在词典中,则去掉子串的最后一个字符,重复步骤2
5. 重复步骤1,直到文本结束

#### 3.2.2 商品相关度计算
根据用户的搜索词,需要计算每个商品与搜索词的相关度,并按相关度排序。

常见的相关度计算模型有:
- TF-IDF
- BM25
- Word2Vec
- LSI 
- LDA

以TF-IDF为例,计算过程如下:
1. 根据商品分词结果,统计每个词在各个商品中的词频TF
2. 根据各个词在所有商品中的出现频率,计算每个词的逆文档频率IDF
3. 用TF与IDF的乘积作为每个词对商品的贡献度
4. 将搜索词中每个词对商品的贡献度相加,得到该商品与搜索词的相关度

#### 3.2.3 商品推荐
除了用户主动搜索商品外,系统还应该根据用户的历史行为,自动推荐用户可能感兴趣的商品。

常见的推荐算法有:
- 基于用户的协同过滤
- 基于物品的协同过滤
- 基于模型的协同过滤
- 基于内容的推荐
- 组合推荐

以基于物品的协同过滤为例,推荐过程如下:  
1. 根据所有用户的历史行为,计算物品两两之间的相似度
2. 对于某个用户,找出他过去感兴趣的物品
3. 根据物品相似度,找出与这些物品最相似的其他物品
4. 去掉用户已经交互过的物品,得到推荐列表

## 4. 数学模型与公式详解

### 4.1 用户信用评分模型

在二手交易平台中,需要对用户的信用进行量化评估,以判断用户的可信程度。

我们可以设计一个用户信用评分模型,影响因素包括:
- 好评率:用户收到的好评数占总评价数的比例
- 购买数:用户购买商品的次数
- 售出数:用户售出商品的次数
- 信用分:初始信用分100,根据用户行为加减分

信用评分计算公式如下:

$$Score = \alpha \times GoodRating + \beta \times BuyCount + \gamma \times SellCount + \delta \times CreditPoints$$

其中:
- $GoodRating$ 表示用户的好评率,取值范围[0,1]
- $BuyCount$ 表示用户的购买数,需要归一化到[0,1]
- $SellCount$ 表示用户的售出数,需要归一化到[0,1]  
- $CreditPoints$ 表示用户的信用分,需要归一化到[0,1]
- $\alpha, \beta, \gamma, \delta$ 为各个因素的权重系数,且$\alpha + \beta + \gamma + \delta = 1$

举例如下:
- 假设某用户收到100个评价,其中95个好评,则$GoodRating=0.95$
- 假设某用户购买了20次商品,售出了10次商品,平台最高记录为100次,则$BuyCount=0.2, SellCount=0.1$
- 假设某用户当前信用分为120分,信用分取值范围为[0,200],则$CreditPoints=0.6$
- 假设我们取$\alpha=0.5, \beta=0.2, \gamma=0.2, \delta=0.1$

则该用户的信用评分为:

$$Score = 0.5 \times 0.95 + 0.2 \times 0.2 + 0.2 \times 0.1 + 0.1 \times 0.6 = 0.595$$

### 4.2 商品推荐模型

在3.2.3节