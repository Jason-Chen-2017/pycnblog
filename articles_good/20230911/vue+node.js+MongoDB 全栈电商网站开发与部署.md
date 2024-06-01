
作者：禅与计算机程序设计艺术                    

# 1.简介
  

随着互联网的发展，越来越多的人开始关注互联网电商这个行业。无论是在线购物、网上约会还是在线服务，都离不开“电商平台”。由于电商网站具有用户参与程度高、信息化程度高等特征，因此也逐渐成为互联网企业必备的基础设施。而基于互联网电商的业务模式快速发展，中小型电商平台也越来越受到企业青睐。近年来，Vue.js 已经成为了构建用户界面的热门框架，它将组件化设计带入前端领域，使得开发过程变得更加灵活可控。因此，我们可以用 Vue 来搭建一个完整的电商网站的前后端分离系统。本文将以 Vue + Node.js + MongoDB 为主要技术进行介绍。Vue 是构建Web应用的新一代 JavaScript 框架，其简洁的模板语法和双向数据绑定特性使得它非常适合开发单页应用。Node.js 是一个基于 Chrome V8 引擎的 JavaScript 运行环境，它使用事件驱动、非阻塞I/O模型及轻量级线程，性能出众，非常适合处理实时性要求高、并发量大的应用场景。MongoDB 是 NoSQL 数据库中的一种，是一个开源的面向文档存储的数据库。通过结合这些技术，我们可以打造出功能丰富、易于扩展的电商网站系统。
# 2.知识点概览
## 2.1. 后台管理系统
### 2.1.1. 用户管理模块
- 用户登录、注册、找回密码
- 修改个人信息
- 查看订单信息
### 2.1.2. 商品管理模块
- 添加、删除商品
- 上下架商品
- 修改商品信息
### 2.1.3. 分类管理模块
- 添加、删除分类
- 修改分类信息
### 2.1.4. 品牌管理模块
- 添加、删除品牌
- 修改品牌信息
### 2.1.5. 管理员管理模块
- 添加、修改管理员信息
### 2.1.6. 权限控制模块
- 分配权限给角色
- 设置权限优先级
### 2.1.7. 角色管理模块
- 创建、修改角色
- 删除角色
- 分配权限给角色
### 2.1.8. 系统设置模块
- 配置站点设置
- 配置邮件发送设置
- 配置短信发送设置
- 配置快递物流查询设置
## 2.2. 前台页面
- 首页（轮播图展示、推荐商品推荐、热门搜索推荐）
- 分类页（列表展示、排序、分页、关键字检索）
- 购物车页（加入购物车、更新购物车、删除购物车）
- 我的订单页（查看订单、确认收货、评价商品）
- 商品详情页（查看商品描述、规格参数、评论区）
- 登录、注册、忘记密码
- 会员中心（订单、地址、优惠券）
# 3.项目结构介绍
本项目的前后端分离架构如下图所示：
整个项目分为两个部分：

1. 后台管理系统：负责运营者进行日常维护管理。

2. 前台页面：包括首页、分类页、购物车页、订单页、会员中心等页面。

分别对应的后端 API 服务（API Gateway）如下图所示：
各个模块的详细功能将在后续的章节进行介绍。
# 4. 技术栈介绍
## 4.1. 后端技术栈
- Node.js：用作服务器编程语言，基于 Chrome V8 引擎，性能出色且轻量级，非常适合开发实时的 web 应用；
- Express：用于构建 Web 应用和 API 的框架，它提供诸如路由、中间件等功能；
- MongoDB：非关系数据库，支持分布式集群、水平扩展和自动故障转移，是当下最流行的 NoSQL 数据库之一；
- Mongoose：连接 Node.js 和 MongoDB 的 ODM 框架，简化了数据模型的定义、查询和操作；
- JWT：JSON Web Token，一种安全认证标准，用于保护 API 请求；
- Passport：提供身份验证策略，支持 OAuth、OpenID Connect、SAML 等多种认证协议；
- Redis：使用内存中的数据结构存储键值对，支持发布订阅、事务等特性；
- GraphQL：用于 API 的查询语言，提供了一种类型化的 API 查询方式；
- RabbitMQ：消息队列，用来实现异步任务的分布式处理；
- Nginx：提供反向代理、负载均衡和 HTTP 服务，同时也可以作为静态资源服务器；
## 4.2. 前端技术栈
- Vue.js：使用 MVVM 模式开发的客户端 JavaScript 框架，其轻量、功能丰富、容易上手，适合快速迭代和开发复杂的 web 应用；
- Element UI：饿了么推出的基于 Vue 2.0 的桌面端组件库，提供了丰富的基础组件和便捷的定制能力；
- axios：基于 Promise 的 HTTP 客户端，用于和后端 API 交互；
- webpack：用于构建、打包 JavaScript 模块，提供模块加载机制和插件机制；
- ESLint：JavaScript 编码风格检查工具，用于检测并修复代码错误；
- PM2：Node.js 进程管理器，用来简化进程管理；
# 5. 项目目录结构
```
├── package.json
├── README.md
├── src // 项目源码文件夹
│ ├── api // API 文件夹
│ │ ├── admin.js // 后台管理系统接口
│ │ ├── category.js // 分类接口
│ │ ├── goods.js // 商品接口
│ │ ├── order.js // 订单接口
│ │ └── user.js // 用户接口
│ ├── config // 配置文件
│ │ ├── index.js // 主配置项
│ │ ├── jwt.js // JWT 配置项
│ │ ├── mongodb.js // MongoDB 配置项
│ │ ├── pm2.js // PM2 配置项
│ │ ├── redis.js // Redis 配置项
│ │ └── smtp.js // SMTP 配置项
│ ├── middleware // 中间件文件夹
│ ├── models // 数据模型文件夹
│ │ ├── Admin.js // 管理员模型
│ │ ├── Category.js // 分类模型
│ │ ├── Goods.js // 商品模型
│ │ ├── Order.js // 订单模型
│ │ └── User.js // 用户模型
│ ├── public // 静态资源文件夹
│ ├── routes // 路由文件夹
│ ├── services // 逻辑层文件夹
│ │ ├── admin.js // 后台管理系统逻辑层
│ │ ├── category.js // 分类逻辑层
│ │ ├── goods.js // 商品逻辑层
│ │ ├── order.js // 订单逻辑层
│ │ └── user.js // 用户逻辑层
│ ├── utils // 工具函数文件夹
│ ├── app.js // 应用入口文件
│ ├── cluster.js // 应用启动脚本
│ └── views // 视图层文件夹
└── test // 测试文件夹
```
# 6. 数据库设计
## 6.1. 数据库表设计
数据库采用 MongoDB 来存储数据。共有五张表：Admin、Category、Goods、Order、User。

### 6.1.1. Admin 表
该表用来存储管理员信息。字段如下：

- _id (ObjectId)：自动生成的记录 ID。
- username (String): 管理员用户名。
- password (String): <PASSWORD>。
- roleId (ObjectId): 管理员角色 ID。
- status (Number): 管理员状态，1 表示启用，0 表示禁用。
- createTime (Date): 创建时间。
- updateTime (Date): 更新时间。

### 6.1.2. Category 表
该表用来存储商品分类信息。字段如下：

- _id (ObjectId)：自动生成的记录 ID。
- name (String): 分类名称。
- parentCategoryId (ObjectId): 父类别 ID。
- level (Number): 分类级别。
- sortNum (Number): 分类排序序号。
- status (Number): 分类状态，1 表示启用，0 表示禁用。
- createTime (Date): 创建时间。
- updateTime (Date): 更新时间。

### 6.1.3. Goods 表
该表用来存储商品信息。字段如下：

- _id (ObjectId)：自动生成的记录 ID。
- title (String): 商品名称。
- subtitle (String): 商品副标题。
- mainImage (String): 商品主图片 URL。
- detailImages ([String]): 商品详情图片 URLs。
- price (Number): 商品价格。
- stockNum (Number): 商品库存数量。
- description (String): 商品描述。
- originPlace (String): 产地。
- productAddress (String): 生产地点。
- weight (Number): 重量。
- size (String): 尺寸。
- color (String): 颜色。
- tags ([String]): 标签。
- categoryIds ([ObjectId]): 所属分类 ID 数组。
- brandName (String): 品牌名。
- status (Number): 商品状态，1 表示启用，0 表示禁用。
- sellCount (Number): 销售数量。
- browseCount (Number): 浏览数量。
- createTime (Date): 创建时间。
- updateTime (Date): 更新时间。

### 6.1.4. Order 表
该表用来存储订单信息。字段如下：

- _id (ObjectId)：自动生成的记录 ID。
- userId (ObjectId): 下单用户 ID。
- addressInfo (Object): 收获地址信息。
- goodsList ([Object]): 商品清单。
- totalPrice (Number): 总金额。
- expressFee (Number): 邮费。
- couponFee (Number): 优惠券金额。
- actualPayAmount (Number): 实际支付金额。
- payStatus (Number): 支付状态，1 表示已支付，0 表示未支付。
- deliveryStatus (Number): 发货状态，1 表示已发货，0 表示未发货。
- confirmStatus (Number): 确认收货状态，1 表示已确认收货，0 表示未确认收货。
- note (String): 订单留言。
- paymentType (Number): 支付方式，1 表示支付宝，2 表示微信，3 表示其他。
- logisticsCode (String): 物流单号。
- logisticsCompany (String): 物流公司。
- sendTime (Date): 发货时间。
- finishTime (Date): 完成时间。
- cancelReason (String): 取消原因。
- cancelTime (Date): 取消时间。
- closeReason (String): 关闭原因。
- closeTime (Date): 关闭时间。
- createTime (Date): 创建时间。
- updateTime (Date): 更新时间。

### 6.1.5. User 表
该表用来存储用户信息。字段如下：

- _id (ObjectId)：自动生成的记录 ID。
- username (String): 用户名。
- nickname (String): 用户昵称。
- avatarUrl (String): 用户头像 URL。
- mobilePhone (String): 手机号码。
- email (String): 邮箱。
- sex (Number): 性别，0 表示女，1 表示男，2 表示未知。
- birthday (Date): 生日。
- regIp (String): 注册 IP。
- loginIp (String): 最后登录 IP。
- lastLoginTime (Date): 最后登录时间。
- status (Number): 用户状态，1 表示正常，0 表示禁用。
- createTime (Date): 创建时间。
- updateTime (Date): 更新时间。
# 7. 后台管理系统实现
## 7.1. 登录模块
登录模块的实现需要实现以下几个步骤：

1. 从请求中获取登录信息。
2. 检查用户名是否存在，若不存在，返回错误信息。
3. 根据用户名和密码验证用户是否正确，若错误，返回错误信息。
4. 生成 JSON Web Token，并返回给客户端。
5. 将 JSON Web Token 保存至浏览器本地，作为后续访问的凭据。

```javascript
const { Admin } = require('../models')
const bcrypt = require('bcryptjs')
const jwt = require('jsonwebtoken')

// 登录
module.exports = async function(ctx) {
  const { body: reqBody } = ctx.request

  if (!reqBody ||!reqBody.username ||!reqBody.password) {
    return ctx.throw(400, '用户名或密码不能为空')
  }

  try {
    let admin = await Admin.findOne({
      username: reqBody.username
    })

    if (!admin) {
      return ctx.throw(401, '用户名或密码错误')
    }

    const isMatch = await bcrypt.compareSync(reqBody.password, admin.password)

    if (!isMatch) {
      return ctx.throw(401, '用户名或密码错误')
    }

    const token = jwt.sign({ id: admin._id }, process.env.JWT_SECRET, { expiresIn: '7 days' })

    ctx.cookies.set('token', token, { maxAge: 7 * 24 * 3600 * 1000, httpOnly: true })

    return ctx.body = { success: true, message: '登录成功' }
  } catch (err) {
    console.error(err)
    return ctx.throw(500, '登录失败')
  }
}
```
## 7.2. 获取管理员列表模块
获取管理员列表模块的实现需要实现以下几个步骤：

1. 从 JWT 中解析出当前用户的 ID。
2. 通过当前用户的 ID 查询管理员列表。
3. 返回管理员列表数据。

```javascript
const { Admin } = require('../models')
const jwt = require('jsonwebtoken')

// 获取管理员列表
module.exports = async function(ctx) {
  const token = ctx.headers['authorization']?.replace(/^Bearer\s/, '')

  if (!token) {
    return ctx.throw(401, '请先登录')
  }

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET)
    const admins = await Admin.find({})
    return ctx.body = { data: admins, success: true, message: '' }
  } catch (err) {
    console.error(err)
    return ctx.throw(401, '请先登录')
  }
}
```
## 7.3. 创建管理员模块
创建管理员模块的实现需要实现以下几个步骤：

1. 从 JWT 中解析出当前用户的 ID。
2. 检查用户名是否已经存在，若存在，返回错误信息。
3. 对密码进行加密，然后插入管理员信息。
4. 返回管理员信息。

```javascript
const { Admin } = require('../models')
const bcrypt = require('bcryptjs')
const jwt = require('jsonwebtoken')

// 创建管理员
module.exports = async function(ctx) {
  const { body: reqBody } = ctx.request

  if (!reqBody ||!reqBody.username ||!reqBody.password ||!reqBody.roleId) {
    return ctx.throw(400, '用户名、密码和角色不能为空')
  }

  try {
    let existsAdmin = await Admin.findOne({
      username: reqBody.username
    })

    if (existsAdmin) {
      return ctx.throw(400, '用户名已被占用')
    }

    const saltRounds = parseInt(process.env.SALT_ROUNDS) || 10

    const hashPassword = bcrypt.hashSync(reqBody.password, saltRounds)

    const newAdmin = new Admin({
      username: reqBody.username,
      password: <PASSWORD>,
      roleId: reqBody.roleId
    })

    await newAdmin.save()

    return ctx.body = { success: true, message: '', data: newAdmin }
  } catch (err) {
    console.error(err)
    return ctx.throw(500, '创建管理员失败')
  }
}
```
## 7.4. 更新管理员模块
更新管理员模块的实现需要实现以下几个步骤：

1. 从 JWT 中解析出当前用户的 ID。
2. 根据管理员 ID 检查是否存在此管理员，若不存在，返回错误信息。
3. 如果密码发生变化，则对密码进行加密。
4. 更新管理员信息。
5. 返回管理员信息。

```javascript
const { Admin } = require('../models')
const bcrypt = require('bcryptjs')
const jwt = require('jsonwebtoken')

// 更新管理员
module.exports = async function(ctx) {
  const { params, body: reqBody } = ctx.request

  if (!params ||!params.id) {
    return ctx.throw(400, '管理员 ID 不能为空')
  }

  try {
    const currentAdmin = jwt.decode(ctx.headers['authorization']?.replace(/^Bearer\s/, ''), process.env.JWT_SECRET)

    const oldAdmin = await Admin.findByIdAndUpdate(params.id, reqBody, { new: true }).select('-password -__v').exec()

    if (!oldAdmin) {
      return ctx.throw(404, '管理员不存在')
    }

    if (currentAdmin && String(currentAdmin.id)!== String(params.id)) {
      return ctx.throw(403, '没有权限')
    }

    if ('password' in reqBody) {
      const saltRounds = parseInt(process.env.SALT_ROUNDS) || 10

      const hashPassword = bcrypt.hashSync(reqBody.password, saltRounds)

      Object.assign(oldAdmin, {
        password: hashPassword
      })
    }

    await oldAdmin.save()

    return ctx.body = { success: true, message: '', data: oldAdmin }
  } catch (err) {
    console.error(err)
    return ctx.throw(500, '更新管理员失败')
  }
}
```