
作者：禅与计算机程序设计艺术                    

# 1.简介
  

MongoDB 是用 C++ 开发的一个开源 NoSQL 数据库。作为一个基于分布式文件存储的数据库，它具备高性能、高可用性、自动分片、易扩展等特点。相比于传统关系型数据库 MySQL 和 Oracle ，它的优势在于更接近自然语言的数据模型及查询语言。此外，由于 MongoDB 使用了 JavaScript 编程语言编写，使得数据处理逻辑与业务逻辑可以实现零耦合。因此，无论是开发应用程序还是构建云平台，都适合采用 MongoDB。本文是对 MongoDB 数据库的基础知识做一个整体性的梳理，其目的是帮助读者了解并掌握 MongoDB 的各种特性、用法、应用场景和优势。
# 2.核心概念
## 2.1 文档（Document）
MongoDB 中数据的基本单位称之为“文档”，它类似于 JSON 对象。一条记录就是一个文档。文档由字段和值组成，其中值可以是任何类型，包括对象或者数组。
## 2.2 集合（Collection）
集合是一个逻辑概念，相当于关系型数据库中的表格。集合中可以存放不同结构和形式的数据。每个集合都会有一个唯一标识符 _id 。集合中的数据是松散结构的，文档不需要有预定义的模式（schema）。也就是说，你可以自由地向集合中插入任何格式的文档。
## 2.3 索引（Index）
索引是一个特殊的数据结构，它以键-值对的方式存储文档中的字段。在创建索引时，MongoDB 会将相应字段上的值转换为一种索引，这样就可以根据索引快速查找数据。对于某些字段，MongoDB 默认会建立索引；但对于某些复杂查询或需要排序的字段，则需要手动建立索引。索引可以提升查询效率，同时降低写入性能。
## 2.4 数据库（Database）
数据库是 Mongo 中的逻辑隔离单元，一个 MongoDB 可以拥有多个数据库。数据库在物理上被存储在文件夹中。不同的数据库可以有相同的集合名，但集合名必须全局唯一。数据库中可以创建用户、角色和权限，并控制访问权限。
## 2.5 事务（Transactions）
MongoDB 提供事务功能，用来支持多文档 ACID 操作。事务是一系列操作，要么全部成功，要么全部失败。事务可以确保一组操作全要么执行，要么完全不执行。
## 2.6 分片集群（Sharded Cluster）
为了解决 MongoDB 数据量过大的问题，可以把数据分布到多个服务器上，这种方式就是分片集群。Mongo 支持基于文档、范围、hash 等方式进行分片，将数据分布到不同的服务器上。分片集群中可以增加节点，提升性能，也可以减少负载。
## 2.7 复制集（Replica Set）
复制集是 MongoDB 的高可用性机制。在复制集中，各个成员服务器彼此互为主从。当主节点出现故障时，系统可以自动选择新的主节点，继续提供服务。复制集还可以配置延迟复制，从而可以提高读取吞吐量。
## 2.8 连接字符串（Connection String）
连接字符串提供了标准的方法来指定数据库服务器的位置，并可选地指定身份验证信息。用于连接到 MongoDB 的 URI 形如 mongodb://[username:password@]host1[:port1][,...hostN[:portN]][/[database][?options]]，如下所示：
```javascript
mongodb://localhost:27017/testdb
```

其中：

- host: 指定服务器主机名。
- port: 指定服务器端口号。如果省略该参数，默认值为 27017。
- database: 指定数据库名称。
- username: 可选用户名。
- password: 可选密码。
- options: 其他选项，比如 authSource 或 replicaSet。

除此之外，还可以通过配置文件 ~/.mongorc.js 来设置默认连接字符串：

```javascript
connect("mongodb://localhost:27017/myDB");
```

# 3.数据类型
## 3.1 数值类型
- Double (double) - 存储浮点值，可使用数学上的“.”表示法。
- Integer (int) - 存储整数值，最大可表示 2147483647。
- Long (long) - 存储长整数值，最大可表示为 9223372036854775807。

例如：

```javascript
{
  "number": NumberDouble(3.14), // double
  "integer": NumberInt(-2147483648), // int
  "long": NumberLong("-9223372036854775808") // long
}
```

注：NumberInt()、NumberLong() 函数是在 MongoDB v3.2+ 版本引入的。

## 3.2 字符串类型
- String (string) - 最常用的字符串类型，通常是 UTF-8 编码的文本。
- Symbol (symbol) - 一个独一无二且不可改变的值。
- ObjectID (object id) - 用于存储文档的唯一 ID。

例如：

```javascript
{
  "_id": ObjectId("5f0c1b1bc7f0e0a0cccafeaa"), // object id
  "name": "Alice", // string
  "symbol": Symbol("foo bar") // symbol
}
```

## 3.3 布尔类型
- Boolean (bool) - true/false 两种取值。

例如：

```javascript
{
  "isStudent": false // bool
}
```

## 3.4 日期类型
- Date (date) - 表示日期和时间。

例如：

```javascript
{
  "birthDate": ISODate("1995-12-17T03:24:00Z") // date
}
```

## 3.5 Binary 数据类型
- BinData (binary data) - 用于存储二进制数据，如图片、视频、压缩包等。

例如：

```javascript
{
  "photo": BinData(0, "VGhpcyBpcyBhIHRlc3Q=") // binary data
}
```

# 4.查询语法
## 4.1 基本查询
查询语言提供了丰富的查询条件，允许客户端在集合中查找满足特定条件的文档。以下是一些示例：

- 查找 name 为 “Alice” 的所有文档：

  ```javascript
  db.collectionName.find({ name: "Alice" })
  ```

- 查找 age 大于等于 20 并且 salary 小于 50000 的所有文档：

  ```javascript
  db.collectionName.find({ age: { $gte: 20 }, salary: { $lt: 50000 } })
  ```

- 查找 name 以 "A" 开头的所有文档：

  ```javascript
  db.collectionName.find({ name: /^A/ })
  ```

- 查找 name 以 "i" 结尾的所有文档：

  ```javascript
  db.collectionName.find({ name: /i$/ })
  ```

- 查找 name 不为空的所有文档：

  ```javascript
  db.collectionName.find({ name: { $ne: null } })
  ```

- 查找 salary 在 30000~50000 之间的所有文档：

  ```javascript
  db.collectionName.find({ salary: { $gt: 30000, $lte: 50000 } })
  ```

- 查找 status 为 "active" 或者 "inactive" 的所有文档：

  ```javascript
  db.collectionName.find({ status: { $in: ["active", "inactive"] } })
  ```

- 查找 status 以 "act" 开头的所有文档：

  ```javascript
  db.collectionName.find({ status: /^act/ })
  ```

## 4.2 条件表达式
条件表达式允许对字段进行比较、计算、逻辑运算和正则匹配等操作。以下是一些示例：

- 查询 age 字段大于 20：

  ```javascript
  { age: { $gt: 20 } }
  ```

- 查询 salary 字段大于等于 30000 并且小于等于 50000：

  ```javascript
  { salary: { $gte: 30000, $lte: 50000 } }
  ```

- 查询 score 字段不等于 80：

  ```javascript
  { score: { $ne: 80 } }
  ```

- 查询 name 字段包含字符串 "hello"：

  ```javascript
  { name: /hello/ }
  ```

- 查询 name 字段以 "Al" 开头：

  ```javascript
  { name: /^Al/ }
  ```

- 查询 name 字段以 "ice" 结尾：

  ```javascript
  { name: /ice$/ }
  ```

- 查询 salary 字段在 [30000, 50000] 之间：

  ```javascript
  { salary: { $gt: 30000, $lt: 50000 } }
  ```

- 查询 score 字段在 [70, 90] 之间：

  ```javascript
  { score: { $gte: 70, $lte: 90 } }
  ```

- 查询 yearOfBirth 字段在当前年龄范围内：

  ```javascript
  {
    yearOfBirth: {
      $mod: [100, current_year % 100] // 得到当前年份的后两位
    }
  }
  ```

## 4.3 更新操作
更新操作用于修改已存在的文档。以下是一些示例：

- 将 age 字段增加 1：

  ```javascript
  db.collectionName.updateOne({}, { $inc: { age: 1 } })
  ```

- 将 salary 字段乘以 1.1：

  ```javascript
  db.collectionName.updateMany({}, { $mul: { salary: 1.1 } })
  ```

- 添加 address 字段并赋值为 “123 Main St”：

  ```javascript
  db.collectionName.updateOne({}, { $set: { address: "123 Main St" } })
  ```

- 删除 status 字段：

  ```javascript
  db.collectionName.updateOne({}, { $unset: { status: "" } })
  ```

- 将 status 修改为 "inactive"：

  ```javascript
  db.collectionName.updateOne({}, { $set: { status: "inactive" } })
  ```

- 设置 name 字段为 "John Doe"：

  ```javascript
  db.collectionName.updateOne({}, { $rename: { oldName: newName } })
  ```

- 把 age 小于 20 的文档删除：

  ```javascript
  db.collectionName.deleteMany({ age: { $lt: 20 } })
  ```

## 4.4 聚合框架
聚合框架提供了对集合中文档进行统计、分组、排序、分页等操作，这些操作返回结果集。以下是一些示例：

- 统计 salary 字段的平均值：

  ```javascript
  db.collectionName.aggregate([
    { $group: { _id: null, avgSalary: { $avg: "$salary" } } }
  ])
  ```

- 对 name 字段按字母顺序进行分组，然后求每组文档的数量：

  ```javascript
  db.collectionName.aggregate([
    { $group: { _id: "$name", count: { $sum: 1 } } },
    { $sort: { _id: 1 } }
  ])
  ```

- 求出年龄最小的文档：

  ```javascript
  db.collectionName.findOne({},{age:-1})
  ```

- 获取前10条数据：

  ```javascript
  db.collectionName.find().limit(10).toArray()
  ```