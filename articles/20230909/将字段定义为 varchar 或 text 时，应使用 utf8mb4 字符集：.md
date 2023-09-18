
作者：禅与计算机程序设计艺术                    

# 1.简介
  

utf8mb4 是 MySQL 和 MariaDB 默认支持的四字节字符集，它可以存储四个字节的 Unicode 编码字符，而不仅限于 ASCII 字符。

对于中文或日文等全角文字来说，一般会占用两个字节。而实际上，Unicode 标准并没有规定单个字符占用多少字节，这就导致了不同系统平台上对同一个 Unicode 字符的处理方式存在差异。为了解决这个问题，MySQL 和 MariaDB 在设计时允许用户指定字段长度时并不指定实际的字符数量，只需要根据指定的长度分配足够的空间即可。这样做虽然简单易用，但在实际应用中可能遇到以下几种情况：

1. 数据保存超过设计长度导致数据截断
2. 插入数据超出设计长度的字符由于数据库本身限制无法正确存储导致报错或者其他异常情况
3. 查询结果返回部分字符被截断的问题
4. 搜索引擎无法正确索引中文或日文等全角文字的问题

因此，当需要存储、检索或搜索全角文字时，建议将字段定义为 utf8mb4 字符集。

此外，MySQL/MariaDB 提供了一个新的存储引擎 InnoDB，它是 MySQL 数据库的默认存储引擎，InnoDB 支持多版本并发控制（MVCC）功能，通过 MVCC 可以实现高效的读写性能，并且保证事务一致性。但是，它也有一些限制，比如，不支持对 TEXT、BLOB 类型字段进行 FULLTEXT 索引。因此，如果要在 InnoDB 表中存储或检索全角文字，建议不要使用 TEXT 或 BLOB 类型字段，建议将这些字段定义为 VARCHAR(n) 类型。

# 2. 相关概念
## 1. 字符集
字符集 (Charset) 是用来表示和传输文本的符号集合及其编码方法的一套规则。UTF-8 是一种常用的字符集，它可以表示任意 Unicode 字符，包括汉字、英文、数字、标点符号、各种符号，支持所有的现代语言和方言。

## 2. 编码
编码 (Encoding) 是指通过某种字符集将计算机中的符号转换成二进制数组的方法。编码其实就是把符号转化成01串，或者反过来，把01串还原成符号。ASCII、GBK、UTF-8都是编码方式。

## 3. Unicode
Unicode 是国际标准化组织(ISO)制定的用于电脑记事本的字符编码方案，它是基于通用字符集(Universal Character Set，UCS)标准，能够完整且唯一地表示世界上所有字符，包括各主要语言、符号、数字和标点符号，具有高度的互操作性。

# 3. utf8mb4 使用场景分析
## 1. MySQL v5.5+ 默认 utf8mb4 字符集
从 MySQL v5.5 版本开始，utf8mb4 是 MySQL 的默认字符集，可以用于存储各种语言、符号、特殊字符。

例如，创建一个数据库并插入一条记录：
```sql
CREATE DATABASE mytest;
USE mytest;
CREATE TABLE mytable (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL DEFAULT '' COMMENT '姓名',
  email VARCHAR(50) CHARACTER SET utf8mb4 COLLATE utf8mb4_bin NOT NULL DEFAULT '' COMMENT '邮箱'
);
INSERT INTO mytable (name, email) VALUES ('张三', '<EMAIL>');
```
上述语句创建了一个名为 mytest 的数据库，其中有一个表 mytable ，其中 id 为自增主键、name 为 VARCHAR(20)，email 为 VARCHAR(50)。这两个字段都使用了 utf8mb4 字符集。

注意，这里的 CHARACTER SET 设置的是 utf8mb4，COLLATE 设置的是 utf8mb4_bin。这两者分别表示字符集和比较规则。

utf8mb4 的最大好处就是可以存储四字节的中文或日文字符，所以推荐将字符串类型字段都设置为 utf8mb4 字符集。

## 2. MySQL v5.5 以前版本兼容性
早期版本 MySQL 不支持 utf8mb4，需要自己编译安装或者升级到最新版，才能正常使用。

比如，CentOS7 中的 MySQL 安装命令如下：

```bash
sudo yum install mysql55-community-release-el7
sudo yum update -y
sudo yum install mysql-server mysql-client
```

这个过程可能会比较长，取决于网络速度和服务器硬件配置。完成之后就可以像上面一样创建、插入数据，并且使用 utf8mb4 字符集。

## 3. MariaDB 默认 utf8mb4 字符集
MariaDB 是一个开源社区版本的 MySQL 数据库，它支持绝大部分 MySQL 命令，而且与 MySQL API 兼容。MariaDB 默认也是使用 utf8mb4 字符集。

## 4. InnoDB 支持 utf8mb4
InnoDB 存储引擎支持对 utf8mb4 字符集的字段索引，并且支持对 TEXT、BLOB 类型字段进行 FULLTEXT 索引。

因此，即使你的 MySQL/MariaDB 数据库版本较旧，也可以安全地使用 utf8mb4 字符集。