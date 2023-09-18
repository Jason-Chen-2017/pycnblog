
作者：禅与计算机程序设计艺术                    

# 1.简介
  


作为一名IT从业者，了解数据库安全是一个必备技能。一个健康、安全的数据库系统应该具备多种防护机制，包括对用户权限管理、数据完整性检查、网络攻击检测、日志审计等方面的保障措施。其中最基础也是最重要的一项就是设定合理的密码复杂性规则。

在MySQL数据库中，如果密码规则设置不当，可能会导致数据库容易受到攻击。常见的密码复杂性规则有以下几点：

1.最小长度限制：密码长度不能少于指定的长度。例如，要求密码至少包含8个字符。

2.必须含有特殊符号或数字：密码必须包含特殊符号（如!@#$%^&*()_+-=[]{}|\\:\";'<>?,./~）或者数字。

3.密码不能太简单：有的网站推荐密码不能只有字母和数字。强烈建议密码要包含大小写字母、数字和特殊符号。

4.应避免重复使用相同密码：使用过于相似的密码会导致账户被盗用。

5.密码保存应加密处理：数据库存储的密码本身就已经加密，无需担心明文密码泄露。

6.更新密码频率控制：为了保证账户安全，对于经常使用的账户，应适当减少修改密码的频率，提高账户安全性。

今天，我们将详细探讨MySQL数据库密码复杂性规则及其实践方法，希望能够帮助大家更好地保障数据库的安全。

# 2.核心概念和术语说明
## 2.1 密码复杂性规则

在数据库系统中，密码复杂性规则指的是设定的密码必须符合一定的要求才能允许登录到数据库。密码复杂性规则通常包括如下四个方面：

1. 密码长度限制：密码长度不能少于指定的长度。

2. 必须含有特殊符号或数字：密码必须包含特殊符号（如!@#$%^&*()_+-=[]{}|\\:\";'<>?,./~）或者数字。

3. 密码不能太简单：密码长度不能太短且不能仅由字母和数字组成。

4. 不允许密码相同：禁止两个账户使用相同的密码。

一般情况下，密码复杂性规则可分为弱密码规则和强密码规则。弱密码规则要求密码长度至少为8位，不能包含用户名、生日、地址、邮编、电话等信息；强密码规则则要求密码必须包含至少两种类型的字符，包括数字、小写字母和大写字母，而且还可以包含特殊字符。

虽然复杂性规则确实能够起到一定程度的保护作用，但仍然存在着暴力破解、字典攻击等常见安全威胁。因此，在实际应用中，还需要配合一些额外的安全防范措施才能确保数据库系统的安全。

## 2.2 哈希函数和加盐

在数据库密码存储过程中，为了防止密码被破解而进行的一种加密方法叫做“加盐哈希”。这里所谓的加盐就是把原始密码加上一个随机字符串，然后再通过某种加密算法计算出一个摘要字符串作为最终的密文。这样的话，即使有人获取了原始密码，也无法直接计算出对应的摘要。

## 2.3 用户认证方式

目前比较流行的数据库用户认证方式有两种：明文和基于哈希值的基于口令。

- 明文：这种认证方式下，用户输入的密码会在服务器端原样保存，因此风险较高。

- 基于哈希值的基于口令：这种认证方式下，用户输入的密码会先用某种哈希算法转换成一个固定长度的值，然后将这个值和其他一些参数一起存放在数据库中，之后每次用户登录时都会发送带有密码的哈希值，服务器会根据当前用户记录的密码和用户提供的哈希值进行比对。由于哈希值难以通过穷举法推导出原始密码，因此这种认证方式非常安全。

# 3. 具体操作步骤
## 3.1 创建数据库并设置用户权限

首先创建一个空白数据库，比如mydatabase，并设置一个具有创建表、插入、删除、更新等权限的用户，假设用户名为mysqluser，密码为<PASSWORD>！
```sql
CREATE DATABASE mydatabase;

USE mydatabase;

CREATE USER'mysqluser'@'localhost' IDENTIFIED BY '123456!';

GRANT SELECT,INSERT,UPDATE,DELETE ON mydatabase.* TO'mysqluser'@'localhost';
```

## 3.2 配置密码复杂性规则

接着，配置MySQL的密码复杂性规则。具体的方法是打开MySQL配置文件my.ini文件，找到password-policy-table-check-username这一行，去掉前面的注释符号，改为如下所示：
```ini
[mysqld]
...
password-policy-table-check-username=off
```

然后重启MySQL服务使设置生效。
```bash
service mysql restart
```

同时，也可以在mydatabase数据库中新建一个名为mysql.user_authentication表，用来存放密码复杂性规则。
```sql
CREATE TABLE `mysql`.`user_authentication` (
  `host` VARCHAR(64) NOT NULL,
  `user` VARCHAR(16) NOT NULL,
  `auth_string` VARCHAR(1024),
  PRIMARY KEY (`user`,`host`),
  UNIQUE INDEX `idx_user_string` (`user`, `auth_string`(767)),
  INDEX `idx_auth_string` (`auth_string`(767))
);
```

默认情况下，该表的结构如下：

- host：允许访问数据库的主机名称，或IP地址。
- user：用户名。
- auth_string：该字段包含用户名、密码哈希值、Salt及其他信息。

## 3.3 添加密码复杂性规则

设置完密码复杂性规则后，就可以添加新的用户或修改已有用户的密码。

### 3.3.1 新增用户

```sql
ALTER USER'mysqluser'@'localhost' IDENTIFIED WITH mysql_native_password BY '123456!';
```

这里我们用的是mysql_native_password认证插件，这个插件支持密码复杂性规则，并且可以使用带密码的哈希值进行验证。

### 3.3.2 修改密码

```sql
SET PASSWORD FOR'mysqluser'@'localhost' = OLD_PASSWORD('123456!')
```

这里我们用了OLD_PASSWORD函数，它可以用来将明文密码转换成密文密码，然后根据密码复杂性规则验证密码是否符合要求。

## 3.4 测试密码复杂性规则

最后，测试一下我们的密码复杂性规则是否有效。
```sql
DROP USER IF EXISTS 'testuser'@'localhost';

CREATE USER 'testuser'@'localhost' IDENTIFIED WITH mysql_native_password BY 'abcde'; --密码太短

CREATE USER 'testuser'@'localhost' IDENTIFIED WITH mysql_native_password BY 'Abcde'; --密码只包含字母

CREATE USER 'testuser'@'localhost' IDENTIFIED WITH mysql_native_password BY 'ABCDE'; --密码只包含大写字母

CREATE USER 'testuser'@'localhost' IDENTIFIED WITH mysql_native_password BY 'abcde123'; --密码没有数字

CREATE USER 'testuser'@'localhost' IDENTIFIED WITH mysql_native_password BY 'aBcDeFGHIjklmnoPqRSTuVwXyZ'; --密码没有特殊符号

CREATE USER 'testuser'@'localhost' IDENTIFIED WITH mysql_native_password BY 'ABCDEREF'; --密码太常见

--测试用户名重复
CREATE USER'mysqluser'@'localhost' IDENTIFIED WITH mysql_native_password BY '<PASSWORD>'; 

FLUSH PRIVILEGES;
```

注意，为了测试方便，这里我并没有删除刚才创建的用户。运行结束后，可以看到有些用户无法创建成功，因为密码不符合复杂性规则。当然，密码复杂性规则只是一项简单的防范措施，真正要保障数据库的安全还得结合其他安全措施一起使用。