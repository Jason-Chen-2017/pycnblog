
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


MySQL是一个关系型数据库管理系统，它的诞生自90年代末期，它是开放源代码软件，采用GPL（General Public License）授权协议。MySQL是最流行的关系型数据库服务器端产品之一。

作为一款开源数据库，MySQL支持各种平台、各种编程语言及开发框架，而且功能非常强大，是一个十分值得信赖的数据库产品。由于其通用性和灵活性，使得MySQL在一些更复杂的场景下得到了广泛应用。因此，本文将介绍MySQL中最重要的数据类型以及如何利用它们建立数据库表。

# 2.核心概念与联系

## 数据类型

### 整数类型

- tinyint(M)：无符号的TINYINT（1字节）整数。取值范围是从0到255。如果超出这个范围，就会产生溢出错误。当把TINYINT设置为UNSIGNED时，允许负值，但是会被自动转换成一个无符号整数。

- smallint(M)：短整型SMALLINT（2字节）整数。取值范围是从-32768到32767。

- mediumint(M)：中等大小整型MEDIUMINT（3字节）整数。取值范围是从-8388608到8388607。

- int(M)或integer(M)：整型INTEGER（4字节）整数。取值范围是从-2147483648到2147483647。

- bigint(M)：大整型BIGINT（8字节）整数。取值范围是从-9223372036854775808到9223372036854775807。

这些整数类型都是有符号的，但是有的情况下需要使用无符号的整数。可以设置整数类型的属性UNSIGNED属性，然后插入的数据值就只能是正数了。例如：

```mysql
CREATE TABLE test (
    id INT UNSIGNED NOT NULL AUTO_INCREMENT PRIMARY KEY,
    num TINYINT UNSIGNED DEFAULT 0
);

INSERT INTO test (num) VALUES (-1); -- 报错，不能插入负数！
```

虽然可以使用无符号整数，但是还是建议使用有符号整数，因为无符号整数容易出现问题。例如，若有一个负值，在有符号整数中，它的二进制表示为全1；而在无符号整数中，它的二进制表示则可能是某种反向补码形式。

```mysql
SELECT -1 AS value; # 表示十六进制数 FFFFFFFF

SELECT CONVERT(-1, UNSIGNED INTEGER) AS unsigned_value; # 表示十进制数 4294967295
```

### 浮点数类型

- float(M,D)：浮点数类型FLOAT。M代表总长度，D代表小数部分的位数。取值范围是从-3.4E+38到+3.4E+38。

- double(M,D)：双精度浮点数类型DOUBLE。取值范围同上。

- decimal(M,D)：高精度数字DECIMAL。它支持用户自定义精度，并且不会丢失精度。

```mysql
CREATE TABLE test (
    price FLOAT(10,2),
    quantity DECIMAL(5,2)
);

INSERT INTO test (price,quantity) VALUES ('3.14', '12.34');
```

浮点数类型和双精度浮点数类型都支持科学计数法。例如：

```mysql
SELECT PI() * 10^3 AS pi; # 返回结果：3140
```

### 日期和时间类型

- date：日期类型DATE。存储的值是一个日期值。如"2021-07-01"。

- time：时间类型TIME。存储的是时间值，包括时、分、秒、微秒。如"10:10:10.000123"。

- datetime：日期时间类型DATETIME。同时保存日期和时间信息，格式为"YYYY-MM-DD HH:mm:ss[.fraction]"。如"2021-07-01 10:10:10.000123"。

- timestamp：时间戳类型TIMESTAMP。保存的是从1970-01-01 00:00:00 UTC 日历所过去的秒数。

```mysql
CREATE TABLE test (
    dt DATE,
    tm TIME,
    dtm DATETIME,
    ts TIMESTAMP
);

INSERT INTO test (dt, tm, dtm, ts) VALUES 
    ('2021-07-01','10:10:10','2021-07-01 10:10:10', NOW());
```

TIMESTAMP类型用于记录事件发生的时间戳，可以保证数据的一致性和完整性。一般来说，TIMESTAMP列应该设为NOT NULL、DEFAULT CURRENT_TIMESTAMP或者ON UPDATE CURRENT_TIMESTAMP。

```mysql
ALTER TABLE test ADD COLUMN created TIMESTAMP DEFAULT CURRENT_TIMESTAMP; 

UPDATE test SET updated = NOW(); 
```

这样，就可以方便地记录创建和更新的时间戳。

### 字符串类型

- char(M):定长字符串CHAR。M代表字符的最大数量。最大可存储65535个字符。

- varchar(M):变长字符串VARCHAR。M代表最大字符数量。最大可存储65535个字符。

- text：文本类型TEXT。可存储大量文本数据。

```mysql
CREATE TABLE test (
    name CHAR(5),
    message VARCHAR(255),
    description TEXT
);

INSERT INTO test (name,message,description) VALUES 
    ('alice','hello world','this is a test message.'),
    ('bob','你好，世界','这是一条测试消息。');
```

## 字段属性

- NOT NULL：不允许该字段为空。

- NULL：允许该字段为空。

- UNIQUE：唯一约束，即该字段的所有值必须不同。

- PRIMARY KEY：主键约束，唯一标识每一行数据。

- AUTO_INCREMENT：只适用于整数类型字段，用来设置自增值，每次插入新数据时，该字段的值都会增加。

- DEFAULT：设置默认值。

- COMMENT：对该字段添加注释。

```mysql
CREATE TABLE test (
    id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(50) NOT NULL,
    content TEXT,
    views INT UNSIGNED NOT NULL DEFAULT 0,
    likes INT UNSIGNED NOT NULL DEFAULT 0,
    status ENUM('draft','published') NOT NULL DEFAULT 'draft' COMMENT '草稿或发布状态'
);
```