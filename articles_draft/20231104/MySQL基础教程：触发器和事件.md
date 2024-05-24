
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


本文将简要介绍触发器（Trigger）和事件（Event）。并通过相关实例介绍如何创建触发器、事件及其功能。
# Trigger
触发器（Trigger）是一个在特定数据库活动时执行的SQL语句。当某些条件被满足时，例如用户插入或更新表中的数据时，会自动执行指定的触发器。触发器可以分为三类：
- DML Trigger：当对数据表进行INSERT、UPDATE、DELETE操作时，就会执行DML Trigger。
- DBCC Trigger：DBCC TRACER命令用来跟踪指定对象的操作，它可以保存到日志文件中。每个数据库都有一个DBCC TRACER命令的默认触发器，这个触发器会记录所有数据库访问的情况。
- Logon/Logoff Trigger：当一个用户登录或者退出数据库时，就会执行Logon/Logoff Trigger。

触发器与存储过程相似，都是可以帮助用户实现复杂逻辑的一种工具。但是两者之间也存在一些区别：
- 执行时机不同：触发器是在特定事件发生时才执行的，而存储过程则是在调用时立即执行。
- 作用对象不同：触发器可以作用于单个表，也可以作用于整个数据库，而存储过程只能作用于单个过程。
- 可用性不同：存储过程可以在任何时候调用，而触发器只有在特定的条件下才能运行。如果需要重启数据库或断开连接，则不会触发触发器。

总体来说，触发器更适合于频繁使用的操作，比如记录用户操作日志；而存储过程则适合于一次性的或复杂的操作，比如计算值、生成报告等。由于触发器无法修改表结构，所以一般情况下建议使用存储过程。
# Event
事件（Event）也叫做通知，是指数据库内部发生的一系列操作，这些操作通常与数据库的状态有关。例如，当数据库出现严重错误时，就可能发送邮件通知管理员。

目前MySQL支持以下四种类型的事件：
- Server启动/停止/崩溃
- 用户登录/退出
- 数据操作（如插入、删除、更新等）
- 慢查询

除了可以使用触发器来响应事件外，还可以通过其他方式来处理事件。例如，可以编写脚本监控服务器日志文件，每当发生特定事件时，就触发相应的动作。

# 创建触发器
## 创建DML Trigger
下面以用户登录日志作为示例，演示如何创建一个DML Trigger。假设用户登录成功后，需要记录一条日志。首先，创建一个新的表user_login_log：

```sql
CREATE TABLE user_login_log (
  id INT PRIMARY KEY AUTO_INCREMENT,
  username VARCHAR(20),
  login_time DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

然后，定义一个名为login_trigger的触发器：

```sql
DELIMITER $$

CREATE TRIGGER login_trigger AFTER INSERT ON user_login_table
FOR EACH ROW
BEGIN
    INSERT INTO user_login_log (username) VALUES (NEW.username);
END$$

DELIMITER ;
```

以上语句定义了一个login_trigger的触发器，该触发器在user_login_table上发生INSERT操作时（AFTER表示在事件完成之后触发），执行本身的SQL语句（INSERT INTO user_login_log (username) VALUES (NEW.username);）。

需要注意的是，TRIGGER语句需要放在CREATE TABLE之前，并且DELIMITER需设置为$$。另外，NEW代表当前正在插入的行的数据，OLD则代表刚刚删除的那一行的数据。因此，上面语句表示在新插入一行数据时，自动添加一行登录日志。

## 创建DBCC Trigger
DBCC TRACER命令用来跟踪指定对象的操作，它可以保存到日志文件中。每个数据库都有一个DBCC TRACER命令的默认触发器，这个触发器会记录所有数据库访问的情况。

这里，我们还是用登录日志举例，创建一个名为dbcc_trigger的DBCC Trigger。

```sql
DELIMITER $$

CREATE TRIGGER dbcc_trigger BEFORE INSERT ON user_login_table
FOR EACH STATEMENT
BEGIN
   DECLARE v_query varchar(100);
   SET v_query = 'DBCC TRACEON(3604)';
   PREPARE stmt FROM @v_query;
   EXECUTE stmt;
   DEALLOCATE PREPARE stmt;
END$$

DELIMITER ;
```

以上语句定义了一个dbcc_trigger的DBCC Trigger，该触发器在user_login_table上发生INSERT操作前（BEFORE表示在事件发生之前触发），执行SQL语句（DECLARE v_query varchar(100); SET v_query = 'DBCC TRACEON(3604)'; PREPARE stmt FROM @v_query; EXECUTE stmt; DEALLOCATE PREPARE stmt;）。

需要注意的是，DBCC TRACER命令需要使用PREPARE和EXECUTE命令进行预编译，而不是直接执行。同时，需要声明变量@v_query并将字符串直接赋值给它。另外，DBCC TRACER命令不能加注释。

## 创建Logon/Logoff Trigger
最后，我们再看一下如何创建一个Logon/Logoff Trigger。假设我们想在用户登录或登出时，记录一条日志。如下所示：

```sql
DELIMITER //

CREATE TRIGGER logon_trigger 
BEFORE INSERT ON user_login_table
FOR EACH ROW
BEGIN
    IF NEW.status='LOGIN' THEN
        INSERT INTO user_login_log (username) VALUES (NEW.username);
    ELSEIF NEW.status='LOGOUT' THEN
        DELETE FROM user_login_log WHERE username=NEW.username AND login_time=(SELECT MAX(login_time) FROM user_login_log WHERE username=NEW.username);
    END IF;
END//

DELIMITER ;
```

以上语句定义了一个logon_trigger的触发器，该触发器在user_login_table上发生INSERT操作时（BEFORE表示在事件发生之前触发），执行IF...ELSE判断语句。

如果新插入的记录的status字段等于LOGIN，则执行INSERT INTO语句插入一条日志；如果status字段等于LOGOUT，则执行DELETE FROM语句删除对应用户名下最近一次登录的日志。

同样地，需要注意DELIMITER命令。