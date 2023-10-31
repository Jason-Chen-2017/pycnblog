
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


触发器（Trigger）是在数据库中设置的一种特殊类型的存储过程，它会在指定数据库表中关于增删改操作时自动执行。触发器经常用于实现对数据行的自动验证、强制性约束、以及多表之间的一致性维护。另外，触发器还可以用于记录用户对数据的操作，进行审计，并实现业务规则引擎等功能。本教程将介绍MySQL的触发器及其常用语法。
# 2.核心概念与联系
## 2.1 触发器基本概念
触发器是一个自定义的事件响应器，当满足一定条件的时候，MySQL会自动执行定义好的SQL语句或存储过程。触发器分为三类：
- INSERT触发器：当一条新纪录插入到数据库表时激活触发器。
- UPDATE触发器：当一条记录被更新时激活触发器。
- DELETE触发器：当一条记录从数据库表中被删除时激活触发器。
每个触发器都有相应的触发条件，当符合触发条件时，触发器就会执行相应的操作。触发器一般是由管理员创建，并赋予特定的触发动作，当触发了该触发条件，就执行相应的SQL语句或者存储过程。

## 2.2 触发器相关语法
下面介绍MySQL的触发器相关语法。
### 2.2.1 创建触发器语法
```sql
CREATE TRIGGER trigger_name
  trigger_event ON table_name
  FOR EACH ROW trigger_timing
  trigger_condition 
  BEGIN
    // trigger body here
  END;
```
其中，trigger_name表示触发器名称；trigger_event表示触发事件，比如INSERT、UPDATE、DELETE等；table_name表示触发器关联的表名；trigger_timing表示触发的时间类型，比如BEFORE、AFTER、INSTEAD OF等；trigger_condition表示触发器生效的条件表达式，若为空，则默认永远有效；BEGIN... END表示触发器主体，即执行触发器时执行的语句。
### 2.2.2 删除触发器语法
```sql
DROP TRIGGER [IF EXISTS] trigger_name;
```
其中，trigger_name表示要删除的触发器名称；IF EXISTS表示如果触发器不存在也不报错。
### 2.2.3 查看触发器语法
```sql
SHOW TRIGGERS [FROM table_name];
```
其中，FROM table_name表示只显示某张表的触发器信息。
### 2.2.4 修改触发器语法
```sql
ALTER TABLE table_name MODIFY COLUMN column_name datatype;
```
其中，table_name表示要修改的表名；column_name表示需要修改的数据列名；datatype表示修改后的数据类型。

## 2.3 使用触发器的注意事项
使用触发器之前，需要考虑以下几个方面：
- 安全性：创建触发器的权限通常较高，需要慎重选择是否给予其他人权限。
- 执行时间：触发器运行频率决定了数据库的性能，过于频繁的运行可能会影响数据库的正常运行。
- 更新效率：对于大量的数据操作，触发器可能会造成严重的性能问题。

## 2.4 触发器实战案例
下面以一个具体例子——学生注册的场景为例，介绍如何利用触发器实现学生信息的校验和插入。假设有一个老师要在学校开设课程，需要先填写一些课程相关的信息。假设此前已经设计好了表结构和表单页面，老师可直接向表单提交数据。但是，为了保证课程信息准确无误，老师希望每填写完一项数据，都会要求输入正确的信息。因此，老师可以利用触发器对表单中的数据进行校验。下面是演示用的学生注册表和触发器定义：
```mysql
-- 学生信息表
CREATE TABLE students (
  id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(50) NOT NULL DEFAULT '',
  age INT UNSIGNED NOT NULL CHECK (age >= 1 AND age <= 120),
  gender ENUM('male', 'female') NOT NULL DEFAULT'male'
);

-- 插入学生信息触发器
DELIMITER $$
CREATE TRIGGER insert_students_trigger BEFORE INSERT ON students FOR EACH ROW
BEGIN
  -- 检验姓名长度
  IF CHAR_LENGTH(NEW.name) > 50 THEN
    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Name is too long';
  END IF;
  
  -- 检验年龄范围
  IF NEW.age < 1 OR NEW.age > 120 THEN
    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Invalid age';
  END IF;
  
  -- 检验性别选项
  IF NEW.gender!='male' AND NEW.gender!= 'female' THEN
    SIGNAL SQLSTATE '45000' SET MESSAGE_TEXT = 'Invalid gender';
  END IF;
END$$
DELIMITER ;
```
上面定义了一个`students`表格和一个`insert_students_trigger`触发器。触发器在新纪录插入到`students`表时会执行，首先判断名字的字符长度是否超过50个字符，如果超过则报出错误码45000，并且给出提示信息。然后判断年龄的取值范围是否在1~120之间，如果小于1或者大于120则也报出错误码45000。最后检查性别选项是否为男或者女。

下面通过插入一些测试数据来验证触发器是否正常工作：
```mysql
-- 插入一条合法的学生记录
INSERT INTO students (id, name, age, gender) VALUES (NULL, 'Alice', 17, 'female'); 

-- 插入一条非法的学生记录，因为名字的长度大于50
INSERT INTO students (id, name, age, gender) VALUES (NULL, 'Bobbie Bubble', 19,'male');

-- 插入一条非法的学生记录，因为年龄超出范围
INSERT INTO students (id, name, age, gender) VALUES (NULL, 'Cindy Xu', 200, 'female');

-- 插入一条非法的学生记录，因为性别选项错误
INSERT INTO students (id, name, age, gender) VALUES (NULL, 'David Wang', 22, 'unknown');
```
可以看到，只有第一条插入语句成功，而第二条到第四条均失败，因为触发器给出了相应的错误信息。