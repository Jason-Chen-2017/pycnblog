
作者：禅与计算机程序设计艺术                    

# 1.简介
         
1995年，瑞典计算机科学家 <NAME> 提出了著名的论断：“互联网的本质就是个数据库。”随后，人们相继提出 MySQL 和 MongoDB 是最佳选择。今天，MySQL 和 MongoDB 都已经成为最流行的关系型数据库。它们都是开源免费的产品，易于部署和使用。因此，掌握这些数据库的用法对于应对日益复杂的 Web 服务应用来说至关重要。这两款数据库之间的差异在于其存储方式和数据处理的方式。MySQL 是一个关系型数据库，它存储结构化的数据；而 MongoDB 则是一个非关系型数据库，它可以存储结构化和半结构化的数据。无论您需要什么样的数据类型（例如文档、图形、键值等），都可以在这两个数据库中找到适合的解决方案。当然，还有更多优秀的数据库产品供您选择。
          在这篇文章中，我将给读者介绍 MySQL、MongoDB 和 Redis 的一些基本概念、术语、功能和使用场景。通过这些知识，读者能够更好地理解这三种数据库的不同之处，并根据自己的实际需求进行正确的选择。
         # 2.数据库基本概念
         1.关系型数据库（RDBMS）
         关系型数据库管理系统（Relational Database Management System，RDBMS）是一种基于关系模型来存储和管理数据的数据库。它将数据保存在表格的形式上，每张表中包括不同的字段（或属性），每条记录都对应唯一的主键（Primary Key）。这种数据模型最大的特点就是数据以表的形式存放在一起，并且表之间存在一定的联系。一个 RDBMS 中一般会提供许多用于数据的查询语言，如 SQL (Structured Query Language)，用来访问和操纵数据库中的数据。
          
         2.非关系型数据库（NoSQL）
         非关系型数据库（NoSQL）是一类新型的数据库系统，诞生于 2009 年，它最初的目标是在不断增长的数据量和高并发情况下，提供快速查询能力。传统的关系型数据库通常由传统的表格结构组成，但是随着业务的发展，不断增加的数据量和用户的增加，数据库的性能和可用性越来越受到挑战。因此，2009 年一位名叫 Reginald Bellamy 领导的 MongoDB 社区，开始探索 NoSQL 数据库的可能性。他把 NoSQL 分为三个主要类型：键值对存储、文档型存储和列存储。
          
         3.ACID 特性
         ACID （Atomicity，Consistency，Isolation，Durability）是关系数据库管理系统最基本的四个属性。ACID 是指事务的基本特征，它强调一系列的原子性、一致性、隔离性、持久性特性。 Atomicity 确保事务是一个不可分割的工作单位，事务开始之前就要做好准备工作。Consistency 确保事务执行的结果必须是正确的，也就是说一个事务不会改变这个数据库的不一致状态。 Isolation 确保多个事务并发执行时，一个事务的中间状态不会被其他事务看到。 Durability 表示一个事务一旦提交，它对数据库所作的更新就永久保存下来。
         操作事务的时候，为了保证 ACID 中的一致性和持久性，必须开启事务。如果操作失败，事务可以回滚到事务开始前的状态，保证数据的一致性。另外，可以通过数据库的备份机制来实现数据安全。
          
         4.索引
         索引（Index）是帮助数据库快速找到数据的数据结构。索引是数据库搜索引擎优化的一项重要手段。当数据库有大量的数据时，索引可以加速数据的检索，减少查询时间。索引的类型有多种，最常用的有 B-Tree 索引、Hash 索引、全文索引和空间索引。B-Tree 索引是目前最常用的索引类型，它的时间复杂度是 O(log n)。
        
        # 3.MySQL 介绍
        ## 3.1 MySQL 安装配置
        1.下载安装包

        ```
        $ wget http://dev.mysql.com/get/Downloads/MySQL-5.7/mysql-5.7.17-linux-glibc2.5-x86_64.tar.gz
        ```

        2.解压压缩包

        ```
        $ tar -zxvf mysql-5.7.17-linux-glibc2.5-x86_64.tar.gz
        ```

        3.移动文件夹

        ```
        $ sudo mv mysql-5.7.17-linux-glibc2.5-x86_64 /usr/local/mysql
        ```

        4.添加环境变量

        ```
        $ vim ~/.bashrc
        ```

        在文件末尾加入以下命令:

        ```
        export PATH=$PATH:/usr/local/mysql/bin
        ```

        5.刷新环境变量

        ```
        source ~/.bashrc
        ```

        6.创建软连接

        ```
        ln -s /usr/local/mysql/bin/mysql /usr/bin/mysql
        ln -s /usr/local/mysql/bin/mysqladmin /usr/bin/mysqladmin
        ln -s /usr/local/mysql/bin/mysqldump /usr/bin/mysqldump
        ln -s /usr/local/mysql/bin/mysqlimport /usr/bin/mysqlimport
        ln -s /usr/local/mysql/bin/mysqlcheck /usr/bin/mysqlcheck
        ln -s /usr/local/mysql/bin/mysqld_safe /usr/bin/mysqld_safe
        ln -s /usr/local/mysql/lib/* /usr/lib64
        ```

        7.启动服务

        ```
        systemctl start mysqld.service
        systemctl enable mysqld.service
        ```

        8.登录mysql

        ```
        mysql -u root -p
        ```

        如果提示密码输入框，请输入root密码。

        9.修改mysql默认字符集

        ```
        ALTER DATABASE `test` CHARACTER SET utf8 COLLATE utf8_general_ci;
        ALTER TABLE `test`.`table` CONVERT TO CHARACTER SET utf8 COLLATE utf8_general_ci;
        FLUSH PRIVILEGES;
        ```

        将所有数据库以及数据库表编码转为UTF-8。

        10.设置密码验证方式

        ```
        set global validate_password_policy=LOW; //设置为弱密码策略
        set global validate_password_length=6; // 设置密码长度限制
        ```

        更改密码验证策略为弱密码，密码长度限制为6位。

        ## 3.2 MySQL 基础语法
        ### 3.2.1 数据定义语言 DDL（Data Definition Language）
        1.创建数据库

        ```sql
        CREATE DATABASE dbname;
        ```

        2.删除数据库

        ```sql
        DROP DATABASE IF EXISTS dbname;
        ```

        3.创建表

        ```sql
        CREATE TABLE table_name (column1 datatype constraints, column2 datatype constraints);
        ```

        例如：

        ```sql
        CREATE TABLE users (
            id INT PRIMARY KEY AUTO_INCREMENT,
            name VARCHAR(50) NOT NULL,
            email VARCHAR(100),
            age INT DEFAULT '0' CHECK (age >= 0 AND age <= 120),
            reg_date DATE DEFAULT CURRENT_DATE
        );
        ```

        4.修改表结构

        ```sql
        ALTER TABLE tablename ADD COLUMN new_column datatype constraints;
        ALTER TABLE tablename MODIFY COLUMN existing_column datatype constraints;
        ALTER TABLE tablename CHANGE old_column new_column datatype constraints;
        ALTER TABLE tablename DROP COLUMN colummname;
        ```

        5.删除表

        ```sql
        DROP TABLE IF EXISTS tablename;
        ```

        6.查看表信息

        ```sql
        DESC tablename;
        SHOW COLUMNS FROM tablename;
        SELECT * FROM information_schema.tables WHERE table_name='tablename';
        ```

        ### 3.2.2 数据操控语言 DML（Data Manipulation Language）
        1.插入数据

        ```sql
        INSERT INTO tablename (col1, col2,...) VALUES (val1, val2,...);
        ```

        插入一条记录：

        ```sql
        INSERT INTO users (id, name, email, age) VALUES ('1', 'John Doe', 'johndoe@example.com', '30');
        ```

        2.插入多条记录

        ```sql
        INSERT INTO tablename (col1, col2,...) VALUES (val1, val2,...),(val1, val2,...),(val1, val2,...);
        ```

        3.更新数据

        ```sql
        UPDATE tablename SET col1 = value1, col2 = value2 WHERE condition;
        ```

        更新一条记录：

        ```sql
        UPDATE users SET name='Jane Smith' WHERE id=1;
        ```

        4.删除数据

        ```sql
        DELETE FROM tablename WHERE condition;
        ```

        删除一条记录：

        ```sql
        DELETE FROM users WHERE id=1;
        ```

        **注意**：如果没有条件语句，那么将删除整个表的内容！

        ### 3.2.3 数据查询语言 DQL（Data Query Language）
        1.查找单条记录

        ```sql
        SELECT columns FROM tablename WHERE conditions;
        ```

        查找 id 为 1 的记录：

        ```sql
        SELECT * FROM users WHERE id=1;
        ```

        2.查找多条记录

        ```sql
        SELECT columns FROM tablename [WHERE conditions] ORDER BY sortby ASC|DESC LIMIT offset,rows;
        ```

        查询 id 大于等于 10 小于等于 20 的记录，并按照 id 排序，每页显示 10 条：

        ```sql
        SELECT * FROM users WHERE id>=10 AND id<=20 ORDER BY id ASC LIMIT 0,10;
        ```

        3.统计数量

        ```sql
        SELECT COUNT(*) FROM tablename [WHERE conditions];
        ```

        获取 users 表的记录总数：

        ```sql
        SELECT COUNT(*) FROM users;
        ```

        4.组合查询

        ```sql
        SELECT column1, column2 FROM tablename1 [, tablename2...]
        INNER JOIN tablenameX ON tablename1.columnN = tablenameX.columnM
        WHERE condition;
        ```

        查询 users 表和 orders 表，通过 user_id 关联 orders 表：

        ```sql
        SELECT u.*, o.* 
        FROM users AS u 
        INNER JOIN orders AS o ON u.id = o.user_id;
        ```

    # 4.MongoDB 介绍
    ## 4.1 MongoDB 安装配置
    1.下载安装包

    ```
    $ wget https://fastdl.mongodb.org/linux/mongodb-linux-x86_64-ubuntu1604-4.0.3.tgz
    ```

    2.解压压缩包

    ```
    $ tar -zxvf mongodb-linux-x86_64-ubuntu1604-4.0.3.tgz
    ```

    3.移动文件夹

    ```
    $ sudo mv mongodb-linux-x86_64-ubuntu1604-4.0.3 /usr/local/mongodb
    ```

    4.创建软链接

    ```
    $ cd /usr/bin/ 
    $ sudo ln -s /usr/local/mongodb/bin/mongod mongod
    $ sudo ln -s /usr/local/mongodb/bin/mongo mongo
    ```

    5.创建日志目录

    ```
    $ sudo mkdir /var/log/mongodb
    ```

    6.编辑配置文件

    ```
    $ vim /etc/mongod.conf
    ```

    添加以下内容：

    ```
    bind_ip = 127.0.0.1
    port = 27017
    dbpath = /data/db
    logpath = /var/log/mongodb/mongod.log
    journal = true
    smallfiles = true
    ```

    配置文件参数说明：

    + bind_ip：绑定的 IP 地址，默认为 0.0.0.0，表示允许外部访问；
    + port：监听端口，默认为 27017；
    + dbpath：数据库目录路径，默认为 /data/db；
    + logpath：日志文件路径，默认为 /var/log/mongodb/mongod.log；
    + journal：开启 Journal ，默认为 false；
    + smallfiles：启用小尺寸文件，默认为 false；

    7.启动 MongoDB 服务

    ```
    $ sudo service mongod start
    ```

    此时，MongoDB 服务应该已经正常启动运行了。

    8.测试 MongoDB 是否安装成功

    使用客户端工具 mongo 连接本地数据库，创建 test 数据库和 collection，插入一条记录：

    ```
    $ mongo
    > use test
    switched to db test
    > db.createCollection('users')
    Created Collection: users
    > db.users.insertOne({name:'John Doe'})
    Inserted 1 record(s) into the "users" collection
    > exit
    ```

    此时，就可以使用 mongo 命令查看数据库内容了：

    ```
    $ mongo
    > use test
    switched to db test
    > show collections
    users
    system.indexes
    > db.users.find()
    { "_id" : ObjectId("5b47e4c718d4f1beebcf8f4a"), "name" : "John Doe" }
    > exit
    ```

    也可以使用 GUI 工具 Robo 3T 查看数据库内容。

    ## 4.2 MongoDB 基础语法
    ### 4.2.1 CRUD 操作
    1.创建集合（collection）

    ```
    db.createCollection("collectionName")
    ```

    2.插入数据

    ```
    db.collectionName.insert({"key":"value"})
    ```

    在 users 集合中插入一条记录：

    ```
    db.users.insert({name:"John Doe",email:"johndoe@example.com",age:30})
    ```

    批量插入多条记录：

    ```
    var list = [{name:"John Doe",email:"johndoe@example.com",age:30},
               {name:"Jane Smith",email:"janesmith@example.com",age:25}];
               
    db.users.insertMany(list)
    ```

    3.读取数据

    ```
    db.collectionName.findOne({"key":"value"})
    ```

    从 users 集合中查询一条记录：

    ```
    db.users.findOne({})
    ```

    返回所有记录：

    ```
    db.users.find()
    ```

    搜索记录：

    ```
    db.users.find({name:/^J.*/})
    ```

    返回指定字段：

    ```
    db.users.find({},{"name":true,"_id":false})
    ```

    4.更新数据

    ```
    db.collectionName.update({"key":"value"},{"$set":{"newKey":"newValue"}})
    ```

    更新一条记录：

    ```
    db.users.update({name:"John Doe"},{$set:{age:31}})
    ```

    对 age 大于等于 30 的记录，将 age 增加 1：

    ```
    db.users.update({age:{$gte:30}},{$inc:{age:1}})
    ```

    5.删除数据

    ```
    db.collectionName.remove({"key":"value"})
    ```

    删除一条记录：

    ```
    db.users.remove({name:"John Doe"})
    ```

    删除集合：

    ```
    db.dropDatabase()
    ```
    
    ## 4.2.2 高级查询
    1.正则表达式匹配

    ```
    db.users.find({name:/^J.*/})
    ```

    返回所有姓名以 J 开头的记录。
    
    2.$in 操作符

    ```
    db.users.find({name:{$in:["John","Jane"]}})
    ```

    返回 name 字段值为 John 或 Jane 的所有记录。
    
    3.$and 操作符

    ```
    db.users.find({$and:[{"name":"John"},{"age":30}]})
    ```

    返回 name 字段值为 John，且 age 字段值为 30 的记录。
    
    4.$or 操作符

    ```
    db.users.find({$or:[{name:"John"},{age:30}]})
    ```

    返回 name 字段值为 John 或 age 字段值为 30 的记录。
    
    5.$nor 操作符

    ```
    db.users.find({$nor:[{name:"John"},{age:30}]})
    ```

    返回 name 字段值不是 John，也不是 age 字段值不是 30 的记录。
    
    6.$exists 操作符

    ```
    db.users.find({age:{$exists:true}})
    ```

    返回 age 字段存在的所有记录。
    
    7.$type 操作符

    ```
    db.users.find({age:{$type:"number"}})
    ```

    返回 age 字段为数字类型的记录。
    
    8.$mod 操作符

    ```
    db.users.find({age:{$mod:[2,0]}})
    ```

    返回 age 字段值为偶数的所有记录。
    
    9.$text 操作符

    ```
    db.users.find({$text:"John Doe"})
    ```

    使用文本搜索来返回包含 “John Doe” 字符串的记录。
    
    10.$where 操作符

    ```
    db.users.find({$where:"function(){ return this.age == 30; }"});
    ```

    执行一个自定义的 JavaScript 函数来返回满足条件的记录。