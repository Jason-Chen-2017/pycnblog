
作者：禅与计算机程序设计艺术                    

# 1.简介
  
：
PostgreSQL（以下简称PG）是一个开源的对象关系数据库管理系统，它的前身是加州大学伯克利分校的研究人员开发的POSTGRES项目，1996年发布了第一个正式版。从9.0版本开始，PG成为世界上最流行的开源数据库之一。目前，PG已经经历了12个主要版本的发布，支持多达几千种编程语言、操作系统以及硬件平台。虽然PG相对其他开源数据库而言更擅长处理事务性工作负载，但对于高吞吐量的分析型或复杂查询的负载，其性能仍然很难令人满意。因此，即使是面向海量数据的场景下，PG也可能会遇到一些性能问题，比如恢复时间过长、崩溃频繁等。因此，如果需要实现在各种场景下的强大的数据库功能，使用PG可能并不合适。不过，作为一款出色的商业数据库产品，它还是值得学习和使用的。本书将详细讲述PG内部的数据结构、执行过程、资源管理机制及相关组件，力争讲透PG的内核技术，为读者提供一个全面的了解。
# 2.基础知识：PostgreSQL是基于关系模型的数据库管理系统，它的关系数据模型由表和行组成，每个表都有若干列定义，每行记录都对应一行数据。PostgreSQL采用的是文档型数据库的设计方式，所有的数据库对象都以文档的方式存储在数据库中，因此数据库可以方便地被查询、修改、删除、备份和复制。除此之外，PostgreSQL还提供了丰富的SQL语言接口，能够轻松地访问和操作数据。PostgreSQL的性能得到了业界的广泛认可，并且具有很多独特的特性，比如高可用性、灵活的数据结构、强大的ACID事务保证、灵活的数据类型支持等。
# 3.基本概念
## 3.1 数据库系统概览
PostgreSQL是一个关系型数据库管理系统，用于管理关系数据，由多个存储库（database）组成。PostgreSQL分为客户端/服务器模式和嵌入式模式两种运行模式。客户端/服务器模式的部署架构包括前端客户端应用和后端服务进程。前端应用通过网络连接到PostgreSQL服务进程，再由PostgreSQL服务进程处理请求，并将结果返回给前端应用。嵌入式模式一般只在特定应用中使用，通常可以提升数据库的性能。
PostgreSQL服务器由多个进程组成，其中包括主服务器进程（称为postgres），还有任意数量的工作进程。工作进程负责接收客户端连接并处理请求。每一个PostgreSQL服务进程都有自己的内存空间，可以同时容纳多个数据库实例，并共享磁盘空间和网络资源。
## 3.2 数据库系统构成
### 3.2.1 数据库
数据库（Database）是存储数据的集合。在PostgreSQL中，数据库就是一个逻辑上的概念。PostgreSQL允许创建、删除、修改、查询数据库。创建数据库时，需要指定数据库名称、权限模式和初始编码。数据库中的对象（表、视图、索引、序列、函数、触发器等）可以通过不同的模式进行组织和管理。
### 3.2.2 表（Table）
表（Table）是数据库中存放关系数据的一种结构。表是矩形结构，类似于Excel表格或者电子表格。每个表都有一系列的字段（Field），用来存储数据。表由主键、唯一键、索引三种约束条件确定的一组字段唯一确定一条记录。表可以有多个索引（Index）。
#### 3.2.2.1 字段（Field）
字段（Field）是表中的一个属性，用来描述和存储数据。表中的每一列都是一个字段。每个字段都有一个名称、数据类型和选项。例如，名称可以是“name”、“age”等，数据类型可以是整数、浮点数、字符、日期等，选项可以包括是否允许空值、是否唯一、是否自增、默认值等。
#### 3.2.2.2 主键
主键（Primary Key）是表中某个字段或者组合字段的集合，其唯一标识表中的每条记录。每个表只能拥有一个主键。主键的选择对表的性能、完整性和查询效率均有重要影响。
#### 3.2.2.3 唯一键
唯一键（Unique Key）也是表中某个字段的集合，但是该字段的值不能重复。通常情况下，唯一键与主键不同，不要求唯一且可以为空。
#### 3.2.2.4 索引
索引（Index）是帮助PostgreSQL快速定位数据位置的一个数据结构。索引主要有两类，一种是B-Tree索引，另一种是散列索引。B-Tree索引是根据索引字段排序好的一棵树，其查找速度快；散列索引则是根据一个哈希函数映射到磁盘地址，访问速度较慢，但对范围查询非常有效。PostgreSQL支持两种类型的索引，一种是普通索引（Normal Index），一种是唯一索引（Unique Index）。
### 3.2.3 意向锁（Intention Locks）
意向锁（Intention Locks）是事务在执行过程中为了满足其隔离级别而对某些资源的锁定方式。PostgreSQL使用意向锁来保障事务的隔离性。在不同隔离级别下，会存在不同的意向锁。在READ COMMITTED隔离级别下，只会获得Share Row Exclusive Lock。在REPEATABLE READ隔离级别下，除了获得Share Row Exclusive Lock外，还会获得Share Update Exclusive Lock。在SERIALIZABLE隔离级别下，所有锁都不会被授予。
### 3.2.4 物理文件
PostgreSQL实际上是一个存储管理系统，它使用底层的文件系统管理物理文件。一个PostgreSQL数据库实例由一个或者多个目录（目录可以是磁盘、磁带机、磁卡等设备）组成，用于存放表数据文件、日志文件、配置文件等。数据库文件会在这些目录中按照名称顺序分配。
## 3.3 事务处理
事务（Transaction）是指一个不可分割的工作单位。事务由一系列的SQL语句或者数据库操作组成，事务必须满足ACID（原子性、一致性、隔离性、持久性）四大特性。ACID全名分别为Atomicity、Consistency、Isolation、Durability，翻译过来就是：原子性、一致性、独立性、持久性。
### 3.3.1 ACID特性
#### 3.3.1.1 原子性（Atomicity）
原子性（Atomicity）是指一个事务是一个不可分割的工作单位，事务中的操作要么全部成功，要么全部失败回滚到事务开始之前的状态。事务应该是DBMS的最小工作单元，这就保证了事务的原子性。
#### 3.3.1.2 一致性（Consistency）
一致性（Consistency）是指事务必须使数据库从一个一致性状态变到另一个一致性状态。一致性与ACID中的一致性类似，但比ACID中要严格。一致性表示事务完成后，所有节点的数据都必须符合预期的规则。
#### 3.3.1.3 隔离性（Isolation）
隔离性（Isolation）是当多个用户并发访问数据库时，每一个用户都只能看到自己所做的改变。隔离性与ACID中的隔离性类似，但比ACID中要弱。隔离性表示并发事务的执行不能互相干扰，一个事务的执行不能看见其他事务中间的操作。
#### 3.3.1.4 持久性（Durability）
持久性（Durability）是指一个事务一旦提交，它对数据库中的数据的改变就永久保存下来，并不会因数据库故障而消失。持久性与ACID中的持久性类似，但比ACID中要弱。持久性表示已提交的事务的结果对其他用户是可见的。
### 3.3.2 隔离级别
隔离级别（Isolation Level）是用来处理并发访问导致数据不一致的问题。PostgreSQL共有五个事务隔离级别：读未提交（Read Uncommitted）、读已提交（Read Committed）、可重复读（Repeatable Read）、串行化（Serializable）。
#### 3.3.2.1 读未提交（Read Uncommitted）
读未提交（Read Uncommitted）是最低的隔离级别，它允许脏读、幻影读、不可重复读和phantom reads。该级别允许一个事务读取尚未提交的数据，另一个事务可以更改这个数据并提交，然后第一个事务再次读取同一行数据，由于第二个事务的更新，得到了两次读取同一行数据的结果不同，出现了幻象读现象。
#### 3.3.2.2 读已提交（Read Committed）
读已提交（Read Committed）是第二低的隔离级别，它在读提交的基础上，进一步提升了隔离性。一个事务只能读取已经提交的事务所做的改变，换句话说，一个事务总是能够读取到数据库中最新一点的状态。该级别避免了脏读、不可重复读和phantom read。
#### 3.3.2.3 可重复读（Repeatable Read）
可重复读（Repeatable Read）是InnoDB和XtraDB的默认隔离级别。它确保同一个事务的多个实例在并发环境中返回同样的记录集合。该级别除了防止脏读、幻读外，还避免了不可重复读。
#### 3.3.2.4 串行化（Serializable）
串行化（Serializable）是最高的隔离级别。它确保事务按照相同的顺序执行，也就是说，序列化隔离在每行数据上加排他锁，所以只能一个事务一个事务的执行，直到执行结束。这种隔离级别通常用在OLTP（Online Transaction Processing，联机事务处理）系统上，它可以避免并行事务的并发执行带来的问题。
## 3.4 SQL解析
PostgreSQL使用词法分析和语法分析两个阶段的SQL解析。词法分析器将SQL语句拆分成词元，语法分析器识别出语句的语法结构。每个词元都会对应着一个节点，语法分析生成语法树。
### 3.4.1 词法分析器
词法分析器（Lexer）扫描输入的SQL语句，将其拆分成一系列的词元。SQL的词法定义如下：
```sql
<keyword> ::= add | all | alter | analyze | and | any | array | as | asc |
              audit | authorization | avg | between | binary | blob | both |
              boolean | by | case | cast | char | character | check | clob | close |
              cluster | coalesce | collate | column | comment | commit | compile |
              concat | connect | constraint | create | cross | current_user | date |
              datetime | decimal | declare | default | delete | desc | deterministic |
              disallow | disconnect | distinct | double | drop | else | end | escape |
              except | exec | execute | exists | exit | external | false | fetch | first | float | for | foreign | from | full | function | general | get | global | grant | group | hash | having | identity | if | in | index | indicator | inner | input | insensitive | insert | int | integer | intersect | interval | into | is | isolation | join | key | language | last | lateral | left | level | like | limit | listagg | localtime | localtimestamp | lock | long | loop | match | maxvalue | member | merge | method | minvalue | minus | minute | modifies | modify | natural | new | no | none | not | nullif | numeric | of | off | offset | old | on | only | open | or | order | outer | output | over | overlaps | pad | partial | partition | percent | permission | placing | position | precision | prepare | preserve | primary | privileges | procedure | public | quote | real | references | restrict | return | returns |revoke | right | row | rows | schema | scroll | second | select | session | set | share | similar | size | smallint | some | space | sql | sqlcode | sqlerror | substring | sum | symmetric | sysdate | system | table | then | time | timestamp | timezone_hour | timezone_minute | to | trailing | transaction | translate | translation | true | union | unique | unknown | update | usage | user | using | validate | value | values | varchar | varying | view | when | where | while | with | without 

<identifier> ::= <nondigit>[<chars><digits>]|<digit>

<nondigit> ::= [_A-Za-z] 

<chars> ::= <char>|<chars><char>

<char> ::= [_A-Za-z0-9] 

<digits> ::= <digit>|<digits><digit>

<digit> ::= [0-9]

<stringliteral> ::= '<chars>' 

<number> ::= <int>|<float> 

<int> ::= [<sign>]<digit>{[<sep><digit>]}<numtype>

<sep> ::= [_AaBbCcDdEeFfGgHhIiJjKkLlMmNnOoPpQqRrSsTtUuVvWwXxYyZz]

<sign> ::= [-+] 

<numtype> ::= bit|bool|boolean|integer|int|smallint|bigint|decimal|dec|numeric|real|float|double|precision|money|currency|date|datetime|timestamp|time|interval|year|month|day|hour|minute|second

<float> ::= [<sign>]<digit>{<sep><digit>}<frac>?<exp>?<floattype> 

<frac> ::= '.'<digit>{<sep><digit>}

<exp> ::= 'E'|'e'<sign>?<digit>{<sep><digit>} 

<floattype> ::= float|double precision|float4|float8
```
如上所示，词法分析器识别出关键字、标识符、字符串、数字、注释等词元，并将它们组装成语法树中的节点。
### 3.4.2 语法分析器
语法分析器（Parser）检查SQL语句的语法结构是否正确。语法分析器使用语法定义将SQL语句的各个词元组合成符合语法规则的表达式。每个表达式都是一个语法树节点。PostgreSQL的语法定义如下：
```sql
<stmt> ::= <ddlstmt> |<dmlstmt>

<ddlstmt> ::= CREATE { TYPE | DOMAIN } name AS ENUM '(' <enumvals> ')' 
            | ALTER TABLE name RENAME COLUMN old_col TO new_col
            | DROP { TABLE | VIEW | INDEX | SCHEMA | SEQUENCE } name [ CASCADE ]
            | TRUNCATE TABLE name [ CASCADE ]
            | COMMENT ON { TABLE | COLUMN | CAST } name IS quoted_string
            | GRANT <privileges> ON <objspec> TO grantees [ WITH GRANT OPTION ]
            | REVOKE [GRANT OPTION FOR ] <privileges> ON <objspec> FROM revokees
            | LOCK TABLE name [ NOWAIT ]
            | UNLOCK TABLES

<dmlstmt> ::= INSERT INTO name [( <columnlist> )] VALUES ( { <exprlist> | DEFAULT } [,...])  
            | UPDATE name SET { <assignments> } [ WHERE <condition> ]
            | DELETE FROM name [ USING name ] [ WHERE <condition> ]
            | SELECT [ ALL | DISTINCT ] [ <targetlist> ] 
              [ FROM <fromclause> ] 
              [ WHERE <searchcondition> ] 
              [ GROUP BY <groupingset> ] 
              [ HAVING <searchcondition> ] 
              [ ORDER BY <sortspecificationlist> ] 
              [ LIMIT <limitoptions> ] 
              [ OFFSET <offset> ]
            | EXECUTE name [ ( <params> ) ]
            
<columnlist> ::= identifier [,...]

<targetlist> ::= target [,...]
                | '*'

<fromclause> ::= relation 
                | relation ',' fromclause
                
<relation> ::= <rangefunc> 
             | larg RELOP rarg
             | alias
             | subquery
             | JOIN type FULL OUTER? JOIN relation join_qual
             | NATURAL? LEFT? RIGHT? OUTER? JOIN relation
             | INTERSECT ALL? querypart
             | UNION ALL? querypart
             | CASE expression WHEN expr THEN result [ ELSE elseresult ] END
             | EXISTS subquery
             | UNIQUE querypart
             | NOT? BETWEEN expr AND expr
             | NOT? IN subquery
             | NULL
             | TRUE
             | FALSE
             | literal

<join_qual> ::= ON searchcondition
              | USING '(' identifier [,...] ')'
              
<larg> ::= relation 
         | scalar
         | expr
         
<rarg> ::= relation 
         | scalar
         | expr
       
<subquery> ::= '( )' 
             | '( <query> )'
             
<scalar> ::= CURRENT OF cursor
           | NEXT VALUE FOR sequence
           | ROW ( number | '(' <rowvaluelist> ')' )
           | <arrayconstr>
           | LATERAL? ARRAY '[' array_expr '] OVER '(' window_defn ')'
           | LATERAL? func_call
           | typed_table
           | qualified_table_name '.' *
           | subquery
           | expr COLLATE collation
           | ROW '(' expr [,...]'(n)'::datatype[args]
           | ROW '(' expr')'
           | structconstr 
           | ROW '(' ')'
           | typecast
           | CASE expr WHEN expr THEN expr [... ] END
           | EXTRACT '(' extract_field FROM datetime_expr ')'
           | PERCENTILE_CONT '(' expr FROM expr ')'
           | POSITION ('(' substr ')') IN expr
           | COALESCE expr [,...]
           | IF expr THEN expr ELSE expr
           | VARIADIC '?''::text[]'
           | attr_deferrable
           | attr_notnull
           | security_label
           | data_type_attrs
           | domain_constraints
           | FOREIGN KEY '(' columnref [,...]')' REFERENCES reftable ['(' refcolumn[,...]')'] action
           | CHECK '(' predicate ')'

<typecast> ::= expr '::' data_type [ ARRAY ]
             | expr :: data_type [ ARRAY ]
             | COLLATION FOR identifier

<typed_table> ::= data_type 
                 | ROW '(' expr [,...]')' AS name
                 | namedrowtype AS name
                 | unnamedrowtype AS name
                 
<unnamedrowtype> ::= '(' col_decl [,...]')'
                  
<namedrowtype> ::= ROW '(' NAMEDATATYPE col_decl [,...]')'
                   
<structconstr> ::= ROW '(' expr val_commalist')' 
                  | ROW '(' val_commalist')' 
                  
<val_commalist> ::= val_comma
                  
<val_comma> ::= expr 
               | expr COMMA val_comma
               
<action> ::= RESTRICT
            | CASCADE
            | NO ACTION
            | SET NULL
            | SET DEFAULT
            | NO MATCH

<data_type_attrs> ::= INTEGER 
                    | CHARACTER VARYING
                    | NUMERIC
                    | DECIMAL
                    | REAL
                    | FLOAT [ PRECISION ]
                    | MONEY
                    | TEXT
                    | VARCHAR [ ( n ) ]
                    | DATE
                    | TIME [ WITHOUT TIME ZONE ] [ TIMESTAMP ] [ WITH TIME ZONE ]
                    | BOOLEAN
                    | INTERVAL [ field ] [ fields ]
                    | BYTEA
                    | DOUBLE PRECISION
                    | UUID
                    | BIT [ ( n ) ]
                    | SERIAL [ datatype ]
                    | BIGSERIAL [ datatype ]
                    | SMALLSERIAL [ datatype ]
                    | CHAR [ ( n ) ]
                    | ARRAY [ '<' data_type '>', dimension ]
                    
<dimension> ::= constant

<col_decl> ::= col_name data_type [ COLLATE collation ] [ opt_col_constraint [...] ]

<opt_col_constraint> ::= CONSTRAINT constraint_name
                          | NOT NULL
                          | NULL
                          | DEFAULT default_expr
                          | GENERATED ALWAYS AS IDENTITY ( start [ increment ] )
                       