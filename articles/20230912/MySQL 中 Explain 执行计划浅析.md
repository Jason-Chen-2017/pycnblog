
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Explain 是 MySQL 用于分析 SQL 查询语句执行过程的命令，可用于分析查询优化、索引选择等问题。本文介绍 explain 执行计划命令的语法及功能，并通过示例进一步分析其中的含义，帮助读者更好地理解 MySQL 的执行计划机制。
# 2.基本概念和术语
## 2.1 Explain 执行计划概述
Explain 命令是 MySQL 中的一个工具，可用来获取 SELECT、INSERT、UPDATE、DELETE 语句的执行计划信息，输出结果中包括该语句的执行序列、数据访问情况、连接关系、是否使用临时表等详细信息。MySQL 提供了 EXPLAIN 关键字来实现对 SQL 语句的分析和优化。
## 2.2 概念术语
### 2.2.1 Table Access Path
Table Access Path（表存取路径）是指 MySQL 在处理查询时，按照一定顺序查找数据表的过程。例如，当查询涉及到多个表时，可能需要多次访问某个表才能返回结果。每条记录在数据库中都有一个唯一的主键值，因此可以通过主键进行快速定位。Table Access Path 可以分成三个部分：

1. Index Range Scan：按照索引列从小到大的顺序进行范围扫描。此方法适用于对指定索引列进行精确匹配的查询，或者对索引列进行范围查询的查询。
2. Index Skip Scan：索引跳过扫描，跳过中间不满足条件的索引记录，仅读取满足条件的索引记录。该方法仅适用于组合索引的查询。
3. Full Table Scan：全表扫描，直接从索引第一行开始，逐行扫描整个表，找到所有满足条件的记录。这种方法效率最低。

### 2.2.2 Nested Loop Join
Nested Loop Join（内层循环联结）是一种广泛使用的算法，可以计算两个关系表之间的笛卡尔乘积，然后基于相关联字段进行过滤。此方法比较耗资源，尤其是在大型表上。Nested Loop Join 有两种实现方式：

1. Block Nested-Loop Join：块嵌套循环联结法，首先将两个表按索引列排序，然后从两端开始遍历，直至找到匹配项。
2. Batched Key Access Join：批处理键访问联结法，采用批量的方式向内存或磁盘读入索引页数据，然后合并匹配的记录。

### 2.2.3 Sort Merge Join
Sort Merge Join（排序归并联结）是一个非常高效的联结方法，它先对两个表进行排序，然后逐行比较，以找到所有匹配的记录。如果两个表大小不同，则会先对较小的表进行合并排序，再对两个表进行归并。Sort Merge Join 的性能一般优于其他联结方法。

### 2.2.4 Hash Join
Hash Join（哈希联结）是一个用哈希表进行联结的方法。它将第一个表的连接字段映射到第二个表的对应位置，以便快速定位匹配项。目前，Hash Join 是 Nested Loop Join 和 Sort Merge Join 的折衷方案，但是它的性能也不是很理想。

### 2.2.5 执行计划树结构
Execution Plan Tree （执行计划树）是 MySQL 使用 Explain 获取到的执行计划中最重要的部分之一。它反映了 MySQL 根据统计信息，解析器生成的执行计划图。执行计划树的层次结构如下图所示：

执行计划树从下到上依次表示执行顺序：从下往上依次是每个节点的时间开销、CPU使用率、数据访问情况；从右到左依次是连接类型、键的使用情况、索引使用情况等。通常情况下，执行计划树由以下几个部分构成：

1. Type：表示访问类型，如 const 表示只访问一次，all 表示全表扫描。
2. Rows Examined：表示已经扫描的数据记录数量。
3. Extra Information：扩展信息，比如 using index 表示使用了覆盖索引等。
4. Filter Condition：表示过滤条件。
5. Projection：表示列的取值范围。

### 2.2.6 概念术语总结
Table Access Path、Nested Loop Join、Sort Merge Join、Hash Join、Execution Plan Tree，这些概念与技术术语比较复杂，但是有助于我们更好地理解 MySQL 的执行计划机制。下面我们开始进入正文。