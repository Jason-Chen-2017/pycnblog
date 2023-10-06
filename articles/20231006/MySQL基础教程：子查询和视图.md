
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


在数据库中，子查询可以实现复杂的数据查询。子查询是指嵌套在另一个查询中的一个SELECT语句，或者从其他表中检索数据的子查询。通过子查询，可以实现更复杂、灵活的条件组合查询。视图则是一个虚拟的表，用来存储对现实世界的某个实体进行汇总和分析。

这两者的功能非常类似，但又存在着不同的地方。子查询是将一条SQL语句的结果作为另一条SQL语句的输入值，也就是说可以让一条语句得到另一条语句所需要的所有数据；而视图是定义的一些列（列可以来自多个表），其作用类似于数据库的表，提供一种虚拟的方式来查看复杂的数据结构。虽然功能相似，但是也有本质上的区别。

因此，了解什么时候该用子查询，什么时候应该用视图，如何使用子查询和视图，是很重要的。下面就结合实际案例，系统atically讲解一下MySQL子查询和视图的基本概念和使用方法。


# 2.核心概念与联系
## 2.1 子查询
子查询就是嵌套在另一个查询中的SELECT语句，或者从其他表中检索数据的子查询。子查询一般分为两种：基于标量值的子查询和基于行值的子查询。前者获取单个值并根据这个值执行判断或运算，后者获取多行数据并根据这些行进行各种处理。子查询的语法如下：
```sql
SELECT column_name FROM table_name WHERE condition = (SELECT scalar_value);
```
或
```sql
SELECT column_name FROM table_name WHERE condition IN (SELECT row_value);
```
子查询中常用的关键字包括：
- SELECT: 获取数据。
- FROM: 指定数据源，一般是当前查询所在的表或者视图。
- WHERE: 对数据进行筛选。
- IN: 从子查询返回的值列表中匹配指定字段。

举个例子，假设有一个需求：查找每个部门的平均工资，同时还要统计每个部门的人数，可以这样做：
```sql
SELECT 
    department_id,
    AVG(salary) AS avg_salary,
    COUNT(*) AS employee_count
FROM employees e
WHERE department_id IN (
    SELECT id FROM departments d WHERE dept_location='NY'
)
GROUP BY department_id;
```
这里，子查询首先过滤出纽约销售部门的ID，然后再使用AVG()函数计算平均工资，COUNT()函数计算人数。最后，使用GROUP BY对结果集进行分组。

## 2.2 视图
视图是一个虚构的表，其结构上是由一个或多个表引用的表的join结果。所以，视图不在物理存储上存在，只存在于逻辑层。视图没有独立的物理文件，它的内容实际上是依赖于其引用的基表生成的。视图的创建，删除，修改，检索都十分方便。视图的语法如下：
```sql
CREATE VIEW view_name AS select_statement;
```
其中，select_statement可以是任何有效的SELECT语句。

例如，如果有一个employees表和departments表，希望把employees表按照department_id分组，并且给每个部门加上相应的部门名称，可以使用视图：
```sql
CREATE VIEW emps_by_dept AS 
SELECT e.*, d.department_name 
FROM employees e JOIN departments d ON e.department_id=d.id;
```
然后就可以像使用employees表一样，通过emps_by_dept视图来访问部门信息了。

## 2.3 区别与联系
### 2.3.1 相同点
子查询与视图都是为了实现复杂的数据查询，并提供了一种虚拟的方式来查看复杂的数据结构。它们之间的相同点主要包括：
- 可以使用WHERE条件进行筛选、排序。
- 都可以通过JOIN关键字连接多个表。
- 支持多种查询方式，如GROUP BY、HAVING等。

### 2.3.2 不同点
子查询用于获取单个值，或者基于标量值的运算；视图是定义的一些列，其结构上是由多个表引用的表的join结果。因此，它们之间的不同点主要有以下几方面：
- 性能上：子查询在计算时会先完成一次外层查询的计算，再把结果传递给子查询进行处理；而视图是在数据库内部完成的，不需要额外的开销，速度较快。
- 数据安全性上：子查询的查询结果受到外层查询影响，可能会造成数据泄露。视图的查询结果不会受外层查询影响，不存在数据泄露的风险。
- 可扩展性上：子查询只能选择当前事务可见的数据，不能实现跨事务的操作；视图可以实现跨事务的操作，但只能读数据。
- 更新策略上：子查询只支持简单更新，而且只有外层查询的UPDATE才会影响到子查询的结果；视图支持复杂的更新策略，支持INSERT/DELETE/UPDATE操作。