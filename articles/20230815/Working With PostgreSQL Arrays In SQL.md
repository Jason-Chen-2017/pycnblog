
作者：禅与计算机程序设计艺术                    

# 1.简介
  

PostgreSQL是一个开源的关系数据库管理系统（RDBMS），也是目前最流行的数据库之一。它提供丰富的数据类型、完整的SQL语言支持、强大的事务处理机制、高性能等诸多优点。但同时，它也带来了一些局限性。由于PostgreSQL基于堆文件存储数据，因此对于具有多个元素的数据类型，比如数组或JSONB对象，可能存在效率低下或者不便的情况。本文将介绍PostgreSQL中对数组类型数据的操作。
在PostgreSQL中，数组类型是一种多值的集合，可以存储多个相同类型的元素。它类似于C/Java中的数组，不过这里面不能保存不同类型的数据。并且，数组类型并不是由单独的一列数据组成，而是由多个列数据一起组成。数组中的每个元素都有一个下标（称为偏移量），通过这个偏移量就可以定位到相应的元素值。因此，数组类型是一类特殊的数据类型。
# 2.PostgreSQL Arrays Concepts and Terminology
PostgreSQL Arrays可以分为以下几种：

1. Single-Dimensioned Array: 一维数组
这种数组只能存储同一种数据类型的数据，且每个元素都是唯一的。如int[]。

2. Multi-Dimensional Array: 多维数组
这种数组可以存储多维数组，其中各个元素可能是不同的类型。如int[][]。

3. Homogeneous Array: 同构数组
这种数组中所有元素必须是相同的类型。如text[]。

4. Varying Array: 可变数组
这种数组的长度可以在运行时改变，这时它的长度就是可变的。如varchar[].

PostgreSQL对数组类型的支持主要依赖于三个系统函数：
- ARRAY[...]: 创建一个新的数组
- ARRAY_LENGTH(array, dimension): 获取数组的维度信息或长度
- ARRAY_TO_STRING(array, delimiter): 将数组转换为字符串形式

除此之外，还有一些其他相关的系统函数，比如索引函数、聚集函数等。
# 3.Working with Arrays in PostgreSQL
## 3.1 Creating an Empty Array
To create a new empty array, you can use the following syntax:

```sql
CREATE TABLE mytable (
    id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL,
    myarray INT[], -- define the type of elements for this array
    otherfield INTEGER
);
```

This will create a table `mytable` with columns `id`, `name`, `myarray` and `otherfield`. The column `myarray` is defined as an integer array which means that it can hold multiple integers at once. You don't need to specify any values yet since we are just creating the table structure. 

You can also add an existing array as a default value using the `DEFAULT` clause when defining the column:

```sql
CREATE TABLE mytable (
    id SERIAL PRIMARY KEY,
    name VARCHAR NOT NULL,
    myarray INT[] DEFAULT '{1,2,3}', -- provide some initial data by default
    otherfield INTEGER
);
```

In this case, the table will be created with three initial integers stored in the array. If no data is provided explicitly during insertion or update operations, these numbers will be used instead. Note that arrays must always be enclosed within `{ }` curly braces, even if they only contain one element. Also note that Postgres uses `,` as the separator between array elements. 

Alternatively, you can create an empty array using the function `ARRAY[]`:

```sql
INSERT INTO mytable (name) VALUES ('John');
UPDATE mytable SET myarray = ARRAY[]::INT[]; -- set the array to an empty array
```

Note that while both methods achieve the same result, the first method has the advantage of allowing us to declare the exact types of each column beforehand. This makes code more readable and easier to maintain. The second method is often useful if you want to store an empty array in a pre-existing table or insert rows without specifying all columns. However, keep in mind that the two approaches may not always be compatible depending on how your database schema was designed. 
## 3.2 Inserting Elements into an Existing Array Column
To insert a single element into an existing array column, you can use the simple assignment operator `=` like so:

```sql
UPDATE mytable SET myarray[1] = 7; -- replace the first element with 7
```

If the array does not exist yet, you can still assign values to specific positions using indexing (`[ ]`). For example:

```sql
INSERT INTO mytable (id, name, myarray) VALUES (42, 'Jane', '{9,8,7}');
```

This would insert a new row into the table with ID=42, Name='Jane' and MyArray={9,8,7}. Note that the array literal needs to be enclosed in single quotes and separated by commas to indicate separate elements. If there were already five elements in the array, those sixth and seventh ones would have been added automatically. To avoid adding extra elements, you could append to the end of the array using the concatenation operator `&` like so:

```sql
UPDATE mytable SET myarray = myarray || ARRAY[1]; -- appends [1] to the end of the array
```

Now, the resulting array will be {9,8,7,1}, assuming that it did not exceed its maximum length. Alternatively, you could insert individual elements using subscripts inside the brackets:

```sql
UPDATE mytable SET myarray[array_length(myarray,1)+1] = 6; -- inserts 6 after the last element
```

This approach allows you to dynamically determine where to insert new elements based on their current position within the array.

Similarly, you can use the JSONB operators `.` and `#>` to modify nested objects or arrays respectively:

```sql
UPDATE mytable SET myjsonbcolumn #> '{"foo", "bar"}' = '[1,2,3]' WHERE...; -- updates foo->bar to [1,2,3]
```

However, note that modifying large nested arrays can be slow because each modification requires rewriting the entire array from scratch. Instead, consider splitting out smaller chunks of the original array into dedicated tables with foreign keys back to the main table. This will allow faster modifications and better performance overall.
## 3.3 Accessing Individual Elements of an Array
Once you have inserted some elements into an array column, you can access them using either subscript notation (`[ ]`) or the functions `array_position()` and `unnest()`. Here's an example:

```sql
SELECT myarray[1], unnest(myarray), array_position(myarray, 8) FROM mytable;
```

The output might look something like this:

```
     myarray    |   unnest   | array_position 
---------------+------------+--------------
 "{9,8,7}"      |        9   |           null 
 "{9,8,7,6,5}"  |       6    |           4  
 "{9,8,7,6,5}"  |       5    |           3  
(3 rows)
```

The `myarray[1]` expression returns the first element of the array, but assumes that the array contains at least one element. If the array is empty, then the query will return `null`. While the `unnest()` function flattens the array into a list of scalar values, including `null` elements if the array contains gaps. Finally, `array_position()` returns the offset of the specified element within the array, or `NULL` if it cannot find the element. Note that offsets start at 1, not zero.