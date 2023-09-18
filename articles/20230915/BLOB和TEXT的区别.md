
作者：禅与计算机程序设计艺术                    

# 1.简介
  

BLOB（Binary Large Object）对象表示二进制大型数据对象，主要包括图片、视频、音频、压缩文件等，在数据库中也叫做BINARY LARGE OBJECT或BLOB字段类型。而TEXT对象则表示存储为文本的数据类型。两者的差别主要体现在对数据的处理方式上。

# 2.不同点
## 2.1 数据格式
BLOB字段类型的数据通常采用字节流的方式进行存储，可以存储任何格式的文件，比如图片、视频、音频或者压缩文件。而TEXT字段类型的数据则是按字符编码的方式进行保存的，只能存储特定格式的文本数据。

## 2.2 索引方式
对BLOB对象来说，一般都需要建立一个独立的索引来加快检索速度，但对于TEXT对象来说，一般都直接将其创建索引就足够了，不需要额外的索引结构。

## 2.3 查询性能
对于BLOB类型的字段，由于数据的大小不固定，无法建立索引，因此查询效率不一定很高。但是对于TEXT类型的字段，只要建好了索引，就可以快速进行全文检索，查询速度非常快。

## 2.4 增删改操作
对于BLOB对象，支持插入、更新、删除操作；对于TEXT对象，支持插入、更新操作，但是不支持删除操作。

## 3.应用场景
- 大量图片、视频等二进制数据插入数据库时适用BLOB字段类型。
- 对文本数据进行排序、统计分析时适用TEXT字段类型。

# 4. BLOB相关的功能
## 4.1 插入
```
INSERT INTO table_name (column_name) VALUES ('value'); 
-- value可以使用一个变量，也可以使用一段字符串或表达式
```

## 4.2 删除
```
DELETE FROM table_name WHERE column_name = 'value';
```

## 4.3 更新
```
UPDATE table_name SET column_name = 'new_value' WHERE condition;
```

## 4.4 查找
```
SELECT * FROM table_name WHERE column_name LIKE '%text%';
```

# 5. TEXT相关的功能
## 5.1 插入
```
INSERT INTO table_name (column_name) VALUES ('value'); 
-- value可以使用一个变量，也可以使用一段字符串或表达式
```

## 5.2 删除
不支持删除操作。

## 5.3 更新
```
UPDATE table_name SET column_name = 'new_value' WHERE condition;
```

## 5.4 查找
```
SELECT * FROM table_name WHERE column_name LIKE '%text%';
```

## 6. 参考资料
2021/10/7 11:00:29<|im_sep|>

# Bibliography and References {-}
<!-- This is an optional section that could be used for bibliographic references or reading lists -->

* MySQL数据库Blob与Text的区别 [@wangxinghui2021mysql]

# Footnotes {-}
<!-- You can add footnotes here if necessary -->

[^1]: Some textual footnote @smith2000smith. 

## About the author

This article was written by **Alex**. More information about Alex can be found [here].