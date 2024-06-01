HBase是一个分布式、可扩展、高性能的列式存储系统，它基于Google的Bigtable设计，采用了LSM-tree数据结构作为存储引擎。HBase支持高效的随机读写操作，适用于存储海量数据和实时数据处理。

HBase的二级索引（Secondary Index）功能允许用户在一个表中创建一个额外的索引，以便在查询过程中加速查找。二级索引可以帮助我们解决一些特定查询场景的问题，例如：需要基于某个非主键列进行快速搜索。

下面是一个HBase二级索引的基本原理和代码实例讲解：

原理：

1. 创建二级索引：在HBase表中创建一个二级索引时，需要指定一个非主键列作为索引列。HBase会为这个索引列创建一个专用的存储文件，并在内存中维护一个索引缓存。

2. 数据写入：当数据写入HBase表时，除了写入数据行和主键列之外，还需要同时写入索引列的值。HBase会将这些值存储在索引文件中，以便进行快速查找。

3. 查询过程：当执行查询时，HBase会首先在内存中的索引缓存中查找索引列的值，如果找到匹配的值，则直接返回对应的数据行。否则，HBase会在磁盘上的索引文件中进行查找。

代码实例：

以下是一个使用HBase二级索引的简单示例：

1. 首先，创建一个HBase表，并指定主键列和二级索引列：
```java
Configuration conf = new Configuration();
HBaseAdmin admin = new HBaseAdmin(conf);
HTableDescriptor tableDescriptor = new HTableDescriptor(TableName.valueOf("exampleTable"));
tableDescriptor.addFamily(new HColumnDescriptor("cf1"));
admin.createTable(tableDescriptor, "key", "indexColumn");

// 创建二级索引
IndexDescriptor indexDescriptor = new IndexDescriptor("indexColumn");
indexDescriptor.setReversed(true);
tableDescriptor.addIndex(indexDescriptor);
admin.modifyTable("exampleTable", tableDescriptor);
```
1. 向HBase表中写入数据，并同时写入二级索引列的值：
```java
HTable table = new HTable(conf, "exampleTable");
Put put = new Put(Bytes.toBytes("row1"));
put.add(Bytes.toBytes("cf1"), Bytes.toBytes("key"), Bytes.toBytes("value1"));
put.add(Bytes.toBytes("cf1"), Bytes.toBytes("indexColumn"), Bytes.toBytes("indexValue1"));
table.put(put);

// 写入更多数据
```
1. 执行查询操作，使用二级索引加速查找：
```java
Scan scan = new Scan();
scan.setFilter(new SingleColumnValueFilter(Bytes.toBytes("cf1"), Bytes.toBytes("indexColumn"), CompareFilter.CompareOp.EQUAL, Bytes.toBytes("indexValue1")));
ResultScanner scanner = table.getScanner(scan);
for (Result result : scanner) {
    System.out.println("Key: " + result.getValue(Bytes.toBytes("cf1"), Bytes.toBytes("key")));
    System.out.println("Value: " + result.getValue(Bytes.toBytes("cf1"), Bytes.toBytes("value")));
}

// 关闭资源
scanner.close();
table.close();
```
通过这个示例，我们可以看到HBase二级索引如何加速查询操作。在查询过程中，HBase首先尝试在内存中的索引缓存中查找索引列的值，如果找到匹配的值，则直接返回对应的数据行。否则，HBase会在磁盘上的索引文件中进行查找。这样，HBase就可以在O(logN)的时间复杂度内完成查询操作。