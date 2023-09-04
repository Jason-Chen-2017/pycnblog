
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Facebook是一个著名的社交网站，其社交网络服务于全球数百亿用户。Facebook基于Google的Bigtable数据库构建了一个超高性能的分布式存储系统来支撑其应用服务，该系统能够存储海量的数据并可快速响应。由于数据量的激增、对数据库查询请求的大量并发，Facebook正在面临巨大的数据库扩展、容灾、备份等问题。因此，Facebook正在寻找一种能够满足其大规模数据存储和处理需求的可靠的、高度可伸缩的分布式存储方案。
为了解决这些问题，Facebook开发了名为Fdb的分布式存储系统，它是一个基于Bigtable的开源分布式存储系统，它可以支持百万级或千万级数据的存储和处理。在本文中，我们将详细阐述Fdb的设计和实现，并分享一些经验教训和对未来的期望。
# 2.基本概念术语说明
首先，我们需要了解一些相关的基本概念和术语。
## 2.1 Google BigTable
Google的Bigtable是一种分布式存储系统，它被设计用于在多个服务器上存储和检索大型的持久化数据集。Bigtable的主要特点包括以下几点：
1. 线性的扩展性：Bigtable通过水平拆分数据存储在不同的服务器节点上，并通过负载均衡使得访问请求能够自动地负载到合适的服务器节点上。
2. 分布式数据存储：Bigtable的数据存储分布在多台服务器上，任何一个节点都可以提供服务，并且各个节点之间通过复制互联互通，形成了一个由无限存储空间组成的整体集群。
3. 自动故障恢复：Bigtable中的每个节点都是独立运行的，当其中某一个节点发生故障时，其他节点会自动检测到这种情况并迅速进行故障切换，保证数据服务的连续性。
4. 可用性高：Bigtable采用多副本机制，即相同的数据存放在不同的服务器上，可以在节点宕机时仍然保持数据的可用性。
5. 高效查询性能：Bigtable利用谷歌的MapReduce计算框架，通过执行海量的并行数据检索，达到极高的查询性能。

## 2.2 HBase
Apache HBase是Bigtable项目的一个开源实现，它是Bigtable的Java版本，能够运行在Apache Hadoop之上。HBase的主要特点包括以下几点：
1. 宽列族存储：HBase中的数据按列族(Column Family)进行分类，每一个列族都可以有不同的属性，例如数据类型、压缩方式等。不同的列族之间可以共享索引信息，有效地减少数据的冗余和存储开销。
2. 实时数据分析：HBase提供了实时的分布式数据分析能力，能够对数十亿条记录进行快速查询。
3. 高性能批量导入导出：HBase支持高性能的批量导入导出，从而降低了数据迁移的耗时。
4. 支持动态schema：HBase支持灵活的schema，可以根据实际情况动态增加或删除列族，有效地节省磁盘资源。
5. 智能读取负载均衡：HBase支持智能的负载均衡策略，能够识别热点读写负载，提升系统的吞吐量和性能。

## 2.3 Fdb
Facebook的分布式存储系统Fdb是一种基于Bigtable的分布式存储系统，它基于开源版本的HBase进行开发。Fdb主要特点包括以下几点：
1. 数据模型优化：Fdb对Bigtable的基础数据模型做了优化，采用了更加灵活和高效的模型。Fdb的表由一个或多个列族组成，不同列族的数据结构可以不同，相互之间不影响；同一列族下的数据按RowKey排序；不同列族下的某个列有不同的版本；
2. 异步写入：Fdb采用异步的日志写入模式，避免了长时间的等待和锁定，可以有效地提升写入性能；
3. 分层冗余存储：Fdb对数据存储采用了三级冗余存储，可以最大程度防止数据丢失；
4. 流式聚合查询：Fdb支持流式聚合查询，可以将多次查询合并为一次批处理，避免多次网络往返，加快查询速度；
5. 强一致性：Fdb保证数据的强一致性，所有写入操作都会直接同步到所有节点上，避免了复杂的分布式事务协议。

## 2.4 其他关键术语
除以上所述的关键术语外，Fdb还包含以下关键术语：
- TLog：事务日志。在Bigtable中，每一次写操作会先写入内存，然后在提交前同步到Journal(Wal)，如果出现失败则重试，直至成功。而Fdb也会在内存中维护一个日志（TLog），用来存储提交的事务。TLog采用WAL的方式同步到底层存储中，保证强一致性。
- Data Block：数据块。数据块是Fdb将数据按照大小划分的最小单元，每个数据块只能存储一个RowKey范围内的数据。数据块大小默认设置为1MB。
- Memtable：内存表。Memtable是Fdb的本地缓存，保存最近写入的数据，它也是一个队列结构，按先进先出(FIFO)的方式被追加到日志中。
- SSTable：SSTable文件。SSTable是Fdb的稀疏存储表，它是Bigtable项目的核心文件，其是一种将数据按照Key-Value形式编码的二进制文件。

# 3.核心算法原理和具体操作步骤以及数学公式讲解
## 3.1 数据存储过程概览
Fdb是Bigtable的开源实现，它的核心思想就是分布式数据存储。分布式数据存储的基本思路是将数据按照Rowkey划分为多个数据块(DataBlock)，并存储在不同的机器上(Server)。每个数据块可以按照列族(ColumnFamily)进行分类，同一列族下的数据具有相同的schema。因此，Fdb将数据按照RowKey进行排序，并且不同列族的schema与数据块是独立的。为了保证高可用性，Fdb采用了三级冗余机制：每个数据块有三个副本分别存储在不同的服务器上，其中一份副本位于本地磁盘，另两份副本位于不同的数据中心，以提供容错保护。


如图1所示，Fdb的存储架构主要分为如下四部分：

1. Master Server：Fdb的Master Server管理整个集群，通过负载均衡器分配数据块的读写请求，并协调数据块的副本分布和故障切换。

2. Data Servers：Data Server负责存储数据块的物理位置，同时协助Master Server完成数据副本的创建、删除、移动等操作。

3. Client Library：客户端库负责将用户请求转发给对应的Data Server，完成数据读写操作。

4. Namenode Server：Namenode Server存储Fdb的元数据，包括表的定义、布局、权限等，这些元数据将作为配置参考数据保存到内存中。

接下来，我们详细介绍一下Fdb的核心数据存储流程。

## 3.2 数据写入流程
### 3.2.1 数据写入阶段
当Client向Fdb发送写请求时，首先会将数据封装成一系列的KeyValue对。在Fdb的API中，写请求是通过put()方法来完成的。当put()方法被调用时，首先会检查表是否存在，若不存在则创建表。在表存在的情况下，put()方法将数据分解为一系列的KeyValue对，并将每个KeyValue对插入到对应的Memtable中。对于每个Memtable，Fdb会维护一个链表(LinkedList)来存储KeyValue对，该链表中保存的是最新写入的数据。对于不同的Memtable，Fdb会将它们分别存储到不同的SSTable文件中。

```java
public void put(String tableName, byte[] rowKey, List<Column> columns) throws IOException {
    Table table = connection.getTable(TableName.valueOf(tableName));

    try {
        // Check if the table exists and create it if not present
        if (!Admin.exists(connection, tableName)) {
            Admin.createTable(connection, new HTableDescriptor(tableName).addFamily(new HColumnDescriptor("data")));
        }

        Put put = new Put(rowKey);
        for (Column column : columns) {
            put.addColumn(Bytes.toBytes(column.getFamily()), Bytes.toBytes(column.getName()), column.getValue());
        }

        table.put(put);
    } finally {
        table.close();
    }
}
```

### 3.2.2 数据提交阶段
当Memtable中的数据积累到一定数量后或者间隔一段时间就会被提交到Journal文件中。Journal文件中保存的是所有的写操作请求，包括客户端发起的put()、delete()等请求。在收到Commit请求时，Fdb会将TLog中的数据逐步写入底层的HBase存储中，同时通知后续的后台线程对数据进行处理。后台处理线程负责将日志中的数据转换为SSTable文件，并将其分布到不同的服务器上。

```java
if (memtables.size() > MEMTABLE_FLUSH_THRESHOLD || System.currentTimeMillis() - lastFlushTime >= FLUSH_FREQUENCY_MS) {
    flushMemTables();
}
private synchronized void flushMemTables() {
    long t0 = System.nanoTime();

    // Sort all memtables by key to ensure that data is written in sorted order
    Collections.sort(memtables, Comparator.comparing((Memtable m) -> ByteBufferUtil.bytesToHex(m.getFirst().getRow()))
                                   .thenComparingLong((Memtable m) -> Long.MAX_VALUE - m.getTimestamp()));

    // Write each memtable to disk as an sstable file and delete its entries from memory
    int count = 0;
    for (Memtable memtable : memtables) {
        String filename = getUniqueSStableFilename();
        LOG.info("Writing memtable {} to {}", memtable.getId(), filename);
        writeMemtableToSstableFile(filename, memtable);
        memtable.clearEntries();
        count++;
    }

    if (count > 0) {
        // Send a notification to the background thread to process the newly flushed files
        notifyBackgroundThread();
    }

    double elapsedSec = (System.nanoTime() - t0) / SECOND_NANOS;
    lastFlushTime = System.currentTimeMillis();
    LOG.info("Flushed {} memtables in {:.3f} sec", count, elapsedSec);
}
```

### 3.2.3 数据提交后处理阶段
当TLog中的数据被写入底层存储后，Fdb会启动一个后台线程来处理新产生的文件。后台线程的工作过程如下：

1. 检查是否有新的SSTable文件需要处理；

2. 将新的SSTable文件移动到不同的服务器上的HBase目录中；

3. 对已有的SSTable文件进行合并操作，将其中的数据归并到一起；

4. 删除旧的SSTable文件。

```java
private void handleNewSStables() {
    try {
        boolean moreFiles = true;
        while (moreFiles &&!shutdownRequested()) {
            FileStatus status = fs.listStatus(getFdbSstablesDir(), null)[0];

            Path path = status.getPath();
            if (path!= null) {
                String name = path.getName();

                // Only consider files with names of this format: "table-<timestamp>-<sequence>.sst"
                Matcher matcher = FILENAME_PATTERN.matcher(name);
                if (matcher.matches()) {
                    // Extract timestamp and sequence number
                    long timestamp = Long.parseLong(matcher.group(1));
                    int seqNum = Integer.parseInt(matcher.group(2));

                    // Skip over any existing sstables which match the same timestamp but lower or equal sequence number
                    boolean foundExistingSSTable = false;
                    for (int i = 0; i < activeMemtables.size(); i++) {
                        Memtable memtable = activeMemtables.get(i);
                        long mtTimestamp = Math.abs(memtable.getTimestamp());
                        if (mtTimestamp == timestamp && seqNum <= memtable.getSequenceId()) {
                            foundExistingSSTable = true;
                            break;
                        }
                    }

                    if (!foundExistingSSTable) {
                        // Move the new sstable into place on all servers
                        moveFileToHdfs(path);

                        // Remove any old versions of this sstable that are no longer needed
                        removeOldSSTables(timestamp, seqNum);

                        // Create a merged sstable using the current set of live files
                        mergeSSTables(activeMemtables, timestamp, seqNum);

                        // Delete the original sstable file from local filesystem
                        deleteLocalFile(path);
                    } else {
                        LOG.debug("{} has already been processed.", name);
                    }
                }
            }

            Thread.sleep(100); // Sleep briefly before checking again for new files
        }
    } catch (Exception e) {
        logErrorAndShutdown("Failed to handle new sstables:", e);
    }
}
```

## 3.3 数据读取流程
### 3.3.1 数据获取阶段
当客户端向Fdb发送读请求时，首先会定位对应的服务器，并将请求转发到该服务器。Client首先会检查本地缓存，如果命中则返回结果。否则，Client会向Master服务器发送Get请求，Master服务器根据指定的规则选择一个Data服务器进行处理。Master服务器会根据数据的负载进行负载均衡，并且为请求生成相应的路由表。

```java
public Result get(String tableName, Get get) throws IOException {
    TableName tn = TableName.valueOf(tableName);
    Table table = connection.getTable(tn);
    Result result = null;

    try {
        result = table.get(get);
        return result;
    } finally {
        table.close();
    }
}
```

### 3.3.2 数据合并阶段
如果多个Data服务器上都有请求的数据，那么Client会接收到多个结果，然后将结果合并为最终结果返回。首先，Client会把结果按照Region切分成多个部分，再把这些部分排序，最后把结果合并为最终结果。

```java
ResultScanner scanner = null;
try {
    Scan scan = new Scan(query.getStartRow(), query.getEndRow());
    addFilterIfNotNull(scan, filterList);
    if (query.isReversed()) {
        scan.setReversed(true);
    }

    final QueryPlan plan = planner.getQueryPlan(tn, scan, query.getMaxVersions(),
                                            query.isSmall(), batchSize, serverCacheConfig, clientTracker);

    scanner = plan.getQueryExecutor().execScan(plan, null);

    results = getMergedResults(scanner, query.isSmall());

    return results;
} finally {
    IOUtils.closeQuietly(scanner);
    masterClient.releasePlan(plan);
}
```

## 3.4 处理动态变化的架构
Fdb采用了自主研发的Hadoop生态系统，它能够在不停机状态下完成动态的数据处理调整。

### 3.4.1 添加/删除节点
添加/删除DataNode节点：Fdb集群在运行过程中可以随时动态增加或删除DataNode节点。当一个DataNode节点加入集群时，它会自动将数据分片分布到其他DataNode节点上，并完成数据同步。当一个DataNode节点离开集群时，它不会影响现有的数据分布，只会暂时停止提供服务。

### 3.4.2 改变服务器配置
Fdb允许管理员修改服务器的配置，例如调整DataNode的内存、磁盘数量、CPU核数等，而不需要停机。Fdb的Master服务器会在短时间内感知到配置文件的变更，并调整相应的数据分布策略。

### 3.4.3 修改schema
Fdb的schema是可以动态修改的。例如，管理员可以通过增加或删除列族、更改数据类型、压缩方式等来调整数据格式。新增或删除列族时，Fdb会根据新的schema创建新的SSTable文件，并将老的SSTable文件留作历史纪录，这样Fdb就能够兼容老的读写请求。

### 3.4.4 更新节点软件
当DataNode节点升级软件时，Fdb的Master服务器会自动感知到该节点的变化，并完成必要的重新分片操作。

# 4.具体代码实例和解释说明
在研究了Fdb的基本算法和存储流程之后，我们来看一下Fdb的代码实现。我们先从Client的角度出发，看一下如何连接Fdb集群并写入数据，然后再到Server的角度出发，看一下Fdb的后台处理线程是如何将SSTable文件处理为最终的HBase文件。

## 4.1 Fdb Client端代码示例
这里我们展示Fdb Java客户端的put()方法的代码实现，用于向Fdb集群中插入数据。put()方法的参数列表如下：

```java
public void put(String tableName, byte[] rowKey, List<Column> columns) throws IOException {
...
}
```

该方法首先检查表是否存在，如果不存在则创建一个空表。然后，将数据封装为Put对象，并将其写入到Memtable中。当Memtable中的数据积累到一定数量后或者间隔一段时间就会被提交到Journal文件中。Journal文件的名称遵循固定的命名规则："table" + "_" + tableName + "_journal_NNN.jnl", 其中NNN是一个自增编号，每次打开Journal文件时将自动递增该编号。

```java
Table table = connection.getTable(TableName.valueOf(tableName));
boolean created = false;
try {
    if (!admin.tableExists(tableName)) {
        admin.createTable(new HTableDescriptor(tableName), new byte[][] {{'d'}});
        created = true;
    }
    Put p = new Put(rowKey);
    for (Column col : columns) {
        p.addColumn(col.family, col.qualifier, col.value);
    }
    table.put(p);
} catch (IOException ex) {
    throw new RuntimeException(ex);
} finally {
    closeTable(table);
    if (created) {
        admin.disableTable(tableName);
        admin.deleteTable(tableName);
    }
}
```

当TLog文件中的数据被写入到底层的HBase存储中时，后台处理线程会处理Journal文件，将其转换为SSTable文件，并将其分布到不同的服务器上。后台处理线程在每次处理完Journal文件后，会将Journal文件名更新为"processed_" + Journal文件名。

```java
while (true) {
    if (hasEnoughSpaceForNextJournal()) {
        synchronized (this) {
            if (!hasEnoughSpaceForNextJournal()) {
                continue;
            }
            String nextJnlName = getNextJournalName();
            startJournalProcessing(nextJnlName);
        }
    }
    Thread.sleep(100);
}
```

## 4.2 Fdb Server端代码示例
Fdb的后台处理线程负责将Journal文件转换为SSTable文件，并将其分布到不同的服务器上。后台处理线程的代码比较长，我们只展示其中重要的功能。

后台处理线程的主要工作流程如下：

1. 检查是否有新的Journal文件需要处理；

2. 将新的Journal文件的内容解析出来，得到对应的RowKey范围、列族信息及其对应的写操作类型等；

3. 根据不同写操作类型，决定将该条目写入到内存还是SSTable文件中；

4. 如果写入SSTable文件中，则将其先写入到内存的Memtable中，并将其按照Key-Value的方式排序，再写入到硬盘中的tmp文件中；

5. 当tmp文件中的数据累计到一定数量后，便将tmp文件合并为新的SSTable文件；

6. 将新的SSTable文件移动到HDFS上的指定目录中，然后删除原有的SSTable文件；

7. 在处理完Journal文件后，如果当前没有更多的Journal文件需要处理，则休眠一段时间后再次检查是否有新的Journal文件需要处理。

```java
void run() {
  boolean running = true;

  while (running) {
    synchronized (this) {
      if (!isThereAnyPendingJournal()) {
        waitForNextCheckInterval();
        continue;
      }

      File jnlFile = popNextJournal();
      processJournal(jnlFile);
    }
  }
}

private void processJournal(File journalFile) {
  InputStream inputStream = null;
  OutputStream outputStream = null;
  
  try {
    LOGGER.info("Processing journal:" + journalFile.getName());
    
    inputStream = new FileInputStream(journalFile);
    BufferedReader reader = new BufferedReader(new InputStreamReader(inputStream));
    BufferedWriter writer = null;

    try {
      outputStream = new FileOutputStream(temporarySSTablePath);
      
      writer = new BufferedWriter(new OutputStreamWriter(outputStream));
      
      String line = "";
      while ((line = reader.readLine())!= null) {
        
        if (line.isEmpty() || line.charAt(0) == '#') {
          continue;
        }
        
        byte[] bytes = Bytes.fromHex(line.split("\\|")[0]);
        KeyValue kv = parseKVFromLine(line);

        if (kv.getType() == OperationType.DELETE) {
          
          if (memtable.contains(kv)) {
            memtable.delete(kv);
          }
          writeToTmp(writer, kv);
          
        } else if (kv.getType() == OperationType.PUT) {
        
          memtable.put(kv);
          
          if (memtable.getSizeInBytes() >= MEMORY_THRESHOLD) {
            
            flushMemoryTable(writer, memtable);
            
          } else if (memtable.getCount() % ROWS_PER_FILE == 0) {
            
            flushMemoryTableToFile(writer, memtable);
            
          }
          
        }
      }
      
      if (memtable.isNotEmpty()) {
        flushMemoryTableToFile(writer, memtable);
      }
      
      tmpToSstable();
      appendManifestToMetaTable();
      deleteJournalFile();
      
    } finally {
      closeIOResources(reader, writer, inputStream, outputStream);
    }
    
  } catch (IOException ioe) {
    LOGGER.error("Error processing journal:" + journalFile.getName(), ioe);
    reportError(ioe);
    stopAsyncProcessing();
  }
}

private static KeyValue parseKVFromLine(String line) {
  String[] tokens = line.split("\\|");
  byte[] row = Bytes.fromHex(tokens[0]);
  byte[] cf = Bytes.fromString(tokens[1]);
  byte[] cq = Bytes.fromHex(tokens[2]);
  long ts = Long.parseLong(tokens[3]);
  long version = Long.parseLong(tokens[4]);
  byte typeByte = Byte.parseByte(tokens[5].trim());
  Type type = Type.codeToType(typeByte);
  long valueLength = Long.parseLong(tokens[6]);
  byte[] value = Bytes.fromHex(tokens[7]);
  
  return new KeyValue(row, cf, cq, ts, type, valueLength, value, version);
}

private static void appendManifestToMetaTable() throws IOException {
  Manifest manifest = ManifestParser.read(manifestFilePath);
  Set<String> sstsInManifest = manifest.getSSTables().stream()
                     .map(s -> s.getName()).collect(Collectors.toSet());

  for (File file : metaTableDir.listFiles()) {
    if (file.isFile() && file.getName().endsWith(".json")) {
      Metadata metadata = MetadataManager.deserialize(file);
      if (metadata!= null && metadata.getColumnFamilyNames()!= null) {
        Set<String> cfsInFile = Arrays.stream(metadata.getColumnFamilyNames())
                             .filter(Objects::nonNull).collect(Collectors.toSet());
        if (!cfsInFile.equals(sstsInManifest)) {
          recreateMetadataJson(file);
        }
      }
    }
  }
}
```

# 5.未来发展趋势与挑战
Fdb目前的技术和架构已经满足了Facebook当前的业务需求。但是，Fdb还有很多未来需要进一步优化的地方。在未来的发展中，Fdb还可以考虑以下方面的改进：

1. 接口改进：当前Fdb的接口定义过于简单，尤其是在对数据进行读写时，只提供了单个get()和put()方法，对于复杂的查询条件和过滤条件支持较弱。为了更好地支持复杂的查询操作，Fdb可以提供更高级的API，例如SCAN()方法或SELECT()方法。

2. 查询优化：Fdb的查询优化器目前还处于初级阶段，仅支持最简单的搜索条件，例如根据RowKey查找、根据列簇和列名查找。为了支持更复杂的查询条件，Fdb可以引入新的查询优化器模块，比如基于表达式树的查询优化器或基于统计信息的查询优化器。

3. 更细粒度的冗余：虽然Fdb的冗余机制可以提供高可用性和容错能力，但其冗余级别较低，可能无法满足某些场景下的需求。比如，当一个区域的结点发生故障时，Fdb仍然可以继续提供服务，但可能会导致延迟增加，这时可以考虑增加一级冗余，即在不同结点部署额外的备份，以提升访问的可靠性。

4. 自动运维工具：为了提升Fdb的易用性，Fdb可以提供一套自动运维工具，帮助管理员快速设置集群、扩容节点、迁移数据等，并监控集群状态，发现异常并及时报警。

5. 系统改进：Fdb目前的系统结构较为简单，存在单点问题。为了提升系统的扩展性和弹性，Fdb可以考虑使用更加成熟的微服务架构或容器技术，并结合云平台和自动化运维工具进行自动化部署、横向扩展等。另外，Fdb还可以进一步优化对磁盘、网络的使用，从而实现更好的性能和资源利用率。

# 6.附录常见问题与解答
## Q：为什么要设计Fdb？
A：Facebook拥有庞大的用户群体，每天都产生海量的数据，而传统的数据库系统无法满足其海量数据的处理要求。因此，Facebook开发了Fdb——一个可伸缩、高性能的分布式数据库系统，来支持其业务的发展。
## Q：什么样的数据可以适合使用Fdb？
A：Fdb可以适用于任意数据模型，而且可以对数据进行有效的索引，因此，任何需要频繁查找或查询的数据都可以被适当地存储在Fdb中。Facebook通常会选取有关联关系的数据存储在Fdb中，比如评论、页面访问数据等，因为这样可以方便快速找到相关的信息。
## Q：Fdb采用哪种存储引擎？
A：Fdb采用了HBase作为其存储引擎。HBase是一个开源的分布式 NoSQL 数据库，它通过Google的Bigtable论文中描述的 Bigtable 架构设计而来。Bigtable 是 Google 的分布式存储系统，能够提供高可用性、高性能和可扩展性。HBase 可以运行在 Apache Hadoop 上，其提供了与 Hadoop MapReduce 框架完全兼容的 API 。此外，Facebook还开发了自己的自定义版本的 HDFS ，以支持其特定需求，比如控制副本数和数据位置分布。
## Q：Fdb的性能如何？
A：Fdb的性能非常好，尤其是在大数据量的情况下。Fdb的所有存储组件都设计成可水平扩展的，因此，可以轻松应对数据量的增长。Facebook的测试表明，Fdb可以存储百万级数据，并在毫秒级的时间内进行实时查询。此外，Fdb还具备低延迟、高吞吐量和自动故障切换能力，能够在不同区域之间的结点发生故障时，仍然提供正常的服务。
## Q：Fdb的开发为什么需要分布式锁机制？
A：分布式锁机制是确保并发访问的重要手段。虽然MySQL和PostgreSQL等传统数据库都提供对事务的原生支持，但是，它们不能保证跨节点的事务的一致性。因此，为了保证跨越多个结点的事务的一致性，Fdb的设计者开发了分布式锁机制。分布式锁机制可以让多个事务按照顺序执行，并且在冲突的时候会等待其他事务释放锁。这样就可以确保跨越多个结点的事务的一致性。