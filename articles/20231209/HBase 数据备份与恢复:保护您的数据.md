                 

# 1.背景介绍

HBase是一个分布式、可扩展、高性能的列式存储系统，基于Google的Bigtable设计。它是Hadoop生态系统的一部分，用于存储大规模的结构化数据。HBase提供了高可用性、高可扩展性和高性能的数据存储解决方案，适用于实时数据访问和分析场景。

数据备份和恢复是保护数据的关键环节之一，尤其是在大规模分布式系统中，数据的丢失和损坏可能导致严重后果。在HBase中，数据备份和恢复是通过HBase的数据复制和恢复机制实现的。本文将详细介绍HBase数据备份与恢复的核心概念、算法原理、具体操作步骤以及数学模型公式。

# 2.核心概念与联系

在HBase中，数据备份和恢复的核心概念包括：

1. HRegionServer：HBase的数据存储和处理单元，负责存储和管理HBase表的数据。
2. HRegion：HRegionServer内部的数据存储单元，负责存储HBase表的一部分数据。
3. HStore：HRegion内部的数据存储单元，负责存储HBase表的一列数据。
4. HFile：HStore内部的数据存储单元，负责存储HBase表的一行数据。
5. Snapshot：HBase的快照功能，用于保存HBase表的当前状态。
6. Compaction：HBase的压缩功能，用于合并多个HFile，减少存储空间和提高查询性能。

这些概念之间的联系如下：

- HRegionServer负责存储和管理HRegion。
- HRegion负责存储和管理HStore。
- HStore负责存储和管理HFile。
- Snapshot用于保存HBase表的当前状态，可以通过HRegionServer和HRegion访问。
- Compaction用于合并多个HFile，可以通过HRegion和HStore访问。

# 3.核心算法原理和具体操作步骤以及数学模型公式详细讲解

## 3.1 数据备份

HBase数据备份的核心算法原理是通过HRegionServer和HRegion的数据复制机制实现的。具体操作步骤如下：

1. 在HBase集群中创建一个新的HRegionServer实例，并将其配置为备份服务器。
2. 在新的HRegionServer实例上创建一个新的HRegion，并将其配置为备份区域。
3. 在原始HRegionServer实例上创建一个新的HRegion，并将其配置为主要区域。
4. 使用HBase的复制API，将原始HRegion的数据复制到新的HRegion。
5. 确保新的HRegionServer实例和新的HRegion的数据复制完成后，进行验证和测试。

数学模型公式详细讲解：

- 数据备份的复制因子（replication factor）：表示原始数据在备份区域中的复制次数。例如，如果复制因子为3，则原始数据在备份区域中会有三个副本。
- 数据备份的延迟（latency）：表示数据复制操作所需的时间。

## 3.2 数据恢复

HBase数据恢复的核心算法原理是通过HRegionServer和HRegion的数据恢复机制实现的。具体操作步骤如下：

1. 在HBase集群中创建一个新的HRegionServer实例，并将其配置为恢复服务器。
2. 在新的HRegionServer实例上创建一个新的HRegion，并将其配置为恢复区域。
3. 在原始HRegionServer实例上创建一个新的HRegion，并将其配置为主要区域。
4. 使用HBase的恢复API，将原始HRegion的数据恢复到新的HRegion。
5. 确保新的HRegionServer实例和新的HRegion的数据恢复完成后，进行验证和测试。

数学模型公式详细讲解：

- 数据恢复的恢复因子（recovery factor）：表示原始数据在恢复区域中的恢复次数。例如，如果恢复因子为3，则原始数据在恢复区域中会有三个副本。
- 数据恢复的延迟（latency）：表示数据恢复操作所需的时间。

# 4.具体代码实例和详细解释说明

在本节中，我们将通过一个具体的代码实例来详细解释HBase数据备份和恢复的具体操作步骤。

## 4.1 数据备份

```java
import org.apache.hadoop.hbase.HBaseConfiguration;
import org.apache.hadoop.hbase.client.Connection;
import org.apache.hadoop.hbase.client.ConnectionFactory;
import org.apache.hadoop.hbase.client.HTable;
import org.apache.hadoop.hbase.client.HTableDescriptor;
import org.apache.hadoop.hbase.coprocessor.BaseRegionObserver;
import org.apache.hadoop.hbase.coprocessor.ObserverContext;
import org.apache.hadoop.hbase.coprocessor.RegionCoprocessorEnvironment;
import org.apache.hadoop.hbase.regionserver.HRegionInfo;
import org.apache.hadoop.hbase.util.Bytes;
import org.apache.hadoop.hbase.client.Put;
import org.apache.hadoop.hbase.client.Scan;
import org.apache.hadoop.hbase.client.Result;
import org.apache.hadoop.hbase.client.ResultScanner;
import org.apache.hadoop.hbase.client.Get;
import org.apache.hadoop.hbase.client.Delete;
import org.apache.hadoop.hbase.client.Durability;
import org.apache.hadoop.hbase.HColumnDescriptor;
import org.apache.hadoop.hbase.TableName;
import org.apache.hadoop.hbase.coprocessor.RegionEvent;
import org.apache.hadoop.hbase.coprocessor.RegionObserverContext;
import org.apache.hadoop.hbase.regionserver.HRegion;
import org.apache.hadoop.hbase.regionserver.HRegionInfo;
import org.apache.hadoop.hbase.regionserver.HStore;
import org.apache.hadoop.hbase.regionserver.HFile;
import org.apache.hadoop.hbase.regionserver.HFileUtil;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionType;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionJob;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionJobStatus;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionObserver;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequest;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestBuilder;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type.Type.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type;
import org.apache.hadoop.hbase.regionserver.compaction.CompactionRequestType.Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type. Type