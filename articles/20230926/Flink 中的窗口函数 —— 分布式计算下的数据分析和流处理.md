
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Apache Flink 是 Apache 软件基金会孵化的一款开源分布式流处理平台，它基于Google 的数据流模型构建，具有低延迟、高吞吐量、强一致性等优点。它提供了流处理、批处理、机器学习、图计算等多种能力，能够满足各种业务场景的需求。

在日常工作中，由于对实时数据的快速处理和分析要求越来越高，所以分布式系统越来越受到重视，很多公司都开始选择采用 Flink 来作为实时数据分析引擎。如今越来越多的公司应用 Flink 来进行复杂事件处理、运营数据分析、金融交易监控等领域的实时数据处理。

本文从窗口函数（Window Function）入手，详细剖析了 Flink 中窗口函数的实现原理及其应用。

# 2.基本概念术语说明
## 概念定义及相关名词
在传统数据库系统中，窗口函数（Window Function）是用来计算某些统计指标（比如，平均值、最大值、最小值等）的函数。它的输入是一个表或查询结果集，输出也是一个表。窗口函数一般都会带有一个 OVER clause（窗口句柄），用来描述窗口的大小、滑动方式、聚合方式、排序方式等。窗口函数在 SQL 和编程语言中都有相应的语法定义。

而在 Flink 中，窗口函数也是一种比较重要的功能。因为 Flink 可以把实时流式数据处理任务拆分成多个步骤并行执行，每个步骤负责不同时间段的数据处理，所以需要保证窗口函数能够将数据划分成多个小的时间片段，然后各个节点分别处理这些时间片段的数据，最后再汇总得到最终的结果。

所以，窗口函数在 Flink 中与其他运算符有着一些区别。首先，窗口函数不是单独的算子，而是属于 Flink 查询执行框架中的一种窗口算子类型。其次，窗口函数不仅可以用于聚合操作，还可以用于过滤、排序、去重等其他操作。另外，窗口函数既可以作用在 Table API，也可以作用在 DataStream API 上。

那么什么是窗口？窗口就是一个连续的时间段。通常情况下，窗口通常被称为 Tumbling Window 或 Sliding Window。Tumbling Window 又称为滚动窗口，它指定了一个固定的时间长度，例如每隔5秒钟产生一次新窗口；Sliding Window 则是在 Tumbling Window 的基础上增加了滑动步长，即每次移动一步，形成一个新的窗口。窗口的长度与时间间隔由用户指定的参数决定。

假设有一个实时流式数据源，他按照时间顺序生成数值序列，这个数据流可以通过窗口函数转换成为多个时间段的集合，每个时间段里的数据流可以用相同的聚合函数计算出多个统计指标。这里，窗口函数就起到了分组和聚合的作用，它将流式数据根据时间划分为多个数据子集，然后针对每个数据子集进行聚合和计算，最后将结果合并输出。

## 时态性
对于窗口函数来说，窗口操作的时间是在数据流上的，而不是应用程序内的系统时钟。这意味着窗口函数在计算的时候，不考虑具体的时间，只要数据进入窗口，就会触发计算。所以窗口的长度以及滑动步长都是可以变化的。窗口开始结束的时间也不是硬编码的，它们依赖于时间驱动的窗口机制。这样做的好处是不需要考虑窗口的准确关闭时间，而且窗口的打开和关闭是自动触发的，无需用户指定。

## 操作类型
窗口函数主要用于以下几类操作：
- 聚合操作（Aggregate Function）：包括 count、sum、avg、min、max、distinct count 等。通过对窗口内的数据进行聚合计算，得到窗口内的数据统计信息。
- 求和操作（Summarization Operation）：包括求和、平均值、计数、和、方差、标准差等。通过对窗口内的数据进行汇总，得到整个数据流的统计信息。
- 排队操作（Queueing Operations）：包括first、last、lead、lag、rank、row_number、percent_rank等。通过对窗口内的数据进行排序，获取排序后的位置信息。
- 窗口内计算（Window Aggregation）：包括 cumulate sum、cumulate avg、expression based window function 等。通过对窗口内的数据进行逐条计算，得到窗口内的复杂统计信息。

其中，聚合操作、求和操作、排队操作属于窗口内计算，其他三个属于窗口外计算。

# 3.核心算法原理和具体操作步骤以及数学公式讲解
## 数据流模型
当一个数据流经过 Flink 后，会先经历过 Source Operator、Operator、Sink Operator 三层架构。源算子接收外部数据源的数据，并将其输入到第一个算子 Operator1 中进行处理。

Operator1 对输入的数据进行处理，同时，它还会向下游传递相同的数据，即 Operator2 和 Sink Operator 将获得相同的数据。依此类推，直到 Sink Operator 将数据输出到外部系统，这是一个持续的过程。



## 窗口函数定义
### 基本概念
窗口函数在 SQL 中定义的是 OVER 关键字，其基本逻辑是分组计算，即对输入数据按照窗口划分，然后在每个窗口内进行聚合计算。

而在 Flink 中，窗口函数的定义更加灵活。它可以作用在任意类型的输入上，并且支持不同的窗口类型，从而实现复杂的窗口计算。

因此，窗口函数除了具备 SQL 中的 OVER 关键字之外，还包含以下几个方面：
1. 函数：窗口函数不仅可以作用在 Table API 或者 DataStream API 上，还可以作用在 DataSet、CoGroupFunction、KeyedProcessFunction 等 Flink Java API 对象上。
2. 窗口类型：窗口函数可以实现不同的窗口类型，包括 Tumbling Windows、Sliding Windows、Session Windows、Global Windows 等。
3. 计算方式：窗口函数可以实现基于窗口内数据的计算和排序，也可以实现基于窗口之间的关系的计算，甚至可以实现不同窗口之间的数据交换和关联。

窗口函数的定义语法如下所示：
```sql
<agg> OVER (PARTITION BY <key>) [ORDER BY <order>] [ROWS|RANGE] BETWEEN <start> AND <end>;
```

以上语法包括：
- `<agg>`：聚合函数。包括 count、sum、avg、min、max、distinct count 等。
- `PARTITION BY <key>`：指定数据分组的方式。如果没有指定，则默认将所有数据放入同一组。
- `[ORDER BY <order>]`：窗口内数据排序的字段。如果没有指定，则默认升序排列。
- `[ROWS|RANGE]`：窗口分割方式。ROWS 表示按固定行数分割窗口，RANGE 表示按固定时间范围分割窗口。
- `BETWEEN <start> AND <end>`：指定窗口的开始和结束位置。

除此之外，窗口函数还可以使用增量聚合（Incremental Aggregation）的方法，提升性能。增量聚合的目的是将窗口的计算分解成多个小任务，只有当任务的所有输入数据均已准备好时，才执行计算任务。增量聚合模式下，窗口函数的结果可能不会像全量聚合模式下一样精确，但可以减少不必要的计算资源开销。

### 支持的数据类型
窗口函数支持以下的数据类型：
- RowData：代表数据记录，是一个二元组数组结构。RowData 可以保存结构化和非结构化的数据，也可以用于嵌套的 RowType。
- TupleX：代表元组结构的数据。
- POJO：代表 java bean 对象结构的数据。

窗口函数目前不支持 Map、List、Array 等复杂类型的数据。

## Tumbling Window（滚动窗口）
Tumbling Window 是最简单的窗口类型，它将数据流划分为固定长度的窗口，并一次性计算窗口中的所有数据。窗口划分完成后，窗口中的数据可以被丢弃，而不会留存于下一个窗口。


### Count Over Tumbling Window（计数窗口）
Count Over Tumbling Window 可以统计某个时间窗口内的数据数量。语法如下：
```sql
COUNT(*) OVER(
  PARTITION BY <key>
  ORDER BY <time_attr> ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW);
```

- `PARTITION BY <key>`：指定数据分组的方式。如果没有指定，则默认将所有数据放入同一组。
- `ORDER BY <time_attr>`：指定时间属性，用于排序数据。如果没有指定，则默认按数据插入时间排序。
- `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`：表示窗口划分方式。UNBOUNDED PRECEDING 表示前面的所有时间数据，CURRENT ROW 表示当前时间数据。

### Sum Over Tumbling Window（求和窗口）
Sum Over Tumbling Window 可以统计某个时间窗口内的数据总和。语法如下：
```sql
SUM(<expr>) OVER(
  PARTITION BY <key>
  ORDER BY <time_attr> ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW);
```

- `SUM(<expr>)`: 指定用于求和的值表达式。
- `PARTITION BY <key>`：指定数据分组的方式。如果没有指定，则默认将所有数据放入同一组。
- `ORDER BY <time_attr>`：指定时间属性，用于排序数据。如果没有指定，则默认按数据插入时间排序。
- `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`：表示窗口划分方式。UNBOUNDED PRECEDING 表示前面的所有时间数据，CURRENT ROW 表示当前时间数据。

### Max Over Tumbling Window（最大值窗口）
Max Over Tumbling Window 可以统计某个时间窗口内的最大值。语法如下：
```sql
MAX(<expr>) OVER(
  PARTITION BY <key>
  ORDER BY <time_attr> ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW);
```

- `MAX(<expr>)`: 指定用于求最大值的的值表达式。
- `PARTITION BY <key>`：指定数据分组的方式。如果没有指定，则默认将所有数据放入同一组。
- `ORDER BY <time_attr>`：指定时间属性，用于排序数据。如果没有指定，则默认按数据插入时间排序。
- `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`：表示窗口划分方式。UNBOUNDED PRECEDING 表示前面的所有时间数据，CURRENT ROW 表示当前时间数据。

### Min Over Tumbling Window（最小值窗口）
Min Over Tumbling Window 可以统计某个时间窗口内的最小值。语法如下：
```sql
MIN(<expr>) OVER(
  PARTITION BY <key>
  ORDER BY <time_attr> ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW);
```

- `MIN(<expr>)`: 指定用于求最小值的的值表达式。
- `PARTITION BY <key>`：指定数据分组的方式。如果没有指定，则默认将所有数据放入同一组。
- `ORDER BY <time_attr>`：指定时间属性，用于排序数据。如果没有指定，则默认按数据插入时间排序。
- `ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW`：表示窗口划分方式。UNBOUNDED PRECEDING 表示前面的所有时间数据，CURRENT ROW 表示当前时间数据。

## Sliding Window（滑动窗口）
Sliding Window 是 Tumbling Window 的升级版本，它在 Tumbling Window 的基础上增加了滑动步长，使得窗口的大小随着时间推移而变短。在每个窗口内，可以看到更多的数据。


### Count Over Sliding Window（计数窗口）
Count Over Sliding Window 可以统计某个时间窗口内的数据数量。语法如下：
```sql
COUNT(*) OVER(
  PARTITION BY <key>
  ORDER BY <time_attr> RANGE BETWEEN INTERVAL 'n' SECOND PRECEDING AND CURRENT ROW);
```

- `PARTITION BY <key>`：指定数据分组的方式。如果没有指定，则默认将所有数据放入同一组。
- `ORDER BY <time_attr>`：指定时间属性，用于排序数据。如果没有指定，则默认按数据插入时间排序。
- `RANGE BETWEEN INTERVAL 'n' SECOND PRECEDING AND CURRENT ROW`：表示窗口划分方式。INTERVAL ‘n’ SECOND PRECEDING 表示窗口大小为 n 秒，TIMESTAMP 'yyyy-MM-dd HH:mm:ss.SSS' 之前的数据。CURRENT ROW 表示当前时间数据。

### Sum Over Sliding Window（求和窗口）
Sum Over Sliding Window 可以统计某个时间窗口内的数据总和。语法如下：
```sql
SUM(<expr>) OVER(
  PARTITION BY <key>
  ORDER BY <time_attr> RANGE BETWEEN INTERVAL 'n' SECOND PRECEDING AND CURRENT ROW);
```

- `SUM(<expr>)`: 指定用于求和的值表达式。
- `PARTITION BY <key>`：指定数据分组的方式。如果没有指定，则默认将所有数据放入同一组。
- `ORDER BY <time_attr>`：指定时间属性，用于排序数据。如果没有指定，则默认按数据插入时间排序。
- `RANGE BETWEEN INTERVAL 'n' SECOND PRECEDING AND CURRENT ROW`：表示窗口划分方式。INTERVAL ‘n’ SECOND PRECEDING 表示窗口大小为 n 秒，TIMESTAMP 'yyyy-MM-dd HH:mm:ss.SSS' 之前的数据。CURRENT ROW 表示当前时间数据。

### Max Over Sliding Window（最大值窗口）
Max Over Sliding Window 可以统计某个时间窗口内的最大值。语法如下：
```sql
MAX(<expr>) OVER(
  PARTITION BY <key>
  ORDER BY <time_attr> RANGE BETWEEN INTERVAL 'n' SECOND PRECEDING AND CURRENT ROW);
```

- `MAX(<expr>)`: 指定用于求最大值的的值表达式。
- `PARTITION BY <key>`：指定数据分组的方式。如果没有指定，则默认将所有数据放入同一组。
- `ORDER BY <time_attr>`：指定时间属性，用于排序数据。如果没有指定，则默认按数据插入时间排序。
- `RANGE BETWEEN INTERVAL 'n' SECOND PRECEDING AND CURRENT ROW`：表示窗口划分方式。INTERVAL ‘n’ SECOND PRECEDING 表示窗口大小为 n 秒，TIMESTAMP 'yyyy-MM-dd HH:mm:ss.SSS' 之前的数据。CURRENT ROW 表示当前时间数据。

### Min Over Sliding Window（最小值窗口）
Min Over Sliding Window 可以统计某个时间窗口内的最小值。语法如下：
```sql
MIN(<expr>) OVER(
  PARTITION BY <key>
  ORDER BY <time_attr> RANGE BETWEEN INTERVAL 'n' SECOND PRECEDING AND CURRENT ROW);
```

- `MIN(<expr>)`: 指定用于求最小值的的值表达式。
- `PARTITION BY <key>`：指定数据分组的方式。如果没有指定，则默认将所有数据放入同一组。
- `ORDER BY <time_attr>`：指定时间属性，用于排序数据。如果没有指定，则默认按数据插入时间排序。
- `RANGE BETWEEN INTERVAL 'n' SECOND PRECEDING AND CURRENT ROW`：表示窗口划分方式。INTERVAL ‘n’ SECOND PRECEDING 表示窗口大小为 n 秒，TIMESTAMP 'yyyy-MM-dd HH:mm:ss.SSS' 之前的数据。CURRENT ROW 表示当前时间数据。

## Session Window（会话窗口）
Session Window 根据数据进入的时间和离开的时间进行分组，每个会话是一个连续的时间窗口。在会话内，可以看到全部的数据，而在会话间切换，窗口的数据就会被清空。


### Count Over Session Window（计数窗口）
Count Over Session Window 可以统计某个时间窗口内的数据数量。语法如下：
```sql
COUNT(*) OVER(
  PARTITION BY <key>
  ORDER BY <session_start>, <event_time>);
```

- `PARTITION BY <key>`：指定数据分组的方式。如果没有指定，则默认将所有数据放入同一组。
- `ORDER BY <session_start>, <event_time>`：指定两个属性，用于排序数据。第一项表示会话的开始时间，第二项表示事件的时间戳。

### Sum Over Session Window（求和窗口）
Sum Over Session Window 可以统计某个时间窗口内的数据总和。语法如下：
```sql
SUM(<expr>) OVER(
  PARTITION BY <key>
  ORDER BY <session_start>, <event_time>);
```

- `SUM(<expr>)`: 指定用于求和的值表达式。
- `PARTITION BY <key>`：指定数据分组的方式。如果没有指定，则默认将所有数据放入同一组。
- `ORDER BY <session_start>, <event_time>`：指定两个属性，用于排序数据。第一项表示会话的开始时间，第二项表示事件的时间戳。

### Max Over Session Window（最大值窗口）
Max Over Session Window 可以统计某个时间窗口内的最大值。语法如下：
```sql
MAX(<expr>) OVER(
  PARTITION BY <key>
  ORDER BY <session_start>, <event_time>);
```

- `MAX(<expr>)`: 指定用于求最大值的的值表达式。
- `PARTITION BY <key>`：指定数据分组的方式。如果没有指定，则默认将所有数据放入同一组。
- `ORDER BY <session_start>, <event_time>`：指定两个属性，用于排序数据。第一项表示会话的开始时间，第二项表示事件的时间戳。

### Min Over Session Window（最小值窗口）
Min Over Session Window 可以统计某个时间窗口内的最小值。语法如下：
```sql
MIN(<expr>) OVER(
  PARTITION BY <key>
  ORDER BY <session_start>, <event_time>);
```

- `MIN(<expr>)`: 指定用于求最小值的的值表达式。
- `PARTITION BY <key>`：指定数据分组的方式。如果没有指定，则默认将所有数据放入同一组。
- `ORDER BY <session_start>, <event_time>`：指定两个属性，用于排序数据。第一项表示会话的开始时间，第二项表示事件的时间戳。

## Global Window（全局窗口）
Global Window 是 Flink 提供的一个特殊窗口，它用于处理整个数据流，也就是说，它没有时间属性。该窗口始终存在，始终不会结束，除非任务被取消或失败。

Global Window 主要用于不需要对齐或窗口操作的数据聚合，例如，在连接或协同过滤时，就可以直接使用 Global Window 。


### Count Over Global Window（计数窗口）
Count Over Global Window 可以统计整个数据流的数据数量。语法如下：
```sql
COUNT(*) OVER();
```

### Sum Over Global Window（求和窗口）
Sum Over Global Window 可以统计整个数据流的数据总和。语法如下：
```sql
SUM(<expr>) OVER();
```

- `SUM(<expr>)`: 指定用于求和的值表达式。

### Max Over Global Window（最大值窗口）
Max Over Global Window 可以统计整个数据流的最大值。语法如下：
```sql
MAX(<expr>) OVER();
```

- `MAX(<expr>)`: 指定用于求最大值的的值表达式。

### Min Over Global Window（最小值窗口）
Min Over Global Window 可以统计整个数据流的最小值。语法如下：
```sql
MIN(<expr>) OVER();
```

- `MIN(<expr>)`: 指定用于求最小值的的值表达式。

## Partition By Key（分组）
Partition By Key 在所有的窗口函数中都能找到，它允许用户指定数据如何分组。如果没有指定，则默认将所有数据放入同一组。

举例来说，如果数据源的数据格式为 `(name string, age int)`，则可以使用 `PARTITION BY name` 来指定数据按姓氏分组。

# 4.具体代码实例和解释说明
## 计数窗口案例
```java
import org.apache.flink.streaming.api.datastream.*;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import static org.apache.flink.api.common.functions.MapFunction.*;

public class CountWindowExample {

    public static void main(String[] args) throws Exception {

        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        SingleOutputStreamOperator<Tuple2<String, Integer>> input =
            env.fromElements("A", "B", "C")
               .map(new Tuple2MapFunction());

        DataStream<Integer> result = input
           .countWindow(3, 1) // 3 个元素的窗口，滑动 1 个元素
           .apply(CountWindowFunction.getCountWindowFunction()) // 使用 CountWindowFunction
            ;

        result.print();
        env.execute("Count Window Example");
    }


    private static class Tuple2MapFunction implements MapFunction<String, Tuple2<String, Integer>> {
        @Override
        public Tuple2<String, Integer> map(String value) throws Exception {
            return new Tuple2<>(value, null);
        }
    }
}

class CountWindowFunction extends WindowFunction<Tuple2<String, Integer>, Integer, String, TimeWindow> {

    @Override
    public void apply(String key, TimeWindow timeWindow, Iterable<Tuple2<String, Integer>> input, Collector<Integer> out) throws Exception {
        int count = 0;
        for (Tuple2 tuple : input) {
            if (tuple!= null &&!tuple.getField(1).equals(-1)) {
                count++;
            }
        }
        out.collect(count);
    }
}
```

输出：
```
15:14:22.827 [main] INFO org.apache.flink.streaming.api.operators.StreamTask - Starting task and initializing inputs. Task Name: Count Window Example.
15:14:22.840 [main] INFO org.apache.flink.runtime.jobgraph.IntermediateResult[uid=3d1b4d6f072d01a3cc9e6f567fb7c1a6]-Source: Custom Source -> (1/1) -> Shuffle -> (1/2) -> Hash Partition -> (1/3) -> Count Window -> Print to Std. Out (1/1) (a9bb69a89fc8fb4b67c5f2636a777c5d) switched from CREATED to RUNNING. (JobID: ccf2c6f0a0e5a8e0df1cb6e5c734b8c3)
15:14:22.840 [main] INFO org.apache.flink.streaming.runtime.tasks.OneInputStreamTask - OneInput stream task for Count Window Example (a9bb69a89fc8fb4b67c5f2636a777c5d) was created
15:14:22.841 [main] INFO org.apache.flink.streaming.api.operators.StreamTask - Initializing operator chain. TaskName: Count Window Example.
15:14:22.841 [main] INFO org.apache.flink.streaming.api.operators.StreamTask - Created type info of operator: DataStream Count Window -> Type: Output Type: INT32
15:14:22.842 [main] INFO org.apache.flink.runtime.executiongraph.ExecutionGraphBuilder$DeploymentInstance-PartitionNode-Source: Custom Source -> (1/1) -> Shuffle -> (1/2) -> Hash Partition -> (1/3) <- PartitionCoordinator <- JobManager
15:14:22.842 [main] INFO org.apache.flink.runtime.executiongraph.ExecutionGraphBuilder$DeploymentInstance-ShuffleChain-Custom Source -> (1/1) -> Shuffle -> (1/2) <- ResultPartitionConsumers -> (1/3) <- InputGateSelector -> (1/1) <- JobManager
15:14:22.843 [main] INFO org.apache.flink.runtime.jobgraph.IntermediateResult[uid=77a18a9e0ef1c7ce867f84bd9c52de14]-Source: Custom Source -> (2/2) -> Shuffle -> (2/2) -> Partition Consumer (1/1) -> (1/1) -> File Source -> Unmanaged Memory Source -> (1/2) -> Iterator (1/1) -> Local Exchange (1/2) -> Forward (1/2) -> (1/2) -> Build Locally & Send -> (1/2) -> Co Location (1/2) -> Broadcast Variable Receiver -> (1/2) -> Merge Buckets (1/2) -> (1/2) -> Rebalance Shuffle (1/2) -> (1/2) -> (1/2) -> Other Parallelism -> Shuffle -> Sink: Unmanaged Memory Sink -> (1/1) -> Output (1/1) -> (1/1) -> Sink: Unmanaged Memory Sink -> Output Selector (1/1) -> Collector (1/1) -> (1/1) -> Count Window -> Print to Std. Out (2/1) (1d3880a73144aa7bf7c6c53f4c0e92c8) switched from CREATED to RUNNING. (JobID: ccf2c6f0a0e5a8e0df1cb6e5c734b8c3)
15:14:22.843 [main] INFO org.apache.flink.runtime.executiongraph.ExecutionGraphBuilder$DeploymentInstance-PartitionNode-Source: Custom Source -> (2/2) -> Shuffle -> (2/2) -> Partition Consumer (1/1) -> (1/1) -> File Source -> Unmanaged Memory Source -> (1/2) -> Iterator (1/1) -> Local Exchange (1/2) -> Forward (1/2) -> (1/2) -> Build Locally & Send -> (1/2) -> Co Location (1/2) -> Broadcast Variable Receiver -> (1/2) -> Merge Buckets (1/2) -> (1/2) -> Rebalance Shuffle (1/2) -> (1/2) -> (1/2) -> Other Parallelism -> Shuffle -> Sink: Unmanaged Memory Sink -> (1/1) -> Output (1/1) -> (1/1) -> Sink: Unmanaged Memory Sink -> Output Selector (1/1) -> Collector (1/1) -> (1/1) <- PartitionCoordinator <- JobManager
15:14:22.843 [main] INFO org.apache.flink.runtime.executiongraph.ExecutionGraphBuilder$DeploymentInstance-ShuffleChain-Custom Source -> (2/2) -> Shuffle -> (2/2) <- ResultPartitionConsumers -> (2/2) <- InputGateSelector -> (1/1) <- JobManager
15:14:22.844 [main] INFO org.apache.flink.runtime.taskmanager.TaskManager - Deploying job Vertex Count Window Example(a9bb69a89fc8fb4b67c5f2636a777c5d) (1/1) (Vertex: Count Window Example)#0 (a9bb69a89fc8fb4b67c5f2636a777c5d) (Attempt: 0)
15:14:22.844 [main] INFO org.apache.flink.runtime.taskexecutor.SlotProviderImpl - Number of available slots changed from [1] to [4].
15:14:22.844 [main] INFO org.apache.flink.runtime.taskmanager.TaskManagerRunner - The slot number is updated to the minimum of max slots per node [4], and total tasks [2] which needs to run in parallel. Final slot number is [4].
15:14:22.845 [main] INFO org.apache.flink.runtime.resourcemanager.slotmanager.SlotManagerImpl - Submitting slot request for [ResourceProfile{cpuCores=1, heapMemoryMB=1024, directMemoryMB=0}] at ResourceManager at localhost. Requested number of free slots is [4]. Currently running jobs: []. Pending requests: []
15:14:22.845 [main] INFO org.apache.flink.runtime.resourcemanager.slotmanager.Scheduler - Received resource offers: [org.apache.flink.runtime.clusterframework.types.ResourceOffer@3cf8e21c]
15:14:22.846 [main] INFO org.apache.flink.runtime.scheduler.SchedulerBase - Resource allocation successful. [ResourceProfile{cpuCores=1, heapMemoryMB=1024, directMemoryMB=0}, ResourceProfile{cpuCores=1, heapMemoryMB=1024, directMemoryMB=0}, ResourceProfile{cpuCores=1, heapMemoryMB=1024, directMemoryMB=0}, ResourceProfile{cpuCores=1, heapMemoryMB=1024, directMemoryMB=0}]
15:14:22.846 [main] INFO org.apache.flink.runtime.scheduler.SchedulerBase - Triggering scheduling of tasks in registered jobs.
15:14:22.846 [main] INFO org.apache.flink.runtime.scheduler.Scheduler - Tasks to trigger: [Physical Slot: Default TaskExecutor (7a990cf9e2cf4da61f4e8e400c68f0b9) - (1/4)]
15:14:22.846 [main] INFO org.apache.flink.runtime.taskmanager.TaskManager - All required resources are allocated. Launching task a9bb69a89fc8fb4b67c5f2636a777c5d on thread Thread-3 (default slot with index 0) (a9bb69a89fc8fb4b67c5f2636a777c5d)
15:14:22.857 [Thread-3] INFO org.apache.flink.streaming.api.operators.CountWindowFunction - ---------- Window: Current time: 1583279662846, Window start: 1583279662746, Window end: 1583279662846 ------------
15:14:22.858 [Thread-3] INFO org.apache.flink.streaming.api.operators.CountWindowFunction - Counter: 0
15:14:22.859 [Thread-3] INFO org.apache.flink.streaming.api.operators.CountWindowFunction - ----------- Window: Previous time: 1583279662746, Previous Window start: 1583279662736, Previous Window end: 1583279662746 -------------
15:14:22.861 [Thread-3] INFO org.apache.flink.streaming.api.operators.CountWindowFunction - ------- Window: Next time: 1583279662847, Next Window start: 1583279662837, Next Window end: 1583279662847 --------
Counter: 1
----------- Window: Previous time: 1583279662746, Previous Window start: 1583279662736, Previous Window end: 1583279662746 -------------
------- Window: Next time: 1583279662847, Next Window start: 1583279662837, Next Window end: 1583279662847 --------
Counter: 2
---------- Window: Current time: 1583279662861, Window start: 1583279662756, Window end: 1583279662861 -----------
Counter: 0
---------------------------- Window: End -----------------------------