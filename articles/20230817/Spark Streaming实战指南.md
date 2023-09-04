
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 概述
Apache Spark™是当前最流行的开源大数据处理框架之一。它提供了丰富的数据处理功能，如数据采集、清洗、转换、分析、机器学习等。而Spark Streaming也是一个在Spark生态系统中的重要组件，其提供的实时计算特性可以让用户实现实时的流式数据处理应用。实时流式数据处理引擎通常会在输入端持续不断地接收到数据，然后进行数据处理并将结果输出到输出端，例如保存到数据库中、更新数据缓存或向其他应用程序发送信息。但是，由于缺乏专业人士对Spark Streaming的深入理解，导致开发者在应用该框架时经常面临很多问题。为此，笔者创作了本文，力图帮助开发者理解Spark Streaming的基础知识、原理和用法，以及如何解决实际问题。希望通过本文能够帮助更多的人熟悉和掌握Spark Streaming，提升开发效率、降低运维成本、增加工作竞争力。


## 作者简介
马俊(Majun) 博士，华东师范大学软件学院研究员。现任Apache基金会软件工程师，主要从事基于Spark的大数据工程师。曾就职于华为技术有限公司任职高级软件工程师和架构师，主导开发华为公司私有云平台，具备十多年的软件设计和开发经验。


## 文章目标读者
对于已经掌握Apache Spark的用户来说，希望通过本文，可以了解Spark Streaming框架的内部运行机制，以及一些常用的算法和实践方法，提升开发效率和解决实际问题能力。对于需要学习Spark Streaming开发的新手来说，本文可以作为Spark Streaming的学习资源，快速了解Spark Streaming的基本概念、原理和用法，加快技能提升。欢迎各位同仁参与评论，共同完善本文。


# 2.概览
## 2.1 大数据背景介绍
大数据是指以可观察到的规模存储海量数据，这些数据包括各种结构化、非结构化数据、及其相关元数据。随着大数据的爆炸性增长，数据处理和分析已经成为当今社会的一个热点话题。由于数据的复杂性，传统的数据处理方法遇到了诸多困难。例如，从海量原始数据中提取有效信息成为了一个技术难题。

Apache Spark™是当前最流行的开源大数据处理框架之一。它提供了丰富的数据处理功能，如数据采集、清洗、转换、分析、机器学习等。其中Spark Streaming 是Spark提供的实时计算引擎，可以让用户实现实时的流式数据处理应用。

Apache Hadoop®是Apache基金会下的开源分布式计算框架。其能够存储、处理海量数据，能够支持批处理和实时计算。Hadoop能够以分布式的方式进行数据处理，通过Hadoop MapReduce编程模型进行批处理，并且Hadoop YARN提供了容错机制以应对集群故障。然而，当流式数据到来的时候，Hadoop MapReduce方法就无法满足需求。

Spark Streaming建立在Spark之上，是一个实时的流式数据处理框架。它利用Spark提供的高性能和容错能力，使得用户能够以较低的延迟时间实时处理数据。Spark Streaming允许用户将实时数据源连接到数据处理程序，并获取实时数据。Spark Streaming通过数据分片的方式对实时数据进行分发，这样就可以把数据流按照时间或者空间划分为多个子集，从而提高数据的处理速度和容错性。另外，Spark Streaming还提供了一个持久化机制，用来存储已处理的数据。Spark Streaming通过容错机制来保证处理数据的正确性和一致性。

## 2.2 Spark Streaming概述
Spark Streaming是Spark提供的实时计算引擎。它可以实时地接收输入数据，并将它们分批、分区后传入到Spark作业中进行处理。Spark Streaming提供了以下几个主要优点：

1. 流式计算：Spark Streaming可以以微批（micro-batch）的方式接收实时数据，使得数据处理的延迟时间可以被减小到秒级甚至更短。
2. 可靠性：Spark Streaming可以使用“检查点”功能来确保数据处理的一致性和可靠性。
3. 弹性：Spark Streaming可以自动动态调整数据处理过程中的并发度，以适应数据输入的速率和处理能力的变化。
4. 实时查询：Spark Streaming可以通过一种高效的方法对数据进行存储，并支持实时查询和分析。
5. 窗口计算：Spark Streaming可以用于对过去一定时间内的数据进行汇总和计算。

Spark Streaming一般分为三个阶段：

1. 数据采集阶段：首先，Spark Streaming从实时数据源接收数据，并将其切分为一系列的小批量。
2. 数据处理阶段：接下来，Spark Streaming根据分批的数据，调用Spark作业对其进行处理。
3. 数据输出阶段：最后，Spark Streaming将处理好的结果输出到文件、数据库或终端等地方。


## 2.3 Spark Streaming架构
Spark Streaming的架构如下所示：


1. SparkContext：用户启动Spark Streaming程序后，会创建一个SparkContext，用于创建RDD、累加器、广播变量等。
2. InputDStream：InputDStream表示输入数据流，InputDStream可以从文件、TCP套接字、Kafka消息队列、Kinesis等多种数据源接收数据，并以固定间隔生成RDDs。
3. Transformation：Transformation表示转换操作，它可以对DStreams进行组合、过滤、窗口聚合等操作。
4. OutputDStream：OutputDStream表示输出数据流，它负责将DStream中处理后的结果数据保存到外部存储中，例如HDFS、Hive、Cassandra等。
5. Checkpointing：Checkpointing用于容错，它记录了数据流的处理进度，如果出现失败情况，可以从最近的检查点重新开始处理。

# 3.核心概念术语
## 3.1 DStream
DStream (Discretized Stream)，又称微批处理流，是Spark Streaming的数据类型。DStream由连续不断的RDD组成，每隔一段时间就会更新一次，所以它是一连串的RDDs，每个RDD代表时间窗口内的数据。DStream可以简单理解为一系列的RDDs集合。

每个DStream包含两个基本属性：

1. Batch interval：DStream以固定长度的时间间隔生成一个RDD，也就是批次间隔。
2. Sliding interval：窗口滑动间隔，即前一个批次完成处理后，下一个批次开始的时间间隔。


## 3.2 Transformations
Transformation是Spark Streaming的核心运算单元，它是对DStream进行操作的函数。Spark Streaming支持丰富的Transformations，具体包括：

1. map()：map()是最简单的Transformation，它接受一个函数，对每个元素执行该函数，返回新的元素。
2. flatMap()：flatMap()与map()类似，但它接受的是一个序列，将该序列展开为多个元素。
3. filter()：filter()可以用来选择出特定条件的元素。
4. reduceByKey()：reduceByKey()可以对相同key的元素进行聚合操作，例如求和、平均值等。
5. window()：window()可以将DStream按时间窗口分组。
6. join()：join()可以将DStream两边的元素进行连接，产生笛卡尔积。
7. updateStateByKey()：updateStateByKey()是状态计算的核心操作，它可以为每个key维护一个状态，并根据窗口的输入数据更新状态。

## 3.3 Input Sources
Input Source是Spark Streaming读取数据的渠道。目前支持的文件输入源有File Input、Directory Input、Socket Input、Kafka Input等。

## 3.4 Checkpointing
Checkpointing 是Spark Streaming提供的容错机制。Checkpointing的主要作用是在处理过程中发生异常时，重启后可以从最近的检查点位置继续处理。

Spark Streaming支持两种类型的检查点机制：

1. **内存检查点**：顾名思义，这种检查点机制只保留在内存里，当程序崩溃或者意外停止时，会丢失所有的检查点信息。因此，这种检查点机制仅供本地测试使用，不能部署到生产环境中。
2. **HDFS检查点**：这种检查点机制将检查点信息写入HDFS中，当程序崩溃或者意外停止时，可以从HDFS中恢复数据。

# 4.核心算法原理与操作步骤
本节将详细阐述Spark Streaming中常用的算法原理与操作步骤。

## 4.1 map()函数
map()函数可以对每个元素执行一个函数，然后生成一个新的元素，常用于映射或过滤操作。

```scala
// 以每个单词的首字母转为大写形式作为键，值设置为1
val wordCounts = lines.flatMap(_.split(" ")).map((_, 1)).reduceByKey(_ + _)
```

```java
JavaDStream<String> javaLines = ssc.textFileStream("/path/to/file"); // 文件目录
JavaPairDStream<String, Integer> pairDs = javaLines.flatMapToPair(new PairFlatMapFunction<String, String, Integer>() {
    @Override
    public Iterable<Tuple2<String, Integer>> call(String line) throws Exception {
        List<String> words = Arrays.asList(line.split(" "));
        return Lists.newArrayList(Iterables.transform(words, new Function<String, Tuple2<String, Integer>>() {
            @Override
            public Tuple2<String, Integer> apply(String input) {
                return new Tuple2<>(input.substring(0, 1).toUpperCase(), 1);
            }
        }));
    }
}).reduceByKey(new Function2<Integer, Integer, Integer>() {
    @Override
    public Integer call(Integer i1, Integer i2) {
        return i1 + i2;
    }
});
```

## 4.2 flatmap()函数
flatmap()函数与map()函数相似，但它接受的是一个序列，将该序列展开为多个元素。常用于将一个元素拆分为多个元素。

```scala
val tweetWords = tweets.flatMap { tweet =>
  tweet.split("\\W+")
}
```

```java
JavaDStream<String> javaTweets = jssc.socketTextStream("localhost", 9999);
JavaDStream<String> javaWordsWithoutPunctuations = javaTweets.flatMap(new FlatMapFunction<String, String>() {
    @Override
    public Iterable<String> call(String t) throws Exception {
        List<String> words = Arrays.asList(t.replaceAll("[^\\p{L}\\p{Nd}]+", "").toLowerCase().split("\\s+"));
        return Iterables.filter(words, Predicates.<String>not(Predicates.isEmpty()));
    }
});
```

## 4.3 filter()函数
filter()函数可以用于选择出特定条件的元素。常用于过滤不需要的元素。

```scala
val filteredRdd = rdd.filter(row => row.value > 10 &&!row.isBad)
```

```java
JavaDStream<Row> javaRows = jssc.queueStream(queues);
JavaDStream<Row> goodRows = javaRows.filter(new Function<Row, Boolean>() {
    @Override
    public Boolean call(Row row) throws Exception {
        return row.getInt(0) > 10 &&!(Boolean) row.getField(1);
    }
});
```

## 4.4 reduceByKey()函数
reduceByKey()函数可以对相同key的元素进行聚合操作，常用于求和、平均值等。

```scala
val result = data.reduceByKey((acc, value) => acc + value)
```

```java
JavaPairDStream<String, Long> pairs = jssc.socketTextStream("localhost", 9999)
       .flatMapToPair(new PairFlatMapFunction<String, String, Long>() {
            @Override
            public Iterable<Tuple2<String, Long>> call(String t) throws Exception {
                String[] parts = t.split(",");
                if (parts.length!= 2) throw new IllegalArgumentException();
                try {
                    return Lists.newArrayList(new Tuple2<>(parts[0], Long.parseLong(parts[1])));
                } catch (NumberFormatException e) {
                    throw new IllegalArgumentException();
                }
            }
        });
        
JavaPairDStream<String, Long> reducedPairs = pairs.reduceByKey(new Function2<Long, Long, Long>() {
            @Override
            public Long call(Long a, Long b) throws Exception {
                return a + b;
            }
        });
```

## 4.5 window()函数
window()函数可以将DStream按时间窗口分组。常用于实时计算过去一定时间范围内的统计指标。

```scala
import org.apache.spark.sql.functions._

val windowedAvg = data.withWatermark("timestamp", "1 minute")
 .groupBy(window($"timestamp", "1 hour"), $"key")
 .agg(mean($"value"))
  
val result = sc.parallelize(Seq(("key1", 1), ("key1", 2)))
 .toDF("key", "value")
 .withWatermark("timestamp", "1 minute")
 .groupBy(window($"timestamp", "1 hour"), $"key")
 .count()
```

```java
import static org.apache.spark.sql.functions.*;
import static org.apache.spark.streaming.api.Time.*;

JavaPairDStream<String, Double> windowedAvg = jssc.queueStream(queues)
       .filter(new Function<String, Boolean>() {
            @Override
            public Boolean call(String t) throws Exception {
                return Math.random() < 0.1; // randomly drop some values to simulate missing data
            }
        }).mapToPair(new PairFunction<String, String, Double>() {
            @Override
            public Tuple2<String, Double> call(String t) throws Exception {
                String[] parts = t.split(",");
                long timestamp = System.currentTimeMillis();
                if (parts.length == 2) {
                    double value = Double.parseDouble(parts[1]);
                    return new Tuple2<>(parts[0], value);
                } else {
                    return null; // ignore invalid records
                }
            }
        })
       .filter(new Function<Tuple2<String, Double>, Boolean>() {
            @Override
            public Boolean call(Tuple2<String, Double> tuple2) throws Exception {
                return tuple2!= null;
            }
        }).groupByKeyAndWindow(slidingWindows(Duration.ofMinutes(60), Duration.ofSeconds(30)),
                                new KeySelector<Tuple2<String, Double>, String>() {
                                    @Override
                                    public String getKey(Tuple2<String, Double> tuple2) throws Exception {
                                        return tuple2._1();
                                    }
                                }, new AggregateFunction2<List<Tuple2<String, Double>>, Optional<Double>, Double>() {
                                    private final Aggregator<Tuple2<String, Double>,?, Double> aggregator =
                                            new AverageAggregator<>();

                                    @Override
                                    public Optional<Double> apply(@SuppressWarnings("unchecked") List<Tuple2<String, Double>> list) throws Exception {
                                        if (!list.isEmpty()) {
                                            return Optional.ofNullable(aggregator.apply(list));
                                        } else {
                                            return Optional.empty();
                                        }
                                    }
                                });

Dataset<Row> countByWindow = sparkSession.readStream().format("csv").schema(StructType.fromDDL("key string, value integer")).load("/path/to/file")
       .selectExpr("_1 as key", "_2 as value")
       .withColumn("timestamp", current_timestamp()).as(Encoders.tuple(StringType(), IntegerType()))
       .groupByKey(window($"timestamp", "1 hour"), col("key"))
       .count();
```

## 4.6 join()函数
join()函数可以将DStream两边的元素进行连接，产生笛卡尔积。常用于基于事件关联的场景。

```scala
val joined = leftData.join(rightData)
```

```java
JavaPairDStream<String, String> javaLeft = jssc.socketTextStream("localhost", 9998)
       .mapToPair(new PairFunction<String, String, String>() {
            @Override
            public Tuple2<String, String> call(String t) throws Exception {
                return new Tuple2<>(UUID.randomUUID().toString(), t);
            }
        });
        
JavaPairDStream<String, String> javaRight = jssc.socketTextStream("localhost", 9999)
       .mapToPair(new PairFunction<String, String, String>() {
            @Override
            public Tuple2<String, String> call(String t) throws Exception {
                return new Tuple2<>(UUID.randomUUID().toString(), t);
            }
        });

JavaPairDStream<String, String> merged = javaLeft.join(javaRight);
```

## 4.7 updateStateByKey()函数
updateStateByKey()函数可以为每个key维护一个状态，并根据窗口的输入数据更新状态。常用于维护状态的聚合操作。

```scala
import org.apache.spark.sql.functions._

val updatedState = stateData.updateStateByKey((seq: Seq[Int], state: Option[Int]) => {
  val sum = seq.sumOption.getOrElse(state.getOrElse(0))
  Some(sum)
})

updatedState.foreachRDD { rdd => 
  rdd.foreach { case (key, value) => println(s"Key $key has an updated value of ${value}") } 
}
```

```java
import static org.apache.spark.sql.functions.*;
import static org.apache.spark.streaming.api.Time.*;

JavaPairDStream<String, String> streamWithKeys = jssc.queueStream(queues)
       .filter(new Function<String, Boolean>() {
            @Override
            public Boolean call(String t) throws Exception {
                return true;
            }
        }).mapToPair(new PairFunction<String, String, String>() {
            @Override
            public Tuple2<String, String> call(String t) throws Exception {
                return new Tuple2<>(UUID.randomUUID().toString(), t);
            }
        });

final JavaPairDStream<String, StateValue> stateValues = streamWithKeys
       .mapToPair(new PairFunction<Tuple2<String, String>, String, StateValue>() {
            @Override
            public Tuple2<String, StateValue> call(Tuple2<String, String> t) throws Exception {
                int newValue = random.nextInt(100);
                StateValue value = new StateValue(System.currentTimeMillis(), newValue);
                return new Tuple2<>(t._1, value);
            }
        }).updateStateByKey(new Function2<Optional<StateValue>, Iterator<Tuple2<String, StateValue>>, Optional<StateValue>>() {
            @Override
            public Optional<StateValue> call(Optional<StateValue> previousValue, Iterator<Tuple2<String, StateValue>> iterator) throws Exception {
                ArrayList<StateValue> buffer = Lists.newArrayList(iterator);
                long currentTimeMillis = System.currentTimeMillis();

                for (StateValue sv : buffer) {
                    if (previousValue.isPresent()) {
                        sv.setSum(sv.getNewValue());
                        sv.setTime(currentTimeMillis);
                    }
                }

                if (!buffer.isEmpty()) {
                    int sum = 0;
                    for (StateValue sv : buffer) {
                        sum += sv.getSum();
                    }

                    long timeStamp = buffer.get(buffer.size() - 1).getTime();
                    StateValue combined = new StateValue(timeStamp, sum);
                    return Optional.of(combined);
                } else {
                    return previousValue;
                }
            }
        });

stateValues.foreachRDD(new VoidFunction<JavaRDD<Tuple2<String, StateValue>>>() {
            @Override
            public void call(JavaRDD<Tuple2<String, StateValue>> stringStateValuePairRDD) throws Exception {
                JavaPairRDD<String, Long> countsPerKey = stringStateValuePairRDD.rdd()
                       .mapToPair(new PairFunction<Tuple2<String, StateValue>, String, Long>() {
                            @Override
                            public Tuple2<String, Long> call(Tuple2<String, StateValue> stringStateValuePair) throws Exception {
                                return new Tuple2<>(stringStateValuePair._1, 1l);
                            }
                        })
                       .reduceByKey(new Function2<Long, Long, Long>() {
                            @Override
                            public Long call(Long a, Long b) throws Exception {
                                return a + b;
                            }
                        });
                
                Dataset<Row> dataset = sparkSession.createDataFrame(countsPerKey.rdd(), Encoders.tuple(StringType(), LongType())).orderBy("_1");
                
                DataFrameWriter writer = dataset.writeStream().outputMode("complete");
                
                // checkpoint every 10 seconds or after inactivity for at most 60 seconds
                writer.option("checkpointLocation", "/tmp/" + UUID.randomUUID().toString()).trigger(processingTime("10 seconds"))
                     .start("/path/to/output/directory").awaitTermination();
            }
        });
```

# 5.代码实例及解释说明
## 5.1 基于文件的word count

### 5.1.1 Scala版本

```scala
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.streaming.{Seconds, StreamingContext}

object WordCount {
  
  def main(args: Array[String]): Unit = {
    
    val conf = new SparkConf().setAppName("Word Count")
    val sc = new SparkContext(conf)

    val ssc = new StreamingContext(sc, Seconds(5))
    
    val lines = ssc.socketTextStream("localhost", 9999)
    val words = lines.flatMap(_.split(" "))
    val wordCounts = words.map((_, 1)).reduceByKey(_ + _)
    
    wordCounts.print()
    
    ssc.start()
    ssc.awaitTermination()
  }
}
```

### 5.1.2 Java版本

```java
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.apache.spark.streaming.api.java.JavaPairDStream;
import org.apache.spark.streaming.api.java.JavaStreamingContext;

public class WordCount {

  public static void main(String[] args) throws InterruptedException {
    SparkConf conf = new SparkConf().setAppName("WordCount");
    JavaStreamingContext jssc = new JavaStreamingContext(conf, Durations.seconds(5));

    JavaDStream<String> lines = jssc.socketTextStream("localhost", 9999);
    JavaDStream<String> words = lines.flatMap(new FlatMapFunction<String, String>() {
      @Override
      public Iterable<String> call(String s) throws Exception {
        return Arrays.asList(s.split(" "));
      }
    });

    JavaPairDStream<String, Integer> wordCounts = words.mapToPair(new PairFunction<String, String, Integer>() {
      @Override
      public Tuple2<String, Integer> call(String s) throws Exception {
        return new Tuple2<>(s, 1);
      }
    }).reduceByKey(new Function2<Integer, Integer, Integer>() {
      @Override
      public Integer call(Integer v1, Integer v2) throws Exception {
        return v1 + v2;
      }
    });

    wordCounts.print();

    jssc.start();
    jssc.awaitTermination();
  }
}
```

## 5.2 基于Kafka的消费实时统计

### 5.2.1 Scala版本

```scala
import kafka.serializer.StringDecoder
import org.apache.spark.streaming.kafka.{KafkaUtils, OffsetRange}

object KafkaStats {
  
  def main(args: Array[String]): Unit = {
    
    val appName = "Kafka Stats"
    val brokers = "localhost:9092"
    val topic = "mytopic"
    
    val sc = new SparkContext(appName, "local[*]")
    val ssc = new StreamingContext(sc, Seconds(5))
    
    val kvs = KafkaUtils.createDirectStream[String, String](
      ssc, 
      Array(topic), 
      kafka.consumer.ConsumerConfig.fromProps(
        Map(
          "bootstrap.servers" -> brokers,
          "group.id"->"test",
          "auto.offset.reset" -> "latest"
        )
      ),
      StringDecoder,
      StringDecoder
    )
    
    
    import org.apache.spark.sql.functions._
    
    val messageLengths = kvs.map(_.value).flatMap(_.split("\\s+"))
                          .map(s => s.replaceAll("""[^\w\s]""", ""))
                          .filter(_.nonEmpty)
                          .map(_.trim.length)
    
    messageLengths.foreachRDD{ rdd => 
      val lengthTotal = rdd.sum()
      val recordCount = rdd.count()
      
      println(f"Length total: $lengthTotal / Record count: $recordCount")
    }
    
    ssc.start()
    ssc.awaitTermination()
  }
}
```

### 5.2.2 Java版本

```java
import java.util.Properties;

import kafka.common.TopicPartition;
import kafka.consumer.ConsumerRecord;
import kafka.consumer.ConsumerRecords;
import kafka.javaapi.consumer.ConsumerConnector;
import kafka.producer.Producer;
import kafka.utils.CommandLineUtils;
import kafka.utils.ZKStringSerializer$;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.streaming.api.java.JavaPairReceiverInputDStream;
import org.apache.spark.streaming.api.java.JavaStreamingContext;

import scala.collection.immutable.HashMap;

public class KafkaStats {

  public static void main(String[] args) {
    String appName = "KafkaStats";
    String brokers = "localhost:9092";
    String topic = "mytopic";

    SparkConf conf = new SparkConf().setAppName(appName).setMaster("local[*]");
    JavaStreamingContext jssc = new JavaStreamingContext(conf, Durations.seconds(5));

    Properties props = new Properties();
    props.put("bootstrap.servers", brokers);
    props.put("group.id", "test");
    props.put("auto.offset.reset", "latest");

    ConsumerConnector consumer = kafka.consumer.Consumer$.MODULE$.createJavaConsumerConnector(new HashMap<>(), props);
    HashMap<TopicPartition, Long> offsets = new HashMap<>();
    offsets.put(new TopicPartition(topic, 0), 0l);

    HashMap<TopicPartition, Long> commitOffsets = consumer.commitOffsets(offsets);
    HashSet<TopicPartition> assignedPartitions = commitOffsets.keySet();

    ConsumerRecords<String, String> records = consumer.poll(1000);
    while(!records.isEmpty()){
      for(ConsumerRecord<String, String> record : records){
        System.out.println(record.partition()+" "+record.offset()+": "+record.value());
      }

      records = consumer.poll(1000);
    }

    // Create direct stream with custom decoder
    JavaPairReceiverInputDStream<String, String> messages = 
        JavaStreamingContext.fromGraph(
            GraphGenerator.generate(appName, brokerZkQuorum, topic, null, null, false, 
                                   ZKStringSerializer$.MODULE$, ZKStringSerializer$.MODULE$)
        ).mapWithState(StorageLevel.MEMORY_AND_DISK_SER, (message, state) -> {
          System.out.println(message.value());
          return "";
        });

    // Filter out empty and non-alphabetic characters, trim whitespace, then compute length
    JavaDStream<Integer> lengths = messages.flatMap(new FlatMapFunction<Tuple2<String, String>, Integer>() {
      @Override
      public Iterable<Integer> call(Tuple2<String, String> tuple) throws Exception {
        return Arrays.stream(tuple._2().split("\\s+"))
                    .map(token -> token.replaceAll("[^A-Za-z]", ""))
                    .filter(token ->!token.isEmpty())
                    .map(token -> token.trim().length()).collect(Collectors.toList());
      }
    });

    lengths.foreachRDD(rdd -> {
      long lengthTotal = rdd.reduce((total, next) -> total + next);
      long recordCount = rdd.count();
      System.out.println("Length total: " + lengthTotal + ", Record count: " + recordCount);
    });

    // Start the computation
    jssc.start();
    jssc.awaitTermination();
  }

  /**
   * The entry point to generate graph from configuration parameters.
   */
  public static class GraphGenerator extends StreamGraphGenerator<String, String>{
    public static JavaDStream<Tuple2<String, String>> generate(
        String jobName, String zkQuorum, String topic, String groupId, 
        Map<String, Object> kafkaParams, boolean createTopicIfNotExist, 
        Class keyDecoderClass, Class valueDecoderClass) {
      Configuration conf = new Configuration();
      conf.set(this.APP_NAME, jobName);
      conf.set(this.ZK_QUORUM, zkQuorum);
      conf.set(this.TOPIC, topic);
      conf.set(this.GROUP_ID, groupId);
      conf.setAll(kafkaParams);
      conf.set(this.CREATE_TOPIC_IF_NOT_EXISTS, Boolean.toString(createTopicIfNotExist));
      conf.setClass(this.KEY_DESERIALIZER, keyDecoderClass, Deserializer.class);
      conf.setClass(this.VALUE_DESERIALIZER, valueDecoderClass, Deserializer.class);
      conf.setBoolean(this.DEDUPLICATION_DISABLED, true);

      StreamExecutionEnvironment env = this.getEnv(conf);
      DataStream<byte[]> bytes = env.addSource(new KafkaSource(conf)).name("directKafkaSource");
      DataStream<Object> obj = bytes.map(bytesObj -> ByteStringParser.parse(bytesObj, conf)).name("deserializeBytesToObject");
      JavaDStream<String> str = (JavaDStream<String>)obj.flatMap(data -> ((Iterable<String>)data)._2).name("extractString");

      return str.mapToPair(strObj -> (Tuple2<String, String>)strObj).name("stringToStringPair");
    }
  }

  /**
   * A simplified parser that converts raw byte array to deserialized object using deserializers specified in configuration.
   */
  public static class ByteStringParser implements Function<byte[], Object> {
    private Deserializer keyDeserializer;
    private Deserializer valueDeserializer;

    public static Object parse(byte[] bytes, Configuration conf) {
      ByteStringParser parser = new ByteStringParser(conf);
      return parser.call(bytes);
    }

    public ByteStringParser(Configuration conf) {
      try {
        keyDeserializer = ReflectionUtils.newInstance(conf.getClass(this.KEY_DESERIALIZER, Deserializer.class), conf);
        valueDeserializer = ReflectionUtils.newInstance(conf.getClass(this.VALUE_DESERIALIZER, Deserializer.class), conf);
      } catch (Exception e) {
        throw new RuntimeException("Failed to initialize serializers", e);
      }
    }

    @Override
    public Object call(byte[] bytes) throws Exception {
      GenericRecord genericRecord = InternalGenericRecord.deserialize(bytes, keyDeserializer, valueDeserializer);
      return Tuple2$.MODULE$(genericRecord.getKey().toString(), genericRecord.getValue().toString());
    }
  }

  /**
   * Wrapper around internal Kafka source used by Flink's own Kafka connector implementation.
   */
  public static class KafkaSource extends RichParallelSourceFunction<byte[]> {
    private Configuration config;
    private ConsumerIterator<byte[], byte[]> it;
    private volatile boolean isRunning = true;

    public KafkaSource(Configuration conf) {
      super();
      this.config = conf;
    }

    @Override
    public void open(Configuration parameters) throws Exception {
      HashMap<TopicPartition, Long> offsets = new HashMap<>();
      offsets.put(new TopicPartition(config.getString(this.TOPIC), 0), 0l);
      this.it = ConsumerIterator.partitions(offsets, config.getString(this.ZK_QUORUM), config.getString(this.GROUP_ID), config, new ByteArrayDeserializer(), new ByteArrayDeserializer());
    }

    @Override
    public void run(SourceContext<byte[]> ctx) throws Exception {
      while (isRunning) {
        synchronized(ctx.getCheckpointLock()) {
          ConsumerRecord<byte[], byte[]> rec = it.next();
          ctx.collect(rec.value());
        }
      }
    }

    @Override
    public void cancel() {
      isRunning = false;
    }
  }
}
```