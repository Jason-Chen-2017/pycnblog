
作者：禅与计算机程序设计艺术                    
                
                
《 Apache Beam：在 Java 中使用 Beam》
================================

概述
--------

Apache Beam是一个用于流式数据处理的编程框架，它支持多种编程语言，包括Java。在这篇文章中，我们将介绍如何在Java中使用Apache Beam，以及相关的实现步骤、技术原理和应用示例。通过本文，读者将了解到如何在Java中使用Beam进行流式数据处理，以及如何优化和改进Beam的性能。

技术原理及概念
-------------

### 2.1. 基本概念解释

在流式数据处理中，数据流是以流的形式不断地输入到系统中的。数据流可以是批处理的，也可以是实时处理的。对于实时处理，数据流通常是连续的。

在Apache Beam中，数据流被称为“Beam”。Beam提供了一种用于定义和操作数据流的方法，以及一组操作，用于将数据流转换为所需的数据结构。

### 2.2. 技术原理介绍: 算法原理，具体操作步骤，数学公式，代码实例和解释说明

在Beam中，数据流的处理是通过一个`PTransform`对象来完成的。PTransform对象将一个数据流转换为另一个数据流。转换的步骤如下：

1. 从输入中读取数据。
2. 对数据进行转换。
3. 将转换后的数据输出到目标。

下面是一个简单的Beam示例，其中PTransform对象将一个文本数据流转换为输出：
```
import org.apache.beam as beam;
import org.apache.beam.options.PTransform;
import org.apache.beam.sdk.io.ReadableByteArray;
import org.apache.beam.sdk.io.ReadableIntArray;
import org.apache.beam.sdk.io.写作.FileSystem;
import org.apache.beam.sdk.io.写作.Hadoop;
import org.apache.beam.sdk.io.Writable;
import org.apache.beam.sdk.io.WritableFn;
import org.apache.beam.runtime.核心.Runtime;
import org.apache.beam.runtime.api.Option;
import org.apache.beam.runtime.api.PTransform;
import org.apache.beam.runtime.java.FunctionPipeline;
import org.apache.beam.runtime.java.FunctionSpace;
import org.apache.beam.runtime.java.jdbc.JDBC;
import org.apache.beam.runtime.java.jdbc.SimpleJDBC;
import org.apache.beam.runtime.java.options.Options;
import org.apache.beam.runtime.java.security.security.PrivilegedAccord;
import org.apache.beam.runtime.java.security.security.PrivilegedScm;
import org.apache.beam.runtime.java.streaming.PTransformStep;
import org.apache.beam.runtime.java.streaming.Streams;
import org.apache.beam.runtime.java.streaming.Table;
import org.apache.beam.runtime.java.util.IntStream;
import org.apache.beam.runtime.java.util.Map;
import org.apache.beam.runtime.java.util.memory.Memory;
import org.apache.beam.runtime.java.util.memory.MemoryStore;
import org.apache.beam.runtime.java.util.random.Random;
import org.apache.beam.runtime.java.util.slidingwindow.SlidingWindows;
import org.apache.beam.runtime.java.util.slidingwindow.Windows;

public class ApacheBeam {

  // 获取输入数据
  private final PTransform<String, String> input;

  // 获取输出数据
  //...

  public ApacheBeam() {
    input = PTransform.create(new SimpleJDBC.JdbcIO("jdbc:hdfs://hdfs://namenode-host:port/path/to/data"))
       .with(new JDBC.Kafka()
           .version("universal")
           .topic("topic")
           .property("bootstrap-servers", "hdfs://namenode-host:port/path/to/bootstrap")
           .property("group-id", "group-id")
           .property("key-prefix", "key-prefix"));
  }

  public PTransform<String, String> getInput() {
    return input;
  }

  public void setInput(PTransform<String, String> input) {
    this.input = input;
  }

  //...

  public void run(Runtime environment) {
    environment.setGroup(new PrivilegedScm.Builder(PrivilegedAccord.create("my-group")));

    // 读取输入数据
    Table<String, String> input = input.get(0);

    // 对数据进行转换
    input = input
       .map(new Map<String, Integer>() {
          @Override
          public void apply(Map<String, Integer> map) {
            map.put("key", 1);
          }
        })
       .map(new PTransform<String, Integer>() {
          @Override
          public Integer apply(String value) {
            return Integer.parseInt(value);
          }
        });

    // 输出数据
    input.print();

    environment.execute();
  }

}
```
上面代码定义了一个简单的Beam程序，它读取来自HDFS的文本数据，并将其转换为整数。Beam还提供了一些其他功能，如流控制、批处理等。

### 2.2. 相关技术比较

与传统的流处理框架（如Apache Flink和Apache Spark等）相比，Apache Beam具有以下优势：

* 基于流处理的编程模型，易于理解和编写。
* 支持多种编程语言（包括Java、Python和Scala等），使得现有的数据处理工作流更易于集成到现有的开发环境中。
* 丰富的SQL语法，使得数据处理更加直观。
* 自动分区和裁剪数据流，提高了处理性能。
* 易于与大数据和云集成。

## 3. 实现步骤与流程

### 3.1. 准备工作：环境配置与依赖安装

要使用Beam在Java中进行流处理，需要完成以下步骤：

1. 安装Java和相关的支持库（如Apache Beam SDK和Apache Flink）。
2. 安装Apache Beam的Java库。

可以使用以下命令来安装Beam Java库：
```
// Maven
mvn dependency:tree -u http://beam.apache.org/release/stable/maven/

// Gradle
implementation 'org.apache.beam:beam-api:latest'
    implementation 'org.apache.beam:beam-transforms-api:latest'
    implementation 'org.apache.beam:beam-runtime:latest'
```
### 3.2. 核心模块实现

要使用Beam进行流处理，需要完成以下步骤：

1. 定义Beam程序。
2. 定义Beam PTransform。
3. 定义Beam应用程序。

下面是一个简单的Beam应用程序实现：
```
import org.apache.beam.api.beam.VoidOutput;
import org.apache.beam.api.options.POptions;
import org.apache.beam.api.transforms.MapKey;
import org.apache.beam.api.transforms.PTransform;
import org.apache.beam.api.transforms.PTransformStep;
import org.apache.beam.runtime.api.Context;
import org.apache.beam.runtime.api.Option;
import org.apache.beam.runtime.api.PTransform.PTransformStep;
import org.apache.beam.runtime.java.BeamApp;
import org.apache.beam.runtime.java.BeamOptions;
import org.apache.beam.runtime.java.Context;
import org.apache.beam.runtime.java.Option;
import org.apache.beam.runtime.java.Table;
import org.apache.beam.runtime.java.Values;
import org.apache.beam.runtime.java.jdbc.JDBC;
import org.apache.beam.runtime.java.jdbc.SimpleJDBC;
import org.apache.beam.runtime.java.options.行级并行。
import org.apache.beam.runtime.java.options.行级并行";
import org.apache.beam.runtime.java.options.行级并行";
import org.apache.beam.runtime.java.options.行级并行";
import org.apache.beam.runtime.java.options.行级并行";
import org.apache.beam.runtime.java.options.行级并行";
import org.apache.beam.runtime.java.options.行级并行";
import org.apache.beam.runtime.java.options.行级并行";
import org.apache.beam.runtime.java.transforms.步长;
import org.apache.beam.runtime.java.transforms.GroupCombine;
import org.apache.beam.runtime.java.transforms.KeyWithin;
import org.apache.beam.runtime.java.transforms.PTransform;
import org.apache.beam.runtime.java.transforms.PTransformStep;
import org.apache.beam.runtime.java.transforms.PTransformWithKey;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithin;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithin;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithin;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithin;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithin;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithin;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithin;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithin;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.KeyWithIn;
import org.apache.beam.runtime.java.transforms.PTransformWithKey.Key;

