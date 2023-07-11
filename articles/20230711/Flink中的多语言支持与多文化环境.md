
作者：禅与计算机程序设计艺术                    
                
                
《73. Flink 中的多语言支持与多文化环境》
=========================

73. Flink 中的多语言支持与多文化环境
----------------------------------------------

### 1. 引言

### 1.1. 背景介绍

Flink 是一个基于流处理的分布式计算框架，旨在实现低延迟、高吞吐、可扩展的流式数据处理。在 Flink 中，流式数据的处理通常涉及大量的使用 Java 和 Scala 等编程语言编写的高级数据处理 API。然而，随着 Flink 社区的不断发展和创新，越来越多的开发者和使用者开始关注 Flink 的多语言支持问题。

### 1.2. 文章目的

本文旨在讲解 Flink 中的多语言支持，以及如何通过多语言环境来支持不同编程语言的使用。首先将介绍 Flink 中的多语言支持相关概念，然后讨论多语言环境在实际应用中的优势和挑战，接着讨论 Flink 中的多语言支持技术和实现步骤，最后给出应用示例和代码实现讲解。

### 1.3. 目标受众

本文的目标读者是具有编程基础和实际项目经验的开发者和使用者，以及对多语言处理和 Flink 感兴趣的读者。

## 2. 技术原理及概念

## 2.1. 基本概念解释

### 2.1.1. Flink 的并行处理模型

Flink 采用了并行处理模型，将数据处理任务拆分成多个子任务，由多个并行的处理单元并行执行。这种并行处理模型可以在不牺牲性能的情况下，提高数据处理的吞吐量和延迟。

### 2.1.2. 多种编程语言支持

Flink 支持多种编程语言，包括 Java、Python、Scala 等。开发者可以根据实际需求选择不同的编程语言来编写 Flink 应用程序。

### 2.1.3. 事件时间（Event Time）

Flink 中的事件时间是一种抽象概念，用于描述数据处理的逻辑顺序。通过定义事件时间，可以确保数据处理的正确性和可靠性。

### 2.1.4. 外键（Foreign Key）

外键是一种数据库概念，用于引用其他数据表中的数据。在 Flink 中，可以通过外键来引用其他数据表的数据，实现数据之间的关联。

## 3. 实现步骤与流程

### 3.1. 准备工作：环境配置与依赖安装

要在 Flink 中使用多语言支持，首先需要确保环境配置正确。然后，安装所需的依赖包。

### 3.2. 核心模块实现

核心模块是 Flink 中的一个核心组件，负责数据处理的计算和存储。在这里，可以使用 Java 和 Scala 等编程语言编写核心模块。

### 3.3. 集成与测试

完成核心模块的编写后，需要进行集成和测试。集成测试通常包括以下步骤：

1. 验证数据源的正确性
2. 验证核心模块的正確性
3. 验证数据处理的正确性

## 4. 应用示例与代码实现讲解

### 4.1. 应用场景介绍

本文将介绍如何使用 Flink 实现一个简单的数据处理应用程序。该应用程序将使用 Java 编程语言编写核心模块，使用 Scala 编程语言编写事件时间和数据处理逻辑。

### 4.2. 应用实例分析

首先，需要创建一个数据源。这里，我们将使用 Google 的 BigQuery 作为数据源。然后，需要创建一个 Flink 应用程序，并使用 Java 和 Scala 编写核心模块。最后，运行应用程序并测试数据处理的正确性。

### 4.3. 核心代码实现

```java
import org.apache.flink.api.common.serialization.Serdes;
import org.apache.flink.api.environment.FlinkContext;
import org.apache.flink.stream.api.datastream.DataStream;
import org.apache.flink.stream.api.environment.StreamExecutionEnvironment;
import org.apache.flink.stream.api.functions.source.SourceFunction;
import org.apache.flink.stream.api.scala.{ScalaFunction, ScalaFunction0};
import org.apache.flink.stream.api.scala.function.{Function1, Function2};
import org.apache.flink.stream.api.java.JavaStream;
import org.apache.flink.stream.api.scala.Tuple1;
import org.apache.flink.stream.api.scala.Tuple2;
import org.apache.flink.stream.api.java.JavaPairFunction;
import org.apache.flink.stream.api.java.JavaPairFunction0;
import org.apache.flink.stream.api.java.JavaSelect;
import org.apache.flink.stream.api.java.JavaTable;
import org.apache.flink.stream.api.java.SQLDataSource;
import org.apache.flink.stream.api.select.Select;
import org.apache.flink.stream.api.table.Table;
import org.apache.flink.stream.api.window.WindowFunction;
import org.apache.flink.stream.api.window.字段级窗口函数.{WindowFunction1, WindowFunction2};
import org.apache.flink.stream.api.java.JavaStream;
import org.apache.flink.stream.api.scala.{ScalaFunction, ScalaFunction0};
import org.apache.flink.stream.api.scala.function.{Function1, Function2};
import org.apache.flink.stream.api.java.JavaPairFunction;
import org.apache.flink.stream.api.java.JavaSelect;
import org.apache.flink.stream.api.java.SQLDataSource;
import org.apache.flink.stream.api.select.Select;
import org.apache.flink.stream.api.table.Table;
import org.apache.flink.stream.api.window.WindowFunction;
import org.apache.flink.stream.api.window.字段级窗口函数.{WindowFunction1, WindowFunction2};

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.SQLException;
import java.util.Properties;

public class FlinkMultiLanguageExample {
    public static void main(String[] args) throws SQLException {
        FlinkContext context = FlinkContext.get();
        StreamExecutionEnvironment env = context.getExecutionEnvironment();

        // 创建数据源
        SQLDataSource dataSource = new SQLDataSource(
                "jdbc:sqlite:///test.db",
                "test_user",
                "test_password");

        // 创建数据处理程序
        DataStream<String> input = env.addSource(dataSource)
               .map("input")
               .groupBy("user")
               .window(new WindowFunction<String, Tuple2<String, Integer>>() {
                    @Override
                    public Tuple2<String, Integer> map(String value, Integer key,
                            Long offset, Long timeout) throws Exception {
                        // 将输入 value 和 key 组合成 Tuple2
                        Tuple2<String, Integer> tuple = new Tuple2<String, Integer>();
                        tuple.getString(0) = value;
                        tuple.getInteger(1) = key;
                        return tuple;
                    }
                });

        // 定义数据处理逻辑
        input.map(value -> value.trim())
               .groupBy("user")
               .windows(new WindowFunction<String, Tuple1<String, Integer>>() {
                    @Override
                    public Tuple1<String, Integer> map(String value, Integer key,
                            Long offset, Long timeout) throws Exception {
                        // 计算统计量
                        Long count = input.count();
                        double avg = count == 0? 0 : count.get() / count;

                        // 计算平均值
                        double sum = 0;
                        for (Integer i : input.keySet()) {
                            sum += value.get(i);
                        }
                        double avg = sum == 0? 0 : sum.get() / input.size();

                        // 返回 Tuple1
                        return new Tuple1<String, Integer>(avg, count);
                    }
                });

        // 定义数据处理策略
        input.foreach((value, key, tuple) -> {
            System.out.println("Received data: " + value);
            System.out.println("User: " + key);
            System.out.println("Count: " + tuple.get Integer(1));
            System.out.println("Average count: " + tuple.getDouble(0));
        });

        // 执行 Flink 应用程序
        env.execute("Flink Multi-Language Example");
    }

}
```
### 4. 应用示例与代码实现讲解

本文将介绍如何使用 Flink 实现一个简单的数据处理应用程序。该应用程序将使用 Java 和 Scala 等编程语言编写核心模块。

首先，需要创建一个数据源。这里，我们将使用 Google 的 BigQuery 作为数据源。然后，需要创建一个 Flink 应用程序，并使用 Java 和 Scala 编写核心模块。最后，运行应用程序并测试数据处理的正确性。

### 4.1. 应用场景介绍

本文将介绍如何使用 Flink 实现一个简单的数据处理应用程序。该应用程序将使用 Java 和 Scala 等编程语言编写核心模块。

首先，需要创建一个数据源。这里，我们将使用 Google 的 BigQuery 作为数据源。然后，需要创建一个 Flink 应用程序，并使用 Java 和 Scala 编写核心模块。最后，运行应用程序并测试数据处理的正确性。

### 4.2. 应用实例分析

首先，需要创建一个数据源。这里，我们将使用 Google 的 BigQuery 作为数据源。然后，创建一个 Flink 应用程序，并使用 Java 和 Scala 编写核心模块。

```java
import org.apache.flink.api.common.serialization.Serdes;
import org.apache.flink.api.environment.FlinkContext;
import org.apache.flink.api.environment.StreamExecutionEnvironment;
import org.apache.flink.stream.api.datastream.DataStream;
import org.apache.flink.stream.api.environment.StreamExecutionEnvironment;
import org.apache.flink.table.api.Table;
import org.apache.flink.table.api.Table.Builder;
import org.apache.flink.table.api.TableEnvironment;
import org.apache.flink.table.descriptors.KvTableDescriptor;
import org.apache.flink.table.descriptors.TableDescriptor;
import org.apache.flink.table.descriptors.TableEnvironment;
import org.apache.flink.table.descriptors.TableDescriptor;
import org.apache.flink.table.descriptors.TableEnvironment;
import org.apache.flink.table.descriptors.TableDescriptor;
import org.apache.flink.table.descriptors.TableEnvironment;
import org.apache.flink.table.descriptors.TableDescription;
import org.apache.flink.table.descriptors.TableEnvironment;
import org.apache.flink.table.descriptors.TableFunction;
import org.apache.flink.table.descriptors.TableFunctionAnnotation;
import org.apache.flink.table.descriptors.TableFunctionDescriptor;
import org.apache.flink.table.descriptors.TableFunctionResult;
import org.apache.flink.table.descriptors.TableFunctionTableEnvironment;
import org.apache.flink.table.descriptors.TableTableEnvironment;
import org.apache.flink.table.descriptors.TableDescriptor;
import org.apache.flink.table.descriptors.TableEnvironment;
import org.apache.flink.table.descriptors.TableFunctionResult;
import org.apache.flink.table.descriptors.TableFunctionTableEnvironment;
import org.apache.flink.table.descriptors.TableTableEnvironment;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableFunctionResult;
import org.apache.flink.table.descriptors.TableFunctionTableEnvironment;
import org.apache.flink.table.descriptors.TableFunctionTableInfo;
import org.apache.flink.table.descriptors.TableTableDescription;
import org.apache.flink.table.descriptors.TableTableEnvironment;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableDescription;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation;
import org.apache.flink.table.descriptors.TableAnnotationDescription;
import org.apache.flink.table.descriptors.TableAnnotationField;
import org.apache.flink.table.descriptors.TableAnnotationStyle;
import org.apache.flink.table.descriptors.TableTableInfo;
import org.apache.flink.table.descriptors.TableTableStyle;
import org.apache.flink.table.descriptors.TableAnnotation

