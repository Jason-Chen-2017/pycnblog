                 

# 《Kafka Connect原理与代码实例讲解》

> 关键词：Kafka Connect，数据集成，流处理，连接器，源Connector，SinkConnector，性能调优，安全配置

> 摘要：本文将深入探讨Kafka Connect的原理，包括其架构、核心组件、连接器类型、应用场景等。此外，还将通过代码实例详细讲解Kafka Connect的开发与实现，帮助读者更好地理解和使用Kafka Connect，以实现高效的数据集成和流处理。

## 第1章 Kafka Connect概述

### 1.1 Kafka Connect的概念与作用

Kafka Connect是一个可扩展的工具，用于连接Kafka集群和其他数据存储系统。它使得在Kafka集群之间进行数据集成变得更加容易，无论是从Kafka读取数据到其他系统，还是将数据从其他系统写入Kafka。

Kafka Connect的主要功能包括：

1. **数据集成**：可以轻松地将数据从各种数据源（如数据库、文件系统、队列等）导入到Kafka主题中，或者从Kafka主题导出到其他系统。
2. **流处理**：支持实时流处理，可以处理和分析数据流，而无需复杂的手动编程。
3. **数据同步**：可以实时同步数据，确保源和目标数据系统之间的数据一致性。
4. **数据导出**：可以将Kafka主题中的数据导出到外部系统中，用于后续分析或处理。

Kafka Connect在Kafka生态系统中的地位非常重要，它充当了Kafka与其他数据源和系统之间的桥梁，使得Kafka能够成为企业级数据处理和流处理平台的核心组成部分。

### 1.2 Kafka Connect架构

Kafka Connect的架构主要由三个核心组件组成：Coordinator、Worker和Connector。

#### Coordinator

Coordinator是Kafka Connect的集中管理组件，负责管理 Workers 和 Connectors。它提供了以下功能：

- **任务管理**：创建、启动、停止和监控任务。
- **配置管理**：存储和配置 Connector 的参数。
- **错误处理**：捕获和记录错误，提供任务状态信息。

#### Worker

Worker是运行在Kafka集群中的一个进程，负责执行Coordinator分配的任务。每个Worker可以运行多个Connector实例，它们从数据源读取数据，然后将数据写入Kafka主题或从Kafka主题读取数据，然后将数据写入目标系统。

#### Connector

Connector是Kafka Connect的核心组件，它负责将数据从源系统读取到Kafka主题或从Kafka主题写入到目标系统。Kafka Connect提供了内置的Connector，同时也支持自定义Connector。

### 1.3 Kafka Connect的类型

Kafka Connect主要分为以下几种类型：

#### 内置连接器

内置连接器是由Kafka Connect团队预编译和测试的，可以随Kafka Connect一起安装。常见的内置连接器包括：

- **File Stream**：从文件系统读取数据并将其写入Kafka主题。
- **JDBC**：从关系型数据库读取数据并将其写入Kafka主题。
- **MongoDB**：从MongoDB数据库读取数据并将其写入Kafka主题。
- **Kafka to Kafka**：从源Kafka主题读取数据并将其写入目标Kafka主题。

#### 自定义连接器

自定义连接器允许用户根据特定的需求编写自己的连接器。自定义连接器需要实现几个关键接口，如`SourceConnector`和`SinkConnector`。

#### 第三方连接器

第三方连接器是由外部组织或公司开发的，通常可以在Kafka Connect的GitHub存储库中找到。这些连接器提供了连接到各种外部系统的能力，如Amazon S3、Google BigQuery等。

### 1.4 Kafka Connect的应用场景

Kafka Connect的应用场景非常广泛，以下是一些典型的使用场景：

- **数据集成**：将数据从各种数据源导入到Kafka主题，以便进行进一步的处理和分析。
- **流处理**：在Kafka Connect中处理流数据，例如进行过滤、转换和聚合。
- **数据同步**：实时同步数据，确保源和目标系统之间的数据一致性。
- **数据导出**：将Kafka主题中的数据导出到外部系统，如数据仓库或分析工具。

## 第2章 Kafka Connect核心组件

### 2.1 Kafka Connect Coordinator

Coordinator是Kafka Connect的核心组件之一，负责管理 Workers 和 Connectors。以下是Coordinator的作用、工作原理和关键组件。

#### Coordinator的作用

- **任务管理**：Coordinator负责创建、启动、停止和监控任务。每个任务可以是一个或多个连接器的实例。
- **配置管理**：Coordinator存储和配置 Connector 的参数，确保连接器按照正确的配置运行。
- **错误处理**：Coordinator捕获和记录错误，提供任务状态信息，帮助用户诊断和解决问题。

#### Coordinator的工作原理

1. **初始化**：Coordinator启动时，会与Kafka集群进行连接，并创建一个专门的Kafka主题来存储任务元数据。
2. **任务分配**：Coordinator根据任务的配置和可用资源，将任务分配给 Workers。
3. **任务监控**：Coordinator定期检查任务的运行状态，如果发现任务失败或异常，会尝试重新启动任务。
4. **任务状态更新**：Coordinator将任务的运行状态更新到Kafka主题，以便其他组件（如 Worker 和 Kafka Tools）可以访问这些信息。

#### Coordinator的关键组件与配置

- **Kafka主题**：Coordinator使用一个专门的Kafka主题来存储任务元数据。这个主题通常具有高持久性和高可用性。
- **ZooKeeper**：Coordinator使用ZooKeeper来存储和同步元数据，确保在分布式环境中的一致性。
- **配置文件**：Coordinator的配置存储在Kafka Connect的配置文件中，这些配置包括 Coordinator 地址、Kafka 集群地址、主题名称等。

### 2.2 Kafka Connect Worker

Worker是Kafka Connect的执行组件，负责运行 Coordinator 分配的任务。以下是Worker的职责、工作流程、启动与配置、监控与故障处理。

#### Worker的职责

- **任务执行**：Worker运行连接器实例，从源系统读取数据并将其写入Kafka主题，或将数据从Kafka主题写入目标系统。
- **错误处理**：Worker捕获和处理连接器实例的错误，并尝试重新启动失败的任务。

#### Worker的工作流程

1. **初始化**：Worker启动时，会连接到 Coordinator，并从 Coordinator 获取任务的元数据。
2. **任务分配**：Worker根据元数据创建连接器实例，并开始执行任务。
3. **任务监控**：Worker定期检查任务的状态，并在任务失败时尝试重新启动。
4. **错误记录**：Worker记录错误信息，并更新任务状态。

#### Worker的启动与配置

- **启动命令**：启动 Worker 时，可以使用以下命令：
  ```bash
  bin/connect-distributed.sh --bootstrap servers=localhost:9092 --config files=connect-config.properties
  ```
- **配置文件**：Worker的配置存储在`connect-config.properties`文件中，这些配置包括 Kafka 集群地址、Coordinator 地址、日志配置等。

#### Worker的监控与故障处理

- **日志监控**：Worker的日志可以用来监控任务的运行状态，如果发现错误或异常，可以查看日志来诊断问题。
- **故障处理**：如果 Worker 进程意外停止，Coordinator 会将其标记为失败，并尝试重新启动它。

### 2.3 Kafka Connect Connector

Connector是Kafka Connect的核心组件，负责连接源系统（如数据库、文件系统等）和目标系统（如Kafka主题等）。以下是Connector的定义、核心类与方法以及生命周期管理。

#### Connector的定义

- **Connector**：是一个可以连接到源系统和目标系统的组件，它负责从源系统读取数据，将其转换，然后写入目标系统。
- **Source Connector**：从源系统读取数据的连接器。
- **Sink Connector**：将数据从源系统写入到目标系统的连接器。

#### Connector的核心类与方法

- **ConnectorConfig**：用于配置 Connector 的参数。
- **SourceConnector**：实现 SourceConnector 接口的类，负责从源系统读取数据。
- **SinkConnector**：实现 SinkConnector 接口的类，负责将数据写入目标系统。

#### Connector的生命周期管理

1. **启动**：Connector 在 Worker 启动时开始运行。
2. **初始化**：Connector 从 Coordinator 获取配置并初始化。
3. **启动连接器实例**：Connector 创建连接器实例并开始执行任务。
4. **关闭**：当 Worker 关闭时，Connector 会关闭连接器实例。

### 2.4 Kafka Connect Source Connector

Source Connector 负责从源系统读取数据并将其写入到 Kafka 主题。以下是 Source Connector 的工作原理、配置与实现以及性能调优。

#### Source Connector 的工作原理

1. **初始化**：Source Connector 从 Coordinator 获取配置并初始化。
2. **读取数据**：Source Connector 从源系统（如数据库、文件系统等）读取数据。
3. **写入 Kafka 主题**：将读取到的数据写入到 Kafka 主题。
4. **更新状态**：将任务的状态更新到 Coordinator。

#### Source Connector 的配置与实现

- **配置**：Source Connector 的配置包括源系统的连接信息、数据格式、批量大小等。
- **实现**：Source Connector 需要实现 SourceConnector 接口，并实现读取数据和写入 Kafka 主题的逻辑。

#### Source Connector 的性能调优

1. **批量大小**：调整批量大小可以影响性能，需要根据源系统和 Kafka 集群的能力来选择合适的批量大小。
2. **并发连接**：增加并发连接可以提高读取数据的速度，但需要考虑网络和数据库的负载。
3. **数据压缩**：使用数据压缩可以减少数据传输的带宽，但会增加 CPU 的负载。

### 2.5 Kafka Connect Sink Connector

Sink Connector 负责将数据从 Kafka 主题写入到目标系统（如数据库、文件系统等）。以下是 Sink Connector 的工作原理、配置与实现以及性能调优。

#### Sink Connector 的工作原理

1. **初始化**：Sink Connector 从 Coordinator 获取配置并初始化。
2. **读取 Kafka 主题**：从 Kafka 主题中读取数据。
3. **写入目标系统**：将读取到的数据写入到目标系统（如数据库、文件系统等）。
4. **更新状态**：将任务的状态更新到 Coordinator。

#### Sink Connector 的配置与实现

- **配置**：Sink Connector 的配置包括目标系统的连接信息、数据格式、批量大小等。
- **实现**：Sink Connector 需要实现 SinkConnector 接口，并实现读取 Kafka 主题和写入目标系统的逻辑。

#### Sink Connector 的性能调优

1. **批量大小**：调整批量大小可以影响性能，需要根据 Kafka 集群和目标系统的能力来选择合适的批量大小。
2. **并发连接**：增加并发连接可以提高写入目标系统的速度，但需要考虑网络和数据库的负载。
3. **数据压缩**：使用数据压缩可以减少数据传输的带宽，但会增加 CPU 的负载。

## 第3章 Kafka Connect连接器开发

### 3.1 连接器开发基础

连接器开发是Kafka Connect的核心内容，它涉及到配置、接口实现和测试等多个方面。以下是连接器开发的基础知识和最佳实践。

#### 连接器开发的准备环境

- **Kafka**：确保已经安装和配置了Kafka集群，包括 ZooKeeper 和 Kafka Broker。
- **Kafka Connect**：下载和安装Kafka Connect，并准备好运行 Worker 的环境。
- **开发工具**：安装Java开发工具（如 IntelliJ IDEA 或 Eclipse），并配置Maven或其他依赖管理工具。

#### 连接器开发的核心步骤

1. **创建项目**：使用 Maven 或 Gradle 创建一个新的项目，并添加 Kafka Connect 相关的依赖。
2. **编写配置类**：创建一个配置类，用于定义连接器的配置参数，如数据库连接信息、Kafka主题名称等。
3. **实现 Connector 接口**：根据需要实现 SourceConnector 或 SinkConnector 接口，实现连接器的主要逻辑，如数据读取、写入和转换等。
4. **测试连接器**：编写单元测试，确保连接器能够正确读取和写入数据，并处理各种异常情况。
5. **构建和部署**：构建连接器并部署到 Kafka Connect，以便进行测试和实际应用。

#### 连接器开发的最佳实践

- **模块化设计**：将连接器分成多个模块，如配置模块、数据读取模块、数据写入模块等，便于维护和扩展。
- **错误处理**：确保连接器能够处理各种异常情况，如网络故障、数据库连接失败等，并记录错误日志。
- **性能优化**：针对连接器的性能进行优化，如批量处理、并发连接等，确保连接器能够高效运行。
- **文档和示例**：编写详细的文档和示例代码，帮助其他开发者理解和使用连接器。

### 3.2 内置连接器分析

Kafka Connect提供了多个内置连接器，以下分析几个常见的内置连接器：FileStream、JDBC和Redis连接器。

#### FileStream 连接器

FileStream 连接器用于读取文件系统中的文件并将其写入Kafka主题。以下是FileStream连接器的基本配置和使用方法。

**配置**：

```properties
# 文件系统路径
file Streamsconnector.file-stream.path=/path/to/files
# Kafka主题
file Streamsconnector.file-stream.topic=my-topic
# 文件模式（追加或覆盖）
file Streamsconnector.file-stream.mode=APPEND
```

**使用方法**：

1. 将文件放置在指定的文件系统路径下。
2. 启动FileStream连接器，连接器会定期读取文件并将其写入到Kafka主题。

#### JDBC 连接器

JDBC 连接器用于从关系型数据库读取数据并将其写入Kafka主题。以下是JDBC连接器的基本配置和使用方法。

**配置**：

```properties
# 数据库连接信息
jdbc connector.jdbc.url=jdbc:mysql://localhost:3306/mydb
jdbc connector.jdbc.user=root
jdbc connector.jdbc.password=mysecret
# Kafka主题
jdbc connector.topic=my-topic
```

**使用方法**：

1. 配置数据库连接信息。
2. 启动JDBC连接器，连接器会定期从数据库中读取数据并将其写入到Kafka主题。

#### Redis 连接器

Redis 连接器用于读取Redis数据库中的数据并将其写入Kafka主题。以下是Redis连接器的基本配置和使用方法。

**配置**：

```properties
# Redis连接信息
redis connector.redis.host=localhost
redis connector.redis.port=6379
# Kafka主题
redis connector.topic=my-topic
```

**使用方法**：

1. 配置Redis连接信息。
2. 启动Redis连接器，连接器会定期从Redis数据库中读取数据并将其写入到Kafka主题。

### 3.3 自定义连接器实现

自定义连接器是Kafka Connect的核心功能之一，它允许开发者根据特定的需求编写自己的连接器。以下是自定义连接器的设计与架构、实现与测试以及部署与监控。

#### 自定义连接器的设计与架构

1. **需求分析**：分析连接器的需求，确定连接器的功能、配置和性能要求。
2. **架构设计**：设计连接器的架构，确定连接器的核心组件和接口。
3. **模块划分**：将连接器划分为多个模块，如配置模块、数据读取模块、数据写入模块等。

#### 自定义连接器的实现与测试

1. **实现接口**：根据架构设计实现连接器的接口，如`SourceConnector`和`SinkConnector`。
2. **编写配置类**：创建配置类，用于定义连接器的配置参数。
3. **实现数据读取和写入逻辑**：实现从源系统读取数据和将数据写入目标系统的逻辑。
4. **编写单元测试**：编写单元测试，确保连接器能够正确读取和写入数据，并处理各种异常情况。

#### 自定义连接器的部署与监控

1. **构建和打包**：使用 Maven 或 Gradle 构建连接器，并生成可执行的 JAR 文件。
2. **部署连接器**：将连接器部署到 Kafka Connect，通常是将 JAR 文件放置在 Kafka Connect 的插件目录下。
3. **启动连接器**：启动 Kafka Connect，连接器将自动加载并开始执行。
4. **监控连接器**：监控连接器的运行状态，包括任务的状态、日志和性能指标。

### 3.4 连接器性能优化

连接器的性能优化是确保连接器能够高效运行的关键。以下是连接器性能评估方法、性能瓶颈分析和优化策略。

#### 性能评估方法

1. **基准测试**：使用基准测试工具（如 Apache JMeter）模拟负载，评估连接器的性能。
2. **压力测试**：增加负载，评估连接器的稳定性和响应时间。
3. **监控指标**：监控连接器的性能指标，如吞吐量、延迟、错误率等。

#### 性能瓶颈分析

1. **网络瓶颈**：检查网络带宽和延迟，确定是否存在网络瓶颈。
2. **数据库瓶颈**：检查数据库的查询性能和索引优化，确定是否存在数据库瓶颈。
3. **Kafka 瓶颈**：检查 Kafka 集群的分区数、副本数和负载，确定是否存在 Kafka 瓶颈。

#### 优化策略

1. **批量处理**：增加批量大小，减少 I/O 操作和网络传输次数。
2. **并发连接**：增加并发连接，提高数据读取和写入的速度。
3. **数据压缩**：使用数据压缩，减少数据传输的带宽，但会增加 CPU 的负载。
4. **缓存使用**：使用缓存技术，减少对数据库的查询次数。
5. **负载均衡**：将负载均衡到多个连接器实例，提高系统的可用性和性能。

## 第4章 Kafka Connect实战应用

### 4.1 数据集成应用案例

数据集成是将数据从不同的数据源（如数据库、文件系统等）导入到Kafka主题的过程。以下是一个数据集成应用案例的详细步骤。

#### 案例需求

假设我们有一个员工信息数据库，需要将员工信息定期导入到Kafka主题中，以便进行实时分析。

#### 实现步骤

1. **数据库连接**：配置JDBC连接器，连接到员工信息数据库。

   ```properties
   jdbc connector.jdbc.url=jdbc:mysql://localhost:3306/employee_db
   jdbc connector.jdbc.user=root
   jdbc connector.jdbc.password=mysecret
   ```

2. **查询员工信息**：编写SQL查询语句，查询员工表中的数据。

   ```sql
   SELECT * FROM employees;
   ```

3. **写入Kafka主题**：配置JDBC连接器，将查询结果写入Kafka主题。

   ```properties
   jdbc.connector.topic=employee-topic
   ```

4. **启动JDBC连接器**：启动JDBC连接器，将员工信息定期导入到Kafka主题。

   ```bash
   bin/connect-distributed.sh --bootstrap servers=localhost:9092 --config files=connect-config.properties
   ```

5. **数据导入监控**：监控JDBC连接器的运行状态，确保数据导入过程正常。

   ```bash
   bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic employee-topic --from-beginning
   ```

#### 案例总结

通过配置JDBC连接器，我们可以轻松地将员工信息数据库中的数据导入到Kafka主题中。这样，我们就可以利用Kafka的实时处理能力，对员工信息进行实时分析。

### 4.2 流处理应用案例

流处理是将Kafka主题中的数据实时处理的过程。以下是一个流处理应用案例的详细步骤。

#### 案例需求

假设我们有一个用户行为日志的Kafka主题，需要实时计算每个用户的点击量。

#### 实现步骤

1. **配置Kafka Connect**：配置Kafka Connect，将用户行为日志导入到Kafka主题。

   ```properties
   file connector.topic=my-log-topic
   file connector.path=/path/to/logs
   ```

2. **编写流处理程序**：使用Apache Flink或Apache Spark等流处理框架，编写流处理程序，计算每个用户的点击量。

   ```java
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0</span>
```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0</span>
```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0
   <span style="color:green;">`<br>```</span>
   DataStream<String> logStream = env.addSource(new FlinkKafkaConsumer0

```<span style="color:green;">`<br

