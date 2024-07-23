                 

## 1. 背景介绍

在深度学习大行其道的今天，Pregel却独树一帜，以图计算为核心的计算框架成为计算机科学领域的一颗明珠。Pregel的设计初衷在于：通过分布式并行计算的方式，处理大规模图结构数据，实现数据上的高效计算和处理。

### 1.1 问题由来
Pregel源于Google于2005年发布的MapReduce论文。在这篇论文中，Google提出了MapReduce模型，标志着分布式计算进入了一个新的阶段。然而，MapReduce只能处理简单结构的数据，对于大规模图数据，其处理效率和灵活性不够。

为了解决这些问题，Google的Jeff Dean在MapReduce的基础上，提出了Pregel算法。Pregel框架主要面向大规模图数据的计算，其设计理念是将大规模图数据分解为小规模子图，利用分布式计算资源处理小规模子图，最终通过聚合操作将结果汇总，形成全局答案。

### 1.2 问题核心关键点
Pregel算法的设计核心在于：

1. **Map-Reduce模型**：通过将图数据分解成多个小规模子图，每个小规模子图作为一个Map任务，进行本地处理。Map任务的输出数据通过Shuffle操作，送到相应的Reduce任务中进行聚合操作。
2. **数据并行处理**：将图数据拆分成多个小规模子图，在多个计算节点上并行处理，显著提升了计算效率。
3. **高效的图处理算法**：通过定义和实现新的图处理算法，可以高效处理大规模图数据，处理复杂图查询，如PageRank算法、社区检测算法等。
4. **自包含的编程模型**：Pregel提供了一套自包含的编程模型，使得用户可以方便地编写和部署分布式图计算任务。

## 2. 核心概念与联系

### 2.1 核心概念概述

Pregel的核心概念包括以下几个关键点：

1. **顶点（Vertex）**：图中的基本节点，表示一个数据实体，可以是一个用户、一条边、一个网页等。
2. **边（Edge）**：连接两个顶点的线，表示顶点之间的关联关系。
3. **顶点计算（Vertex Compute）**：定义在每个顶点上的计算任务，通常由用户自定义。
4. **消息传递（Message Passing）**：顶点的计算任务中，通过发送消息传递数据给相邻的顶点。
5. **超时机制（Timeout）**：在消息传递过程中，设定一个超时机制，防止死锁和计算节点资源耗尽。
6. **迭代次数（Iteration Count）**：定义算法需要迭代的次数，通常为1。

这些概念构成了Pregel的核心框架，为大规模图计算提供了基本的模型和算子。

### 2.2 核心概念原理和架构的 Mermaid 流程图

```mermaid
graph TB
    A[顶点 (Vertex)] --> B[边 (Edge)]
    B --> C[顶点计算 (Vertex Compute)]
    C --> D[消息传递 (Message Passing)]
    D --> E[超时机制 (Timeout)]
    E --> F[迭代次数 (Iteration Count)]
```

## 3. 核心算法原理 & 具体操作步骤
### 3.1 算法原理概述

Pregel算法的核心思想是：将大规模图数据分解为多个小规模子图，在每个子图上执行顶点计算，通过消息传递将计算结果汇总，最终得到全局结果。其基本步骤如下：

1. **初始化**：将整个图数据分解为多个小规模子图，每个子图作为一个计算节点。
2. **迭代计算**：在每个计算节点上，执行顶点计算和消息传递，生成新的顶点计算结果。
3. **汇总输出**：将每个计算节点的计算结果汇总，生成全局结果。

### 3.2 算法步骤详解

Pregel算法主要包含以下几个步骤：

#### 3.2.1 初始化

- **设置图数据**：定义图数据，包括顶点集合 $V$ 和边集合 $E$。
- **顶点集合**：定义顶点集合 $V$，包括所有顶点 $v \in V$。
- **边集合**：定义边集合 $E$，包括所有边 $e \in E$。
- **顶点状态**：初始化每个顶点的状态，通常为空。

#### 3.2.2 迭代计算

- **消息传递**：在每个计算节点上，定义顶点计算任务，通过消息传递算法计算新状态。
- **顶点计算**：在每个顶点上执行计算任务，更新顶点状态。
- **消息传递**：在顶点状态更新完成后，通过消息传递算法，将计算结果传递给相邻的顶点。
- **超时机制**：在消息传递过程中，设定一个超时机制，防止死锁和计算节点资源耗尽。

#### 3.2.3 汇总输出

- **汇总结果**：将每个计算节点的计算结果汇总，生成全局结果。

### 3.3 算法优缺点

Pregel算法具有以下优点：

1. **高效处理大规模图数据**：通过将图数据分解为多个小规模子图，在多个计算节点上并行处理，显著提升了计算效率。
2. **灵活性高**：通过用户自定义顶点计算任务和消息传递算法，可以灵活处理各种复杂的图查询。
3. **可扩展性强**：Pregel框架支持分布式计算，可以通过增加计算节点来扩展计算能力。

同时，Pregel算法也存在一些局限性：

1. **依赖于消息传递**：Pregel算法主要通过消息传递实现数据交换，对于大规模数据的交换，通信开销较大。
2. **自包含性不足**：Pregel框架没有提供丰富的图处理函数库，需要用户自定义实现。
3. **编程复杂度较高**：Pregel算法要求用户自定义顶点计算任务和消息传递算法，编程复杂度较高。

### 3.4 算法应用领域

Pregel算法主要应用于大规模图数据的计算和处理，具有广泛的应用场景，包括：

1. **社交网络分析**：通过分析社交网络中的用户关系和活动，可以发现社区、识别网络中的异常行为等。
2. **推荐系统**：通过分析用户和物品之间的关系，可以为用户推荐个性化的物品。
3. **搜索优化**：通过优化搜索结果的排序和过滤，提升用户搜索体验。
4. **路由优化**：通过优化网络路由路径，提高网络传输效率。
5. **网络分析**：通过分析网络流量和拓扑结构，发现网络中的异常行为和漏洞。

## 4. 数学模型和公式 & 详细讲解 & 举例说明

### 4.1 数学模型构建

Pregel算法的主要数学模型包括顶点计算和消息传递两个方面。

#### 4.1.1 顶点计算

定义顶点 $v$ 的状态为 $\mathbf{v}_t$，其中 $t$ 表示迭代次数。顶点计算函数为 $\mathcal{V}_t(v, \mathcal{N}(v))$，其中 $\mathcal{N}(v)$ 表示顶点 $v$ 的邻居集合。顶点计算函数为：

$$
\mathbf{v}_{t+1} = \mathcal{V}_t(\mathbf{v}_t, \mathcal{N}(v))
$$

#### 4.1.2 消息传递

定义顶点 $v$ 接收到的消息为 $m_{in}(v)$，发送的消息为 $m_{out}(v)$。消息传递算法定义了消息传递函数 $\mathcal{M}(v)$，消息传递函数为：

$$
\mathcal{M}(v) = (m_{in}(v), m_{out}(v))
$$

### 4.2 公式推导过程

#### 4.2.1 顶点计算公式

顶点计算函数的推导过程如下：

$$
\mathbf{v}_{t+1} = \mathcal{V}_t(\mathbf{v}_t, \mathcal{N}(v))
$$

其中 $\mathcal{V}_t$ 为顶点计算函数，$\mathbf{v}_t$ 为顶点 $v$ 在迭代次数 $t$ 的状态，$\mathcal{N}(v)$ 为顶点 $v$ 的邻居集合。

#### 4.2.2 消息传递公式

消息传递函数的推导过程如下：

$$
\mathcal{M}(v) = (m_{in}(v), m_{out}(v))
$$

其中 $m_{in}(v)$ 为顶点 $v$ 接收到的消息，$m_{out}(v)$ 为顶点 $v$ 发送的消息。

### 4.3 案例分析与讲解

#### 4.3.1 单源最短路径算法

单源最短路径算法是Pregel算法的经典应用之一，用于计算从一个顶点到其他所有顶点的最短路径。

- **初始化**：将起点状态 $s_0$ 设置为0，其他顶点状态设置为无穷大。
- **顶点计算**：对于每个顶点 $v$，如果 $v$ 为起点，则 $\mathbf{v}_t = s_0$，否则 $\mathbf{v}_t = \min(\mathbf{v}_{t-1}, \mathcal{N}(v)_{in}(v))$。
- **消息传递**：对于每个顶点 $v$，将 $\mathbf{v}_t$ 发送给所有邻居顶点 $u$。
- **超时机制**：设定超时时间为 $k$，如果消息传递超时，则更新消息为无穷大。

#### 4.3.2 PageRank算法

PageRank算法是Google搜索引擎的核心算法之一，用于计算网页的重要性。

- **初始化**：将每个顶点的状态 $r_0$ 设置为1。
- **顶点计算**：对于每个顶点 $v$，$\mathbf{r}_{t+1}(v) = \sum_{u \in \mathcal{N}(v)} \frac{\mathbf{r}_t(u)}{|\mathcal{N}(u)|} \times 1 - \alpha$。
- **消息传递**：对于每个顶点 $v$，将 $\mathbf{r}_t(v)$ 发送给所有邻居顶点 $u$。
- **超时机制**：设定超时时间为 $k$，如果消息传递超时，则更新消息为0。

## 5. 项目实践：代码实例和详细解释说明

### 5.1 开发环境搭建

#### 5.1.1 安装Pregel

Pregel框架需要依赖于Java、Hadoop和Giraph。因此，需要先安装这些依赖。

1. 安装Java
```bash
sudo apt-get update
sudo apt-get install openjdk-8-jdk
```

2. 安装Hadoop
```bash
wget http://apache-hadoop.s3.amazonaws.com/hadoop-2.7.1/hadoop-2.7.1.tar.gz
tar -xzvf hadoop-2.7.1.tar.gz
cd hadoop-2.7.1
```

3. 安装Giraph
```bash
git clone https://github.com/apache/giraph.git
cd giraph
mvn clean package
```

4. 配置环境变量
```bash
export HADOOP_HOME=/path/to/hadoop-2.7.1
export HADOOP_CLASSPATH=$HADOOP_CLASSPATH:$GIRAPH_HOME/target/hadoop-project/hadoop-core-2.7.1.jar:$GIRAPH_HOME/target/hadoop-project/hadoop-giraph-2.7.1.jar:$GIRAPH_HOME/target/hadoop-project/hadoop-examples-2.7.1.jar:$GIRAPH_HOME/target/hadoop-project/hadoop-examples-2.7.1.jar:$GIRAPH_HOME/target/hadoop-project/hadoop-shell-2.7.1.jar
```

#### 5.1.2 配置Hadoop

1. 配置hdfs-site.xml
```xml
<configuration>
  <property>
    <name>dfs.replication</name>
    <value>1</value>
  </property>
  <property>
    <name>dfs.namenode.name.dir</name>
    <value>hdfs://localhost:9000</value>
  </property>
</configuration>
```

2. 配置yarn-site.xml
```xml
<configuration>
  <property>
    <name>yarn.resourcemanager.address</name>
    <value>yarn-resourcemanager:8121</value>
  </property>
  <property>
    <name>yarn.nodemanager.address</name>
    <value>yarn-nodemanager:8000</value>
  </property>
</configuration>
```

3. 启动Hadoop
```bash
start-dfs.sh
start-yarn.sh
```

#### 5.1.3 配置Pregel

1. 下载Pregel代码
```bash
wget http://www.inf.ed.ac.uk/hdd/publication/data/122.1.0.zip
unzip 122.1.0.zip
cd 122.1.0
```

2. 配置giraph-site.xml
```xml
<configuration>
  <property>
    <name>giraph.container</name>
    <value>giraph</value>
  </property>
  <property>
    <name>giraph.task.manager</name>
    <value>localhost:12345</value>
  </property>
  <property>
    <name>giraph.task.manager.max.property</name>
    <value>5</value>
  </property>
  <property>
    <name>giraph.task.executor</name>
    <value>giraph</value>
  </property>
  <property>
    <name>giraph.task.executor.max.property</name>
    <value>5</value>
  </property>
  <property>
    <name>giraph.task.executor.stdout</name>
    <value>true</value>
  </property>
  <property>
    <name>giraph.task.executor.stderr</name>
    <value>true</value>
  </property>
  <property>
    <name>giraph.task.heartbeat.interval</name>
    <value>10</value>
  </property>
</configuration>
```

3. 启动Giraph
```bash
start-giraph.sh
```

### 5.2 源代码详细实现

#### 5.2.1 单源最短路径算法

```java
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop.io.PointWritable;
import org.apache.hadoop

