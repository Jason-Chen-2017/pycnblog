
作者：禅与计算机程序设计艺术                    

# 1.简介
  

GeoSpark是Apache Spark上针对地理数据处理的一套开源框架。GeoSpark提供对空间数据的处理、分析、统计等功能。支持Scala、Java、Python、R语言的API接口。GeoSpark包含的主要模块包括：

1.SpatialRDD：提供对空间数据的底层存储结构及相关分析方法；

2.SpatialOperator：提供了对空间数据进行各种运算操作的函数，如空间连接、拼接、缓冲、重投影、几何关系计算等；

3.ST_GeomFromWKT：将WKB或WKT表示的空间数据解析成Geometry类型；

4.SpatialPartitioner：用于根据给定的索引划分策略对空间数据进行分区并行化操作，可以提升计算效率；

5.ShapefileReaderWriter：基于shapefile文件格式读写空间数据。

GeoSpark依赖于Apache Sedona项目，Sedona是Apache Sedona (incubating)的简称，是一个开源的分布式图形引擎，它由一系列基于Apache Spark的子项目组成，能够在内存中处理复杂的地理空间数据。

本文主要介绍如何安装GeoSpark。首先，下载GeoSpark对应的版本。从https://repo1.maven.org/maven2/org/datasyslab/geospark/geospark-core/页面找到最新的版本号，然后访问https://repo1.maven.org/maven2/org/datasyslab/geospark/geospark-core/<version>/geospark-core-<version>-jar-with-dependencies.jar获取相应的包。下载完毕后，将包解压到本地磁盘目录。接下来，配置环境变量，编辑配置文件。最后，验证GeoSpark是否成功安装。

# 2.系统要求
系统环境需求: 

1. Java Development Kit (JDK), version 8 or later; 
2. Apache Spark, version 2.4.x or later. 

推荐安装方式：使用Apache Spark的链接进行下载和安装。

 # 3.安装步骤
## 3.1 下载GeoSpark
从https://github.com/DataSystemsLab/GeoSpark/releases 页面选择适合自己机器的版本进行下载。

GeoSpark的分支结构如下：
```
GeoSpark
    |- geospark-project
        |- core: GeoSpark Core APIs and Implementation
        |- kryo: Kryo support for SpatialRDD
        |- proj4j: PROJ4J library wrapper with added methods for JTS Geometry conversion
        |- sql: SQL support using Apache Spark SQL and DataFrame API 
        |- tools: Tools to generate PartitioningStrategy and SpatialPartitioners
```

下载最新版本的源码压缩包并解压，GeoSpark-All-Modules文件夹就在你的下载目录。

## 3.2 配置环境变量
### 为GeoSpark配置Maven仓库
打开 ~/.m2/settings.xml 文件（没有则创建一个），如果里面没有以下内容，添加进去：
```
<repositories>
    <repository>
        <id>geospark-maven</id>
        <url>https://dl.bintray.com/datasystemslab/maven/</url>
    </repository>
</repositories>
```

### 为GeoSpark配置环境变量
在 ~/.bashrc 文件末尾加上以下内容：
```
export SPARK_HOME=/path/to/your/spark/installation
export PATH=$SPARK_HOME/bin:$PATH
export CLASSPATH=$CLASSPATH:/path/to/GeoSpark-All-Modules/target/*:/path/to/GeoSpark-All-Modules/external/datasketches/target/*:/path/to/GeoSpark-All-Modules/external/spatial4j/target/*:/path/to/GeoSpark-All-Modules/external/jts/target/*:/path/to/GeoSpark-All-Modules/external/hadoop/target/*:/path/to/GeoSpark-All-Modules/external/avro/target/*:/path/to/GeoSpark-All-Modules/external/json4s/target/*:/path/to/GeoSpark-All-Modules/external/kryo/target/*:/path/to/GeoSpark-All-Modules/external/metrics/target/*:/path/to/GeoSpark-All-Modules/external/scala-logging/target/*:/path/to/GeoSpark-All-Modules/external/scopt/target/*:/path/to/GeoSpark-All-Modules/external/geotrellis/target/*:/path/to/GeoSpark-All-Modules/external/proj4j/target/*:/path/to/GeoSpark-All-Modules/external/sedona/target/*:/path/to/GeoSpark-All-Modules/external/scalaj-http/target/*:/path/to/GeoSpark-All-Modules/external/scalatest/target/*:/path/to/GeoSpark-All-Modules/external/spark-sql-kafka-0-10/target/*:/path/to/GeoSpark-All-Modules/external/case-classes/target/*
```
其中：

SPARK_HOME 是你Spark的安装路径；

替换 `/path/to/` 为实际路径；

GeoSpark-All-Modules/target/ 目录下有所有JAR包；

GeoSpark-All-Modules/external/xxx/target/ 下的JAR包是在运行外部工具时需要用到的；

例如：`/path/to/GeoSpark-All-Modules/external/datasketches/target`目录下的所有JAR包都要加入到 classpath 中。

保存好之后，执行 `source ~/.bashrc` 命令使之生效。

## 3.3 检验GeoSpark是否正确安装