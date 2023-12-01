                 

# 1.背景介绍

Redis是一个开源的高性能key-value存储系统，它支持数据的持久化，可基于内存（Redis）或磁盘（Redis-Disk）。Redis 提供多种语言的 API，包括：Ruby、Python、Java、C 和 C#等。

Redis 是一个使用 ANSI C 语言编写、遵循 BSD 协议的开源 ( BSD ) 软件。 Redis 的根目录下有许多 C 文件，这些文件定义了 Redis 服务器的核心功能。 Redis 使用起来非常简单，因为它提供了丰富的数据类型，例如字符串 (string)、列表 (list)、集合 (sets)、有序集合 (sorted sets) 以及哈希 (hash)。

Redis是一个基于内存的数据库，所以它不适合长时间保存大量数据。但是在需要快速访问和操作数据时，Redis非常适合使用。例如：实时统计和监控、缓存等场景。

## Redis与其他NoSQL产品对比
|                 | Redis                   | Memcached               | MongoDB                    | HBase                     | Cassandra                   | Neo4j                      | Elasticsearch            | InfluxDB                   | Riak KV               | Couchbase Server          | DataStax Enterprise       | OrientDB                | ArangoDB                | DynamoDB              | Hadoop HDFS             | Hadoop HBase            | Accumulo                | Solr                     | MarkLogic              | FlockDB                | Couchbase Mobile        | Couchbase Lite          | RavenDB             		| Riak CS              		| ScyllaDB          		| Foundation DB        		| OrientDB            		| Aerospike           		| TokuMX for MongoDB   		| Hypertable           	| VoltDB            	   	| Google Cloud Datastore   | Azure Table Storage      | Amazon Simple DB         | Cosmos DB               | Oracle NoSQL Database    | SAP HANA Vora            | Spanner                  │ ClustrixDatabase      │ MongoRiver        │ Voldemort         │ Apache Ignite        │ Apache Druid         │ Apache Geode        │ Apache Samza         │ Apache Kafka          │ Apache Flink          │ Apache Beam          │ Alluxio             |||||||||||||||||||||||||| || || || || || || || || || || || || ||