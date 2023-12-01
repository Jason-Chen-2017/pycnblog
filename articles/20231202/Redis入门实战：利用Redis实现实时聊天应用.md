                 

# 1.背景介绍

Redis是一个开源的高性能key-value存储系统，它支持数据的持久化，可基于内存（in-memory）进行操作。Redis 提供多种语言的API。

Redis 是一个使用 ANSI C 语言编写、遵循 BSD 协议的开源 (open source)、跨平台 (Cross Platform) 的日志型、Key-Value (Key-Value) 存储数据库，并提供多种语言的 API。Redis 可以在内存中进行操作 (In-Memory),并且还提供了持久性的数据存储功能。

Redis支持各种语言（如：Python, Java, Go, Node.js等），并且具有丰富的客户端API和命令集合，这使得开发者可以轻松地与Redis进行交互和操作。此外，Redis还提供了许多高级功能，例如发布/订阅、消息队列、事务等，这些功能使得Redis成为一个非常强大和灵活的数据库解决方案。

## Redis与其他NoSQL数据库产品对比
|                    | Redis                                      | Memcached                               | MongoDB                               | HBase                              | Cassandra                       | Couchbase                     | Neo4j                        | Elasticsearch                | InfluxDB                     | Riak KV(Basho Technologies)   | Amazon DynamoDB(AWS)         | Google Cloud Datastore(Google)| Couchbase Mobile             | Couchbase Lite(Couchbase)    | Riak CS(Basho Technologies)   | DataStax Enterprise          | Apache Solr(Lucene)          | Apache Druid(Incubating at ASF)| OpenTSDB(Apache Incubator at ASF)| OpenStack Swift             | Trove(OpenStack Database as a Service)| Hadoop HBase (Hadoop Ecosystem)| Accumulo (Apache Project)| Aerospike (Aerospike Inc.)| TokuMX (Tokutek)            | MongoRiver (MongoDB Inc.)    | OrientDB (Orient Technologies)| ArangoDB (ArangoDB GmbH)&nbsp;| RethinkDB&nbsp;(RethinkDB Inc.)| Voldemort&nbsp;(Twitter)&nbsp;| Project Voldemort&nbsp;      || Key-Value Store & Pub/Sub & Stream & Geospatial & HyperLogLog & Lua Scripting & Cluster Formation & Master-Slave Replication & Sentinel Monitoring || Key-Value Store && Pub/Sub && Geospatial && HyperLogLog && Lua Scripting && Cluster Formation && Master-Slave Replication || Key-Value Store && Pub/Sub && Geospatial && HyperLogLog && Lua Scripting || Key-Value Store || Document Database || Time Series Database || Graph Database || Search Engine || Wide ColumnStore