
作者：禅与计算机程序设计艺术                    

# 1.简介
  

随着互联网业务的快速发展，海量数据正在涌现出巨大的价值。如何高效、低延迟地存储和消费这些海量数据，成为了企业必须面对的难题之一。
Apache Kafka就是一个开源分布式流处理平台，它通过提供高吞吐率、低延迟、可扩展性和容错能力，满足了上述需求。本书从零开始带领读者一起学习Kafka，全面掌握其核心机制、典型应用场景和实践技巧，使得读者能够更好地理解Kafka并运用到实际工作中。
Kafka是个“小而美”的消息队列系统，它的设计目标就是快速、可靠、持久化地存储和传递数据，可以用于各种实时事件流处理、日志收集、运维告警等场景。通过精心设计的API，Kafka允许用户在数据源和数据处理流程之间灵活切换，实现实时的复杂数据处理。
本书共分为三个部分，第一章节介绍Kafka的基本概念和架构，第二章节详细阐述Kafka核心机制及其最佳实践方法，第三章节根据不同场景介绍实践案例并给出相应的最佳实践建议，希望读者能够从中受益。
本书适合具有一定编程经验、有一定相关知识储备的技术人员阅读。对于非技术人员，本书还可以作为Kafka的入门教程，让他们熟悉Kafka并快速掌握其核心知识。同时，作者会结合自己的实际工作经历和学习心得，进一步完善本书的内容，力争打造一本真正易于阅读和实践的Kafka书籍。
本书内容如下：
# 一、前言 
# 二、Kafka概述
# 三、核心机制 
# 四、主要组件
# 五、典型应用场景
# 六、Kafka集群部署
# 七、Kafka安全配置
# 八、Kafka高可用集群部署
# 九、Kafka水平扩展
# 十、Kafka性能测试和调优
# 十一、Kafka源码分析
# 十二、Kafka社区生态及开源框架
# 十三、Kafka实战案例分享
# 十四、后记 
# 参考资料： 
# https://book.douban.com/subject/27152149/ 
# https://blog.csdn.net/u011402419/article/details/103485686 
# http://www.cnblogs.com/hester/p/11213117.html 
 # Kafka 简单示例-生产者消费者模式
```python
from kafka import KafkaProducer

producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                         value_serializer=lambda x: str(x).encode('utf-8'))

for i in range(10):
    producer.send('my-topic', b'some_message')

producer.close()
```
# 配置文件 kafka.properties 

```java
# broker设置
broker.id=0
listeners=PLAINTEXT://127.0.0.1:9092
log.dirs=/tmp/kafka-logs

# zookeeper设置
zookeeper.connect=localhost:2181

# 是否开启sasl认证（默认为false）
security.inter.broker.protocol=SASL_PLAINTEXT

# sasl配置（选择PLAINTEXT时无需配置）
sasl.mechanism.inter.broker.protocol=PLAIN
sasl.username=testuser
sasl.password=<PASSWORD>
```