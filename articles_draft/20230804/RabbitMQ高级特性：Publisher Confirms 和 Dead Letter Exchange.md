
作者：禅与计算机程序设计艺术                    

# 1.简介
         
 RabbitMQ是一个开源的AMQP（Advanced Message Queuing Protocol）协议实现消息队列中间件，其提供多个功能特性，例如可靠性传递，确保消息被投递一次且仅一次，支持广播、组播等消息模式，并且它还具有高度可扩展性和容错能力。本文将详细介绍RabbitMQ中的两个高级特性——Publisher Confirms 和 Dead Letter Exchange。
          # 2.基本概念术语说明
          ## Publisher Confirms
          在 AMQP 中，publisher-acknowledgements 是指当一个消息发布到队列后，会向消费者发送确认消息（Basic.Ack），表示已收到消息并处理完成。而 publisher confirms 是 RabbitMQ 提供的一个插件机制，用来确认消息是否已经正确地进入到队列中。在 RabbitMQ 的 publisher confirms 模式下，一个消息被发布到队列后，RabbitMQ 会将该消息持久化到磁盘上，同时返回一个唯一标识符（delivery tag）给发布者。消费者接收到消息后，可以根据 delivery tag 来检查确认消息是否已经到达队列中。如果确认消息没有到达，则可以重新发送该消息。
          
          启用了 publisher confirms 特性的队列的行为方式如下：
           - 当生产者发送消息时，生产者会等待直到所有发送者和交换机都通过了确认消息；
           - 如果所有的消费者都确认了消息，则生产者发送确认消息；
           - 如果有一个或多个消费者没确认消息，则生产者会发送 Nack 报文；
           
          当 producer confirmations 没有得到足够的确认时，producer 将会停止等待，从而避免资源耗尽。
          
          ### 操作步骤
          1. 配置 RabbitMQ 启用 publisher confirms 插件
             rabbitmq-plugins enable rabbitmq_amqp1_0

           2. 创建确认队列
             使用命令：
             
                rabbitmqctl add_queue name=confirms durable=true auto_delete=false

                参数含义：

                 name：确认队列名称，一般为“confirms”;

                 durable：是否持久化存储消息，即消息不丢失;

                 auto_delete：是否自动删除此队列，即关闭服务器后，队列也会消失。
                 
          3. 设置 publisher confirms 属性
             通过设置队列属性“x-confirm-mode”的值为 “persistent”，开启 publisher confirms 特性。

              rabbitmqctl set_queue_attributes queue=<确认队列名> x-confirm-mode=persistent

          ## Dead Letter Exchange 
          为了解决消息未能成功投递的问题，RabbitMQ 允许创建 Dead Letter Exchange （DLX）。DLX 是一种特殊类型的交换器，用来接收那些无法路由到任何队列（exchange、routing key）的消息。RabbitMQ 根据 DLX 指定的路由规则，将这些消息重新投递到其他目的地。
          
         ### 操作步骤:
          1. 创建 DLX
             使用命令：
             
                 rabbitmqctl add_exchange name=dlx type=fanout
                  
             参数含义：

                 name：DLX 名称，如 dlx;
                 
          2. 为队列指定 DLX 
             使用命令：
             
                rabbitmqctl set_policy name=my_queue policy=my_policy pattern="*" apply-to=queues definition={"dead-letter-exchange":"dlx"}
                  
              参数含义：

                 name：队列名，如 my_queue;
                 
                 policy：策略名称，如 my_policy;

                 pattern：匹配规则，这里使用星号表示任意的 routing key；

                 apply-to：应用到队列还是 exchange 上面，这里设置为 queues 表示只对队列生效。

                 definition：定义 dead letter exchange，格式为 JSON 字符串。

             比如：创建一个名字叫做 "hello" 的队列，并且将这个队列的死信交换器设置为 "dlx"。

              rabbitmqctl add_queue name=hello
                
               ......
                
            rabbitmqctl set_policy name=hello policy=my_policy pattern="*" apply-to=queues definition={"dead-letter-exchange":"dlx"}

          3. 测试消费者
             在配置好了队列和 DLX 以后，需要测试一下消费者。下面是一些关于如何消费 Dead Letter Queue (DLQ) 的注意事项。

             1. 在配置消费者时，要告诉它从哪个 DLX 上消费。
             2. 需要设置一些超时时间来防止死锁。
             3. 一旦发生死锁，可以尝试杀死相应的进程或者重启 RabbitMQ 服务。
             
             以下是一个示例 consumer 代码片段：

              ```java
              ConnectionFactory factory = new ConnectionFactory();
              factory.setHost("localhost");
              
              // Create a connection with the broker
              Connection connection = factory.newConnection();
              
                Channel channel = connection.createChannel();
                String queueName = "hello";
                String dlQName = "amq.rabbitmq.dead-letter-exchange";
            
                channel.basicConsume(dlQName + "." + queueName, true, new DeliverCallback() {
                    @Override
                    public void handle(String s, Delivery delivery) throws IOException {
                        String message = new String(delivery.getBody());
                        System.err.println("Got DLQ message '" + message + "'");
                    }
                }, consumerTag -> {});
              
              // Close the connection and channels
              connection.close();
              ```

              在上面的代码中，consumer 从 DLX 中消费 hello 队列的所有未成功投递的消息。使用 “amq.rabbitmq.dead-letter-exchange.”+队列名称作为 consumer 的 queue 名称即可。



             
             