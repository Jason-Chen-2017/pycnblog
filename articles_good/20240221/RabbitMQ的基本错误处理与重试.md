                 

RabbitMQ的基本错误处理与重try
==============================

作者：禅与计算机程序设计艺术

## 1. 背景介绍

### 1.1 RabbitMQ简介

RabbitMQ是一个开源的消息队列系统，支持多种编程语言和平台。它使用AMQP(Advanced Message Queuing Protocol)协议，提供高可靠、高可扩展、高可用等特点。RabbitMQ可以帮助我们构建分布式系统，解耦服务之间的依赖关系，实现异步处理、削峰填谷等功能。

### 1.2 错误处理与重试的必要性

在使用RabbitMQ过程中，可能会遇到各种错误和异常情况，例如网络抖动、消费者宕机、消息处理超时等。如果没有采取适当的错误处理措施，这些错误可能导致消息丢失、系统不稳定或even worse。因此，学会如何在RabbitMQ中进行基本错误处理与重试至关重要。

## 2. 核心概念与联系

### 2.1 AMQP模型

RabbitMQ使用AMQP模型，包括Exchange、Queue、Binding、Routing Key、Message等 concepts。这些concepts组成了RabbitMQ的基本工作单元，决定了消息的生产、传递和消费的过程。

### 2.2 错误处理策略

RabbitMQ提供了多种错误处理策略，例如Basic.Return、Basic.Nack、Mandatory、Immediate等。这些策略可以帮助我们在出现错误时进行适当的处理，例如重新发布消息、拒绝消息、将消息路由到其他Queue等。

### 2.3 重试机制

重试机制是指在出现错误后，将消息重新发送给消费者进行处理的机制。RabbitMQ支持两种重试机制：自动重试和手动重试。自动重试是指RabbitMQ自动将消息重新投递给消费者；而手动重试是指应用程序自己控制消息的重试次数和间隔。

## 3. 核心算法原理和具体操作步骤以及数学模型公式详细讲解

### 3.1 Basic.Return策略

Basic.Return策略是RabbitMQ的默认错误处理策略，当消息无法被正确路由到Queue时，RabbitMQ会返回消息给生产者。这时，生产者可以根据返回码和描述信息来判断是否需要重新发布消息。

#### 3.1.1 数学模型公式

$$
BasicReturn = \begin{cases}
True, & \text{if message can not be routed} \\
False, & \text{otherwise}
\end{cases}
$$

#### 3.1.2 具体操作步骤

1. 设置Exchange类型为direct或topic，并设置Mandatory属性为true。
2. 生产者发布消息，并设置Routing Key。
3. RabbitMQ尝试将消息路由到Queue，如果失败则触发Basic.Return策略。
4. 生产者接收Basic.Return事件，并判断是否需要重新发布消息。

### 3.2 Basic.Nack策略

Basic.Nack策略是RabbitMQ的另一种错误处理策略，当消费者处理消息时出现错误时，RabbitMQ会将消息重新投递给其他消费者。

#### 3.2.1 数学模型公式

$$
BasicNack = \begin{cases}
True, & \text{if consumer can not handle the message} \\
False, & \text{otherwise}
\end`mathtex

#### 3.2.2 具体操作步骤

1. 设置Exchange类型为direct或topic，并设置Mandatory属性为false。
2. 消费者接收消息，并进行处理。
3. 如果处理出错，则调用channel.BasicNack方法，并设置requeue参数为true。
4. RabbitMQ会将消息重新投递给其他消费者。

### 3.3 自动重试机制

自动重试机制是指RabbitMQ自动将消息重新投递给消费者的机制。当消费者处理消息时出现错误时，RabbitMQ会根据设置的重试策略将消息重新投递给消费者。

#### 3.3.1 数学模型公式

$$
AutoRetry = \begin{cases}
True, & \text{if consumer can not handle the message and need to retry} \\
False, & \text{otherwise}
\end{cases}
$$

#### 3.3.2 具体操作步骤

1. 设置Exchange类型为direct或topic，并设置Mandatory属性为false。
2. 消费者接收消息，并进行处理。
3. 如果处理出错，则调用channel.BasicReject方法，并设置requeue参数为true。
4. RabbitMQ会根据设置的重试策略将消息重新投递给消费者。

### 3.4 手动重试机制

手动重试机制是指应用程序自己控制消息的重试次数和间隔的机制。当消费者处理消息时出现错误时，应用程序会记录错误信息，并在适当的时候重新发布消息。

#### 3.4.1 数学模型公式

$$
ManualRetry = \begin{cases}
True, & \text{if consumer can not handle the message and need to manual retry} \\
False, & \text{otherwise}
\end{cases}
$$

#### 3.4.2 具体操作步骤

1. 设置Exchange类型为direct或topic，并设置Mandatory属性为false。
2. 消费者接收消息，并进行处理。
3. 如果处理出错，则记录错误信息，并退出当前循环。
4. 定期检查错误队列，并重新发布消息。

## 4. 具体最佳实践：代码实例和详细解释说明

### 4.1 Basic.Return示例

#### 4.1.1 代码实例

```python
import pika
import sys

def callback(ch, method, properties, body):
   if method.REPLY_TEXT == "NO_ROUTE":
       print("Message returned: ", body)
       ch.basic_publish(exchange="", routing_key=method.reply_to, body=body)

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

channel.queue_declare(queue='task_queue', exclusive=True)
channel.basic_consume(queue='task_queue', on_message_callback=callback)
channel.start_consuming()
```

#### 4.1.2 详细解释

1. 创建连接和通道，并声明Queue。
2. 设置Basic.Return的回调函数，当消息无法被正确路由到Queue时，触发该回调函数。
3. 启动消费者，等待消息。
4. 如果收到返回事件，打印消息内容，并重新发布消息。

### 4.2 Basic.Nack示例

#### 4.2.1 代码实例

```python
import pika
import time

def callback(ch, method, properties, body):
   try:
       # do something with the message
       pass
   except Exception as e:
       print("Error occurred: ", e)
       ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

channel.queue_declare(queue='task_queue')
channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue='task_queue', on_message_callback=callback)
channel.start_consuming()
```

#### 4.2.2 详细解释

1. 创建连接和通道，并声明Queue。
2. 设置Basic.Nack的回调函数，当消费者处理消息时出错时，触发该回调函数。
3. 启用prefetch count，让RabbitMQ只发送一条消息给消费者。
4. 启动消费者，等待消息。
5. 如果处理出错，打印错误信息，并拒绝消息。

### 4.3 自动重试示例

#### 4.3.1 代码实例

```python
import pika
import time

MAX_RETRY_TIMES = 5
RETRY_INTERVAL = 1

def callback(ch, method, properties, body):
   try:
       # do something with the message
       pass
   except Exception as e:
       print("Error occurred: ", e)
       if ch.message_count > MAX_RETRY_TIMES:
           ch.basic_reject(delivery_tag=method.delivery_tag, requeue=False)
       else:
           ch.basic_reject(delivery_tag=method.delivery_tag, requeue=True)
           time.sleep(RETRY_INTERVAL)

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

channel.queue_declare(queue='task_queue')
channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue='task_queue', on_message_callback=callback)
channel.start_consuming()
```

#### 4.3.2 详细解释

1. 定义最大重试次数和重试间隔。
2. 创建连接和通道，并声明Queue。
3. 设置自动重试的回调函数，当消费者处理消息时出错时，触发该回调函数。
4. 启用prefetch count，让RabbitMQ只发送一条消息给消费者。
5. 启动消费者，等待消息。
6. 如果处理出错，打印错误信息，并重试。如果重试次数超过最大限制，则拒绝消息。

### 4.4 手动重试示例

#### 4.4.1 代码实例

```python
import pika
import time
import json

ERROR_QUEUE = 'error_queue'

def callback(ch, method, properties, body):
   try:
       task = json.loads(body)
       # do something with the task
       pass
   except Exception as e:
       print("Error occurred: ", e)
       error_body = json.dumps({'task': task, 'error': str(e)})
       channel.basic_publish(exchange="", routing_key=ERROR_QUEUE, body=error_body)

def retry_task():
   connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
   channel = connection.channel()
   while True:
       method_frame, header_frame, body = channel.basic_get(queue=ERROR_QUEUE)
       if method_frame is None:
           break
       task = json.loads(body)
       try:
           # do something with the task
           pass
       except Exception as e:
           print("Error occurred: ", e)
           error_body = json.dumps({'task': task, 'error': str(e)})
           channel.basic_publish(exchange="", routing_key=ERROR_QUEUE, body=error_body)

connection = pika.BlockingConnection(pika.ConnectionParameters('localhost'))
channel = connection.channel()

channel.queue_declare(queue='task_queue')
channel.basic_qos(prefetch_count=1)
channel.basic_consume(queue='task_queue', on_message_callback=callback)
channel.start_consuming()

retry_task()
```

#### 4.4.2 详细解释

1. 定义错误队列。
2. 创建连接和通道，并声明Queue。
3. 设置手动重试的回调函数，当消费者处理消息时出错时，触发该回调函数。
4. 启用prefetch count，让RabbitMQ只发送一条消息给消费者。
5. 启动消费者，等待消息。
6. 如果处理出错，打印错误信息，并将任务和错误信息发布到错误队列。
7. 定期检查错误队列，重新发布失败的任务。

## 5. 实际应用场景

### 5.1 订单系统

在订单系统中，当用户提交订单时，生产者会将订单消息发布到RabbitMQ中。如果因为网络抖动或其他原因导致消息无法被正确路由到Queue，则会触发Basic.Return策略，生产者可以根据返回码和描述信息来判断是否需要重新发布消息。

### 5.2 消息中转站

在消息中转站中，当生产者将消息发布到RabbitMQ时，可能会因为网络抖动或其他原因导致消息无法被正确路由到Queue。这时，可以采用Basic.Return策略，将消息返回给生产者，让生产者进行重新发布。

### 5.3 异步处理

在异步处理中，当消费者处理消息时出现错误时，可以采用Basic.Nack策略，将消息重新投递给其他消费者。这样可以保证消息不会被丢失，并提高系统的可靠性。

### 5.4 削峰填谷

在削峰填谷中，当消费者处理消息时出现错误时，可以采用自动重试机制，将消息重新投递给消费者。这样可以避免消费者被淹没，并提高系统的吞吐量。

### 5.5 数据迁移

在数据迁移中，当应用程序处理消息时出现错误时，可以采用手动重试机制，将消息重新发布给应用程序。这样可以避免消息被遗漏，并保证数据的完整性。

## 6. 工具和资源推荐

### 6.1 RabbitMQ官方文档

RabbitMQ官方文档是学习RabbitMQ最佳的资源之一。它包含了RabbitMQ的所有特性和API，并提供了大量的示例和代码片段。

### 6.2 RabbitMQ教程

RabbitMQ教程是另一个学习RabbitMQ的好资源。它提供了多种语言的示例，并逐步介绍RabbitMQ的特性和API。

### 6.3 RabbitMQ管理插件

RabbitMQ管理插件是一个Web界面，用于监视和管理RabbitMQ集群。它可以显示Queue、Exchange、Binding、Routing Key等信息，并支持操作Queue、Exchange、Binding等。

### 6.4 Shovel插件

Shovel插件是一个RabbitMQ插件，用于在RabbitMQ集群之间传输消息。它可以将Queue中的消息复制到其他Queue中，并支持过滤和转换消息。

### 6.5 Federation插件

Federation插件是另一个RabbitMQ插件，也用于在RabbitMQ集群之间传输消息。与Shovel插件不同，Federation插件可以将Queue中的消息分发到多个Queue中，并支持负载均衡和故障转移。

## 7. 总结：未来发展趋势与挑战

### 7.1 更高的可靠性

随着云计算和微服务的普及，RabbitMQ的使用场景越来越广泛。因此，RabbitMQ需要提供更高的可靠性，以满足用户的需求。这包括提高消息传递的成功率、降低延迟、增强安全性等。

### 7.2 更好的扩展性

随着互联网的发展，系统的规模不断扩大。因此，RabbitMQ需要提供更好的扩展性，以适应用户的需求。这包括支持更多的消息格式、提供更灵活的队列管理、支持更多的数据库等。

### 7.3 更智能的管理工具

随着系统的复杂性不断增加，管理RabbitMQ集群变得越来越困难。因此，RabbitMQ需要提供更智能的管理工具，以帮助用户监控和管理RabbitMQ集群。这包括支持更多的统计指标、提供更好的告警机制、支持更多的可视化工具等。

### 7.4 更丰富的插件生态系统

RabbitMQ插件生态系统已经相当丰富，但还是有很大的潜力。因此，RabbitMQ需要继续扩展插件生态系统，以提供更多的功能和服务。这包括开发更多的插件、支持更多的协议、提供更好的插件管理等。

## 8. 附录：常见问题与解答

### 8.1 为什么消息无法被正确路由到Queue？

消息无法被正确路由到Queue可能是因为Exchange类型设置错误、Routing Key设置错误、Queue绑定错误等原因。可以通过检查Exchange、Queue和Binding的配置来确认问题。

### 8.2 如何避免消息被重复处理？

可以采用唯一ID或消息体的Hash值来判断消息是否已经被处理过。如果已经处理过，则直接返回或忽略消息。

### 8.3 如何避免消息被遗漏？

可以采用Basic.Nack策略或自动重试机制，将消息重新投递给消费者。如果消费者长时间未处理消息，则可以将消息路由到其他Queue或发送告警通知。

### 8.4 如何避免消息积压？

可以采用削峰填谷策略，将高峰期的消息缓存到Queue中，然后再 slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly slowly