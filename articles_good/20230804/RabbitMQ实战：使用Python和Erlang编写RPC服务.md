
作者：禅与计算机程序设计艺术                    

# 1.简介
         
 在本篇文章中，我们将使用RabbitMQ实现一个简单但完整的RPC服务。首先，我们会先对RabbitMQ的一些基础概念进行阐述，然后通过Python和Erlang分别实现服务端和客户端。最后，我们还会结合实践场景，分析并解决一些实际问题，比如分布式服务的可用性、网络延迟等。
          # 为什么选择RabbitMQ？
           RabbitMQ是一个开源的AMQP协议的消息代理软件，具有稳定、可靠、高效的特性，被广泛应用于分布式系统中，如：任务队列、异步处理、事件驱动等。其提供了多种语言的客户端库支持，包括Python、Java、C#、Ruby、PHP等。
           RabbitMQ官方宣称，在处理超过万亿条消息，每秒数百万的消息时，它是世界上最快的企业级消息中间件。本文基于RabbitMQ实现RPC服务，所以为了更好的理解RabbitMQ，建议读者阅读一下RabbitMQ官网关于它的基本介绍。
          # 安装RabbitMQ
           本文使用的是RabbitMQ最新稳定版(v3.7.7)，下载地址：https://www.rabbitmq.com/download.html
           1.解压下载包到指定目录
           2.进入bin目录下，创建名为rabbit的数据文件夹
           3.启动rabbitmq-server: bin\rabbitmq-server.bat
           4.打开浏览器，访问http://localhost:15672 ，默认用户名密码 guest guest
           # 2.RabbitMQ基本概念
          RabbitMQ有很多重要的概念，本文将对它们进行简单介绍。
          ## Broker：消息代理服务器
            RabbitMQ中的Broker就是消息代理服务器，它负责接收和转发消息。每个RabbitMQ节点都可以作为Broker运行，也可以由其他程序充当Broker。Broker根据配置决定是否把消息持久化到磁盘或者内存中，并且支持多种传输协议，如STOMP、MQTT、HTTP等。
          ## Exchange：交换机
          RabbitMQ中Exchange用于接收生产者发送的消息，并将这些消息路由给正确的队列。RabbitMQ共有四种Exchange类型：direct、topic、headers和fanout，不同的Exchange类型决定了如何向队列分发消息。
          ### direct exchange
          Direct Exchange根据routing key将消息投递到binding key（可以理解为队列名称）相同的所有队列中。如果没有符合条件的binding key，则不会投递任何消息。这是一种简单的交换机类型。
          ### topic exchange
          Topic Exchange根据routing key中的词汇进行匹配，此处的词汇可以指多个单词，并且每个词之间用点号隔开。消息的routing key应该和binding key模式匹配才会投递到对应的队列中。这种类型的交换机很灵活，但是性能较差。
          ### headers exchange
          Headers Exchange不依赖于routing key，而是根据消息的headers进行匹配。生产者设置headers信息，消费者监听特定的header关键字，匹配成功后，将消息投递到对应的队列。
          ### fanout exchange
          Fanout Exchange将所有的消息分发到所有绑定的队列中。该类型的Exchange不管接收到多少个 routing key，只要绑定了该exchange，那么消息都会被路由到该exchange下的所有队列中。通常用在广播场景，不需要关注消息的内容，只需要知道有一个消息发生了。
          ## Queue：队列
          RabbitMQ中的队列用于存储消息。每个消息都只能被投递到一个队列中，同一个队列中的消息具有FIFO（先进先出）的顺序。可以在声明队列时指定消息最大长度和溢出策略。
          ## Virtual host：虚拟主机
          每个virtual host相当于一个独立的逻辑环境，不同用户可以使用不同的权限控制。RabbitMQ默认创建一个virtual host“/”。
          ## Connection：连接
          连接是建立在TCP之上的socket链接，用于客户端和broker之间的通信。
          ## Channel：信道
          信道是建立在连接之上的虚拟连接，用于通讯双方的指令传输及确认。
          # Python RPC 服务端实现
          ## 服务端准备工作
          在开始实现RPC服务之前，我们需要完成以下准备工作：
          1.安装pip：我们需要安装pip命令行工具，用于管理python的包依赖关系。
          2.安装pika：Pika是RabbitMQ官方提供的Python客户端库，用于实现Python和RabbitMQ之间的通信。
          3.创建项目文件夹：创建一个名为“rpc_server”的文件夹，用来存放我们的代码文件。
          4.创建配置文件：创建名为config.py的文件，用来保存RabbitMQ的连接参数。

          ```python
          import os

          class Config:
              RABBITMQ_HOST = 'localhost'
              RABBITMQ_PORT = int(os.environ.get('RABBITMQ_PORT', 5672))
              RABBITMQ_USERNAME = 'guest'
              RABBITMQ_PASSWORD = 'guest'
              RABBITMQ_VIRTUAL_HOST = '/'
              EXCHANGE = 'rpc_exchange'

              def __init__(self):
                  self._url = "amqp://{username}:{password}@{host}:{port}/{vhost}".format(
                      username=Config.RABBITMQ_USERNAME,
                      password=Config.RABBITMQ_PASSWORD,
                      host=Config.RABBITMQ_HOST,
                      port=Config.RABBITMQ_PORT,
                      vhost=Config.RABBITMQ_VIRTUAL_HOST
                  )

                  print("Connecting to RabbitMQ:", self._url)

              @property
              def url(self):
                  return self._url

          config = Config()
          ```

          配置文件中，设置了RabbitMQ连接相关的参数，包括hostname、port、用户名密码、virtual host和RPC交换机的名称。

          5.创建消息模型类：我们定义了一个MessageModel类，用来表示我们要传送的消息，例如：函数调用请求、响应结果等。

          ```python
          from typing import Dict

          class MessageModel:
              id: str
              func: str
              args: list
              kwargs: Dict[str, object]

              def __init__(self, message_id: str, function_name: str, arguments: list, keyword_arguments: Dict[str, object]):
                  self.id = message_id
                  self.func = function_name
                  self.args = arguments
                  self.kwargs = keyword_arguments
          ```

          6.创建RPC服务端类：我们定义了一个RpcServer类，用于处理接收到的RPC请求，并返回相应的响应结果。该类的构造方法接受一个connection对象，用于处理网络连接，调用channel对象的queue_declare()方法创建名为rpc_queue的队列，以及exchange对象，用于发布消息。

          ```python
          import pika
          from rabbitmq.config import config
          from rabbitmq.models import MessageModel


          class RpcServer:
              _instance = None

              def __new__(cls, *args, **kwargs):
                  if not cls._instance:
                      cls._instance = super().__new__(cls)

                      credentials = pika.PlainCredentials(config.RABBITMQ_USERNAME, config.RABBITMQ_PASSWORD)
                      connection = pika.BlockingConnection(
                          pika.ConnectionParameters(
                              host=config.RABBITMQ_HOST,
                              port=config.RABBITMQ_PORT,
                              virtual_host=config.RABBITMQ_VIRTUAL_HOST,
                              credentials=credentials
                          )
                      )
                      channel = connection.channel()

                      try:
                          channel.exchange_declare(exchange=config.EXCHANGE, exchange_type='direct')
                          queue = channel.queue_declare('rpc_queue', durable=True)
                          binding_key = f"rpc.{config.RABBITMQ_VIRTUAL_HOST}"
                          channel.queue_bind(exchange=config.EXCHANGE,
                                              queue=queue.method.queue,
                                              routing_key=binding_key)

                          channel.basic_qos(prefetch_count=1)
                          cls._instance.connection = connection
                          cls._instance.channel = channel
                          cls._instance.callback_queue = ''
                          cls._instance.consumer_tag = channel.basic_consume(on_message_callback=cls._instance._response_handler,
                                                                             queue=cls._instance.callback_queue)
                      except Exception as e:
                          raise e
                  return cls._instance

              def __init__(self, *args, **kwargs):
                  pass

              @classmethod
              def get_instance(cls):
                  return cls._instance

              def call(self, function_name: str, *args, **kwargs):
                  message_id = str(uuid.uuid4())[:8]
                  request = MessageModel(message_id, function_name, list(args), dict(**kwargs))

                  correlation_id = str(uuid.uuid4())
                  response = {'result': '', 'error': ''}
                  self.channel.basic_publish(
                      exchange='',
                      routing_key=f"rpc.{config.RABBITMQ_VIRTUAL_HOST}",
                      properties=pika.BasicProperties(reply_to=self.callback_queue,
                                                      correlation_id=correlation_id,),
                      body=json.dumps({'request': vars(request)})
                  )

                  while True:
                      self.connection.process_data_events()
                      result = json.loads(response['result']) if len(response['result']) > 0 else None
                      error = json.loads(response['error']) if len(response['error']) > 0 else None

                      if (not result and not error) or \
                              ('error' in response and response['error']!= '') or \
                              ('result' in response and response['result']!= ''):
                          continue
                      break

                  if error is not None:
                      raise ValueError(error)

                  return result

              def stop(self):
                  self.channel.close()
                  self.connection.close()

              def _response_handler(ch, method, props, body):
                  if self.corr_id == props.correlation_id:
                      ch.basic_ack(delivery_tag=method.delivery_tag)
                      response = json.loads(body)['response']
                      response_dict = {
                         'result': response['result'],
                          'error': response['error'],
                      }
                      self.response = response_dict
          ```

          rpc_server文件夹结构如下所示：

          ```
          rpc_server/
              __init__.py
              models.py        // 定义消息模型类
              server.py        // 定义RPC服务端类
              config.py        // 定义配置类
          ```

        此时的代码还不能正常运行，因为还缺少实现请求消息的处理逻辑。
        ## 请求消息处理逻辑实现
       创建一个新模块`processor`，该模块用于处理收到的请求消息，并生成相应的响应消息。

       ```python
       from typing import Any, Callable, List, Dict
       from types import MethodType
       from functools import wraps
       import uuid
       import json
       import inspect

       import rabbitmq.config as conf
       from rabbitmq.server import RpcServer
       from rabbitmq.models import MessageModel


       class Processor:
           _instances = {}

           @staticmethod
           def register(function_name):
               """
               Decorator for registering processor functions with the server instance
               :param function_name: The name of the function to register
               :return: A decorator that registers a decorated function with the server instance
               """

               def decorator(fnc):
                   fnc.__doc__ = fnc.__doc__ + "
 Registered with Processor."
                   @wraps(fnc)
                   def wrapper(*args, **kwargs):
                       return fnc(*args, **kwargs)

                   setattr(wrapper, "__is_registered", True)
                   WrapperClass = type(fnc).__name__
                   if WrapperClass not in Processor._instances:
                        Processor._instances[WrapperClass] = {}
                   Processor._instances[WrapperClass][function_name] = fnc
                   return wrapper
               return decorator

           @staticmethod
           def process():
               """
               Process requests on registered processors
               :return: None
               """
               def callback(ch, method, properties, body):
                   message = json.loads(body)["request"]
                   module_path, func_name = message["func"].split(":")
                   module = __import__(module_path, globals(), locals(), [""])
                   function = getattr(module, func_name)

                   response = {"result": "", "error": ""}

                   try:
                       result = function(*message["args"], **{k: v for k, v in message["kwargs"].items()})
                       response["result"] = json.dumps({"result": result})
                   except BaseException as e:
                       response["error"] = json.dumps({"error": "{}: {}".format(e.__class__.__name__, str(e)),
                                                       "traceback": traceback.format_exc().strip()})

                   reply_props = pika.BasicProperties(correlation_id=properties.correlation_id,)
                   ch.basic_publish(exchange="",
                                     routing_key=properties.reply_to,
                                     properties=reply_props,
                                     body=json.dumps({
                                         "response": response
                                     }))

                   ch.basic_ack(delivery_tag=method.delivery_tag)

               credentials = pika.PlainCredentials(conf.config.RABBITMQ_USERNAME, conf.config.RABBITMQ_PASSWORD)
               connection = pika.BlockingConnection(
                           pika.ConnectionParameters(
                               host=conf.config.RABBITMQ_HOST,
                               port=conf.config.RABBITMQ_PORT,
                               virtual_host=conf.config.RABBITMQ_VIRTUAL_HOST,
                               credentials=credentials
                           ))
               channel = connection.channel()

               channel.queue_declare(queue="requests", durable=True)
               channel.queue_bind(exchange=conf.config.EXCHANGE,
                                   queue="requests",
                                   routing_key=f"{conf.config.RABBITMQ_VIRTUAL_HOST}.request.*")

               channel.basic_qos(prefetch_count=1)
               channel.basic_consume(queue="requests",
                                      auto_ack=False,
                                      on_message_callback=callback)
               channel.start_consuming()

           @staticmethod
           def start_processing():
               while True:
                    for WrapperClassName, instances in Processor._instances.items():
                         if not hasattr(getattr(Processor, WrapperClassName), "__is_registered"):
                             continue

                         obj = getattr(Processor, WrapperClassName)()
                         methods = inspect.getmembers(obj, predicate=inspect.ismethod)
                         for name, method in methods:
                             doc = getattr(method, '__doc__', None)
                             if doc is not None and "Registered with Processor." in doc:
                                 signature = inspect.signature(method)
                                 arg_names = list(signature.parameters)[1:]
                                 processors = []

                                 for attr in dir(obj):
                                    attribute = getattr(obj, attr)
                                    if isinstance(attribute, MethodType) and callable(attribute) and hasattr(attribute, "__is_processor__"):
                                        processor_name = ".".join([WrapperClassName, attr])
                                        processors.append((processor_name, attribute))

                                 for processor_name, processor_callable in processors:
                                     argspec = inspect.getfullargspec(processor_callable)
                                     defaults = {}
                                     annotations = {}
                                     if argspec.defaults is not None:
                                          num_required_args = len(arg_names) - len(argspec.defaults)
                                          required_args = arg_names[:num_required_args]
                                          optional_args = arg_names[num_required_args:]
                                          defaults = dict(zip(optional_args, argspec.defaults))
                                     annotations = argspec.annotations
                                     required_annotation = annotations.pop(next(iter(required_args)))
                                     optional_annotations = {}
                                     for opt_arg in optional_args:
                                          if opt_arg in annotations:
                                               optional_annotations[opt_arg] = annotations.pop(opt_arg)
                                     processed_args = []
                                     processed_kwargs = {}
                                     i = 0
                                     for req_arg in required_args:
                                          if req_arg in message_model_dict:
                                               processed_args.append(message_model_dict[req_arg])
                                          elif req_arg in defaults:
                                               processed_args.append(defaults[req_arg])
                                          else:
                                               processed_args.append(None)
                                          
                                       for opt_arg in optional_args:
                                          if opt_arg in message_model_dict:
                                               processed_kwargs[opt_arg] = message_model_dict[opt_arg]
                                          elif opt_arg in defaults:
                                               processed_kwargs[opt_arg] = defaults[opt_arg]
                                          else:
                                               processed_kwargs[opt_arg] = None

                                      result = processor_callable(*processed_args, **processed_kwargs)
                                      response = {
                                          "status": "success",
                                          "result": result
                                      }
                                      response_str = json.dumps(response)
                                      ch.basic_publish(exchange='',
                                                        routing_key=props.reply_to,
                                                        properties=pika.BasicProperties(
                                                            correlation_id=props.correlation_id),
                                                         body=response_str)
                     break
       ```

      `processor` 模块定义了两个静态方法，`register()` 和 `process()`. 

      `register()` 方法是修饰器，可以注册处理器函数到服务实例中。修饰器内的 `wrapper()` 函数增加了一个属性 `__is_registered`, 用以标记该函数已注册到 Processor 对象中。修饰器调用 `setattr()` 方法添加属性 `__is_registered`，值为 True。

      使用 `isinstance()` 判断调用方是否继承自 Processor 类。若不存在父类 `Processor` 的实例，则创建 `Processor` 子类实例。实例缓存到 `_instances` 中，键值对形式为 (`WrapperClassName`: `dict`)，其中 `dict` 的键为处理器名称(`processor_name`), 值为处理器函数。
      
      `process()` 方法负责监听 RabbitMQ 的 `requests` 队列，等待接收请求消息。通过回调函数 `callback()` 处理请求消息，解析消息数据，获取处理器名称，导入模块并执行处理器函数，生成响应消息，发送至 RabbitMQ 的 `replies` 队列，并标记为已确认。

      当请求消息过来时，从缓存中查找对应的处理器函数，获取请求参数，依据参数位置索引或取默认值，执行函数并返回结果。
      
      通过 `__is_processor__` 属性判断是否是处理器函数。假设存在多个处理器函数，则按函数签名，按顺序调用各处理器函数，按需填入处理器参数，生成返回值。

      以上就是请求消息处理逻辑的全部代码实现，下面看一下客户端的实现。
      # Python RPC 客户端实现
      ## 客户端准备工作
      在开始实现RPC客户端之前，我们需要完成以下准备工作：
      1.安装pip：我们需要安装pip命令行工具，用于管理python的包依赖关系。
      2.安装pika：Pika是RabbitMQ官方提供的Python客户端库，用于实现Python和RabbitMQ之间的通信。
      3.创建项目文件夹：创建一个名为“rpc_client”的文件夹，用来存放我们的代码文件。
      4.创建配置文件：创建名为config.py的文件，用来保存RabbitMQ的连接参数。
      
      ```python
      import os

      class Config:
          RABBITMQ_HOST = 'localhost'
          RABBITMQ_PORT = int(os.environ.get('RABBITMQ_PORT', 5672))
          RABBITMQ_USERNAME = 'guest'
          RABBITMQ_PASSWORD = 'guest'
          RABBITMQ_VIRTUAL_HOST = '/'
          EXCHANGE = 'rpc_exchange'

          def __init__(self):
              self._url = "amqp://{username}:{password}@{host}:{port}/{vhost}".format(
                  username=Config.RABBITMQ_USERNAME,
                  password=Config.RABBITMQ_PASSWORD,
                  host=Config.RABBITMQ_HOST,
                  port=Config.RABBITMQ_PORT,
                  vhost=Config.RABBITMQ_VIRTUAL_HOST
              )

              print("Connecting to RabbitMQ:", self._url)

          @property
          def url(self):
              return self._url

      config = Config()
      ```

      配置文件中，设置了RabbitMQ连接相关的参数，包括hostname、port、用户名密码、virtual host和RPC交换机的名称。

    ## RPC客户端类
    创建一个新的模块 `client`，该模块包含了 `RpcClient` 类，用来与 RPC 服务端进行通信。
    
    ```python
    import time
    import uuid
    import json
    import pika
    from concurrent.futures import ThreadPoolExecutor
    from rabbitmq.config import config


    class RpcClient:
        def __init__(self):
            self.connection = None
            self.channel = None

            self._executor = ThreadPoolExecutor(max_workers=1)
            self._responses = {}
            self._response_lock = threading.Lock()
            self._futures = set()
            self._futures_lock = threading.Lock()

        def connect(self):
            if self.connection is None or self.channel is None:
                credentials = pika.PlainCredentials(config.RABBITMQ_USERNAME, config.RABBITMQ_PASSWORD)
                self.connection = pika.BlockingConnection(
                    pika.ConnectionParameters(
                        host=config.RABBITMQ_HOST,
                        port=config.RABBITMQ_PORT,
                        virtual_host=config.RABBITMQ_VIRTUAL_HOST,
                        credentials=credentials
                    )
                )
                self.channel = self.connection.channel()

                result = self.channel.queue_declare(queue='', exclusive=True)
                self._callback_queue = result.method.queue

                self.channel.basic_consume(
                    queue=self._callback_queue,
                    on_message_callback=lambda ch, method, properties, body:
                            self._response_handler(json.loads(body)),
                    auto_ack=True
                )

        def disconnect(self):
            self.channel.stop_consuming()
            self.connection.close()

        def call(self, function_name: str, *args, **kwargs):
            message_id = str(uuid.uuid4())[:8]
            future = self._executor.submit(self._blocking_call, message_id, function_name, args, kwargs)
            self._futures_lock.acquire()
            self._futures.add(future)
            self._futures_lock.release()
            return future

        def _blocking_call(self, message_id, function_name, args, kwargs):
            while True:
                self.connect()
                self._send_request(message_id, function_name, args, kwargs)
                response = self._wait_for_response(message_id)
                self.disconnect()
                if 'error' in response:
                    raise ValueError(response['error']['exception'])
                else:
                    return response['result']

        def _response_handler(self, response):
            self._response_lock.acquire()
            msg_id = response['id']
            if msg_id in self._responses:
                self._responses[msg_id].set_result(response)
                del self._responses[msg_id]
            self._futures_lock.acquire()
            futures_to_remove = set()
            for fut in self._futures:
                if not fut.done():
                    continue
                try:
                    r = fut.result()
                    if r['id'] == msg_id:
                        futures_to_remove.add(fut)
                except KeyError:
                    pass
            for fut in futures_to_remove:
                self._futures.discard(fut)
            self._futures_lock.release()
            self._response_lock.release()

        def _send_request(self, message_id, function_name, args, kwargs):
            message = {
                "id": message_id,
                "func": function_name,
                "args": args,
                "kwargs": kwargs
            }
            content_type = 'application/json'
            properties = pika.BasicProperties(content_type=content_type,
                                               delivery_mode=2)
            self.channel.basic_publish(exchange=config.EXCHANGE,
                                       routing_key=f'{config.RABBITMQ_VIRTUAL_HOST}.request.{config.RABBITMQ_VIRTUAL_HOST}',
                                       properties=properties,
                                       body=json.dumps({"request": message}))
            self._response_lock.acquire()
            self._responses[message_id] = asyncio.Future()
            self._response_lock.release()

        def _wait_for_response(self, message_id):
            tic = time.time()
            timeout = 10  # seconds
            while time.time() < tic+timeout:
                self._response_lock.acquire()
                future = self._responses.get(message_id)
                if future is not None and future.done():
                    response = future.result()
                    self._response_lock.release()
                    return response
                self._response_lock.release()
                time.sleep(0.1)
            raise TimeoutError(f"Timeout waiting for response for message ID '{message_id}'")
    ```

  `client` 模块定义了 `RpcClient` 类，包括 `connect()`, `disconnect()`, `call()` 和私有方法 `_blocking_call()`, `_response_handler()`, `_send_request()`, `_wait_for_response()` 。
  
  `connect()` 方法用于建立与 RPC 服务端的连接，`disconnect()` 方法用于断开与 RPC 服务端的连接。
  `call()` 方法用于同步地调用 RPC 服务端的方法，并返回其结果。
  `_blocking_call()` 是异步地调用 RPC 服务端的方法，其内部循环检测服务端是否可用，直到超时为止。
  `_response_handler()` 是 RabbitMQ 消息回调函数，当服务端响应某个消息时，触发该函数，并更新本地缓存的响应结果。
  `_send_request()` 是向服务端发送请求消息的底层函数。
  `_wait_for_response()` 是等待响应结果的底层函数。

  在整个过程中，客户端始终保持与 RPC 服务端的长连接。

  以上就是客户端的全部代码实现。
  # 分布式服务可用性问题
  RPC服务在分布式服务架构中扮演着至关重要的角色。可用性问题的表现一般包括服务不可用、延迟增大、过载以及严重错误。因此，我们需要对RPC服务端和客户端的实现进行适当优化，确保服务的可用性。
  ## RabbitMQ集群部署
  RabbitMQ支持基于镜像、主备、主主、多副本等模式部署集群。本文采用主备模式，即有两台机器构成RabbitMQ集群。

  第一台机器作为主节点，第二台机器作为备份节点。

  操作步骤如下：
  1.安装依赖：apt install rabbitmq-server
   
  2.停止RabbitMQ服务: service rabbitmq-server stop
   
  3.编辑配置文件 /etc/rabbitmq/rabbitmq.conf 中的 cluster_nodes 参数，加入两台机器IP地址，例如："cluster_nodes.rabbit@node1,rabbit@node2"

  4.启动RabbitMQ服务: systemctl restart rabbitmq-server

  以上，我们完成了RabbitMQ集群的部署。
  
  ## RabbitMQ HAProxy配置
  RabbitMQ集群中，我们可以通过HAProxy组件实现自动故障切换。我们将HAProxy放在两台机器的前端，并在前端做负载均衡和请求分发。
  
  操作步骤如下：
  
  1.安装haproxy: sudo apt-get update && sudo apt-get install haproxy
  
  2.编辑配置文件 /etc/haproxy/haproxy.cfg，示例如下：
  
  ```
  global
     daemon
     group  haproxy
     maxconn  100000

  listen rabbitmq_cluster
     bind *:5672
     mode tcp

     balance roundrobin
     option tcpka

     default-server inter 3s fastinter 10s downinter 5s rise 2 fall 3 slowstart 60s maxconn 10000 ssl crt /etc/ssl/certs/ca-certificates.crt
     server node1 192.168.0.10:5672 check
     server node2 192.168.0.11:5672 check backup

  frontend http-in
     bind *:80

     default_backend servers

  backend servers
     balance     roundrobin
     server    server1 192.168.0.10:80 cookie server1 weight 1
     server    server2 192.168.0.11:80 cookie server2 weight 1
  ```
  
  3.启动haproxy: sudo systemctl start haproxy
  
  以上，我们完成了RabbitMQ的HAProxy配置。
  # 网络延迟与请求超时问题
  在实际的生产环境中，可能由于网络拥塞、故障或者流量洪峰导致RabbitMQ出现无法响应的问题。在实际生产环境中，我们需要设计好超时机制，使得服务在一定时间内能够正常响应。否则，调用客户端可能会出现异常，甚至造成系统崩溃。下面就讨论这个问题。
  ## 网络延迟
  有两种情况会导致网络延迟：硬件、软件。硬件因素比如路由器，软件因素比如操作系统、网络栈，可以引起网络延迟。

  我们可以通过以下方式降低网络延迟：
  - 使用专用网络连接：如光纤连接、万兆网卡。
  - 使用CDN加速：内容分发网络，可以减少网络延迟。
  - 使用CDN加速的HTTPDNS服务：智能DNS解析，可以快速定位物理IP地址。
  - 使用QoS保证高吞吐率：避免网络拥塞造成丢包。

  ## 请求超时
  服务端侧超时设置的两个主要参数：
  - handshake_timeout：客户端等待服务端握手消息的超时时间。
  - negotiate_timeout：客户端等待服务端协商关闭连接的超时时间。

  客户端侧的超时设置：
  - send_timeout：客户端等待服务端响应数据的超时时间。
  - recv_timeout：客户端等待服务端数据返回的超时时间。

  对于RabbitMQ来说，可以设置各个参数的值，同时设置比TCP超时短的时间。

  更详细的RabbitMQ超时设置参考官方文档：https://www.rabbitmq.com/networking.html#timeouts

  设置示例如下：

  服务端侧：

  rabbitmq.conf：

  ```
 ...
  networking_tick_interval = 30
  heartbeat_interval = 10
  delegate_count = 10
  frame_max = 131072
  sock_recv_size = 1024
  sock_send_size = 1024
  handshake_timeout = 5000
  negotiate_timeout = 5000
 ...
  ```

  client.py：

  ```python
  import pika
  import socket

  conn_params = pika.ConnectionParameters(host='localhost',
                                            port=5672,
                                            connection_attempts=3,
                                            retry_delay=2,
                                            socket_timeout=5)

  try:
      connection = pika.BlockingConnection(conn_params)
      channel = connection.channel()
     ...
      channel.basic_publish(...)
     ...
      connection.close()
  except socket.timeout as ex:
      print(ex)
  finally:
      connection.close()
  ```