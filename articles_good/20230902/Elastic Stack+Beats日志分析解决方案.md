
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## （一）Elasticsearch、Beats、Kibana简介及其之间的关系
### Elasticsearch
Elasticsearch是一个开源分布式搜索和分析引擎，它的目的是帮助用户发现数据中的隐藏信息，并对数据进行实时地全文检索、可视化分析，最终得出有价值的信息。它提供了一个分布式文档存储，索引，搜索和分析能力的全套解决方案。由于它可以高度扩展，支持各种数据类型，比如对象或者文本，使之能够快速检索到想要的数据；并且可以自动完成数据分析任务，分析海量数据，从而生成各类报表和图表。目前，它已经成为开源搜索引擎领域的主流产品。
### Beats
Beats是一组轻量级的数据采集器（Data Collectors）。它主要用于实时的日志、监控和事件数据收集，且无需安装 Agent 。同时还提供了许多内置模块来处理常见的场景，如文件旁路、传输代理、远程控制等功能。其中最著名的应该就是 Filebeat ，它的作用是从各类日志源（如 Nginx、Apache、Docker 等）收集日志，然后通过一个管道传输到 Elasticsearch 中。另外还有 Metricbeat、Packetbeat 和 Auditbeat 等。
### Kibana
Kibana是一个开源数据分析和可视化平台，它利用Elasticsearch中存储的日志和指标数据，为用户提供可视化的交互界面，让用户能够对数据进行多维度分析。用户可以直观地看到数据的变化趋势、关联性以及异常行为。此外，Kibana还具备强大的可视化分析能力，可以通过高级的图表工具快速创建各类自定义仪表盘。
### Elasticsearch+Beats+Kibana三者关系
Elastic Stack由Elasticsearch、Logstash、Beats、Kibana四个部分组成。Elasticsearch作为数据存储系统，负责存储所有的数据。Logstash负责从外部源接收数据并将其发送给Elasticsearch。Beats则是一个轻量级的数据采集器，可以从各类日志源接收日志数据，然后通过管道传送到Elasticsearch。Kibana是一个基于Web的可视化平台，用来对日志和指标数据进行分析和展示。这样整个栈就可以完美的运行起来，对日志数据进行索引、收集、存储、分析，并通过Kibana进行展示。


## （二）Elasticsearch入门
### 安装Elasticsearch
下载地址：https://www.elastic.co/downloads/elasticsearch

选择对应版本下载，当前最新版本为7.10.1，下载压缩包后解压至服务器指定目录。

```bash
cd /usr/local/software
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.10.1-linux-x86_64.tar.gz
tar -zxvf elasticsearch-7.10.1-linux-x86_64.tar.gz
mv elasticsearch-7.10.1/ /usr/local/app/es/   # 将解压后的文件夹移动到指定目录下
vi /etc/profile    # 添加环境变量
export ES_HOME=/usr/local/app/es/elasticsearch-7.10.1
export PATH=$ES_HOME/bin:$PATH
source /etc/profile
```

启动Elasticsearch服务

```bash
cd $ES_HOME/bin
./elasticsearch
```

验证是否成功启动

```bash
curl http://localhost:9200/
{
  "name" : "DCEglb",
  "cluster_name" : "my-application",
  "cluster_uuid" : "wWl_eLRxTmKm6EysTZkswg",
  "version" : {
    "number" : "7.10.1",
    "build_flavor" : "default",
    "build_type" : "tar",
    "build_hash" : "f2dd4d9",
    "build_date" : "2021-01-15T01:06:35.218447Z",
    "build_snapshot" : false,
    "lucene_version" : "8.7.0",
    "minimum_wire_compatibility_version" : "6.8.0",
    "minimum_index_compatibility_version" : "6.0.0-beta1"
  },
  "tagline" : "You Know, for Search"
}
```

以上安装配置过程，仅供参考，如果部署到生产环境，建议根据需要进行参数优化和安全设置。

### 创建索引
索引（Index）类似于数据库中的表，文档（Document）类似于行记录，字段（Field）类似于列。

创建一个索引，命令如下：

```bash
curl -X PUT "http://localhost:9200/test?pretty"
```

返回结果：

```json
{
  "acknowledged" : true,
  "shards_acknowledged" : true,
  "index" : "test"
}
```

创建第一个文档

```bash
curl -H 'Content-Type: application/json' -X POST "http://localhost:9200/test/_doc/?pretty" -d'
{
  "title": "Hello World!",
  "message": "Welcome to the test index."
}
'
```

返回结果：

```json
{
  "_index" : "test",
  "_type" : "_doc",
  "_id" : "AVKljAuvtXeU2dnjXXM4",
  "_version" : 1,
  "result" : "created",
  "_shards" : {
    "total" : 2,
    "successful" : 1,
    "failed" : 0
  },
  "_seq_no" : 0,
  "_primary_term" : 1
}
```

查询索引中的文档

```bash
curl -X GET "http://localhost:9200/test/_search?q=*&pretty"
```

返回结果：

```json
{
  "took" : 5,
  "timed_out" : false,
  "_shards" : {
    "total" : 1,
    "successful" : 1,
    "skipped" : 0,
    "failed" : 0
  },
  "hits" : {
    "total" : {
      "value" : 1,
      "relation" : "eq"
    },
    "max_score" : null,
    "hits" : [
      {
        "_index" : "test",
        "_type" : "_doc",
        "_id" : "AVKljAuvtXeU2dnjXXM4",
        "_score" : null,
        "_source" : {
          "title" : "Hello World!",
          "message" : "Welcome to the test index."
        }
      }
    ]
  }
}
```

## （三）Filebeat安装配置
### 安装配置Filebeat
1. 下载地址：https://www.elastic.co/downloads/beats/filebeat

2. 选择对应版本下载，当前最新版本为7.10.1，下载压缩包后解压至服务器指定目录。

3. 配置Filebeat配置文件

```bash
vi filebeat.yml
---------------------------
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - "/var/log/*.log"   # 指定要收集的日志路径

output.elasticsearch:
  hosts: ["localhost:9200"]   # 指定输出端为elasticsearch

setup.kibana:
  host: "localhost:5601"      # 指定kibana地址

logging.level: debug          # 设置日志级别
---------------------------
```

4. 启动Filebeat服务

```bash
./filebeat -e -c filebeat.yml
```

5. 检查Filebeat服务状态

```bash
curl -s http://localhost:5066/?pretty
{
  "host": "yourHost",
  "version": "7.10.1",
  "branch": "",
  "build_date": "2021-01-15T01:06:35.218447Z",
  "build_id": "f2dd4d9fbaa6a76bfabcccf51c3b0a04cfde455a",
  "go_version": "go1.15.5",
  "name": "youNameHere",
  "os": {
    "arch": "amd64",
    "family": "darwin",
    "platform": "darwin",
    "version": "10.15.7"
  },
  "product": {
    "name": "beats",
    "edition": "",
    "type": "beat"
  },
  "state": "running",
  "temperature": 0.00008166000000000001,
  "path": {
    "config": "/usr/local/app/filebeat/filebeat.yml",
    "data": "/usr/local/var/lib/filebeat/",
    "home": "/usr/local/app/filebeat",
    "logs": "/usr/local/var/log/filebeat/"
  },
  "stats": {
    "events": {
      "added": 10,
      "done": 10
    },
    "harvester": {
      "closed": 2,
      "opened": 1,
      "started": 2
    },
    "input": {},
    "libbeat": {
      "config": {
        "module": {
          "running": 0,
          "starts": 0,
          "stops": 0
        },
        "reloads": 0
      },
      "dns": {
        "lookups": 0
      },
      "output": {},
      "pipeline": {}
    },
    "pipeline": {
      "clients": 1,
      "events": {
        "active": 0,
        "filtered": 0,
        "published": 0,
        "retry": 0,
        "total": 0
      },
      "queue": {
        "acked": 0,
        "max_events": 4096,
        "events": 0
      }
    },
    "scheduler": {
      "events": 0
    },
    "memstats": {
      "gc_next": 4194304,
      "heap_alloc": 16112,
      "heap_idle": 1052672,
      "heap_inuse": 376832,
      "heap_objects": 513,
      "heap_released": 0,
      "heap_sys": 4034560,
      "num_gc": 3,
      "stack_inuse": 703928,
      "stack_sys": 703928,
      "sys_bytes": 7275264,
      "total_alloc": 5272
    }
  },
  "registrar": {
    "states": {
      "cleanup": 0,
      "current": 0,
      "update": 0
    }
  },
  "syslog": {
    "device_mapper": {},
    "tcp": {},
    "udp": {}
  }
}
```

### 数据收集测试
这里以nginx日志收集为例，日志目录为/var/log/nginx/*error.log。

1. 在nginx配置文件中添加日志打印，修改日志级别为debug。

   ```conf
   error_log logs/error.log debug;
   ```

2. 在/var/log/nginx/error.log文件中增加一条日志记录

   ```text
   Jun 15 15:04:52 localhost nginx[1234]: 2021/06/15 15:04:52 [crit] 1234#0: *3 connect() to unix:/run/php-fpm.sock failed (2: No such file or directory) while connecting to upstream, client: xxxxxxxx, server: xxx.com, request: "GET / HTTP/1.1", upstream: "fastcgi://unix:/run/php-fpm.sock:", host: "xxx.com"
   ```

3. 查看Elasticsearch中的日志记录

   ```bash
   curl http://localhost:9200/filebeat*/_search?pretty
   ```

   返回结果：

   ```json
   {
     "took" : 3,
     "timed_out" : false,
     "_shards" : {
       "total" : 1,
       "successful" : 1,
       "skipped" : 0,
       "failed" : 0
     },
     "hits" : {
       "total" : {
         "value" : 1,
         "relation" : "eq"
       },
       "max_score" : 1.3862944,
       "hits" : [
         {
           "_index" : "filebeat-7.10.1-2021.06.15-000001",
           "_type" : "_doc",
           "_id" : "DWTBfriByJw2yev-WZGC",
           "_score" : 1.3862944,
           "_source" : {
             "@timestamp" : "2021-06-15T10:04:52.000Z",
             "@version" : "1",
             "agent" : {
               "ephemeral_id" : "xxx",
               "hostname" : "localhost",
               "id" : "xxx",
               "type" : "filebeat",
               "version" : "7.10.1"
             },
             "clientip" : "xxxxxx",
             "event" : {
               "dataset" : "nginx.access",
               "duration" : 131072,
               "end" : "2021-06-15T10:04:52.000Z",
               "start" : "2021-06-15T10:04:52.000Z",
               "timezone" : "-07:00"
             },
             "fileset" : {
               "name" : "access"
             },
             "geoip" : {
               "city_name" : "-",
               "continent_code" : "--",
               "country_iso_code" : "--",
               "location" : [
                 0,
                 0
               ],
               "region_name" : "-"
             },
             "input" : {
               "type" : "log"
             },
             "labels" : {
               "beats_controller" : "bbb5a69b-f156-424a-befa-c99b90b1d7ca",
               "beats_job" : "filebeat"
             },
             "log" : {
               "offset" : 15924,
               "file" : {
                 "path" : "/var/log/nginx/error.log"
               }
             },
             "message" : "[crit] 1234#0: *3 connect() to unix:/run/php-fpm.sock failed (2: No such file or directory) while connecting to upstream, client: xxxxxxxx, server: xxx.com, request: \"GET / HTTP/1.1\", upstream: \"fastcgi://unix:/run/php-fpm.sock:\", host: \"xxx.com\"",
             "nginx" : {
               "access" : {
                 "body_bytes_sent" : 292,
                 "clientip" : "xxxxxx",
                 "duration" : 1.31,
                 "headers" : {
                   "Accept" : "*/*",
                   "Connection" : "keep-alive",
                   "Host" : "xxx.com",
                   "User-Agent" : "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36 Edg/90.0.818.62",
                   "X-Forwarded-For" : "xxxxxxxxxxxxxxxxxxxxxxxxx"
                 },
                 "method" : "GET",
                 "port" : "80",
                 "protocol" : "HTTP/1.1",
                 "query" : "/",
                 "referer" : "-",
                 "response" : {
                   "status_code" : "404",
                   "content_length" : ""
                 },
                 "upstream_addr" : "-",
                 "upstream_cache_status" : "-",
                 "uri" : "/"
               }
             },
             "offset" : 15924,
             "prospector" : {
               "type" : "log"
             },
             "server" : {
               "address" : "xxx.com",
               "ip" : "127.0.0.1",
               "port" : "80"
             },
             "service" : {
               "type" : "nginx"
             },
             "source" : "/var/log/nginx/error.log",
             "tags" : [
               "nginx"
             ],
             "timestamp" : "2021-06-15T10:04:52.000Z",
             "type" : "doc"
           }
         }
       ]
     }
   }
   ```

## （四）Kibana安装配置
### 安装配置Kibana
1. 下载地址：https://www.elastic.co/downloads/kibana

2. 选择对应版本下载，当前最新版本为7.10.1，下载压缩包后解压至服务器指定目录。

3. 修改配置文件kibana.yml，开启跨域请求。

   ```yaml
   server.port: 5601
   server.host: "localhost"
   server.name: kibana

   ## Set the default application for Kibana
   # Default app to load
   #kibana.defaultAppId: "home"

   # When running kibana without a custom domain, use this settings to specify the port and protocol
   server.rewriteBasePath: false
   server.basePath: "/"

   # Specifies whether a trailing slash should be appended to all URLs generated by Kibana's router in non-CORS environments
   # By default, Kibana will render routes without a trailing slash when accessing them directly from the URL bar of a browser.
   # This can cause problems with reverse proxies that rewrite requests based on the path component of incoming URLs.
   # Setting this option to `true` will ensure that all generated routes have a trailing slash.
   #server.appendTrailingSlash: false

   # Specifies how Kibana should behave if multiple instances are running on the same hostname and port.
   # To prevent conflicts between applications running on different ports, set this value to "multiple". If you only run one instance, keep it as is.
   # By default, Kibana will not start if another process has bound to its configured port.
   # In some cases, binding to TCP port 5601 may still succeed even if there is no Kibana listening on that port, especially if previous instances have terminated abruptly.
   # The tradeoff here is that multiple instances can potentially interfere with each other if they access files stored in the same location or make changes to the same resources.
   server.port: 5601
   #server.pidfile: pid.txt
   #server.pidfile.forceOverwrite: false

   # Optional setting that specifies where to redirect users when Kibana doesn't recognize their session.
   # If unset, defaults to `${server.options. basePath}/login`. You might want to set it to `${server.options. basePath}` to provide a more user-friendly experience.
   #server.xsrf.redirect: ${server.options. basePath}

   # Optional settings to enable SSL for outgoing connections from the Kibana server to Elasticsearch and Canvas.
   # These options must match the corresponding settings used by Elasticsearch and Canvas. See the documentation at https://www.elastic.co/guide/en/kibana/master/ssl-tls.html for details.
   #kibana.ssl.certificateAuthorities: [ '/path/to/CA.pem' ]
   #kibana.ssl.certificate: /path/to/kibana.crt
   #kibana.ssl.key: /path/to/kibana.key

   # A list of URLs that Kibana uses to communicate securely with Elasticsearch.
   #elasticsearch.hosts: ['http://localhost:9200']

   # Enable cross-origin requests for frontend rendering
   kibana.cors.enabled: true

   # Whitelist of originating URLs allowed to make cross-origin requests to Kibana's frontend
   kibana.cors.origin: "*"

   # Comma-separated list of headers that browsers are allowed to access during cross-origin requests. Defaults to "authorization,content-type".
   #kibana.cors.headers: authorization,content-type

   # Comma-separated list of methods that can be used during cross-origin requests. Defaults to "OPTIONS,HEAD,GET,POST,PUT,DELETE".
   #kibana.cors.methods: OPTIONS,HEAD,GET,POST,PUT,DELETE
   ```

4. 启动Kibana服务

   ```bash
  ./kibana --allow-root --silent
   ```

5. 浏览器打开 http://localhost:5601 进入Kibana首页。

## （五）Beats与Logstash对比
### Logstash
Logstash是一个基于Java开发的开源日志采集工具，具有插件化设计。它可以从多种数据源采集日志数据，经过过滤、转换、和分析之后写入目标存储系统，比如Elasticsearch、MySQL、HDFS、和HBase。Logstash官方宣称，它的性能超越了传统的工具，尤其在较大的集群或实时数据处理需求方面。

### Beats
Beats是一组轻量级的数据采集器（Data Collectors），它主要用于实时的日志、监控和事件数据收集，且无需安装 Agent 。同时还提供了许多内置模块来处理常见的场景，如文件旁路、传输代理、远程控制等功能。Beats 也可用于对接不同的数据源，如 Elasticsearch、Redis、MongoDB、Kafka、S3、StatsD、Cloudwatch、Zabbix、GCP Logs等。

### Beats VS Logstash
#### 对比一：部署方式
Beats 是采用独立的模式来运行的，需要单独的部署，而 Logstash 的部署比较复杂，需要在 Elasticsearch 中单独安装并启动一个 Logstash 插件。

#### 对比二：功能
Beats 提供了更加简单易用，不需要学习额外的配置项，同时适用于不同的平台和环境，而且没有任何依赖，可以使用 Docker 容器的方式来运行。但 Beats 不支持其他的输入源，如 Kafka 和 RabbitMQ，同时也不支持文件过滤，只能跟踪单个文件的日志。Logstash 支持更多的输入源，支持日志过滤、转换、解析、聚合等丰富的功能。

#### 对比三：易用性
Logstash 需要编写配置文件，而 Beats 使用 YAML 文件来定义 Beat 模块。Beats 有两种类型的配置文件，一种是通用的配置文件，如 filebeat.yml；另一种是特定模块的配置文件，如 nginx.yml。对于初次使用的人来说，Logstash 会很难上手，但是 Beats 只需要修改配置文件就可以实现各种功能。

#### 对比四：可靠性
Beats 可靠性一般会优于 Logstash，因为它只会影响到几个节点而不是整个集群。如果某些节点发生故障，那么这个节点上的日志将无法被 Beats 消费，但不会影响其他节点的正常工作。Logstash 在整个集群中起到的作用就相当于一个路由器，如果其中某台机器发生故障，那么它可能会影响整个集群的工作。因此，在实际生产环境中，使用 Logstash 或少量的 Beats 更加合适。