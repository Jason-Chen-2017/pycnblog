
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Elasticsearch 是开源分布式搜索引擎，能够快速地、高效地存储、检索数据。由于其插件机制，它可以支持多种数据类型（例如文本、数值、全文等）及其相关分析功能。本文将通过一个实际案例，详细介绍 Elasticsearch 的安装部署过程，包括集群规划、配置节点数量、设置内存分配、选择硬件配置、压缩选项、启动、停止、重启、日志管理等内容。
# 2. Elasticsearch 简介
Elasticsearch是一个基于Lucene库的开源搜索服务器，具有RESTful Web接口。旨在解决大数据集的海量数据存储、查询和分析问题。Elasticsearch是用Java语言开发的，并使用Lucene作为其核心来实现所有索引、搜索、分析的功能。它对外提供简单灵活的RESTful API接口，方便客户端调用。同时，它还提供了强大的搜索语法，使得搜索变得更加简单快速。
# 3. 安装环境准备
为了部署 Elasticsearch ，首先需要准备好安装环境。如图所示，此次实验环境如下：
# 3.1 操作系统环境
Windows Server 2012 R2
# 3.2 Java 运行时环境
JDK 1.8或更新版本
# 3.3 磁盘空间要求
至少50G以上
# 3.4 RAM 要求
至少2GB以上
# 3.5 网络连接要求
网络连通性要求较低，一般无需配置防火墙
# 3.6 安装包下载地址
https://www.elastic.co/downloads/elasticsearch
# 4. Elasticsearch 安装部署
本节介绍如何在 Windows 上安装 Elasticsearch 5.x 。
# 4.1 安装前准备
下载安装包到安装目录下。默认安装路径 C:\Program Files\ElasticSearch ；也可以自定义安装路径。
# 4.2 安装 Elasticsearch
双击 elasticsearch-5.5.2.msi 安装 Elasticsearch ，根据提示安装即可。
# 4.3 修改配置参数
安装完成后，进入bin目录找到 elasticsearch.yml 文件，打开文件修改以下参数。

	cluster.name: my-application        # 设置集群名称
	node.name: node-1                  # 设置节点名称
	path.data: D:\esData                # 指定数据文件的存放位置
	path.logs: D:\esLogs                # 指定日志文件的存放位置
	network.host: 192.168.1.10         # 指定绑定的 IP 地址
	http.port: 9200                    # 指定 HTTP 服务端口号
	
其中 cluster.name 为集群名称，node.name 为节点名称，path.data 和 path.logs 为指定数据和日志的存放位置，network.host 为绑定 IP 地址，http.port 为 HTTP 服务端口号。
# 4.4 启动 Elasticsearch
在命令行窗口执行以下命令，启动 Elasticsearch :

	cd <install directory>\bin
	.\elasticsearch.bat

启动成功后会看到类似以下输出信息：

	[2017-07-05T02:31:52,380][INFO ][o.e.n.Node ] [node-1] initialized
	[2017-07-05T02:31:52,382][INFO ][o.e.n.Node ] [node-1] starting...
	[2017-07-05T02:31:52,440][WARN ][o.e.b.BootstrapChecks ] [node-1] maximum file descriptors [65536] for Elasticsearch may be too low, increase to at least [65536]
	[2017-07-05T02:31:53,747][INFO ][o.e.t.TransportService ] [node-1] publish_address {192.168.1.10:9300}, bound_addresses {[::]:9300}
	[2017-07-05T02:31:53,757][INFO ][o.e.b.BootstrapChecks ] [node-1] bound or publishing to a non-loopback address, enforcing bootstrap checks
	[2017-07-05T02:31:53,884][INFO ][o.e.c.s.ClusterService ] [node-1] new_master {node-1}{tnyDrTsGTJmuv6seEuvVZg}{L6xeH0m4SXapotjrfidBtg}{192.168.1.10}{192.168.1.10:9300}, reason: zen-disco-elected-as-master ([0] nodes joined, [(0,node-1)] elected leader)
	[2017-07-05T02:31:54,169][INFO ][o.e.h.HttpServer     ] [node-1]publish_address {192.168.1.10:9200}, bound_addresses {[::]:9200}
	[2017-07-05T02:31:54,170][INFO ][o.e.n.Node ] [node-1] started

如果出现如下错误信息：

	bootstrap check failure [shards started on [[node-1][0]] primaries haven't been assigned to nodes yet]

可能是因为 Elasticsearch 没有完全启动，所以在启动之后会花费几分钟时间才开始处理请求。等待几分钟后再试。
# 4.5 测试 Elasticsearch
测试 Elasticsearch 是否已经正常工作，可以使用浏览器访问 http://localhost:9200/_cat/health?v 查看集群状态，如果返回 "status" 字段的值为 "green" 则表示 Elasticsearch 已经正常工作。如果出现黄色的 "yellow" 或红色的 "red" ，则表示 Elasticsearch 有问题，需要排查原因。
# 5. 创建索引
创建索引非常简单，只需要发送一个 PUT 请求到 http://localhost:9200/<index> URL ，然后发送 JSON 文档给服务器就可以了。下面创建一个名为 products 的索引，并添加一些样例数据：

	PUT /products
	{
	  "mappings": {
	    "product": {
	      "properties": {
	        "title": {"type": "string"},
	        "description": {"type": "string"}
	      }
	    }
	  }
	}
	
	POST /products/product/1
	{
	  "title": "My Product",
	  "description": "This is a great product."
	}
	
	GET /products/_search
	{
	  "query": {
	    "match_all": {}
	  }
	}
	
上面的第一条语句定义了一个名为 products 的索引，第二条和第三条语句分别向该索引插入两条产品数据，第四条语句使用 match_all 查询方式获取所有的产品数据。结果应该类似于这样：

	{
	  "took": 4,
	  "timed_out": false,
	  "_shards": {
	    "total": 5,
	    "successful": 5,
	    "failed": 0
	  },
	  "hits": {
	    "total": 2,
	    "max_score": null,
	    "hits": [
	      {
	        "_index": "products",
	        "_type": "product",
	        "_id": "1",
	        "_score": null,
	        "_source": {
	          "title": "My Product",
	          "description": "This is a great product."
	        }
	      }
	    ]
	  }
	}