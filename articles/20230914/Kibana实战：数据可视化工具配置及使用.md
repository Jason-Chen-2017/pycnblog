
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 1.1 为什么要写这个主题？

在互联网行业中，搜索引擎（或称之为索引服务器）已经成为各个公司的必备基础设施。而数据可视化（Data Visualization）也是近年来最热门的话题之一。一般情况下，数据可视化工具往往需要根据业务特点进行定制开发。如果想要将数据可视化工具应用到公司内部工作流程中，并提升效率，那么就需要对它进行配置、管理以及使用技巧的培训。因此，作为一名技术专家，我认为在这方面我可以提供一些帮助。

## 1.2 Kibana 是什么？
Kibana是一个开源的高级数据可视化平台。它基于ElasticSearch打造，提供了数据的分析、可视化功能，具有强大的查询语言，能够轻松地实现复杂的数据分析。它的界面简洁、直观，并且集成了众多插件，能够满足不同场景下的需求。

## 1.3 本书如何学习？
本书的内容主要分为以下六章：

1. Kibana的安装与配置
2. 数据导入
3. 数据过滤
4. 可视化效果展示
5. 仪表盘创建与管理
6. 插件与扩展

每一章节都将详细地介绍相关知识，并配以相应的代码实例。通过学习完整个教程之后，读者不仅会掌握Kibana的基本用法，还能够熟练地运用其各种功能解决实际的问题。当然，也会受益于作者提供的经验之谈。

## 1.4 作者是谁？
李卓桓（又名NLPers），目前就职于百度基础搜索团队，担任搜索服务研发工程师。除了在Elasticsearch/Kibana方面的研究外，他还有丰富的项目实践经验。有时候，他喜欢分享一些个人感悟，所以写下这本《Kibana实战》系列教程也是一项不错的尝试。他将持续更新和维护此系列教程，欢迎感兴趣的读者关注。

# 2.Kibana的安装与配置

Kibana 可以单独安装或者与 Elasticsearch 安装在同一台机器上，这一章节将介绍怎样安装 Kibana 。如果你已经安装了 Elasticsearch ，则直接跳到“配置 Elasticsearch”这一小节即可。 

## 2.1 操作系统要求
Kibana 支持 Linux、macOS 和 Windows 操作系统。本教程基于 CentOS7 环境编写。

## 2.2 安装Java运行时环境
由于 Kibana 需要 Java 运行环境支持，所以首先需要安装Java运行时环境。

```bash
sudo yum install java-1.8.0-openjdk -y
```

## 2.3 安装Kibana
Kibana 的安装包可以在 Kibana 官网下载。由于国内网络限制，访问Kibana官网可能比较慢，所以建议下载镜像源上的安装包。

```bash
wget https://artifacts.elastic.co/downloads/kibana/kibana-7.9.3-linux-x86_64.tar.gz
tar zxvf kibana-7.9.3-linux-x86_64.tar.gz && rm -f kibana-7.9.3-linux-x86_64.tar.gz
mv kibana-7.9.3-linux-x86_64 /usr/local/kibana
```

## 2.4 配置Kibana
Kibana 默认监听在 `http://localhost:5601` ，可以通过修改配置文件`/etc/kibana/kibana.yml` 来修改端口号或者绑定地址等参数。

```yaml
server.port: 5602 # 修改端口号
server.host: "0.0.0.0" # 允许外部访问
elasticsearch.hosts: ["http://localhost:9200"] # 指定 ElasticSearch 地址
```

## 2.5 启动Kibana
启动 Kibana 服务：

```bash
nohup./bin/kibana &
```

启动成功后，打开浏览器输入 `http://<hostname>:5602/` ，出现登录页面说明 Kibana 已经正常启动。默认用户名密码都是 `admin`/`changeme`。



## 2.6 配置 Elasticsearch 
当我们安装好 Kibana 并启动成功后，需要做一些必要的设置才能让 Kibana 和 Elasticsearch 进行通信。

### 2.6.1 安装并启动 Elasticsearch
如果您已经安装并启动了 Elasticsearch ，则可以跳过这一步。

```bash
wget https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.9.3-linux-x86_64.tar.gz
tar zxvf elasticsearch-7.9.3-linux-x86_64.tar.gz && rm -f elasticsearch-7.9.3-linux-x86_64.tar.gz
mv elasticsearch-7.9.3 /usr/local/es
cd /usr/local/es && bin/elasticsearch
```

启动成功后，通过浏览器访问 `http://localhost:9200` ，如果看到类似 `{"name":"cluster_name","cluster_name":"elasticsearch"}` 的输出，则证明 Elasticsearch 已启动。否则请检查 Elasticsearch 是否正确安装及启动。

### 2.6.2 创建索引模板
Kibana 只能连接已存在的 Elasticsearch 集群，所以我们需要先把需要可视化的数据导入 Elasticsearch 中。这里我们创建一个空的索引模板，所有导入到 Elasticsearch 中的数据都会按照我们指定的映射规则被解析、存储。

```bash
curl -H 'Content-Type: application/json' -XPUT http://localhost:9200/_template/my_index_template -d@./my_index_template.json
```

其中 my_index_template.json 文件的内容如下：

```json
{
  "index_patterns": [
    "*"
  ],
  "order": 0,
  "settings": {
    "number_of_shards": 1
  },
  "mappings": {}
}
```

这样我们就可以通过 Kibana 连接 Elasticsearch 了。

# 3.数据导入
Kibana 使用 Elasticsearch 的 RESTful API 接口对数据进行导入和管理。为了使得大家更加容易理解，我们接下来使用 Elasticsearch Python 库对数据进行导入，然后再通过 Kibana 可视化展示。

## 3.1 安装 Elasticsearch Python 客户端
因为 Python 有许多数据处理、分析、可视化工具库，所以我们选择安装 Elasticsearch Python 客户端来连接 Elasticsearch 服务器。

```bash
pip install elasticsearch==7.9.3
```

## 3.2 连接 Elasticsearch 服务器
导入 Elasticsearch 之前，我们需要连接 Elasticsearch 服务器。

```python
from elasticsearch import Elasticsearch

es = Elasticsearch(hosts=['http://localhost:9200'])
```

## 3.3 创建索引
Elasticsearch 中数据是以索引的形式存储的。我们可以指定一个索引名称，然后向该索引中插入文档。

```python
index_name ='my_index'
doc = {'author': 'Alice', 'text': 'Hello world'}
res = es.index(index=index_name, doc_type='_doc', id='1', body=doc)
print(res['result']) # 会打印 'created' 或 'updated'
```

## 3.4 查询数据
可以使用 Elasticsearch 提供的查询 DSL 来查询索引中的数据。

```python
query = {"match_all":{}}
res = es.search(index=index_name, body={'query': query})
for hit in res['hits']['hits']:
    print(hit['_source'])
```

## 3.5 导入数据
对于批量导入数据来说，我们可以使用 Streaming Bulk API 来优化性能。

```python
import csv
import json

def read_data():
    with open('data.csv') as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield {
                '_index': index_name,
                '_id': row['id'],
                '_source': row
            }

responses = helpers.streaming_bulk(es, read_data())
count = 0
for success, info in responses:
    if not success:
        print('Doc failed to insert:', info)
    else:
        count += 1
print(f'{count} documents inserted.')
```

# 4.数据过滤
Kibana 中的数据过滤器可以帮助用户快速找到所需的数据。在这里，我们将详细介绍如何使用数据过滤器来过滤数据。

## 4.1 创建过滤器
点击左侧导航栏中的 Filters 按钮，进入 Filters 页面。然后，点击右上角的 Create a new filter 按钮。


填写过滤器信息：

1. Filter name：过滤器名称，可以随意命名；
2. Index pattern：索引模式，用来指定匹配哪些索引；
3. Query string：查询字符串，用来指定匹配哪些数据。


保存过滤器：点击右上角的 Save button 按钮保存过滤器设置。

## 4.2 使用过滤器
过滤器创建完成后，就可以在 Dashboard 页面中使用过滤器。只需在 Metrics 页面点击 Add Filter 按钮，然后选择刚才创建的过滤器，即可对数据进行过滤。


# 5.可视化效果展示
Kibana 通过不同的可视化效果来显示数据，包括饼图、柱状图、线图、散点图等。在这里，我们将详细介绍这些可视化效果的用法。

## 5.1 概览图
概览图（Discover）主要用于查看数据总览，主要用途是在一个页面浏览所有的数据。

### 5.1.1 查看数据总览
点击导航栏的 Discover 按钮，进入 Discover 页面。


查看数据总览：

1. Search bar：可以自由输入查询条件；
2. Time picker：可以选择时间范围；
3. Refresh Interval：可以自定义刷新间隔；
4. Results：展示结果。

### 5.1.2 自定义显示字段
点击导航栏的 Columns 按钮，进入 Columns 页面。


自定义显示字段：

1. Show columns：可以开启或关闭显示列；
2. Rearrange columns：可以重新排列显示顺序；
3. Rename column：可以重命名显示列；
4. Edit column：可以编辑显示字段。

### 5.1.3 使用聚合
聚合（Aggregations）可以对数据进行统计、汇总等操作。点击导航栏的 Aggregations 按钮，进入 Aggregations 页面。


使用聚合：

1. Type of aggregation：可以选择聚合方式；
2. Field：可以选择聚合字段；
3. Aggregation label：可以设置聚合名称；
4. Options：可以自定义聚合方式。

## 5.2 指标分析
指标分析（Visualize）用于呈现数据的分布、变化趋势、关联关系等。

### 5.2.1 创建指标分析
点击导航栏的 Visualize 按钮，进入 Visualize 页面。


点击右上角的 Create visualization 按钮，创建新的指标分析。


### 5.2.2 选择数据源
选择数据源：

1. Index pattern：选择需要可视化的数据；
2. Query string：可输入查询条件。

### 5.2.3 设置可视化类型
设置可视化类型：

1. Visualization type：选择可视化类型；
2. Buckets：可以自定义分组方式；
3. Split by field：可以设置按字段分组；
4. Color scheme：可以设置颜色风格；
5. X-axis：可以设置 X 轴；
6. Y-axis：可以设置 Y 轴。

### 5.2.4 添加过滤器
添加过滤器：

1. Add filter：选择过滤器；
2. Value selection：选择过滤值；
3. Apply and add another：增加多个过滤器；
4. Clear filters：清除过滤器。

### 5.2.5 设置交叉筛选器
设置交叉筛选器：

1. Drag fields into the panel or click Add cross-field filter；
2. Choose operator；
3. Set value；
4. Click Apply changes。

### 5.2.6 添加标签
添加标签：

1. Labels：选择标签；
2. Select metric or dimension；
3. Set labels；
4. Click Apply changes。

### 5.2.7 设置编辑模式
设置编辑模式：

1. Edit mode：选择编辑模式。

### 5.2.8 保存视图
保存视图：

1. View Name：输入视图名称；
2. Save view。

### 5.2.9 在 Dashboard 中展示可视化
在 Dashboard 中展示可视化：

1. Dashboard 页面；
2. AddVisualization：点击 + 按钮；
3. 选择可视化；
4. Customize visualization：设置可视化属性；
5. Positioning：拖动位置；
6. Title：设置标题；
7. Save dashboard。

# 6.仪表盘创建与管理
仪表盘（Dashboard）是 Kibana 的核心组件，可以将多个可视化效果整合到一个视图中，并提供快捷的导航入口。在这里，我们将详细介绍仪表盘的用法。

## 6.1 创建仪表盘
点击导航栏的 Dashboard 按钮，进入 Dashboard 页面。


点击右上角的 Create new dashboard 按钮，创建一个新的仪表盘。


## 6.2 添加可视化元素
添加可视化元素：

1. AddPanel：点击 + 按钮；
2. 选择可视化；
3. Customize visualization：设置可视化属性；
4. Panel title：设置面板标题；
5. Positioning：拖动位置；
6. Save dashboard。

## 6.3 查看仪表盘
查看仪表盘：

1. 点击 Navigation 按钮；
2. 选择仪表盘。

## 6.4 保存仪表盘
保存仪表盘：

1. 点击 Save button 按钮；
2. Input dashboard name；
3. Click Save As...。

## 6.5 删除仪表盘
删除仪表盘：

1. 点击 More actions 按钮；
2. 选择 Delete dashboard 菜单项；
3. Confirm deletion。

# 7.插件与扩展
Kibana 提供了丰富的插件系统，你可以安装第三方插件，通过插件提供的可视化效果，来丰富你的 Kibana 体验。本文将介绍 Kibana 插件的两种分类——可视化插件和安全插件。

## 7.1 可视化插件
可视化插件是基于 Elasticsearch 技术的，它可以实现复杂的数据可视化功能。一般来说，可视化插件都可以通过命令行安装，也可以通过 UI 安装。

### 7.1.1 命令行安装可视化插件
使用 Elasticsearch 插件安装器来安装可视化插件。

```bash
bin/kibana-plugin install <plugin>
```

### 7.1.2 UI 安装可视化插件
Kibana 用户界面提供了安装插件的功能。

#### 7.1.2.1 转到 Kibana 安装插件页面
登录 Kibana 用户界面，点击左侧导航栏中的 Management > Stack Management > Kibana > Install New Plugin。

#### 7.1.2.2 选择插件源
选择插件源：

1. OSS (Open Source Software)：官方插件源；
2. Commercial：商业插件源。

#### 7.1.2.3 搜索插件
搜索插件：

1. Enter keyword to search plugins；
2. Sort by popularity or alphabetically；
3. Filter by category（可选）。

#### 7.1.2.4 选择插件
选择插件：

1. Check plugin checkbox；
2. Select version（可选）。

#### 7.1.2.5 安装插件
安装插件：

1. Click Install；
2. Wait until installation completes。

#### 7.1.2.6 启用插件
启用插件：

1. Go back to Dashboard page；
2. Enable newly installed plugins from the Settings tab（可选）。

### 7.1.3 使用可视化插件
使用可视化插件：

1. 查找插件提供的可视化效果；
2. 将插件提供的可视化效果添加到仪表盘中；
3. 自定义可视化效果的属性；
4. 保存仪表盘。

## 7.2 安全插件
安全插件可以保护你的 Kibana 实例免受攻击。比如身份验证、授权控制、敏感信息加密等。

### 7.2.1 命令行安装安全插件
使用 Elasticsearch 插件安装器来安装安全插件。

```bash
bin/kibana-plugin install <plugin>
```

### 7.2.2 UI 安装安全插件
Kibana 用户界面提供了安装插件的功能。

#### 7.2.2.1 转到 Kibana 安装插件页面
登录 Kibana 用户界面，点击左侧导航栏中的 Management > Stack Management > Kibana > Install New Plugin。

#### 7.2.2.2 选择插件源
选择插件源：

1. OSS (Open Source Software)：官方插件源；
2. Commercial：商业插件源。

#### 7.2.2.3 搜索插件
搜索插件：

1. Enter keyword to search plugins；
2. Sort by popularity or alphabetically；
3. Filter by category（可选）。

#### 7.2.2.4 选择插件
选择插件：

1. Check plugin checkbox；
2. Select version（可选）。

#### 7.2.2.5 安装插件
安装插件：

1. Click Install；
2. Wait until installation completes。

#### 7.2.2.6 启用插件
启用插件：

1. Go back to Dashboard page；
2. Enable newly installed plugins from the Settings tab（可选）。

### 7.2.3 使用安全插件
使用安全插件：

1. 查找插件提供的安全特性；
2. 根据安全特性，配置 Kibana；
3. 浏览 Kibana，确认插件的作用。