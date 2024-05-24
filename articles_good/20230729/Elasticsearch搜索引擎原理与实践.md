
作者：禅与计算机程序设计艺术                    

# 1.简介
         
　　Elasticsearch 是开源分布式搜索引擎，提供搜素、分析、数据可视化等功能。它是一个基于 Lucene 的全文搜索服务器，能够把结构化或非结构化的数据经过索引生成一个索引库，使其可以被搜索到。
         　　在现代 Web 应用中，搜索功能已经成为不可或缺的一项功能。但是传统上，传统搜索方式需要依赖于数据库查询或者其他复杂的查询接口。而 Elasticsearch 提供了一种高效、稳定的、快速的方式进行数据的检索。
         　　本书以 Elasticsearch 为核心，深入浅出地阐述 Elasticsearch 在实际生产环境中的应用及原理。希望通过对 Elasticsearch 的原理及特性的讲解，帮助读者快速掌握 Elasticsearch 的使用技巧和最佳实践，提升工作效率和解决实际问题的能力。
         　　为了更好地阅读和理解本书的内容，建议读者具有扎实的计算机基础知识、熟悉 Linux 操作系统和 HTTP/RESTful 协议，具备一定的数据结构、算法能力和动手能力。同时，也期待读者能够积极参与到 Elasticsearch 社区的建设中来，共同推进 Elasticsearch 的发展和变革。
         
         # 2.搜索引擎的分类及特点
         　　搜索引擎分为基于信息检索模型的搜索引擎、基于内容关系模型的搜索引擎、基于用户行为模型的搜索引擎等。其中，基于信息检索模型的搜索引擎和基于内容关系模型的搜索引擎都属于垂直搜索引擎，根据搜索需求的不同，分别拥有不同的特色。
         ## 2.1. 基于信息检索模型的搜索引擎
         　　基于信息检索模型的搜索引擎(IR-based search engine)是最初级、低级别的搜索引擎。这种搜索引擎的索引一般由搜索词条、短语、句子、文档等构成，然后通过对这些元素的相似性计算，给予它们相应的权重，从而实现信息检索。主要特点如下：

         - 查询灵活性强：支持多种查询语法、模糊查询、布尔运算符、正则表达式、范围查询、排序、分页等多种查询条件；
         - 数据量小、静态性强：由于采用全文索引，所以索引大小和数据量都比较大；
         - 索引更新慢、准确性差：由于每次更新都要重新建立索引，所以索引更新速度缓慢；
         - 没有点击率预测机制：没有针对用户查询日志进行点击率预测，导致排名不准确；

         ## 2.2. 基于内容关系模型的搜索引擎
         　　基于内容关系模型的搜索引擎(CR-based search engine)是中间级、中等级别的搜索引擎。这种搜索引擎的索引一般包括网页、帖子、图片、视频、音频等多种类型的资源，并将其按内容关联起来。相关内容会被索引并给予较高的权重。主要特点如下：

         - 用户界面友好：具有便捷的用户界面，用户可以直观地找到所需的信息；
         - 结果准确性高：对网页、论坛帖子、图片、视频等各种形式的资源，均可给予较高的权重；
         - 有反馈机制：对用户的反馈意见进行适当处理，提升搜索引擎的效果；
         - 更新及时：对于内容发生变化的网站，搜索引擎的索引会及时更新；

         ## 2.3. 基于用户行为模型的搜索引擎
         　　基于用户行为模型的搜索引擎(UBM-based search engine)是高级别的搜索引擎，这种搜索引擎将用户行为和搜索结果结合起来，可以根据用户的历史记录、偏好及兴趣等因素，对相关内容进行推荐。主要特点如下：

         - 个性化推荐：用户可以通过搜索引擎进行个性化推荐，发现更多符合自己的兴趣爱好的内容；
         - 信息发现及传递：提供丰富的网站、博客、视频、音乐等信息源，让用户能够快速获取和传递信息；
         - 反映社交动态：通过网络社交工具，可以了解到热门话题的最新消息；
         - 更为有效的营销：提供更精准的广告投放，促进互联网商业模式的成长。

         　　综上所述，基于信息检索模型的搜索引擎、基于内容关系模型的搜索引擎、基于用户行为模型的搜索引擎各有特色，在互联网领域都扮演着重要角色。传统的搜索引擎更多是“傻瓜式”的，比如百度、Google，这类搜索引擎只提供简单的关键字搜索功能。当代的搜索引擎则更加智能化、复杂化，具有一定的自然语言处理能力，可以根据用户的搜索习惯和喜好提供合适的搜索结果。而 Elasticsearch 通过丰富的功能、特色和优化，逐步取代传统搜索引擎成为主流。

         　　此外，Elasticsearch 还提供了 RESTful API 和 Java 客户端开发包，可方便集成到现有的服务中，满足不同场景下的需求。另外，Elasticsearch 提供分布式部署、集群管理、水平扩展等能力，能够快速应对海量数据的搜索请求。因此，Elasticsearch 已经成为当前最热门、最知名的搜索引擎之一。

　　　　# 3.基本概念及术语
         　　Elasticsearch 使用一些基础概念和术语，如索引（Index）、类型（Type）、文档（Document）、分片（Shard）、副本（Replica）、字段（Field）、映射（Mapping）、路由（Routing）、节点（Node）、集群（Cluster）等。
         ## 3.1. 索引（Index）
         　　索引（Index）是一个存储数据的地方，类似于数据库中的表，每个索引都有一个名称，可以使用这个名称进行索引的增删改查。索引存储数据的位置和如何去索引这些数据，同时也可以定义字段数据类型、是否索引、索引存储等属性。
         ## 3.2. 类型（Type）
         　　类型（Type）是索引的一个逻辑上的分类，一个索引可以有多个类型，每种类型下又可以存储多个文档。例如，博客站点可以创建 article、comment、tag 三个类型，对应不同的文档。类型不是固定不变的，可以随时增加新的类型。
         ## 3.3. 文档（Document）
         　　文档（Document）是一个可查询的数据项，就是一组键值对。例如，一条评论就是一个文档。每个文档都有一个唯一标识 ID，文档中可以包含多个字段（field）。
         ## 3.4. 分片（Shard）
         　　分片（Shard）是 Elasticsearch 中用来分布式存储数据的方式。一个索引可以分布到多个分片中，这样可以将数据横向扩展。每个分片本身就是一个最小的独立的 Elasticsearch 实例，可以横向扩展到多台机器上。
         ## 3.5. 副本（Replica）
         　　副本（Replica）是分片（Shard）的另一种复制方式。每个索引都可以指定一个冗余度（Replication factor），表示每个分片需要保存几份副本。副本的存在保证了数据可用性。
         ## 3.6. 字段（Field）
         　　字段（Field）是文档（Document）的组成部分，一个字段可以有很多属性，比如字段名、数据类型、是否索引、是否存储等。
         ## 3.7. 映射（Mapping）
         　　映射（Mapping）是定义字段（Field）及其属性的过程。映射定义了一个文档的结构，决定了该文档可以有哪些字段，以及这些字段的属性。
         ## 3.8. 路由（Routing）
         　　路由（Routing）是 Elasticsearch 中的一个参数，用来决定将请求路由到的分片。
         ## 3.9. 节点（Node）
         　　节点（Node）是一个运行 Elasticsearch 服务的机器，可以是一个物理机也可以是虚拟机。一个集群可以由多个节点组成。
         ## 3.10. 集群（Cluster）
         　　集群（Cluster）是一个 Elasticsearch 实例集合。一个集群中可以有多个索引、节点。通常情况下，一个集群包含多个节点，一个节点运行一个 Elasticsearch 服务。

         # 4.核心算法原理及具体操作步骤
         　　本节介绍 Elasticsearch 的主要算法原理和相关操作步骤，包括数据添加、删除、更新、查询、排序、聚合等操作。
         ## 4.1. 数据添加
         　　数据添加（Index）的流程可以简单描述为以下步骤：

         - 创建一个索引（Index）；
         - 指定映射（Mapping）；
         - 插入文档（Document）；

         当执行索引的时候，ElasticSearch 会创建一个索引文件夹来存放数据。索引文件夹中的名称即为索引名称，文档将作为 JSON 文件保存在这个文件夹中。在插入文档的时候，ElasticSearch 会首先校验文档的结构与之前创建的映射是否匹配，如果匹配则创建新的文档文件，否则抛出异常终止操作。

         ## 4.2. 数据删除
         　　数据删除（Delete）的流程可以简单描述为以下步骤：

         - 删除文档（Document）；

         执行删除命令的时候，ElasticSearch 只需要将指定的文档标记为已删除即可。对于仍保留在硬盘中的文档，ElasticSearch 会在后台异步清除，不影响正常的业务操作。

         ## 4.3. 数据更新
         　　数据更新（Update）的流程可以简单描述为以下步骤：

         - 修改文档（Document）；

         当修改一个已存在的文档时，ElasticSearch 会用新文档覆盖旧文档。

         ## 4.4. 数据查询
         　　数据查询（Query）的流程可以简单描述为以下步骤：

         - 从索引中获取文档列表；
         - 对文档列表进行排序、过滤、分页等操作；

         ElasticSearch 支持丰富的查询语法，如 term、match、bool、range、sort、facets、suggester 等。term 查询根据文档中某一个字段的值进行匹配，match 查询则是对整个文档进行全文检索。

         ## 4.5. 数据排序
         　　数据排序（Sort）的流程可以简单描述为以下步骤：

         - 根据条件对文档列表进行排序；

         可以根据任意字段对文档列表进行排序，ElasticSearch 会根据该字段的值对文档进行排序。

        ## 4.6. 数据聚合
        数据聚合（Aggregations）的流程可以简单描述为以下步骤：

         - 对文档进行分组、汇总统计；

         Elasticsearch 提供对数据进行分组、汇总统计的能力，可以很容易地得到统计数据，如平均值、最大值、最小值等。

        # 5.代码实例和解释说明
        本章节介绍 Elasticsearch 在实际工程中的应用和代码实例，如 Java 客户端编程、DSL 搜索语法、RESTful API 调用等。
         ## 5.1. Java 客户端编程
         　　Java 客户端用于访问 Elasticsearch 集群，可以通过官方客户端或第三方客户端来访问。本章节介绍 Elasticsearch 官方 Java 客户端的使用方法。
         ### 5.1.1. 安装
         　　ElasticSearch Java 客户端安装非常简单，只需要添加 Maven 依赖即可。Maven 配置如下：

         ```xml
            <dependency>
                <groupId>org.elasticsearch.client</groupId>
                <artifactId>elasticsearch-rest-high-level-client</artifactId>
                <version>${elasticsearch.version}</version>
            </dependency>

            <!-- https://mvnrepository.com/artifact/com.fasterxml.jackson.core/jackson-databind -->
            <dependency>
                <groupId>com.fasterxml.jackson.core</groupId>
                <artifactId>jackson-databind</artifactId>
                <version>2.9.9.1</version>
            </dependency>
         ```

         ${elasticsearch.version} 表示 Elasticsearch 的版本号。
         　　为了更方便地管理依赖，建议将 Elasticsearch Java 客户端的依赖管理工具设置为 Gradle 或 SBT。
         ### 5.1.2. 初始化客户端
         　　初始化客户端的方法有两种：

         1. 创建客户端对象并传入集群地址：

         ```java
           RestHighLevelClient client = new RestHighLevelClient(RestClient.builder(HttpHost.create("http://localhost:9200")));
         ```

         2. 创建客户端对象并读取配置文件：

         ```java
           RestHighLevelClient client = new RestHighLevelClient(RestClients.create(Settings.builder().loadFromPath(Paths.get("/path/to/config")).build()));
         ```

         　　这里假定配置文件路径为 /path/to/config ，配置文件示例如下：

         ```yaml
            cluster.name: my-cluster
            node.name: node-1
            path.data: /var/elasticsearch/data
            http.port: 9200
         ```

         这段配置设置了集群名称、节点名称、数据目录、HTTP 端口号等信息。
         　　注意：如果采用第 2 种方法，需要确保配置文件中包含连接到 Elasticsearch 集群的必要信息，如集群名称、HTTP 端口号等。
         ### 5.1.3. 添加数据
         　　通过索引（index）操作可以向 Elasticsearch 中添加数据，代码如下：

         ```java
            // 创建索引
            CreateIndexRequest createIndexRequest = new CreateIndexRequest("my_index");
            try {
                AcknowledgedResponse response = client.indices().create(createIndexRequest, RequestOptions.DEFAULT);
                if (response.isAcknowledged()) {
                    System.out.println("索引创建成功！");
                } else {
                    System.out.println("索引创建失败！");
                }
            } catch (IOException e) {
                e.printStackTrace();
            }

            // 创建映射
            String mapping = "{
" +
                              "\"properties\": {
" +
                                "    \"name\": {
" +
                                  "        \"type\": \"text\",
" +
                                  "        \"analyzer\": \"ik_max_word\",
" +
                                  "        \"search_analyzer\": \"ik_smart\"
" +
                                "    },
" +
                                "    \"age\": {
" +
                                  "        \"type\": \"integer\"
" +
                                "    }
" +
                              "}
" +
                            "}";
            PutMappingRequest putMappingRequest = new PutMappingRequest("my_index").source(mapping, XContentType.JSON);
            try {
                AcknowledgedResponse response = client.indices()
                                                     .putMapping(putMappingRequest, RequestOptions.DEFAULT);
                if (response.isAcknowledged()) {
                    System.out.println("映射创建成功！");
                } else {
                    System.out.println("映射创建失败！");
                }
            } catch (IOException e) {
                e.printStackTrace();
            }

            // 添加文档
            IndexRequest indexRequest = new IndexRequest("my_index")
                                          .id("1")
                                          .source("{\"name\":\"张三\", \"age\": 30}", XContentType.JSON);
            try {
                IndexResponse response = client.index(indexRequest, RequestOptions.DEFAULT);
                if (response.status() == RestStatus.CREATED || response.status() == RestStatus.OK) {
                    System.out.println("文档添加成功！");
                } else {
                    System.out.println("文档添加失败！");
                }
            } catch (IOException e) {
                e.printStackTrace();
            }
         ```

         以上代码创建了一个名为 my_index 的索引，并添加了一篇文章《张三，30岁生日快乐》。其中，文章的作者姓名和年龄两个字段分别使用 text 和 integer 类型。创建索引和映射需要指定 mapping 对象，并解析为 JSON 字符串。
         　　注：由于中文分词器 ik_max_word 可能无法正确处理中文文本，因此需要将 analyzer 设置为 “ik_smart”，也就是中文分词器 ik_smart 。
         ### 5.1.4. 删除数据
         　　通过 delete 操作可以从 Elasticsearch 中删除数据，代码如下：

         ```java
             DeleteRequest deleteRequest = new DeleteRequest("my_index", "1");
             try {
                 DeleteResponse response = client.delete(deleteRequest, RequestOptions.DEFAULT);
                 if (response.status() == RestStatus.OK) {
                     System.out.println("文档删除成功！");
                 } else {
                     System.out.println("文档删除失败！");
                 }
             } catch (IOException e) {
                 e.printStackTrace();
             }
         ```

         此处通过 document ID 来删除一条文档。
         ### 5.1.5. 更新数据
         　　通过 update 操作可以更新 Elasticsearch 中已存在的数据，代码如下：

         ```java
             UpdateRequest updateRequest = new UpdateRequest("my_index", "1")
                                              .doc("{\"name\":\"李四\", \"age\": 31}", XContentType.JSON);
             try {
                 UpdateResponse response = client.update(updateRequest, RequestOptions.DEFAULT);
                 if (response.status() == RestStatus.OK) {
                     System.out.println("文档更新成功！");
                 } else {
                     System.out.println("文档更新失败！");
                 }
             } catch (IOException e) {
                 e.printStackTrace();
             }
         ```

         此处通过 document ID 来更新一条文档，并替换 name 字段为 “李四”，age 字段为 31。
         ### 5.1.6. 查询数据
         　　通过 search 操作可以从 Elasticsearch 中查询数据，代码如下：

         ```java
             SearchRequest searchRequest = new SearchRequest("my_index");
             SearchSourceBuilder sourceBuilder = new SearchSourceBuilder();

             // bool query example
             BoolQueryBuilder boolQueryBuilder = QueryBuilders.boolQuery();
             boolQueryBuilder.must(QueryBuilders.termQuery("name", "张三"));
             boolQueryBuilder.should(QueryBuilders.termQuery("age", 30));
             boolQueryBuilder.minimumShouldMatch(1);
             sourceBuilder.query(boolQueryBuilder);

             // match all query example
             MatchAllQueryBuilder matchAllQueryBuilder = QueryBuilders.matchAllQuery();
             sourceBuilder.query(matchAllQueryBuilder);

             // sort by age ascending example
             sourceBuilder.sort("age");

             searchRequest.source(sourceBuilder);

             try {
                 SearchResponse response = client.search(searchRequest, RequestOptions.DEFAULT);
                 long totalHits = response.getHits().getTotalHits().value;
                 List<String> names = new ArrayList<>();
                 for (SearchHit hit : response.getHits()) {
                     Map<String, Object> sourceAsMap = hit.getSourceAsMap();
                     Integer id = (Integer) sourceAsMap.get("id");
                     String name = (String) sourceAsMap.get("name");
                     int age = (int) sourceAsMap.get("age");

                     names.add(name);
                 }

                 System.out.println("命中：" + totalHits + " 条，名字：" + Arrays.toString(names.toArray()));
             } catch (IOException e) {
                 e.printStackTrace();
             }
         ```

         上面的代码创建了一个 bool query 和 match all query ，并按照 age 字段升序排序。执行 search 请求后，得到命中的总条数和命中文档的名字列表。
         ### 5.1.7. 聚合数据
         通过 aggregation 操作可以对 Elasticsearch 中已存在的数据进行聚合统计，代码如下：

         ```java
             SearchRequest searchRequest = new SearchRequest("my_index");
             SearchSourceBuilder sourceBuilder = new SearchSourceBuilder();

             TermsAggregationBuilder termsAggBuilder = AggregationBuilders
                                                        .terms("agg_by_age")
                                                        .field("age")
                                                        .size(100);
             sourceBuilder.aggregation(termsAggBuilder);

             searchRequest.source(sourceBuilder);

             try {
                 SearchResponse response = client.search(searchRequest, RequestOptions.DEFAULT);
                 List<? extends Bucket> buckets = ((Terms) response.getAggregations().get("agg_by_age")).getBuckets();
                 for (Bucket bucket : buckets) {
                     String keyAsString = bucket.getKeyAsString();
                     long docCount = bucket.getDocCount();

                     System.out.println(keyAsString + ": " + docCount);
                 }
             } catch (IOException e) {
                 e.printStackTrace();
             }
         ```

         此处先创建了一个 terms 聚合，并指定聚合字段为 age，size 为 100。执行 search 请求后，得到 age 的值和命中条数的映射关系。
         ## 5.2. DSL 搜索语法
         DSL 搜索语法（Domain Specific Language Search Syntax）用于定义、构造复杂查询语句，如布尔查询、全文检索、排序、聚合等。Elasticsearch 提供了一套基于 JSON 的 DSL 搜索语法。
         ### 5.2.1. 语法结构
         　　DSL 搜索语句的基本结构如下：

         ```json
            {
               "query": {...},          // 查询条件
               "_source": ["fields"],   // 选择返回字段
               "aggs": {...},           // 聚合统计
              ...                      // 其他选项
            }
         ```

         每个搜索语句至少包含一个 query ，且只能有一个 query 。query 的详细语法和类型请参考 Elasticsearch 官方文档。
         ### 5.2.2. 查询语法
         　　Elasticsearch 提供了丰富的查询语法，包括 term 查询、match 查询、bool 查询、range 查询、regexp 查询、fuzzy 查询、prefix 查询、wildcard 查询、geo 查询等。

         　　示例：

         ```json
            {
               "query":{
                  "match":{
                     "title":"Elasticsearch"
                  }
               }
            }
         ```

         这个例子使用 match 查询搜索 title 字段包含关键字 Elasticsearch 的所有文档。
         　　除了查询语法之外，Elasticsearch 还支持组合查询，可以把多个查询条件合并为一个查询语句，形成复合查询。复合查询可以嵌套使用，且支持 AND、OR、NOT 等操作符。
         　　示例：

         ```json
            {
               "query":{
                  "bool":{
                     "must":[
                        {"match":{"title":"Elasticsearch"}},
                        {"range":{"publish_time":{"gte": "2017-01-01"}}}
                     ],
                     "filter":[{"term":{"category":"tech"}}]
                  }
               }
            }
         ```

         此例使用 bool 查询组合了 match 查询和 range 查询，查询条件是：文档的 title 字段含有关键字 Elasticsearch，并且 publish_time 大于等于 2017 年 1 月 1 日，文档的 category 字段值为 tech 。NOT 操作符还可以在 filter 层级上使用，不过不能用于 nested 文档中。
         ### 5.2.3. 返回字段
         　　默认情况下，Elasticsearch 只会返回匹配查询的字段。如果想返回指定字段，可以通过 _source 参数来指定，如：

         ```json
            {
               "_source":["title","url"]
            }
         ```

         上面这段查询将只返回 title 和 url 这两个字段。
         　　如果想要返回所有的字段，可以省略 _source 选项。
         ### 5.2.4. 排序
         Elasticsearch 支持对搜索结果进行排序，通过 sort 参数指定排序字段和方向。

         　　示例：

         ```json
            {
               "sort":[
                  {"age":"desc"},
                  {"_score"}
               ]
            }
         ```

         此例按照 age 字段降序和默认的评分（_score）排序。排序字段可以是任何字段，默认情况下 _score 降序排序。
         ### 5.2.5. 分页
         Elasticsearch 默认只返回前 10 条匹配结果，可以通过 size 和 from 参数控制结果数量和偏移。

         　　示例：

         ```json
            {
               "from":10,
               "size":20
            }
         ```

         此例返回第 10~30 条匹配结果。
         ### 5.2.6. 高亮
         Elasticsearch 支持对搜索结果进行高亮显示，通过 highlight 参数指定高亮字段和标签。

         　　示例：

         ```json
            {
               "highlight":{
                  "pre_tags":["<b>"],
                  "post_tags":["</b>"],
                  "fields":{
                     "content":{
                        "fragment_size":150,
                        "number_of_fragments":3,
                        "no_match_size":300
                     },
                     "title":{
                        "fragment_size":50,
                        "number_of_fragments":1
                     }
                  }
               }
            }
         ```

         此例开启 content 字段的高亮功能，高亮标签为 <b></b> ，指定 fragment_size 为 150，number_of_fragments 为 3，无匹配时 fragment 的长度为 no_match_size （默认为 100）。
         ## 5.3. RESTful API 调用
         Elasticsearch 提供了基于 HTTP 的 RESTful API，允许外部程序调用。本节介绍 Elasticsearch 中使用的几个典型的 API ，如索引、搜索、查询、集群管理等。
         ### 5.3.1. 创建索引
         通过 POST /{index}/ HTTP 方法创建索引。
         　　请求示例：

         ```bash
            curl -XPOST 'http://localhost:9200/my_index' \
              -H 'Content-Type: application/json' \
              -d '{
                   "settings": {
                       "number_of_shards": 3,
                       "number_of_replicas": 2
                   },
                   "mappings": {
                       "doc": {
                           "properties": {
                               "title": {"type": "keyword"},
                               "content": {"type": "text"}
                           }
                       }
                   }
               }'
         ```

         此例创建一个名为 my_index 的索引，设置 shards 为 3 个，replicas 为 2 个。索引类型为 doc ，字段 title 和 content 分别设置为 keyword 和 text 类型。
         ### 5.3.2. 查看索引列表
         通过 GET / HTTP 方法查看所有索引列表。
         　　请求示例：

         ```bash
            curl -XGET 'http://localhost:9200/'
         ```

         返回结果：

         ```json
            {
               "my_index" : {
                  "aliases" : {},
                  "mappings" : {
                     "doc" : {
                        "properties" : {
                           "title" : {
                              "type" : "keyword"
                           },
                           "content" : {
                              "type" : "text"
                           }
                        }
                     }
                  },
                  "settings" : {
                     "index" : {
                        "creation_date" : "1537921275128",
                        "uuid" : "tjvVOFUGTNW3Cpz0qJiMxg",
                        "number_of_shards" : "3",
                        "number_of_replicas" : "2",
                        "version" : {
                           "created" : "6050199"
                        },
                        "provided_name" : "my_index"
                     }
                  }
               }
            }
         ```

         ### 5.3.3. 查看索引设置
         通过 GET /{index}/_settings HTTP 方法查看索引设置。
         　　请求示例：

         ```bash
            curl -XGET 'http://localhost:9200/my_index/_settings?pretty'
         ```

         返回结果：

         ```json
            {
               "my_index" : {
                  "settings" : {
                     "index" : {
                        "number_of_shards" : "3",
                        "number_of_replicas" : "2",
                        "uuid" : "TGuBduvETxWhhwNoetA5fA",
                        "version" : {
                           "created" : "6050199"
                        },
                        "creation_date" : "1537921275128",
                        "provided_name" : "my_index"
                     }
                  }
               }
            }
         ```

         ### 5.3.4. 插入文档
         通过 POST /{index}/{type}/{id} HTTP 方法插入文档。
         　　请求示例：

         ```bash
            curl -XPUT 'http://localhost:9200/my_index/doc/1' \
              -H 'Content-Type: application/json' \
              -d '{
                   "title": "Hello World!",
                   "content": "This is the first post."
               }'
         ```

         此例插入一条标题为 Hello World!，内容为 This is the first post. 的文档到索引 my_index 的 doc 类型中，ID 为 1。
         ### 5.3.5. 删除文档
         通过 DELETE /{index}/{type}/{id} HTTP 方法删除文档。
         　　请求示例：

         ```bash
            curl -XDELETE 'http://localhost:9200/my_index/doc/1'
         ```

         此例删除索引 my_index 中的文档 ID 为 1 的内容。
         ### 5.3.6. 查询文档
         通过 POST /{index}/{type}/_search HTTP 方法查询文档。
         　　请求示例：

         ```bash
            curl -XPOST 'http://localhost:9200/my_index/doc/_search?pretty' \
              -H 'Content-Type: application/json' \
              -d '{
                   "query": {
                      "match": {
                         "content": "first post"
                      }
                   }
               }'
         ```

         此例查询索引 my_index 中的 doc 类型的所有文档，查询内容是包含关键词 first post 的内容。
         ### 5.3.7. 更新文档
         通过 PUT /{index}/{type}/{id} HTTP 方法更新文档。
         　　请求示例：

         ```bash
            curl -XPATCH 'http://localhost:9200/my_index/doc/1' \
              -H 'Content-Type: application/json' \
              -d '{
                   "script": "ctx._source.likes += params.inc",
                   "params": {
                      "inc": 1
                   }
               }'
         ```

         此例更新索引 my_index 中的文档 ID 为 1 的 likes 属性值，累加 1 。
         ### 5.3.8. 获取集群健康状态
         通过 GET /_cat/health HTTP 方法获取集群健康状态。
         　　请求示例：

         ```bash
            curl -s 'http://localhost:9200/_cat/health?v&pretty'
         ```

         此例获取集群 health 状态。
         ### 5.3.9. 管理集群
         Elasticsearch 提供了集群管理 API，允许外部程序管理集群的各项操作。

