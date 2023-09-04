
作者：禅与计算机程序设计艺术                    

# 1.简介
         
Apache Kylin是一个开源的分布式分析引擎，其设计目标是在多维数据集上构建统一的数据模型、OLAP服务器及Cube技术，提供丰富的RESTful API接口及SQL支持能力，支持海量数据高性能查询。它的设计基于贯穿于整个分析生命周期的所有环节，包括数据导入、数据清洗、切分、聚合、计算、存储、以及OLAP服务器的动态扩展等。通过有效利用资源管理、查询优化、访问权限控制等方面，Apache Kylin能够帮助企业降低成本、提升效率、提高质量，并可支撑海量数据的快速查询分析需求。
# 2.Apache Kylin架构
Apache Kylin的架构如下图所示：
从上图中可以看出，Apache Kylin主要由两大模块构成：

- OLAP Server 模块：用于查询Cube的元数据信息，包括Cube列表、Cube定义、维度定义等。并且OLAP Server还负责将数据切分、聚合、汇总等过程。
- Query Engine 模块：采用先进的多线程查询处理方式，充分利用集群硬件资源，支持多种查询语言，如SQL、Hive SQL和Java API，支持复杂的业务逻辑及多种数据源接入。同时还提供了API接口，允许用户自行开发定制的应用。
因此，Apache Kylin整体架构具有高灵活性、高伸缩性和易维护等特点。但同时也存在诸多局限性：

1. 依赖Hadoop生态：由于OLAP Server与HDFS紧密耦合，只能运行于Hadoop生态系统之上；而不能兼容Spark或Flink等其他计算引擎。
2. 数据模型固定：Apache Kylin默认使用星型模型（即所有字段都作为维度），不支持自由组合字段、自定义字段等。而且在模型层面，只支持少数的统计函数，缺乏对业务的深度理解。
3. 功能较弱：Apache Kylin的功能还比较简单，只有Cube的定义、计算以及基于SQL的查询功能，缺乏完整的计算引擎，比如MLLib、GraphX等。同时缺少对第三方组件的支持。
# 3.Apache Kylin在百度的实践
目前，我司已有多个基于Apache Kylin的大数据分析平台。其中，数十亿条广告数据量的媒资库，每天产生数千万次的访问请求，需要实时反馈推荐。我在该项目的研发过程遇到过很多问题，并成功克服了它们。
## 3.1 数据准备
我们首先从离线计算平台抽取了近5年的数据。主要包含以下几个方面：
- 广告相关的数据：广告组、广告单元、广告特征等。
- 用户画像数据：包含用户基本属性、兴趣偏好、行为习惯等。
- 位置信息：用户所在的城市、省份、县区等信息。
- 操作日志数据：记录用户的搜索、点击等操作记录。
- 设备指纹数据：包括IMEI码、MAC地址等信息。

为了使数据规模适中，我们仅选取了2020年1月至7月的数据进行分析。各个表的样例如下：

| AdGroup | Campaign | DisplayNetwork | DeviceType | AdUnit | AdType |...|
|---|---|---|---|---|---|---|
| G1 | C1 | ADNETWORK_1 | PC | AU1 | IMG |...|
| G1 | C1 | ADNETWORK_1 | PHONE | AU1 | TEXT |...|
| G1 | C1 | ADNETWORK_1 | TABLET | AU1 | VIDEO |...|

| UserProfile |...|
|---|---|
| U1 |...|
| U2 |...|

| ClickLog | SearchLog |...|
|---|---|---|
| CL1 | SL1 |...|
| CL2 | SL2 |...|

| IMEICode | MacAddress | IPAddress |...|
|---|---|---|---|
| IM1 | MA1 | IP1 |...|
| IM2 | MA2 | IP2 |...|

## 3.2 数据导入
Apache Kylin官方提供了一个叫做HFileLoader的工具，它可以将各种格式的原始数据转换为Apache Kylin可直接使用的二进制格式的HFile文件。我们使用该工具将抽取的数据转换为HFile文件。

但是Apache Kylin的HFileLoader无法将文本类型的数据正确地解析为字符串，会导致一些统计结果异常。所以，我们编写了自己的脚本，使用命令行工具hive完成数据转换工作。
```shell
hive -f hive_ddl.sql -i userdata.txt;
```
我们根据实际情况编写了hive_ddl.sql文件，设置每个列的类型及字段长度等。
```sql
create external table if not exists adgroup (
    adgroup string comment 'Ad group ID',
    campaign string comment 'Campaign ID',
    displaynetwork string comment 'Display network ID',
    devicetype string comment 'Device type',
    adunit string comment 'Ad unit ID',
    adtype string comment 'Ad type'
) stored as parquet location '/path/to/adgroup';

load data inpath '/path/to/userprofile/' into table userprofile;
load data inpath '/path/to/clicklog/' into table clicklog;
load data inpath '/path/to/searchlog/' into table searchlog;
load data inpath '/path/to/imeicode/' into table imeicode;
load data inpath '/path/to/macaddress/' into table macaddress;
```
这样，我们就将所有需要的数据都导入到了HBase表中。这些数据包括：广告数据、用户画像数据、点击数据、搜索数据、设备指纹数据等。我们为不同的表分别创建了不同的Hive分区，便于后续查询。
## 3.3 数据准备和导入之后，下一步就是进行数据清洗。因为Apache Kylin的OLAP功能依赖于HBase表，所以数据预处理很重要。清洗阶段主要完成以下几项工作：
- 对数值型数据进行标准化；
- 删除无效或重复的数据；
- 将同类数据拆分为不同维度，方便Cubing；
- 补全缺失的数据；
- 修改数据类型的错误；

清洗完的数据示例如下：

| AdGroup | Campaign | DisplayNetwork | DeviceType | AdUnit | AdType | PositionCityId | PositionProvinceId | PositionDistrictId | SearchKeyword | KeywordMatchType | IsSearchClick | Month | DayOfWeek | HourOfDay | DailyVisitCount | TotalVisitCount | Gender | AgeBracket | InterestCategory | AverageDailyVisitsPerUser |...|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| G1 | C1 | ADNETWORK_1 | PC | AU1 | IMG | 1 | 11 | null | 搜索词A | exact_match | true | 202001 | 1 | 0 | 10 | 20 | Male | age_under_18 | Books and Literature | 10 |...|
| G1 | C1 | ADNETWORK_1 | PHONE | AU1 | TEXT | 2 | 22 | 2201 | 搜索词B | phrase_match | false | 202001 | 2 | 12 | 20 | 15 | Female | age_over_18_and_below_60 | Entertainment | 5 |...|

## 3.4 创建Cube
Cube是一种特殊的OLAP数据结构。它由一个或多个Measures和一个或多个Dimensions组成。在我们的场景中，我们需要创建一个广告效果的Cube。我们选择如下维度：AdGroup、Campaign、DeviceType、PositionCityId、PositionProvinceId、Gender、AgeBracket、InterestCategory，以及Clicks、Impressions、Conversions、Cost、AverageCpc、TotalConversions、ViewableImpressions、VisibleAverageDuration等指标。

其中，AdGroup、Campaign、DeviceType、PositionCityId、PositionProvinceId分别是广告组、广告计划、设备类型、广告位置城市ID、广告位置省份ID；Gender、AgeBracket、InterestCategory分别是用户性别、年龄范围、感兴趣分类；Clicks、Impressions、Conversions、Cost分别是广告点击、展现、发生 conversions 的数量、费用；AverageCpc、TotalConversions、ViewableImpressions、VisibleAverageDuration分别是平均CPC、总conversions、可见展现数、平均可见时长。

然后，我们按照如下方式创建了Cube：

- 创建Cube配置文件：在conf文件夹下创建kylin.properties文件，添加如下配置项：

```ini
kylin.env=DEV # 测试环境
kylin.hbase.zookeeper.quorum=hbase-host:2181
kylin.hbase.cluster.fs=/hbase
kylin.storage.url=hdfs://nameservice1/kylin
kylin.metadata.url=http://kylin-rest-host:7070/kylin/api/admin
```

- 在metadata数据库下创建CUBE表：Kylin要求每个Cube都要对应一个独立的元数据表，包含Cube相关的信息。
```sql
CREATE TABLE `DEFAULT`.KYLIN_CUBE
(
  `UUID` varchar(50) NOT NULL,
  `NAME` varchar(50) NOT NULL,
  `DESCRIPTION` varchar(1000),
  `CATEGORY` varchar(50),
  `CREATION_TIME` datetime DEFAULT CURRENT_TIMESTAMP COMMENT '',
  `LAST_MODIFIED` datetime DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '',
  `OWNER` varchar(50) DEFAULT '' COMMENT '',
  PRIMARY KEY (`UUID`),
  UNIQUE KEY `UK_NAME` (`NAME`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin;

CREATE TABLE `DEFAULT`.KYLIN_CUBE_DESC
(
  `UUID` varchar(50) NOT NULL,
  `CUBE_NAME` varchar(255) NOT NULL,
  `COLUMNS` text NOT NULL,
  `AGGREGATIONS` text NOT NULL,
  `FILTERS` text,
  `DIMENSIONS` text NOT NULL,
  `MEASURES` text NOT NULL,
  `MODEL_CAPACITY` int(11) DEFAULT NULL,
  PRIMARY KEY (`UUID`,`CUBE_NAME`),
  CONSTRAINT `FK_KYLIN_CUBE_DESC_TO_CUBE` FOREIGN KEY (`UUID`, `CUBE_NAME`) REFERENCES `DEFAULT`.`KYLIN_CUBE` (`UUID`, `NAME`) ON DELETE NO ACTION ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8 COLLATE=utf8_bin;
```
- 在元数据数据库中创建CUBE的定义。由于我们创建的是广告效果的Cube，名称为"advertiser_performance_cube",我们定义如下：

```sql
INSERT INTO kylin_cube(`uuid`, `name`, `description`, `category`, `owner`) VALUES ('d5d0c02b-cfdd-489a-b4b2-0e4fb4ab5db1', 'advertiser_performance_cube', 'This is a Cube for Advertiser Performance Analysis.', 'default', '');
```

- 根据我们的维度和指标定义，修改CUBE的定义。我们选择如下维度：AdGroup、Campaign、DeviceType、PositionCityId、PositionProvinceId、Gender、AgeBracket、InterestCategory，以及Clicks、Impressions、Conversions、Cost、AverageCpc、TotalConversions、ViewableImpressions、VisibleAverageDuration等指标。修改CUBE的定义为：

```sql
UPDATE kylin_cube SET description='This is a Cube for Advertiser Performance Analysis.' WHERE name = 'advertiser_performance_cube';

INSERT INTO kylin_cube_desc (`uuid`, `cube_name`, `columns`, `aggregations`, `dimensions`, `measures`) VALUES 
  ('d5d0c02b-cfdd-489a-b4b2-0e4fb4ab5db1', 'advertiser_performance_cube', 'AdGroup|Campaign|DeviceType|PositionCityId|PositionProvinceId|Gender|AgeBracket|InterestCategory|Clicks|Impressions|Conversions|Cost|AverageCpc|TotalConversions|ViewableImpressions|VisibleAverageDuration', 
   "SUM_ALL,'COUNT_DISTINCT','COUNT','SUM','MIN','MAX'",
   'AdGroup|Campaign|DeviceType|PositionCityId|PositionProvinceId|Gender|AgeBracket|InterestCategory',
   "'SumOfClicks','SumOfImpressions','AvgOfCost','SumOfConversions','AvgOfAverageCpc','SumOfTotalConversions','SumOfViewableImpressions','AvgOfVisibleAverageDuration'");
```

- 配置该CUBE为可见。此处暂略。

- 启用该CUBE。由于我们只是测试，所以让这个CUBE是自动启用状态即可。

- 检查CUBE是否已经完全建立完毕。检查方式有两种：
1. 通过web界面查看CUBE页面：通过浏览器打开http://host:7070/kylin/cubes?project=default查看Cube列表，找到“advertiser_performance_cube”，点击进入该Cube的页面。在该页面可以看到该Cube的状态、维度、模型大小、切片状态、最近一次build情况等信息。如果出现问题，可以通过查看日志解决；
2. 通过查看元数据数据库：登录元数据数据库，查询kylin_segment_table表和kylin_segments表。kylin_segment_table中的SEGMENTS列包含该CUBE的所有segment信息，kylin_segments表中的DATA_SOURCE_NAME列包含该CUBE的物理模型路径。如果出现问题，可以通过查看日志和Cube定义文件解决。

## 3.5 查询数据
由于Apache Kylin自带的SQL支持，所以查询数据非常容易。这里演示如何查询广告效果最差的几个广告组：

```sql
SELECT AdGroup FROM advertiser_performance_cube 
WHERE CONVERSIONS > 0 AND Impressions < (AVG(Impressions)*1.2);
```

这个查询的含义是查找所有满足条件的广告组，条件是Conversions大于0，Impressions比平均值1.2倍小。得到的结果是广告效果最差的几个广告组。