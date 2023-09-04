
作者：禅与计算机程序设计艺术                    

# 1.简介
  

地理信息系统(GIS)是一个基于地理坐标进行空间描述的系统。在GIS中，人们可以利用计算机对空间特征进行数字化、整理分析，并且把相关的空间数据可视化。地理位置数据是指关于地理空间中事物位置的信息。比如，中国各省市区边界、建筑工程位置、道路等；海岸线、河流等水域边界线；森林覆盖范围、山脉等地形图；主要矿产资源分布区域、采矿业企业位置等。地理位置数据可用于精确测绘地理空间分布规律，从而支持许多GIS应用，如地理信息系统开发、农业监测、旅游行程跟踪、生态环境保护、景观环境评价、地质灾害应对、公共安全管理等。
对于地理位置数据的处理、分析与可视化，目前最流行的数据处理工具就是开源数据科学计算库NumPy以及其数据可视化工具matplotlib。但是，当我们需要处理复杂的地理位置数据时，这些工具就显得力不从心了。这时，我们需要借助第三方的地理位置数据处理工具，比如开源数据可视化库GeoPandas。GeoPandas是基于pandas构建的，专注于地理位置数据处理和可视化。它提供了易于使用的接口和方法，可帮助我们轻松处理、分析和可视化地理位置数据。本教程将会以介绍如何安装、加载和可视化GeoPandas中的地理位置数据为例，演示GeoPandas的强大功能。
# 2.基本概念术语说明
## 2.1 GeoPandas简介
GeoPandas（Geographic Pandas）是一个开源的项目，专注于对GIS中地理位置数据的处理和可视化。它基于Pandas数据分析库，提供了高级地理空间数据结构和操作，并利用matplotlib构建高效的可视化工具。GeoPandas当前最新版本为0.7.0。
GeoPandas提供了两种主要的数据类型：
1. GeoSeries：封装单个地理区域的属性数据，包括点、线、面等几何对象和其他属性。
2. GeoDataFrame：封装多个地理区域的属性数据，既可以包含Point类型的元素，也可以包含Polygon、MultiLineString、MultiPoint、MultiPolygon类型的元素。
GeoSeries和GeoDataFrame是GeoPandas中两个最重要的抽象数据类型，也是学习和理解GeoPandas的关键。在实际应用中，我们往往用不同的抽象层次对地理位置数据进行组织和分类。比如，我们可以将不同地区的出租车收费站点聚集到一个GeoSeries中，然后合并到一个GeoDataFrame中，就可以方便地对出租车收费站点进行数据统计、分析和可视化。同样的，我们也可以将若干个城市或国家的矿产资源分布区域聚集到一个GeoDataFrame中，进行分析、预测、可视化等工作。总之，GeoPandas通过封装不同抽象数据类型，极大的简化了地理位置数据处理和可视化的难度。
## 2.2 抽象数据类型
GeoPandas提供两种抽象数据类型：GeoSeries和GeoDataFrame。
### 2.2.1 GeoSeries
GeoSeries是GeoPandas中最基本的抽象数据类型，可以用来封装单个地理区域的属性数据。GeoSeries主要由两个成员变量组成：index和geometry。
#### index
Index是一个特殊的数组，用于标记每个元素的位置。在GeoSeries中，Index表示的是每一个元素的唯一标识符。如果GeoSeries是根据点、线、面等几何对象进行构造的，那么Index通常是一个整数序列，分别对应着每个元素的编号。如果GeoSeries是根据DataFrame列进行构造的，那么Index则是一个列名序列。
#### geometry
Geometry是GeoPandas中用来封装几何对象的成员变量。GeoSeries可以存放Point、LineString、Polygon等几何对象，以及一些其他属性。
### 2.2.2 GeoDataFrame
GeoDataFrame是GeoPandas中第二个最重要的抽象数据类型，可以用来封装多个地理区域的属性数据。GeoDataFrame由两部分构成：一是表格形式的列，二是由不同类型几何对象组成的列。
#### Column
Column是GeoPandas中最基本的结构单元，可以用来存储和描述地理空间数据的一部分信息。一般来说，Column可以是原始数据类型（数值、字符串、日期等），也可以是几何对象。在GeoDataFrame中，我们可以用列的名称来标记不同的列。
#### Geometry column
Geometry column是GeoDataFrame中用来存放几何对象的列。GeoDataFrame中的每个元素都有一个与之对应的几何对象。Geometry column是一个特殊的Column，其中包含由不同类型几何对象组成的Series。例如，一个GeoDataFrame可以包含多个Point类型的元素，那么它的Geometry column就会包含多个Point对象。
## 2.3 文件格式说明
GeoPandas读取和写入文件有多种格式，包括：ESRI Shapefile、GeoJSON、GPKG、CSV、WKT、GeoPackage、PostGIS数据库。其中Shapefile格式是最通用的一种，后三种格式都是基于Shapefile扩展出来的。本教程将采用GeoJSON格式作为示例。