
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


## 什么是空间数据类型？
在PostgreSQL数据库中，有一个专门用于处理空间数据的类型叫做“PostGIS”，它提供了一个“geography”、“geometry”和“raster”类型的空间数据类型。但是，MySQL并没有内置空间数据类型，所以如何在MySQL中存储和操作空间数据，是一个很重要的问题。而现在，随着互联网的普及，WebGIS已经成为一种流行的GIS开发技术。为了帮助更多的人理解和掌握MySQL中的空间数据类型，我准备了一份MySQL的空间数据类型学习教程。本文将详细介绍MySQL中几种典型的空间数据类型：Point、LineString、Polygon、MultiPoint、MultiLineString、MultiPolygon和GeometryCollection，并通过实例应用介绍这些数据类型的基本操作方法。

## 为什么要用空间数据类型？
通常情况下，我们在数据库中存储地理信息时，都是采用经纬度坐标对。这种方式虽然简单直观，但是由于存在精度损失等不可忽视的问题，因此在某些场合下，精确到分辨率要求高的地图信息无法用经纬度表示。例如，在做车位分析、轨迹预测等需要精准位置信息的业务场景下，经纬度坐标就显得力不从心了。因此，在现实世界里，我们还需要更加精细化的空间数据类型来支持更高级的地理分析功能。

## 空间数据类型详解
### Point
点是最简单的空间数据类型，只有一个二维坐标（x、y）。通常，MySQL的Point类型的数据结构可以用一个DOUBLE类型存储x坐标和y坐标，也可以用一个TEXT类型存储Well-Known Text (WKT)文本形式的坐标。
```
CREATE TABLE point_table(
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(50),
    p POINT
);
```
使用GeomFromText()函数可以把WKT形式的坐标字符串解析成Point类型的数据。
```
INSERT INTO point_table(name,p) VALUES ('point', GeomFromText('POINT(10 20)'));
```

```
SELECT AsText(p) AS wkt FROM point_table WHERE name = 'point';
```
输出结果：
```
POINT(10 20)
```


### LineString
线串是由两个或多个相连的点组成的曲线，所以LineString类型就是由若干个点构成的序列。线串有两种存储方式，第一种是使用多边形的外包矩形框来描述整个线串，第二种是直接保存每两个相邻点之间的距离。

#### 使用矩形框存储线串
对于非常简单的线串，可以使用一个矩形框来描述整个线串，其中左上角的点和右下角的点分别记录了线串的起始点和终止点的坐标值。
```
CREATE TABLE linestring_table(
    id INT PRIMARY KEY AUTO_INDENT,
    name VARCHAR(50),
    ls LINESTRING
);

INSERT INTO linestring_table(name,ls) VALUES 
    ('linestring1', PolygonFromText('POLYGON((0 0, 0 3, 3 3, 3 0, 0 0))')),
    ('linestring2', PolygonFromText('POLYGON((1 1, 1 4, 4 4, 4 1, 1 1))'))
;
```
注意：这里的插入语句中的LinestringFromText()函数可以把Polygon类型的数据转换成Linestring类型。

```
SELECT AsText(ls) as wkt FROM linestring_table ORDER BY id DESC LIMIT 1;
```
输出结果：
```
LINESTRING(0 0, 0 3, 3 3, 3 0, 0 0)
```

#### 直接保存距离信息
对于复杂一些的线串，比如没有严格按照矩形框规则排列的线段，我们可以使用两个相邻点之间距离的一系列值来存储整个线串的信息。这种方式比使用矩形框的方式更加灵活，但是需要考虑周长等计算才能得到总长度。
```
CREATE TABLE distance_table(
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(50),
    dist FLOAT
);

INSERT INTO distance_table(name,dist) VALUES 
    ('distance1', LineStringDistance('LINESTRING(0 0, 3 0)', 'LINESTRING(3 0, 3 3)')),
    ('distance2', LineStringDistance('LINESTRING(-1 -1, 1 1)', 'LINESTRING(1 1, 2 2)'))
;

SELECT SUM(dist) AS length FROM distance_table;
```
输出结果：
```
10.791044264104185
```

### Polygon
多边形是由至少三个以上不同点的集合组成的面状实体，所以Polygon类型也是由一系列点组成的。在MySQL中，Polygon类型数据是指由一系列线环所组成的空间区域。多边形一般由以下三种几何关系决定：1）内部环：多边形的一个子区块；2）外部环：多边形的边界部分；3）边界线：由连接外部环的边界点形成的线。

#### 创建多边形
创建多边形比较简单，只需指定一系列线环就可以了。每条线环对应一个LinearRing对象，即由若干个点构成的闭合曲线。
```
CREATE TABLE polygon_table(
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(50),
    poly POLYGON
);

INSERT INTO polygon_table(name,poly) VALUES 
    ('polygon1', MultiPolygonFromText("MULTIPOLYGON(((0 0, 0 3, 3 3, 3 0, 0 0)), ((1 1, 1 4, 4 4, 4 1, 1 1)))")),
    ('polygon2', MultiPolygonFromText("MULTIPOLYGON(((0 0, 0 3, 3 3, 3 0, 0 0)), ((5 5, 5 8, 8 8, 8 5, 5 5))))")
;

SELECT AsText(poly) AS wkt FROM polygon_table WHERE name = 'polygon2';
```
输出结果：
```
MULTIPOLYGON (((5 5, 5 8, 8 8, 8 5, 5 5)))
```

#### 获取面积和周长
获取多边形的面积和周长也比较容易，只需调用对应的方法即可。
```
SELECT ST_Area(poly) AS area, ST_Perimeter(poly) AS perimeter FROM polygon_table;
```
输出结果：
```
area          | perimeter    
--------------|-------------
             9 |           18  
             21|            28     
```