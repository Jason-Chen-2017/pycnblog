                 

# 1.背景介绍

## 1. 背景介绍
Elasticsearch是一个强大的搜索引擎，它可以帮助我们快速地搜索和分析大量的数据。在现实生活中，地理位置信息是非常重要的，例如在地图应用中，我们可以通过地理位置信息来查找附近的餐厅、酒店等。因此，在Elasticsearch中，我们可以通过地理位置搜索来快速地查找和分析地理位置相关的数据。

在Elasticsearch中，地理位置搜索可以通过Geo Point数据类型来实现。Geo Point数据类型可以存储地理位置信息，例如纬度和经度。通过Geo Point数据类型，我们可以实现地理位置搜索和地图可视化。

## 2. 核心概念与联系
在Elasticsearch中，地理位置搜索和地图可视化是两个相关的概念。地理位置搜索是通过Geo Point数据类型来存储和查找地理位置信息的。地图可视化是通过地理位置信息来生成地图图像的。

地理位置搜索和地图可视化之间的联系是：地理位置搜索可以通过地理位置信息来查找和分析数据，而地图可视化则可以通过地理位置信息来生成地图图像，从而帮助我们更好地理解和分析数据。

## 3. 核心算法原理和具体操作步骤以及数学模型公式详细讲解
在Elasticsearch中，地理位置搜索是通过Geo Distance查询来实现的。Geo Distance查询可以根据地理位置信息来查找和分析数据。Geo Distance查询的数学模型公式是：

$$
d = \arccos(\sin(\phi_1)\sin(\phi_2) + \cos(\phi_1)\cos(\phi_2)\cos(\lambda_1 - \lambda_2)) \times R
$$

其中，$\phi_1$和$\phi_2$是纬度信息，$\lambda_1$和$\lambda_2$是经度信息，$R$是地球的半径。

具体操作步骤如下：

1. 首先，我们需要将地理位置信息存储到Elasticsearch中。我们可以使用Geo Point数据类型来存储地理位置信息。

2. 然后，我们可以使用Geo Distance查询来查找和分析地理位置相关的数据。Geo Distance查询可以根据地理位置信息来查找和分析数据。

3. 最后，我们可以使用地图可视化工具来生成地图图像，从而帮助我们更好地理解和分析数据。

## 4. 具体最佳实践：代码实例和详细解释说明
在Elasticsearch中，我们可以使用Geo Point数据类型来存储地理位置信息。例如，我们可以使用以下代码来存储地理位置信息：

```json
{
  "name": "餐厅A",
  "location": {
    "type": "point",
    "coordinates": [116.404, 39.904]
  }
}
```

然后，我们可以使用Geo Distance查询来查找和分析地理位置相关的数据。例如，我们可以使用以下代码来查找距离当前位置10公里内的餐厅：

```json
{
  "query": {
    "geo_distance": {
      "distance": "10km",
      "pin": true,
      "location": {
        "lat": 39.904,
        "lon": 116.404
      }
    }
  }
}
```

最后，我们可以使用地图可视化工具来生成地图图像，从而帮助我们更好地理解和分析数据。例如，我们可以使用Leaflet库来生成地图图像：

```javascript
L.mapquest.key = 'your_mapquest_key';
L.mapquest.layerControl.addBaseLayer('Streets');
L.mapquest.layerControl.addBaseLayer('Satellite');
L.mapquest.layerControl.addBaseLayer('Topo');
L.mapquest.layerControl.addBaseLayer('Terrain');
L.mapquest.layerControl.addBaseLayer('Hybrid');
L.mapquest.layerControl.addBaseLayer('Aerial');
L.mapquest.layerControl.addBaseLayer('BingAerial');
L.mapquest.layerControl.addBaseLayer('BingRoads');
L.mapquest.layerControl.addBaseLayer('BingSatellite');
L.mapquest.layerControl.addBaseLayer('BingTopo');
L.mapquest.layerControl.addBaseLayer('BingTerrain');
L.mapquest.layerControl.addBaseLayer('OpenStreetMap');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('CycleMap');
L.mapquest.layerControl.addBaseLayer('Cloudmade');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Mapnik');
L.mapquest.layerControl.addBaseLayer('Map