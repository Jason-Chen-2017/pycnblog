
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Plotly和Bokeh都是Python中用于制作数据可视化的流行库，在本文中我们将对其进行比较分析并讨论它们之间的区别以及它们在可视化方面的优缺点。同时，通过实际案例演示如何使用这两个库制作出具有交互性的数据可视化。
# 2.Python绘图库的比较
## 2.1 Plotly
Plotly是一个基于Web的开源库，它可以轻松地创建交互式的复杂数据可视化。它的功能包括：
- 创建丰富的图表类型：线图、散点图、柱状图、堆积图、盒须图等
- 直观易懂的主题和配色方案
- 支持多种编程语言：包括Python、R、JavaScript、Java等
- 可自定义图标的属性
- 提供交互功能：支持缩放、平移、筛选、拖动、缩放标记、点击高亮、hover提示、下载图片等
- 可以嵌入到网页中
- 数据导出功能

## 2.2 Bokeh
Bokeh是一个纯Python的交互式可视化库，它可以用于快速创建交互式的复杂数据可视化。它的功能包括：
- 使用PyData stack开发：使用pandas、NumPy、Matplotlib等构建的数据处理工具栈
- 强大的渲染引擎：Bokeh能够使用GPU和矢量化渲染，使得可视化图形更快捷
- 灵活的可视化组件：包括线、面、圆圈、矩形、条形、箱型图、饼图、蜘蛛图、气泡图、热力图、文本、颜色渐变等
- 高度可定制化的布局：允许调整图形大小、位置、顺序及透明度
- 支持JavaScript回调函数：可以编写JS代码，用以响应用户事件
- 数据驱动：可以使用NumPy或Pandas中的数据进行可视化的生成
- 更方便的高级图形：可以将多个图形层叠在一起，实现复杂的可视化效果

## 2.3 Plotly VS Bokeh
虽然两者都很优秀，但二者之间也存在一些差异。下表列举了Plotly和Bokeh之间的一些区别：

|         | Plotly                    | Bokeh           |
|---------|---------------------------|-----------------|
| 目标领域 | 数据可视化                | 交互式可视化    |
| 语言     | Python                    | Python          |
| API      | 直观的高级API             | 简洁的低级API   |
| 数据输入 | pandas DataFrame          | NumPy数组       |
| 定制能力 | 比较强                      | 不如Plotly强     |
| 渲染性能 | GPU渲染 + WebGL加速        | CPU渲染 + WebGL |
| JS回调   | 支持                       | 不支持          |
| 图像输出 | PNG/SVG/JPEG/HTML formats | HTML Canvas     |

# 3.具体案例：创建社区关系网络可视化
我们将使用两个库——Plotly和Bokeh，分别展示如何使用这两个库创建一个社区关系网络可视化。社区关系网络（CRN）是一个网络，其中节点代表个体或实体，边代表这些个体间的关系，通常包括友谊、合作、仇恨、竞争、信任等。CRN可用于研究不同群体之间的关系、社会运动、组织结构、影响力流通等。
## 3.1 Plotly版本
首先，我们用Plotly库来实现一个简单的社区关系网络可视化。假设有一个社区里有三个参与者，他们彼此之间形成了一个三角形社区结构。为了模拟这个场景，我们可以建立一个包含这三个节点和四条边的图对象。

```python
import plotly.graph_objects as go

nodes = ["A", "B", "C"]
edges = [("A", "B"), ("B", "C"), ("A", "C")]

fig = go.Figure()

for node in nodes:
    fig.add_trace(go.Scatter(x=[node], y=[0], mode="markers+text", text=[node]))
    
for edge in edges:
    x0, y0 = nodes.index(edge[0]), 1
    x1, y1 = nodes.index(edge[1]), 1
    
    if (y0 - y1) > 0:
        xm, ym = ((x0 + x1)/2., y0 + abs(y1 - y0))
        side = "left"
    else:
        xm, ym = ((x0 + x1)/2., y1 + abs(y0 - y1))
        side = "right"
        
    fig.add_annotation(x=xm, y=ym, ax=x0-0.5, ay=-abs(y1-y0), xref="x", yref="y",
                       showarrow=True, arrowhead=side, arrowwidth=1, arrowcolor="#ccc")
```

上述代码创建了一个空白的图对象`fig`，然后通过循环添加三个节点，每个节点的坐标都是[0, 0]。然后，再通过另一个循环添加四条边，边的起点和终点是通过遍历`nodes`列表找到的索引值。要画一条连接两个节点的直线，我们需要确定一条水平垂直于这两点的直线，然后找出与这条线最接近的一点。如果该点的横坐标值小于等于中间节点的横坐标值，那么它就在左侧；反之则在右侧。

接着，我们用`fig.update()`方法更新图形属性。例如，设置画布的宽度和高度：

```python
fig.update_layout(width=700, height=700)
```

最后，我们调用`fig.show()`方法显示最终的结果。如下所示：


## 3.2 Bokeh版本
下面，我们用Bokeh库来实现同样的一个社区关系网络可视化。相比之下，Bokeh的代码会更简单、直接一些，因为它提供了一些预定义好的图形元素。比如，可以用`Circle`图元来表示节点，用`MultiLine`图元来表示边。但是，由于Bokeh的高级定制能力不如Plotly，所以我们可能需要更多的代码来实现想要的效果。

```python
from bokeh.models import ColumnDataSource, Circle, MultiLine

nodes = ["A", "B", "C"]
edges = [(0, 1), (1, 2), (0, 2)]

xs, ys = [], []
for i in range(len(nodes)):
    xs.append([0])
    ys.append([i*2])
    
sources = {"x": [[] for _ in range(len(nodes))],
           "y": [[] for _ in range(len(nodes))]
          }

for source, pos in zip(["A", "B", "C"], [[0]*2, [-2,-1], [-1,1]]):
    sources["x"][nodes.index(source)], sources["y"][nodes.index(source)].extend([[pos[0]], [pos[1]]])
        
for a, b in edges:
    line_data = dict(xs=[], ys=[])
    line_data["xs"].append((sources["x"][a][0], sources["x"][b][0]))
    line_data["ys"].append((sources["y"][a][0], sources["y"][b][0]))

    sources["x"][a].append([])
    sources["x"][b].append([])
    sources["y"][a].append([])
    sources["y"][b].append([])
    
data = {k: v for k, v in sources.items()}

circles = {}
lines = {}

for n in range(len(nodes)):
    circles[n] = Circle(x="x", y="y", size=10, fill_color="blue", name=f"{nodes[n]}_{n}")
    lines[n] = MultiLine(xs="xs", ys="ys", line_alpha=0.5, line_width=1.5, 
                         line_color=["green", "red", "orange"][n%3], name=f"line{n}")

source = ColumnDataSource(data)

p = figure(tools="", width=700, height=700)
p.add_glyph(source, circles[0])
p.add_glyph(source, circles[1])
p.add_glyph(source, circles[2])

for n in range(len(nodes)):
    p.add_glyph(source, lines[n])
    taptool = p.select(type=TapTool)
    callback = CustomJS(args=dict(s=source, c=circles[n]), code="""
            var data = s.data;
            var index = cb_obj.selected['1d'].indices[0];

            // highlight the circle
            for (var i = 0; i < data['x'].length; i++) {
                data['fill_color'][i] = 'white';
                s.change.emit();
            }
            data['fill_color'][index] = '#FF5E5E';
            s.change.emit();

            console.log(`Selected ${c.name}`);
        """)
    taptool.callback = callback 

show(p)
```

先前的代码中，我们已经提取出一些共用的代码，创建了一个节点数据源和一条边数据的字典，里面包含每个节点的X轴坐标、Y轴坐标、多条边上的X轴坐标、多条边上的Y轴坐标。然后，我们用`ColumnDataSource`对象包装这些数据，创建一些带名字的图元，包括圆和多边形。

最后，我们用Bokeh的`CustomJS`机制来绑定一个点击事件处理函数，当用户点击某个节点时，它会改变该节点的颜色，并将其他所有节点的颜色恢复默认状态。这样就可以清晰地看到某个节点被选中时，其他节点的样式发生了变化。