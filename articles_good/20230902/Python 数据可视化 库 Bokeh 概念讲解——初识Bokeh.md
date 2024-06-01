
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## Bokeh 是什么？
Bokeh是一种开源的交互式Web数据可视化库，它支持多种多样的数据可视化类型，包括线图、柱状图、散点图、气泡图等。它提供强大的交互功能使得用户可以轻松地进行缩放、平移、旋转、显示隐藏数据系列、选择显示范围、添加工具提示和注释等。其用Python语言编写而成，具有简单易用、部署方便、跨平台运行等特性。
## 为什么要学习Bokeh？
作为一个优秀的开源项目，学习Bokeh能够帮助开发者更好地了解其内部工作机制和数据处理流程，提升数据可视化分析能力。在日益复杂的web应用中，有必要掌握各种数据可视化方案以应对不同场景下的需求。与其他数据可视化库相比，如matplotlib、plotly、seaborn等，Bokeh有更多高级的功能和可定制化选项，因此是一种值得考虑的工具。此外，由于Bokeh由社区驱动开发并维护，它的迭代速度也远快于其它工具，并拥有良好的文档和教程资源，通过官方网站、github仓库、StackOverflow等渠道获得帮助和支持也是不错的选择。
# 2.基本概念术语说明
## 画布（figure）、轴（axis）、标记（glyphs）
Bokeh是一个声明性的绘图库，用户无需指定底层的图形渲染方式，只需要将数据绑定到绘图组件上即可完成绘图任务。
在Bokeh中，整个绘图过程被分解为三个主要的部分：画布（Figure），坐标轴（Axis）和标记（Glyphs）。

### 画布（Figure）
画布是用来绘制数据的区域。默认情况下，Bokeh画布大小为800x600像素，可以通过`width`、`height`参数设置大小。用户可以使用Figure的`title`，`xlabel`，`ylabel`，`background_fill_color`，`border_fill_color`属性来自定义画布样式。

```python
from bokeh.plotting import figure, output_file, show

# create a new plot with a title and axis labels
p = figure(title="Time-Series Data", x_axis_label='Time', y_axis_label='Value')

output_file("line.html") # 将绘图保存为html文件
show(p)                   # 在浏览器中打开绘图
```

### 坐标轴（Axis）
坐标轴用来表示数据的维度信息，通常在两个方向上存在，分别对应着数据中的X轴和Y轴。坐标轴的相关属性包括：`axis_line_color`、 `axis_line_width`、 `major_tick_line_color`、 `major_tick_line_width`、`minor_tick_line_color`、 `minor_tick_line_width`、`major_label_text_color`、 `major_label_text_font`、 `ticker`。其中，`ticker`是一个调节刻度值的对象，用于控制坐标轴的尺度、位置和标签。

```python
from bokeh.models import LinearAxis, Range1d

yrange=Range1d(start=-10, end=10)   # 设置Y轴范围
yaxis=LinearAxis(y_range_name="foo", axis_label='Value (units)', major_tick_in=0)    # 创建Y轴实例

# add the y-axis to the plot
p.add_layout(yaxis, 'left')
```

### 标记（Glyphs）
标记即实际用于呈现数据的形状、颜色、大小等效果的图元元素。不同的标记代表了不同的可视化效果，例如散点图、折线图、柱状图、饼状图等。每个标记都有一个`glyph`函数，用来描述其具体外观。

```python
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, Circle, HoverTool

source = ColumnDataSource(data=dict(x=[1, 2, 3], y=[1, 2, 3]))
hover = HoverTool()
hover.tooltips = [("index", "$index"), ("(x,y)", "($x, $y)")]

p = figure(tools=["pan,wheel_zoom," hover])

circle = p.circle('x', 'y', source=source, size=10, color='red')   # 添加圆点图层

output_file("scatter.html") # 将绘图保存为html文件
show(p)                    # 在浏览器中打开绘图
```
# 3.核心算法原理和具体操作步骤以及数学公式讲解
Bokeh的核心是Python语言及其强大的绘图工具包matplotlib。在Bokeh中，绘制任何图表都需要建立一个图层、源数据集和一个glyph function，而这些都是matplotlib所固有的。对于Bokeh来说，这些概念依旧适用。但是，Bokeh添加了许多高级特性，如可交互性，事件处理等。同时，Bokeh还提供了许多新颖的高级可视化效果，如滑动窗口、层次式图例、网格线等。

## 绘制简单的图表
在开始进行数据可视化之前，我们先看一下如何创建最简单的图表，包括直方图、条形图、饼图、散点图、气泡图、折线图等。

### 直方图
直方图是一种用于显示变量分布情况的统计图表，其横轴表示数据的范围，纵轴表示变量的频率。Bokeh提供了`quad`、`hist`和`step`等方法来生成直方图。

#### quad
```python
import numpy as np
from bokeh.layouts import column, row
from bokeh.models import CustomJS, ColumnDataSource
from bokeh.plotting import Figure, output_file, show

N = 500
x = np.random.rand(N)*10 - 5         # 生成随机数据
y = np.random.normal(size=N)+np.linspace(-2, 2, N)      # 生成带有噪声的连续数据

# Create a histogram with bottom edges at each x position
hist, edges = np.histogram(y, density=True, bins=int(N/5))     # 对数据进行直方图计算
bottom_edges = np.insert(edges[:-1], 0, edges[0]-np.diff(edges)/2.)

# Set up data sources for plotting
hist_source = ColumnDataSource({'top': hist, 'right': edges[-1]*np.ones(len(hist)),
                                 'left': edges[:-1], 'bottom': bottom_edges})

quad_source = ColumnDataSource({'x': [], 'y': []})

# Define the callback to update the quads when the histogram selection changes
callback = CustomJS(args={'hist_source': hist_source,
                          'quad_source': quad_source},
                    code="""
    var geometry = cb_obj['geometry'];
    var top = hist_source.data['top'][geometry];
    var right = hist_source.data['right'][geometry];
    var left = hist_source.data['left'][geometry];
    var bottom = hist_source.data['bottom'][geometry];

    // Update the Quad glyphs
    quad_source.data = {'x': [[left, right, right, left]],
                        'y': [[bottom, bottom, top, top]]};
""")

# Set up the layout and plots
fig = Figure(match_aspect=False, tools='')
fig.toolbar.logo = None             # Remove the Bokeh logo

# Add a histogram plot with transparent background fill color
hist = fig.quad(top='top', right='right', left='left', bottom='bottom',
                line_color='white', alpha=0.7, source=hist_source)

# Add an invisible RectangleSelection tool that updates quads based on selection in the histogram
select = fig.select_one(type=ColumnDataSource)          # Get the first rectangular selection tool on the figure
select.callback = callback                              # Attach the custom JS callback to it

# Draw a line showing the mean of the distribution
mean = np.average(y)
fig.line([mean, mean], [-6, 6], color='#F4A582', line_width=2)
fig.xaxis.axis_label = 'Value'
fig.yaxis.visible = False                                # Hide the Y-axis label and ticks

# Show the results in the notebook
output_file("quad.html")
show(column(row(hist), row(fig)))                         # Arrange the figures in two rows
```

#### hist
```python
import numpy as np
from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, output_file, show

N = 500
x = np.random.rand(N)*10 - 5        # Generate random data
y = np.random.normal(size=N)+np.linspace(-2, 2, N)       # Generate noisy continuous data

hist, edges = np.histogram(y, density=True, bins=int(N/5))           # Calculate the histogram using NumPy functions
bottom_edges = np.insert(edges[:-1], 0, edges[0]-np.diff(edges)/2.)

source = ColumnDataSource({'top': hist, 'right': edges[-1]*np.ones(len(hist)),
                           'left': edges[:-1], 'bottom': bottom_edges})

TOOLS = ''            # No toolbar or tools displayed

# Initialize the figure objects with various parameters
p1 = figure(title='Histogram', tools=[], match_aspect=True)
p1.quad(top='top', bottom='bottom', left='left', right='right',
        line_color='white', alpha=0.7, source=source)

p2 = figure(title='Density Curve', tools=['save'], height=300, width=300,
            x_range=p1.x_range)
p2.line(x, np.exp(-(x-np.mean(x))**2/(2*np.var(x))), color="#F4A582", line_width=2)


# Put the two plots into a grid layout
grid = gridplot([[p1, None], [None, p2]])

# Display the results
output_file("hist.html")
show(grid)
```

#### step
```python
import numpy as np
from bokeh.io import curdoc
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, output_file, show

N = 500
x = np.sort(np.random.rand(N)*10 - 5)               # Sort randomly generated data
y = np.random.normal(size=N)+np.linspace(-2, 2, N)      # Generate noisy continuous data

source = ColumnDataSource({'x': x, 'y': y})

p = figure(title='Step Chart', tools='', match_aspect=True)
p.step('x', 'y', mode='after', color="#F4A582", line_width=2,
       source=source)

curdoc().add_root(p)                               # Add the root object to the Document
curdoc().title = "Step"                            # Set the window title

output_file("step.html")                           # Save the HTML file
show(p)                                             # Open the plot in the default browser
```

### 条形图
条形图（bar chart）是一种非常常见的图表类型，其横轴表示分类或因子变量的值，纵轴表示量变量的值或数量。Bokeh提供了`vbar`和`hbar`方法来生成条形图。

#### vbar
```python
import numpy as np
from bokeh.layouts import column, row
from bokeh.models import CustomJS, ColumnDataSource
from bokeh.plotting import Figure, output_file, show

N = 50
fruits = ['Apples', 'Pears', 'Nectarines', 'Plums', 'Grapes']
counts = np.random.randint(10, 30, size=N)        # Random integer counts between 10 and 30

source = ColumnDataSource({'fruits': fruits, 'counts': counts})

# Set up the plot and style properties
p = figure(x_range=fruits, sizing_mode='stretch_both', toolbar_location=None)
p.vbar(x='fruits', top='counts', width=0.9, source=source,
       line_color='white', alpha=0.7)

p.xgrid.grid_line_color = None
p.y_range.start = 0
p.x_range.range_padding = 0.1
p.xaxis.major_label_orientation = np.pi/2

# Set up callbacks for selecting individual bars and updating totals
code = """
const indices = selected['1d'].indices;
if (indices.length === 0) {
  total_counts.text = '';
} else {
  const selected_counts = [];
  for (let i = 0; i < indices.length; i++) {
    selected_counts.push(counts.data[indices[i]]);
  }
  total_counts.text = selected_counts.reduce((a, b) => a + b);
}
"""

total_counts = p.text([], [], text=[], text_font_size='13px',
                     text_align='center', text_baseline='middle',
                     y_offset=0, x_offset=0)

selected = p.select_one(type=CustomJS)
selected.args = dict(counts=source.data['counts'], total_counts=total_counts)
selected.code = code

# Plot the results in multiple columns and rows
output_file("vbar.html")
show(column(p))
```

#### hbar
```python
import pandas as pd
from bokeh.charts import Bar, output_file, show

df = pd.DataFrame({'apples': [3, 2, 1, 4, 5],
                   'bananas': [1, 4, 5, 6, 7]})

bar = Bar(df, cat=['apples', 'bananas'])
bar.title('Fruit Counts').xlabel('')              # Remove the X-axis label
bar.legend.orientation = 'horizontal'                # Rotate the legend to be horizontal instead of vertical

output_file("hbar.html")                             # Save the HTML file
show(bar)                                           # Open the plot in the default browser
```

### 饼图
饼图（pie chart）是一种常见的面积占比图表，用于表示某个特定事物中各个部分的相对大小。在Bokeh中，可以通过`wedge`方法来生成饼图。

#### wedge
```python
import math
import numpy as np
from bokeh.models import ColumnDataSource
from bokeh.palettes import Spectral5
from bokeh.plotting import figure, output_file, show

N = 4                        # Number of slices to draw
theta = np.random.uniform(0, 2*math.pi, N)   # Random angles for the starts of slices
colors = Spectral5[:N]                     # Choose some colors from the spectrum

# Use polar coordinates to arrange the wedges around the circle
r = np.array([3+np.random.randn(N)]).flatten()
theta += np.array([-math.pi/2, 0, 0, 0])[::-1][:N]

source = ColumnDataSource({'r': r, 'theta': theta, 'colors': colors})

p = figure(width=400, height=400, toolbar_location=None,
           tooltips="@colors: @percent{0.00%} (@value)")

p.wedge(x=0, y=0, radius='r', start_angle='theta', end_angle='theta',
        color='colors', alpha=0.7, source=source)

p.axis.major_tick_line_color = None
p.axis.minor_tick_line_color = None
p.outline_line_color = None

output_file("wedge.html")
show(p)
```

### 散点图
散点图（scatter plot）是一种二维图表，其中每一个点都表示两个变量之间的关系。Bokeh提供了`scatter`方法来生成散点图。

#### scatter
```python
import numpy as np
from bokeh.plotting import figure, output_file, show

N = 100
x = np.random.standard_normal(N)
y = np.random.standard_normal(N)
radii = np.random.random(N) * 0.1 + 0.1  # Radius values between 0.1 and 0.2

p = figure(tools="", background_fill_color="#fafafa")
p.scatter(x, y, radius=radii,
          fill_alpha=0.6, fill_color="navy", line_color="black", line_alpha=0.5)

output_file("scatter.html")
show(p)
```

### 气泡图
气泡图（bubble chart）是一种三维图表，其中每一个点都可以表示三个变量之间的关系，包括横坐标（x）、纵坐标（y）、大小（radius）。Bokeh提供了`scatter`方法来生成气泡图。

#### bubble
```python
import numpy as np
from bokeh.plotting import figure, output_file, show

N = 100
x = np.random.standard_normal(N)
y = np.random.standard_normal(N)
sizes = np.random.random(N) * 10 + 10  # Sizes values between 10 and 20

p = figure(tools="", background_fill_color="#fafafa")
p.scatter(x, y, radius=sizes, fill_alpha=0.6, fill_color="navy", line_color="black", line_alpha=0.5)

output_file("bubble.html")
show(p)
```

### 折线图
折线图（line chart）是一种常见的图表类型，其中纵轴表示某些变量随着时间或其他维度变化的趋势。Bokeh提供了`line`方法来生成折线图。

#### line
```python
import numpy as np
from bokeh.plotting import figure, output_file, show

N = 100
x = np.arange(N)
y = np.random.random(N) * 10 + np.sin(x/10)

p = figure(tools="")
p.line(x, y, line_width=2, color="navy", alpha=0.7)

output_file("line.html")
show(p)
```