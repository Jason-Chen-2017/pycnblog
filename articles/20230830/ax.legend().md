
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## ax.legend()是Matplotlib中的图例函数，可以用来自动生成图例。
## Matplotlib库是一个功能强大的Python绘图库，它提供了大量用于绘制各种类型的图表和图形的API接口。其中ax.legend()就是用于自动生成图例的函数，通过对图表元素进行分类并在图上标注出名称，能够极大地增强图表的可读性、易用性。
## 本文主要介绍如何使用ax.legend()函数创建图例，并提供一些实例展示如何自定义图例样式。
# 2.基本概念和术语
## Axes（坐标轴）：在Matplotlib中，我们经常将一个图表看作由多个轴组成，其中有两个重要的轴对象分别为x轴和y轴。每一个轴都有一个坐标轴(axis)和标签(label)。一般情况下，我们会将y轴作为纵坐标，而x轴作为横坐标，但是也可以随意设置。
## Legend（图例）：在Matplotlib中，图例(Legend)指的是在绘图时为不同的数据系列添加注释的元素。在Axess和Figures中，图例分为三种类型：图形、文本和其他类型。
- 图形图例(Graphic legend)：图形图例是在图中显示颜色条或标记符号，并标注数据系列的名称。图形图例可以帮助快速识别图表中所含数据的类型，便于了解各个数据之间的关系和联系。
- 文本图例(Textual legend)：文本图例是在图中显示文字标签，用来表示数据系列的名称或者数值。文本图例通常不仅仅显示图表中的数据信息，还可以添加额外的信息，如阈值范围、单位等。
- 其他类型图例(Other types of legends)：除了上述两种图例之外，还有第三种类型图例——概要图例(Overview legend)，它在整个图中位置固定，显示整体的整体情况，可能包括图表的主题信息，以及图例说明等。
## Line2D（折线）：折线(Line2D)是最常用的图形类别之一，它代表着一条连接起始点和终止点的曲线。在Matplotlib中，折线可以用来绘制线形图、散点图等。
# 3.核心算法原理和操作步骤
## 创建图例
### 方法1：直接使用matplotlib.pyplot中的函数创建图例
```python
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [2, 4, 1])
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles, labels)
plt.show()
```
上述代码中，首先使用`plt.plot()`函数绘制了一个线性回归的图。接下来，使用`plt.gca()`获取当前轴(Axes object)的句柄和标签列表。然后调用`plt.legend()`函数创建图例，并传入句柄和标签列表作为参数。最后使用`plt.show()`函数显示图表。

### 方法2：通过设置loc参数指定图例位置
当我们调用`plt.legend()`函数时，可以通过设置loc参数指定图例的位置。默认情况下，图例处于"best"位置，即在最合适的位置出现。如果设置了某些选项，比如ncol参数指定每行放置多少个图例，那么图例也会自动调整。例如：
```python
import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0., 5., 0.2)
s1 = np.sin(2*np.pi*t)
s2 = np.exp(-t)
s3 = s1 * s2

fig, ax = plt.subplots()

ax.plot(t, s1, label='Sine')
ax.plot(t, s2, '--', color='orange', label='Exponential')
ax.plot(t, s3, ':', linewidth=2, label='Product')

ax.set(xlabel='time (s)', ylabel='voltage (mV)')
ax.grid()

leg = ax.legend(loc='upper right', shadow=True)
leg.get_frame().set_facecolor('white')

plt.show()
```
上述代码绘制了三条曲线，并在后面使用`plt.legend()`函数创建了图例。通过设置loc参数，图例被定位在右上角，shadow参数使得阴影效果生效。

### 方法3：设置fancybox参数来美化图例
在上面的例子中，图例是默认黑色的方框，比较单一。通过设置fancybox参数为真，可以让图例变成圆角矩形，并且边框颜色和背景色可以根据rcParams配置进行自定义。例如：
```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'Arial' # 设置字体
rcParams['xtick.direction'] = 'in' # 设置X轴刻度方向
rcParams['ytick.direction'] = 'out' # 设置Y轴刻度方向

t = np.arange(0., 5., 0.2)
s1 = np.sin(2*np.pi*t)
s2 = np.exp(-t)
s3 = s1 * s2

fig, ax = plt.subplots()

ax.plot(t, s1, label='Sine')
ax.plot(t, s2, '--', color='orange', label='Exponential')
ax.plot(t, s3, ':', linewidth=2, label='Product')

ax.set(xlabel='time (s)', ylabel='voltage (mV)')
ax.grid()

leg = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

plt.show()
```
上述代码中的rcParams字典可以修改全局的Matplotlib参数，比如字体、线宽等。另外，通过设置bbox_to_anchor参数和borderaxespad参数可以让图例位于图表的右上角，并防止边界重叠。