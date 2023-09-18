
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Matplotlib是基于Python的2D绘图库，提供了一系列功能强大的用于制作各种图表、图片、散点图、线图等的函数。它具有跨平台的性质，可以运行于Windows、Linux、Mac OS X等多种操作系统中，并兼容各种Python版本（2.7及以上）和多种绘图后端，包括Tkinter、GTK+、wxPython、Qt4/5、cairo-based backends和SVG输出。Matplotlib的目标就是创建出一个简单易用的python环境下的2D绘图工具箱。

## Matplotlib安装
Matplotlib支持Python2.x和Python3.x版本，依赖numpy。建议安装最新版本的Anaconda或Miniconda。如果已安装，可通过如下命令进行安装：

``` python
pip install matplotlib 
```

或者：

``` python
conda install matplotlib
```

## Matplotlib基本概念
### Figure和Axes
Matplotlib中主要有两种对象：Figure和Axes。

- Figure：在Matplotlib中，Figure是一个容器对象，它负责容纳Axes、Axis、LineCollection等子对象，并通过管理者locator坐标轴、颜色映射和子窗口。一个Figure可以包含多个Axes。
- Axes：在Matplotlib中，Axes是一个坐标系，通常用来绘制图像。在一个Figure里，Axes默认情况下共享刻度(tick)、标签(label)，但也可以设置自己的刻度和标签。每个Axes包含两条轴(Axis)：X轴和Y轴。一个Axes可以包含多个坐标系。


### Plotting函数
Matplotlib中的最基本绘图函数是plot()函数。它可以用于绘制单个曲线或折线图，还可以用不同的标记方式表示数据点，如圆点、方块、星号等。该函数允许传入多个参数，包括X轴值和Y轴值数组，以及可选的参数如线宽、颜色、标记类型等。 

下面的例子展示了如何使用plot()函数绘制一条线图：

``` python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
y_sin = np.sin(x)
y_cos = np.cos(x)

plt.figure(figsize=(8,4)) # 设置绘图区域大小

plt.subplot(1, 2, 1)   # 分成1行2列，第一个子图
plt.plot(x, y_sin)     # 绘制正弦曲线
plt.title("Sine")      # 设置子图标题
plt.xlabel("$x$")       # 设置X轴标签
plt.ylabel("$y$")       # 设置Y轴标签

plt.subplot(1, 2, 2)   # 分成1行2列，第二个子图
plt.plot(x, y_cos)     # 绘制余弦曲线
plt.title("Cosine")    # 设置子图标题
plt.xlabel("$x$")       # 设置X轴标签
plt.ylabel("$y$")       # 设置Y轴标签

plt.tight_layout()     # 自动调整子图间距
plt.show()             # 显示图形
```


### Bar chart函数
Matplotlib提供了一个bar()函数用于生成条形图。该函数接收两个参数，一个是X轴的值列表，另一个是Y轴的值列表，并将X轴上的值映射到Y轴上的值上。下面的例子展示了如何使用bar()函数生成条形图：

``` python
import numpy as np
import matplotlib.pyplot as plt

N = 5
menMeans = (20, 35, 30, 35, 27)
womenMeans = (25, 32, 34, 20, 25)

ind = np.arange(N)    # the x locations for the groups
width = 0.35         # the width of the bars

fig, ax = plt.subplots()

rects1 = ax.bar(ind - width/2, menMeans, width, label='Men')
rects2 = ax.bar(ind + width/2, womenMeans, width, label='Women')

ax.set_ylabel('Scores')
ax.set_xticks(ind)
ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))
ax.legend()

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

plt.show()
```


### Subplots函数
Matplotlib的subplots()函数可以创建多幅子图。该函数接收三个参数，分别是行数、列数和位置索引。返回值是一个元组，包含每张子图的Axes对象。下面的例子展示了如何使用subplots()函数创建两幅子图：

``` python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

fig, axes = plt.subplots(nrows=1, ncols=2)

axes[0].plot(x, y1)
axes[0].set_title('Sine Wave')
axes[0].set_xlabel('$x$')
axes[0].set_ylabel('$y$')

axes[1].plot(x, y2)
axes[1].set_title('Cosine Wave')
axes[1].set_xlabel('$x$')
axes[1].set_ylabel('$y$')

plt.show()
```
