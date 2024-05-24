
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Matplotlib是一个Python绘图库，它具有很强大的2D绘图功能。它的底层依赖于其他第三方绘图库，比如Qt或WxWidgets，但是为了更加方便的控制和配置，Matplotlib封装了这些底层库并进行了一系列优化。因此，Matplotlib能够支持各种高级3D图形、动画、Web输出等功能。
Matplotlib可以用于各种领域的绘图需求，包括科学研究、工程应用、数据可视化、机器学习等领域。Matplotlib提供了简单易用、丰富的图表类型和样式选项，并且能够生成多种不同形式的图片文件。本文介绍Matplotlib的基础知识，包括安装及基本操作。
# 2.安装Matplotlib
Matplotlib可以通过pip命令安装：
```python
pip install matplotlib
```
或者从源代码安装：
```python
git clone https://github.com/matplotlib/matplotlib.git
cd matplotlib
sudo python setup.py install # 安装系统所有用户
```
# 3.基本概念术语说明
## 3.1 Figure 和 Axes
Matplotlib 中最重要的对象是Figure和Axes。一个Figure可以理解成一张图，一个Axes对象就是在这个图中的一个坐标系，你可以在不同的坐标系中添加多个子图。

## 3.2 轴线、刻度、标签和文本
当你创建Figure对象时，会默认创建一个Axes对象。每一个Axes对象都有x轴（或纵轴）、y轴（或横轴），还有一个色条，用来显示颜色映射值。

轴线（Axis Line）: x轴和y轴上的直线；

刻度（Ticks）：每个坐标轴上刻度的值，通常由数字或字符表示；

标签（Labels）：坐标轴的名称，通常位于坐标轴两侧；

标题（Title）：一般位于图框之上，用来描述图表的内容。

坐标文本（Coordinate Text）：在坐标轴上显示的值，通常通过matplotlib函数autolabel()自动设置。

## 3.3 Line、Scatter、Bar、Histogram等图表对象
Matplotlib提供了多种图表对象，比如Line、Scatter、Bar、Histogram等等，用来绘制各种类型的图表。这里重点介绍一下Line。

### 3.3.1 Line图
Line图是最常用的一种图表。它可以用来绘制一系列数据的点之间的连接线。Line图常见的属性有颜色、线宽、样式、透明度等。如下面这个例子所示：

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.1)   # x轴刻度间隔
y = np.sin(x)             # y值计算

plt.plot(x, y)            # 创建Line图

plt.xlabel('X')           # 设置X轴标签
plt.ylabel('Y')           # 设置Y轴标签
plt.title('Sine Wave')    # 设置标题
plt.show()                # 显示图形
```

结果如图所示：

### 3.3.2 Scatter图
Scatter图也叫散点图，主要用来展示两种变量之间关系的图表。它可以用来展示大量的数据点，但数据点不能连成曲线。Scatter图常见的属性有颜色、大小、透明度等。如下面这个例子所示：

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)          # 设置随机数种子

x = np.random.rand(50)     # 生成50个随机数作为x轴数据
y = np.random.rand(50)     # 生成50个随机数作为y轴数据

plt.scatter(x, y, c='red', alpha=0.5)      # 创建Scatter图

plt.xlabel('X')                           # 设置X轴标签
plt.ylabel('Y')                           # 设置Y轴标签
plt.title('Random Points')               # 设置标题
plt.show()                                # 显示图形
```

结果如图所示：

### 3.3.3 Bar图
Bar图又称柱状图，用来展示一组数据的分布情况。它可以用来展示一段时间内的不同项目的数量或值。Bar图常见的属性有颜色、宽度、透明度等。如下面这个例子所示：

```python
import numpy as np
import matplotlib.pyplot as plt

n_groups = 5                      # 分类数量
means_men = (20, 35, 30, 35, 27)   # 各分组平均值
std_men = (2, 3, 4, 1, 2)         # 各分组标准差

fig, ax = plt.subplots()

index = np.arange(n_groups)       # X轴刻度索引
bar_width = 0.35                 # 柱子宽度

opacity = 0.4                    # 不透明度
error_config = {'ecolor': '0.3'}  # 误差线配置项

rects1 = ax.bar(index, means_men, bar_width,
                alpha=opacity, color='b',
                yerr=std_men, error_kw=error_config, label='Men')

plt.xlabel('Group')              # 设置X轴标签
plt.ylabel('Scores')             # 设置Y轴标签
plt.title('Scores by group and gender')        # 设置标题
plt.xticks(index + bar_width / 2, ('A', 'B', 'C', 'D', 'E'))  # 设置X轴刻度标签
plt.legend()                     # 添加图例
plt.tight_layout()               # 紧凑布局
plt.show()                       # 显示图形
```

结果如图所示：

### 3.3.4 Histogram图
Histogram图也叫频率分布图，主要用来显示一组数据的分布情况。它主要用于展示数据中某个连续变量的概率密度分布。Histogram图常见的属性有颜色、柱子宽度、边缘颜色等。如下面这个例子所示：

```python
import numpy as np
import matplotlib.pyplot as plt

mu, sigma = 100, 15                          # 分布均值和标准差
x = mu + sigma * np.random.randn(10000)       # 生成符合正态分布的随机数

n, bins, patches = plt.hist(x, 50, density=True, facecolor='g', alpha=0.75)   # 创建直方图

plt.xlabel('Smarts')                     # 设置X轴标签
plt.ylabel('Probability')                # 设置Y轴标签
plt.title('Histogram of IQ')             # 设置标题
plt.text(60,.025, r'$\mu=100,\ \sigma=15$')   # 添加文字注释
plt.axis([40, 160, 0, 0.03])                   # 设置坐标轴范围
plt.grid(True)                              # 添加网格
plt.show()                                 # 显示图形
```

结果如图所示：