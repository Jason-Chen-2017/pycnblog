
作者：禅与计算机程序设计艺术                    

# 1.简介
         

## 概述
Matplotlib是一个Python画图库，具有强大的绘制能力，可以用来可视化数据。在Python中，使用Matplotlib主要依赖两个模块——`matplotlib.pyplot` 和 `numpy`，前者负责数据的绘制，后者用于数值计算。本文将从基础知识、安装配置、实例教程三个方面对Matplotlib进行详细介绍。
## 安装配置
### Windows系统

注意：不同版本的Python可能会存在兼容性问题，请安装对应版本的Anaconda。
### macOS系统
直接通过Homebrew（或Macports）安装最新版本的Python和Matplotlib即可：

```bash
brew install python3
pip3 install matplotlib # pip 或 pip3 可切换到Python3环境安装
```

若想同时安装其他一些常用扩展库（如numpy、pandas等），则可以使用Anaconda集成环境来完成安装：

```bash
brew cask install anaconda
conda install numpy pandas scikit-learn pillow
```

### Linux系统
Python通常默认安装，若要安装Matplotlib则需要另外安装：

```bash
sudo apt-get install python3-tk    #Tkinter支持，用于GUI绘图
sudo apt-get install libagg-dev   #Cairo支持，用于高级绘图
pip3 install matplotlib         #Python3环境下安装Matplotlib
```

### Jupyter Notebook（可选）
为了更好地了解Matplotlib的用法，还可以选择安装Jupyter Notebook并结合IPython的魔法函数交互模式进行学习：

```bash
pip3 install notebook ipykernel      #安装Notebook和IPython内核
ipython kernel install --user       #创建并设置IPython内核
jupyter notebook                     #启动Notebook客户端
```

打开浏览器访问 `http://localhost:8888` ，可以看到一个Notebook文件列表界面。新建Python3 Notebook文档，输入以下代码测试一下：

```python
import matplotlib.pyplot as plt
plt.plot([1, 2, 3], [2, 4, 1])
plt.show()
```

按下“Shift+Enter”组合键执行代码，可以看到一条曲线图在窗口中显示出来了。更多关于Jupyter Notebook的使用方法请参考相关文档。
## 实例教程
### 数据准备
假设我们要绘制一条曲线，其数据点由 x 和 y 坐标表示：

```python
x = [1, 2, 3]
y = [2, 4, 1]
```

### 最简单的折线图
最简单也最常用的图形是折线图，只需调用 `plot()` 函数即可绘制，如下例所示：

```python
import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [2, 4, 1]

plt.plot(x, y)
plt.show()
```

运行结果如下图所示：


### 添加图例（可选）
如果希望在图上添加图例，可以通过 `label` 参数指定每个数据系列的名称，再调用 `legend()` 方法添加图例，如下例所示：

```python
import matplotlib.pyplot as plt

x = [1, 2, 3]
y1 = [2, 4, 1]
y2 = [1, 3, 4]

plt.plot(x, y1, label='First')
plt.plot(x, y2, label='Second')
plt.legend()
plt.show()
```

运行结果如下图所示：


### 设置横纵轴标签、刻度、范围（可选）
如果不指定横纵轴标签、刻度、范围，则默认采用Matplotlib自动生成的样式。如果希望自定义这些设置，可以调用 `xlabel()`, `ylabel()`, `xticks()`, `yticks()`, `axis()` 方法，如下例所示：

```python
import matplotlib.pyplot as plt

x = [1, 2, 3]
y1 = [2, 4, 1]
y2 = [1, 3, 4]

plt.plot(x, y1, 'o-', label='First', markersize=7)
plt.plot(x, y2, '^--', label='Second', lw=2)
plt.xlabel('X Label')
plt.ylabel('Y Label')
plt.title('Line Plot Demo')
plt.grid(True)
plt.axis([0, 4, 0, 5])
plt.legend()
plt.show()
```

运行结果如下图所示：


### 散点图
类似于折线图，也可以绘制散点图。Matplotlib提供了两种绘制散点图的方法，一种是使用 `scatter()` 函数，另一种是使用 `plot()` 函数加上特殊符号。

#### scatter() 方法
该方法的使用方式和 `plot()` 方法一致，只是不需要指定线型。传入 `marker` 参数指定绘制的图案类型。例如，以下代码绘制了四种图案类型的散点图：

```python
import matplotlib.pyplot as plt

n = 1024
x = np.random.normal(0, 1, n)
y = np.random.normal(0, 1, n)

colors = np.random.rand(n)
sizes = 100 * np.random.rand(n)

plt.scatter(x, y, c=colors, s=sizes, alpha=0.5)
plt.colorbar()
plt.clim(-0.5, 1)
plt.show()
```

运行结果如下图所示：


#### plot() 方法 + 特殊符号
除了 `scatter()` 方法外，还可以使用 `plot()` 方法加上特殊符号绘制散点图。例如，以下代码使用 `plot()` 方法绘制两条曲线图，再分别加上小圆点和十字标记，共得到一个散点图：

```python
import matplotlib.pyplot as plt

np.random.seed(0)

n = 1024
x = np.random.normal(0, 1, n)
y = np.random.normal(0, 1, n)

plt.plot(x, y, '.', color='#00bfff', ms=1)        # 小圆点
plt.plot(x, -y, '.', color='#ff9900', ms=1)     # 小十字点

ax = plt.gca()                                     # 获取当前的Axes对象
ax.spines['right'].set_color('none')                # 隐藏右边框
ax.spines['top'].set_color('none')                  # 隐藏上边框
ax.xaxis.set_ticks_position('bottom')               # 将底部坐标轴设置为横轴
ax.yaxis.set_ticks_position('left')                 # 将左侧坐标轴设置为纵轴

plt.xlabel('$x$')                                   # 横轴标签
plt.ylabel('$y$')                                   # 纵轴标签
plt.title('Scatter Plot Demo')                      # 标题
plt.show()
```

运行结果如下图所示：


### 饼图
Pie chart （甜甜圈图）是一个非常常见的图表形式，Matplotlib提供了 `pie()` 函数用于绘制饼图。例如，以下代码绘制了一组随机生成的数据，再绘制了一个不规则的饼图：

```python
import matplotlib.pyplot as plt
import random

data = [random.randint(1, 10) for _ in range(10)]

labels = ['Label%d' % i for i in range(len(data))]

explode = [0, 0, 0.1, 0, 0, 0, 0, 0, 0, 0.1]          # 指定每块扇区偏离中心距离

fig1, ax1 = plt.subplots()                                # 创建子图
ax1.pie(data, explode=explode, labels=labels, autopct='%1.1f%%',
shadow=False, startangle=90)                       # 绘制饼图
ax1.axis('equal')                                          # 使得饼图变成一个正圆

plt.show()
```

运行结果如下图所示：


### 分布图
Histogram（直方图）是描述一组数据的频率分布的重要图表。Matplotlib提供了 `hist()` 函数用于绘制直方图。例如，以下代码生成了1000个服从正态分布的随机样本，然后绘制了一组直方图：

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)
data = np.random.randn(1000)

fig, axs = plt.subplots(nrows=2, sharex=True, figsize=(6, 8))

axs[0].hist(data, bins=20, density=True, histtype='stepfilled',
edgecolor='None')                                       # 第一组直方图
axs[0].set_title('Stepfilled Histogram')                        # 标题

axs[1].hist(data, bins=20, density=True, histtype='step',
cumulative=True, linewidth=2, label='Empirical')        # 第二组直方图
axs[1].hist(np.random.randn(1000), bins=20, density=True,
histtype='step', cumulative=-1, linewidth=2, linestyle='dashed',
label='Theoretical')                                      # 第三组直方图
axs[1].set_title('Cumulative Distribution Function')            # 标题
axs[1].legend(loc='upper left')                                 # 图例位置

for ax in axs:
ax.set_ylim(ymax=0.5)                                         # 纵轴范围

plt.show()
```

运行结果如下图所示：


### Box Plot（箱型图）
Box Plot（箱形图）是一种常用的图形化显示方法，它能够很好的反映出一组数据的中位数、上下四分位数和五百分位数。Matplotlib提供了 `boxplot()` 函数用于绘制箱型图。例如，以下代码生成了1000个服从正态分布的随机样本，再绘制了一个箱型图：

```python
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(19680801)
data = np.random.randn(1000)

fig, ax = plt.subplots()

bp = ax.boxplot(data)                                              # 绘制箱型图
ax.set_xticklabels(['']*len(bp['boxes']))                          # 清除箱型图上面的刻度

plt.show()
```

运行结果如下图所示：
