
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Matplotlib是一个python第三方库，用于绘制图形、动画、图像等多种形式。它支持跨平台显示，能够同时生成高质量的静态图形、交互式的绘图应用、公式编辑、嵌入式科学计算等功能。

Matplotlib由两部分组成：
1. Matplotlib.pyplot模块，它提供了类似于MATLAB的绘图命令集合；
2. Matplotlib.backends模块，它实现了不同类型的输出格式，如PNG，PDF，SVG，PS等。这些文件可以被用来在其他环境中导入使用，如web页面或后期处理软件。

本文将对matplotlib.pyplot模块进行详细介绍，并结合实例代码对该模块的基本用法作出介绍。

# 2.基本概念
## 2.1. pyplot模块介绍

pyplot模块主要提供了两种方法：

1. `plt.plot(x, y)`，用于绘制线图，其中`x`，`y`分别为两个列表或者数组，表示横轴和纵轴的数据。如果只有一条曲线需要画，则只需传入一个数组作为参数即可。

2. `plt.scatter(x, y)`，用于绘制散点图，其中`x`，`y`分别为两个列表或者数组，表示横轴和纵轴的数据。如果需要标注每一个数据点，可以使用第三个参数`c`，表示每个数据点对应的颜色。

3. `plt.bar(x, height)`，用于绘制条形图，其中`x`为标签，`height`为高度值，一般来说会设置为一个列表或者数组，表示每个柱子的高度。

4. `plt.hist(data)`，用于绘制直方图，其中`data`为列表或者数组，表示各个数据点的频率。

5. `plt.imshow(arr)`，用于绘制图像，其中`arr`为二维数组，表示图像矩阵。

6. `plt.title(s)`，设置图片的标题，`s`为字符串类型。

7. `plt.xlabel(s)`，设置横坐标的标签文本，`s`为字符串类型。

8. `plt.ylabel(s)`，设置纵坐标的标签文本，`s`为字符串类型。

9. `plt.legend()`，设置图例。

以上只是pyplot模块中的一些常用的函数，还有很多其它高级特性和函数，但这些基础的方法足以应付日常需求了。

## 2.2. pyplot模块安装

Matplotlib库可以直接通过pip命令安装：

```bash
$ pip install matplotlib
```


# 3. 代码示例

下面我们以绘制简单曲线图、简单散点图、条形图、直方图为例，展示pyplot模块的基本用法。

## 3.1. 曲线图

如下代码，绘制一个简单的曲线图：

```python
import numpy as np
import matplotlib.pyplot as plt

t = np.arange(0., 5., 0.2) # 生成从0~5间隔0.2的数组
s = 1 + np.sin(2 * np.pi * t) # 使用正弦函数生成数据
plt.plot(t, s)              # 绘制曲线
plt.show()                  # 显示图表
```

结果：


## 3.2. 散点图

如下代码，绘制一个简单的散点图：

```python
import numpy as np
import matplotlib.pyplot as plt

n = 1024    # 数据点个数
X = np.random.normal(0, 1, n)     # 生成标准正态分布数据
Y = np.random.normal(0, 1, n)

T = np.arctan2(Y, X)      # 根据正弦定理计算角度
C = np.sqrt(X**2 + Y**2)   # 计算半径

plt.scatter(X, Y, c=T)             # 用角度值表示颜色
plt.colorbar()                    # 添加色带
plt.axis('equal')                 # 设置坐标轴比例一致
plt.show()                        # 显示图表
```

结果：


## 3.3. 柱状图

如下代码，绘制一个简单的条形图：

```python
import matplotlib.pyplot as plt

men_means = (20, 35, 30, 35, 27)
women_means = (25, 32, 34, 20, 25)

ind = np.arange(len(men_means))  # x轴刻度位置

plt.bar(ind, men_means, width=0.35, label='Men')        # 绘制男性数据的柱状图
plt.bar(ind+0.35, women_means, width=0.35, label='Women')   # 绘制女性数据的柱状图
plt.xticks(ind+0.35/2, ('G1', 'G2', 'G3', 'G4', 'G5'))     # 设置x轴刻度标签
plt.legend()                                            # 添加图例
plt.show()                                              # 显示图表
```

结果：


## 3.4. 直方图

如下代码，绘制一个简单的直方图：

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(19680801) # 设置随机种子

mu = 100         # 均值
sigma = 15       # 标准差
x = mu + sigma*np.random.randn(10000)           # 产生一千万个服从正态分布的数据点

num_bins = 50                                  # 分为多少个区间
fig, ax = plt.subplots()                      # 创建一个Figure对象和Axes对象
ax.hist(x, num_bins, density=True)            # 绘制直方图
ax.set_xlabel('Smarts')                       # 设置横坐标标签
ax.set_ylabel('Probability density')          # 设置纵坐标标签
ax.set_title(r'Histogram of IQ: $\mu=100$, $\sigma=15$') # 设置图表标题

# Tweak spacing to prevent clipping of ylabel
fig.tight_layout()                            
                                                  
plt.show()                                    # 显示图表
```

结果：
