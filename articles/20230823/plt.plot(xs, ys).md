
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 概览
Matplotlib是一个开源的Python库，提供用于创建静态，交互式图形的函数。从2007年发布至今，已经成为许多数据科学、机器学习领域的基础工具包。 Matplotlib最初被设计用来处理2D图像绘制，随后添加了3D图形支持，可视化、统计数据分析等功能，目前已经成为最流行的数据可视化库。 Matplotlib的语法类似于MATLAB，但更加面向对象，能够轻松生成各种复杂的可视化效果。本文将详细介绍Matplotlib中的`plot()`函数。

## 安装
Matplotlib可以直接通过pip进行安装：
```python
pip install matplotlib
```

也可以下载源码编译安装。

## 使用
### 绘制折线图
首先，我们需要导入matplotlib模块：
```python
import matplotlib.pyplot as plt
```
然后，我们创建一个Figure对象，并在其中添加一个Axes对象。Axes对象主要负责绘制坐标轴、刻度以及图例，如下所示：
```python
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
```
创建好Axes之后，就可以用`plot()`函数绘制折线图。`plot()`函数有两种调用方式：

1. `plot(x, y)`：将给定的x和y坐标列表作为输入，绘制折线图。
2. `plot([x1, x2,...], [y1, y2,...])`: 将多个折线组成一条曲线，其坐标点由x和y组成的列表表示。

例如，绘制一张简单的折线图，如下所示：
```python
x = range(1, 6)   # 设置x轴数据
y = [i**2 for i in x]   # 设置y轴数据
ax.plot(x, y)    # 用plot函数绘制折线
ax.set_title('Line Chart')   # 设置图表标题
ax.set_xlabel('X Label')     # 设置x轴标签
ax.set_ylabel('Y Label')     # 设置y轴标签
plt.show()      # 显示图表
```


### 添加误差条
要添加误差条到折线图中，可以使用`fill_between()`函数。它的参数包括两个x轴范围，即包含误差条起始位置的左端点和结束位置的右端点；第二个参数指定y轴范围，即包含误差条的上端和下端。如果仅传入单个参数，则默认绘制区域从0至该值。

例如，我们可以在同一张图中添加一个平滑的红色误差条，用来显示实际值的上下限：
```python
import numpy as np
error = np.random.rand(len(x)) * 0.1 + 0.1
upper = y + error
lower = y - error
ax.fill_between(x, upper, lower, facecolor='red', alpha=0.3)
plt.show()
```
