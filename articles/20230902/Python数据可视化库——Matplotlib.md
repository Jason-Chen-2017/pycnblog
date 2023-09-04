
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## Matplotlib是什么?

Matplotlib是一个基于Python的开源数据可视化库，用于创建2D图形、图表和图片。Matplotlib主要提供了四个级别的接口：

1. pyplot接口：matplotlib中最简单的绘制图像的方式，直接导入matplotlib.pyplot模块即可使用，通过多个函数对figure对象进行操作即可实现各种二维图形的绘制。
2. 对象级接口：通过Figure()函数创建一个顶层的figure对象，在其上进行各子组件的添加，最后调用show()方法呈现出来。
3. OO接口：面向对象的绘图方式，可以更精细地控制图形的布局、大小、风格等。
4. 基于GUI的接口：如果需要将图形输出到屏幕或打印设备，可以调用matplotlib.backends接口下的各类绘图后端接口进行绘制。

Matplotlib既可以做静态图形展示，也可以生成动态交互图形，如mplfinance、bokeh等，但本文只讨论其基础功能。

## Matplotlib安装及环境配置

Matplotlib支持Windows、Mac OS X、Linux平台。安装Matplotlib非常简单，直接用pip命令安装即可：

```python
pip install matplotlib
```

### Matplotlib基本示例

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2*np.pi, 20) # x轴数据
y = np.sin(x)                  # y轴数据

plt.plot(x, y)                 # 画出线条
plt.xlabel('x')                # 设置X轴标签
plt.ylabel('sin(x)')           # 设置Y轴标签
plt.title('sin function')      # 设置标题
plt.grid(True)                 # 显示网格

plt.show()                     # 显示绘图结果
```



### 3D绘图

Matplotlib还支持3D绘图，我们可以使用mpl_toolkits包中的mplot3d模块，这个包默认不随着matplotlib一起安装，需要额外安装：

```python
pip install mpl_toolkits
```

然后导入该模块即可：

```python
from mpl_toolkits import mplot3d

fig = plt.figure()               # 创建一个figure对象
ax = fig.add_subplot(projection='3d')   # 添加一个三维坐标轴

# 数据准备
u = np.linspace(-np.pi, np.pi, 100)        # x轴数据
v = np.linspace(-np.pi, np.pi, 100)        # y轴数据
x = u * np.cos(v)                         # z轴数据
y = u * np.sin(v)

# 绘制三角面片
ax.plot_trisurf(x, y, u+v, cmap='viridis', linewidth=0.2)    # 绘制三角面片

ax.set_xlabel('X')       # 设置X轴标签
ax.set_ylabel('Y')       # 设置Y轴标签
ax.set_zlabel('Z')       # 设置Z轴标签

plt.show()               # 显示绘图结果
```
