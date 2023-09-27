
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Matplotlib是基于Python语言的2D绘图库，可以用于创建各种二维图表和图形，如折线图、散点图、直方图等。Matplotlib由<NAME>在2003年创建，最初名字叫做matplotlib.py, 是IPython的一个子项目。


Matplotlib的主要特色有以下几点：
- 支持不同的输出格式，包括PNG、PDF、SVG、EPS、JPEG、TIFF等；
- 提供高质量的向量图形输出，能够自动去除坐标轴上不需要的边框和刻度；
- 针对网格型数据的优化展示功能，能够方便地控制每个网格单元的颜色、透明度、样式及大小；
- 拥有丰富的图像类型，如条形图、饼状图、雷达图、箱型图等；
- 可以通过调用Matlab风格的语法简化图表的创建过程；
- 可以创建精美的三维图形，并且支持动画效果。

# 2.基本概念术语说明
## 2.1 数据结构
Matplotlib中用到的主要数据结构有两类：
- Figure（图）：一个Figure对象表示一个完整的图形画布，它包含图中的所有元素，比如：图例、轴、子图等；
- Axes（坐标轴）：一个Axes对象是一个绘制区域，包含X轴、Y轴和图例。Axes对象通常包含不同的数据类型，比如：直方图、散点图、柱状图、等高线图、条形图等。

## 2.2 使用方法
Matplotlib的使用方法如下：
1. 在当前脚本或者模块中导入Matplotlib包：
```python
import matplotlib.pyplot as plt
```
2. 创建Figure和Axes对象并设置样式属性：
```python
fig = plt.figure()    # 创建Figure对象
ax = fig.add_subplot(111)   # 添加Axes对象并指定其位置参数
ax.set_xlabel('x label')     # 设置X轴标签
ax.set_ylabel('y label')     # 设置Y轴标签
ax.set_title('Title')        # 设置标题
```
3. 将数据可视化：
```python
plt.plot([1,2,3],[4,5,6],'r')    # 用红色的线条连接点(1,4),(2,5),(3,6)
plt.axis([-2,10,-2,10])          # 设置坐标范围
plt.show()                      # 显示图形
```
4. 可选：自定义图例、添加注释、保存图片。

## 2.3 示例应用
下面来看一些Matplotlib绘图的具体例子。
### 2.3.1 直方图
下面用Matplotlib绘制一个直方图：
```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(19680801)
data = np.random.randn(1000)

plt.hist(data, bins=20, normed=True)      # bins：直方图的宽，normed：是否将得到的频率密度分布函数归一化到概率密度分布函数
plt.grid(True)                             # 添加网格
plt.show()
```
结果如下：

### 2.3.2 折线图
下面的例子演示了如何绘制折线图：
```python
import matplotlib.pyplot as plt

x = [1, 2, 3]
y = [2, 4, 1]
labels = ['A', 'B', 'C']

plt.plot(x, y, marker='o', linestyle='--', color='g', label='Line 1')         # 指定曲线的符号、线条、颜色以及对应的名称
plt.legend(loc='upper left')                                                    # 为图表添加图例

for i in range(len(x)):
    plt.text(x[i]+0.1, y[i]-0.1, labels[i])                                       # 在每根线上添加文字标签
    
plt.xlabel('X Label')                                                            # 设置X轴标签
plt.ylabel('Y Label')                                                            # 设置Y轴标签
plt.title('Title')                                                               # 设置标题
plt.show()                                                                       # 显示图形
```
结果如下：

### 2.3.3 散点图
下面的例子演示了如何绘制散点图：
```python
import numpy as np
import matplotlib.pyplot as plt

n = 1024
x = np.random.normal(0, 1, n)
y = np.random.normal(0, 1, n)

plt.scatter(x, y, s=75, c=None, alpha=0.5)       # 以半透明形式绘制散点
plt.grid(True)                                  # 添加网格
plt.xlim(-1.5, 1.5)                            # 设置X轴范围
plt.ylim(-1.5, 1.5)                            # 设置Y轴范围
plt.xticks(())                                 # 不显示刻度标记
plt.yticks(())                                 # 不显示刻度标记
plt.show()                                      # 显示图形
```
结果如下：