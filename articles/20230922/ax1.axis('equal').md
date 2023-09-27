
作者：禅与计算机程序设计艺术                    

# 1.简介
  

ax1.axis('equal')是一个matplotlib的函数，可以实现等轴比例图的绘制。用这个函数，可以使得曲线、直线或坐标轴的长度等相近。一般情况下，直角坐标系中的数据点之间存在距离差异，导致坐标轴不等长，无法比较曲线之间的关系。而等轴比例图则通过设置不同的坐标刻度范围，使得各个坐标轴上的单位长度相同。这样便于观察不同数据的变化规律。



# 2.概念
在直角坐标系中，坐标轴的长度不能相等，否则将影响到数据可视化效果。为了解决这一问题，最简单的方法就是采用等轴比例坐标系，其中坐标轴的长度均等，即$x,y$轴上的单位长度都是一致的。

下面我们先举一个简单的例子来说明该函数如何工作。



# 3.示例
假设我们要绘制两个正弦曲线$y=sin(x)$和$y=\cos(x)$，但由于它们是不同周期的函数，所以它们的图像不会在同一坐标系上。因此，需要先对坐标轴进行变换，使得坐标轴长度相等。

```python
import matplotlib.pyplot as plt

# create data
x = np.linspace(-np.pi * 2, np.pi * 2)
y_sin = np.sin(x)
y_cos = np.cos(x)

fig, axes = plt.subplots(nrows=1, ncols=2)

# plot sin curve in origin coordinate system
axes[0].plot(x, y_sin)
axes[0].set_xlabel("X")
axes[0].set_ylabel("Y")
axes[0].grid()

# use equal axis function to make two plots have the same x and y range
axes[1] = plt.subplot(1, 2, 2, sharex=axes[0], sharey=axes[0]) # set the second subplot share with first one
plt.axis('equal') # enable equal axis mode for second subplot
axes[1].plot(x, y_cos)
axes[1].set_xlabel("X")
axes[1].set_ylabel("Y")
axes[1].grid()

plt.show()
```




如图所示，在左侧坐标系中，sin曲线的斜率较大，因此轴长度比较长；而在右侧坐标系中，cos曲线的斜率较小，轴长度较短，此时使用equal函数后两条曲线呈现出了等轴比例图的效果。

# 4.总结及思考

本文主要介绍了matpotlib中的`ax1.axis('equal')`函数，用于实现等轴比例图的绘制。主要作用是通过设置不同的坐标刻度范围，使得各个坐标轴上的单位长度相同，从而使得不同坐标系下的曲线或直线等数据更容易被观察。并提供了一个实际例子说明了该函数的使用方法。

在实际应用中，建议先确定好自己的数据的表示形式。如果是数据的连续性，例如$x,y$轴代表的物理量随时间变化，那么应采用等轴比例坐标系；反之，如果是数据的离散性，例如$x,y$轴代表的整数值，这种情况下就不需要采用等轴比例坐标系。因此，在决定是否启用等轴比例坐标系前，应该首先考虑清楚自己的数据类型，才能做出正确的选择。