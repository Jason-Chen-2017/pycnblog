
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Matplotlib (中文翻译为绘图库) 是 Python 中一个著名的画图工具包，很多数据科学、机器学习领域的大牛们都在用 Matplotlib 来进行数据可视化分析。近年来 Matplotlib 的生态也越来越好，功能和定制化逐渐增多。从 Matplotlib 的历史发展可以看出，它是一个基于 NumPy 和其他第三方库（如 SciPy、Pillow）构建的开源项目，在高性能和动态交互性上提供了强劲的支持。

本文主要介绍 Matplotlib 中的 ylim() 函数，用来设置 Y 轴的最小值和最大值。

# 2.函数说明
## 2.1 matplotlib.pyplot.ylim(ymin=None, ymax=None)
### 参数：
- ymin: float or None，可选参数，最小值，默认为最小数据的负3倍标准差。
- ymax: float or None，可选参数，最大值，默认为最大数据的正3倍标准差。
返回值：matplotlib.axes._subplots.AxesSubplot 对象。

该函数用于设置 Y 轴的最小值和最大值，并返回当前子图的 Axes 对象。

### 用法
ylim() 函数设置 Y 轴的最小值和最大值。其中，ymin 和 ymax 为数值类型或 None，默认值为 None。当 ymin 或 ymax 为 None 时，系统会自动计算相应的值。如果设置了 ymin 或 ymax，则不会再自动计算值。但是如果同时设置了 ymin 和 ymax，且 ymin 大于等于 ymax，那么就会显示异常信息。

### 案例
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 5, 10)
y_positive = x ** 2
y_negative = -y_positive
plt.figure()
plt.subplot(121)
plt.title('Y Limits With Auto Scale')
plt.plot(x, y_positive)
plt.plot(x, y_negative)
plt.subplot(122)
plt.title('Y Limits Without Auto Scale')
plt.plot(x, y_positive)
plt.plot(x, y_negative)
plt.ylim(ymax=70) # 设置 Y 轴最大值为 70
plt.show()
```
结果如下所示：

上图中，左边的子图中 Y 轴的范围是由数据自身决定的，而右边的子图中手动设置了 Y 轴的最大值为 70，所以两幅图中的曲线没有重合。

通过设置 `ylim()` 函数也可以调整 Y 轴的范围，使得曲线显示的更加符合实际需求。比如，将上述例子中手动设置 Y 轴最大值的 `plt.ylim(ymax=70)` 替换为 `plt.ylim(0, 70)` ，这样就确保了 Y 轴的最大值为 70，并且对齐了两个子图的 Y 轴范围。

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 5, 10)
y_positive = x ** 2
y_negative = -y_positive
plt.figure()
plt.subplot(121)
plt.title('Aligned Y Limits')
plt.plot(x, y_positive)
plt.plot(x, y_negative)
plt.ylim(0, 70) # 设置 Y 轴最小值和最大值
plt.subplot(122)
plt.title('Non-aligned Y Limits')
plt.plot(x, y_positive)
plt.plot(x, y_negative)
plt.ylim(ymax=70) # 设置 Y 轴最大值
plt.show()
```
结果如下所示：

可以看到，当设置了 `ylim()` 函数后，两个子图中的 Y 轴范围已经对齐。