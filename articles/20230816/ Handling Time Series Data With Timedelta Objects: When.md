
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Timedelta是一个Python中用来表示时间差值（duration）或时间间隔的类，它可以表示如“两年零三个月零十天”这样的时间长度。在处理时间序列数据时，对日期和时间进行加减计算，或者与其他日期或时间进行比较等运算时，需要用到Timedelta对象。

本文将介绍Python中的Timedelta对象及其常用的属性和方法。通过示例，展示如何用Timedelta对日期和时间进行各种计算。

# 2.基本概念术语
## 2.1 Timedelta对象
Timedelta对象是Python内置的一个类，它提供了表示时间间隔的功能。

## 2.2 属性
### days、seconds、microseconds
Timedelta对象包含以下三个属性，分别对应时间间隔的天数、秒数、微秒数：

1. `days` 代表时间间隔的天数。
2. `seconds` 代表时间间隔的秒数，注意不是总共的秒数，而是相对于起始时间的秒数。
3. `microseconds` 代表时间间隔的微秒数。

```python
from datetime import timedelta

delta = timedelta(days=2, seconds=30, microseconds=10)
print("Days:", delta.days)      # Days: 2
print("Seconds:", delta.seconds)    # Seconds: 30
print("Microseconds:", delta.microseconds)   # Microseconds: 10
```

### total_seconds()方法
该方法用于获取Timedelta对象的总的秒数，返回的是一个浮点型数值。

```python
delta = timedelta(days=2, hours=12)
total_seconds = delta.total_seconds()
print(total_seconds)     # 97200.0
```

### min、max类属性
min和max是Timedelta类的类属性，分别代表了Timedelta的最小值和最大值。

```python
import datetime as dt

print(dt.timedelta.min)       # 负两周
print(dt.timedelta.max)       # 10675199 days, 106751 hours, 43200 minutes, 86399.999999 seconds
```

其中min值即是`-datetime.timedelta(days=2*7)`的值，表示"负两周"；而max值则是`+datetime.timedelta(days=10675199, hours=106751, minutes=43200, seconds=86399, microseconds=999999)`的值，表示10675199天106751小时43200分钟86399.999999秒，超出了Python所能表示的范围。

## 2.3 方法
### from_days()类方法
从天数创建Timedelta对象。

```python
td = dt.timedelta.from_days(2)
print(td)    # 2 days, 0:00:00
```

### from_seconds()类方法
从秒数创建Timedelta对象。

```python
td = dt.timedelta.from_seconds(30)
print(td)    # 0:00:30
```

### __add__()方法
向两个Timedelta对象相加。

```python
delta1 = timedelta(days=2, seconds=30, microseconds=10)
delta2 = timedelta(hours=12)
result = delta1 + delta2
print(result)        # 4 days, 12:30:40.000010
```

### __sub__()方法
从两个Timedelta对象相减。

```python
delta1 = timedelta(days=2, seconds=30, microseconds=10)
delta2 = timedelta(hours=12)
result = delta1 - delta2
print(result)        # 0:59:27.999990
```

### __mul__()方法
对Timedelta对象乘积。

```python
delta = timedelta(days=2, seconds=30, microseconds=10)
result = delta * 3
print(result)        # 6 days, 0:00:00.999990
```

### __truediv__()方法
对Timedelta对象除法。

```python
delta1 = timedelta(days=2, seconds=30, microseconds=10)
delta2 = timedelta(hours=12)
result = delta1 / delta2
print(result)        # 1.6666666666666667
```