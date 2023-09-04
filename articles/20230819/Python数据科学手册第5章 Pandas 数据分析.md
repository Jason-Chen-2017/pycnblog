
作者：禅与计算机程序设计艺术                    

# 1.简介
  

在过去的十多年里，开源数据科学计算项目如NumPy、SciPy、Pandas等成为了进行数据处理、分析和可视化的一流工具，特别是在Python编程语言中应用广泛。本书通过对Pandas库的详细介绍，以及相关的统计、机器学习等领域的数据分析工具的使用技巧，帮助读者更加熟练地运用Pandas库实现数据的清洗、转换、分析及可视化。本书可以作为数据分析入门课程，也可作为进阶课堂、工作坊等高级训练营课程的配套教材。
# 2.基础知识
## 2.1 pandas简介
pandas是一个开源的，强大的，基于Python的数据分析包。它提供高效地数据结构和各种分析功能。其中pd即是pandas的简称。
Pandas提供了两种主要的数据结构：Series（一维数组）和DataFrame（二维表格）。两者都可用于做基本的数值计算、合并、切分等操作，但两者又有区别：

1. Series只能包含一个数据类型；而DataFrame可以包含不同的数据类型，每列可以包含不同的属性或标签。

2. DataFrame的列可以被赋予标签（label），而Series只能没有标签，因此它不适合于当某一列仅需要单独标注时。

DataFrame是由很多Series组成的集合，每个Series代表了DataFrame的一列数据。Pandas除了具有以上优点外，还提供了方便的数据导入导出、数据过滤、数据聚合等功能，使得其成为一种非常灵活的工具。而且，Pandas提供丰富的API接口，使得其能够与其他Python库无缝整合。

## 2.2 安装pandas
pandas可以在Anaconda、pip、源码安装三种方式进行安装。这里给出Anaconda安装方法：
首先下载Anaconda安装程序，如Anaconda3-5.3.1-Windows-x86_64.exe。
双击运行Anaconda安装程序，根据提示安装Anaconda，默认安装路径选择C:\Users\用户名\Anaconda3（注意替换为自己的用户名）。
打开Anaconda Prompt命令行，输入conda list确认是否成功安装。若能显示conda相关信息，则表示安装成功。

安装pandas:
在命令行输入以下命令：
```python
pip install pandas
```
若出现PermissionError，则需要管理员权限。在命令行输入：
```python
sudo pip install pandas
```
若安装失败，可能是因为缺少依赖包。可以使用下面的命令尝试解决：
```python
sudo apt update
sudo apt install libatlas-base-dev
sudo apt install libgfortran3
sudo pip install pandas --no-cache-dir --ignore-installed
```
最后一条命令会重新编译pandas，这样可以避开可能存在的问题。

## 2.3 pandas数据结构
### 2.3.1 Series
Series是pandas中的一种基本的数据结构，它类似于一维数组，但是可以包含多个数据类型。Series一般由相同长度的数组或列表构成，可以通过索引访问元素。Series可以存储任何类型的数据，包括数字、字符串、日期等。
创建Series的方法如下：
```python
import pandas as pd

data = [1, 2, 3, 4]    # 整数列表
s = pd.Series(data)     # 通过列表创建Series
print(s)

data = {'a': 0., 'b': 1., 'c': 2.}   # 字典
s = pd.Series(data)                  # 通过字典创建Series
print(s)

dates = pd.date_range('20190101', periods=6)      # 创建日期序列
values = np.random.randn(6)                      # 创建随机数序列
df = pd.DataFrame({'A': values}, index=dates)       # 将数据放到DataFrame中
print(df['A'])                                    # 获取Series
```
输出结果：
```
  0  
0  1  
1  2  
2  3  
3  4  
  a   b   c  
0  0.0  1.0  2.0
```

### 2.3.2 DataFrame
DataFrame是pandas中的一种二维的数据结构，它包含了一组相同长度的Series，并带有一个索引。索引通常是Series或者字符串，用于标记DataFrame中的各个记录。DataFrame可以用来处理表格型的数据。
创建DataFrame的方法如下：
```python
import pandas as pd
import numpy as np

# 从字典创建DataFrame
data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
        'year': [2000, 2001, 2002, 2001, 2002],
        'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
frame = pd.DataFrame(data)
print(frame)

# 从numpy数组创建DataFrame
data = np.array([[1, 2], [3, 4]])
frame = pd.DataFrame(data, index=['a', 'b'], columns=['one', 'two'])
print(frame)

# 从Series创建DataFrame
d1 = {"a": [1, 2], "b": [3, 4]}
d2 = {"a": ["a", "b"], "c": [5, 6]}
df1 = pd.DataFrame(d1)
df2 = pd.DataFrame(d2)
df3 = pd.concat([df1, df2], axis=1)
print(df3)
```
输出结果：
```
    state  year  pop
0   Ohio  2000  1.5
1   Ohio  2001  1.7
2   Ohio  2002  3.6
3  Nevada  2001  2.4
4  Nevada  2002  2.9
```

```
       one  two
a  1.0  3.0
b  2.0  4.0
```

```
         a  b  c
0  1.0  3.0 NaN
1  2.0  4.0 NaN
```