
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Pandas是一个开源的数据分析工具，其提供了高性能、易用性的处理数据的能力。本文将从Series和DataFrame两个最常用的pandas数据结构的基础操作进行介绍，包括合并、分组、排序等常用操作。希望通过阅读本文可以帮助读者对pandas有一个更加深刻的了解并应用到实际工作中。

# 2.Series
Pandas中的Series类似于NumPy中的数组(ndarray)，它是一个带标签的一维数组。如下图所示： 



Series是一种有序的、可索引的数组类型。在Pandas中，可以通过很多方式创建Series对象。如从列表、字典或Numpy数组创建，也可以从文件读取数据生成Series对象。另外，也可以通过指定索引来创建Series对象。例如：

```python
import pandas as pd
import numpy as np

data = {'city': ['Beijing', 'Shanghai', 'Guangzhou'],
        'population': [10000000, 20000000, 30000000],
        'area': [10000, 20000, 30000]}
        
df = pd.DataFrame(data) # Create a dataframe object from the dictionary data
print(type(df)) # Output: <class 'pandas.core.frame.DataFrame'>

s = df['city'] # Create a series object by selecting one column of the dataframe
print(type(s)) # Output: <class 'pandas.core.series.Series'>

arr = np.random.randn(10) # Generate an array with random values
s2 = pd.Series(arr) # Create a series object by passing in the array
print(type(s2)) # Output: <class 'pandas.core.series.Series'>
```

创建一个Series时，可以给每个元素指定一个标签（index），否则会自动分配默认标签。

```python
labels = ['a', 'b', 'c']
my_series = pd.Series([1, 2, 3], index=labels)
print(my_series) 
# Output: 
# a    1
# b    2
# c    3
# dtype: int64
```

可以使用下标访问或者标签访问某个元素。标签不存在时会报错。

```python
print("Label-based access:", my_series['a']) # Output: 1
print("Index-based access:", my_series[0])   # Output: 1
```

Series的索引可以是任意不可变对象，但如果不是整数值，则不能用于切片。

```python
my_series = pd.Series(['apple', 'banana', 'cherry', 'date'], index=['x', 'y', 'z', 't'])
print(my_series[['x', 'z']]) # Output: x       apple
                  z       cherry
dtype: object
                  
print(my_series[1:])    # Output: y      banana
                 t     date

```

Series对象的运算包括两个方面：标量运算和向量运算。标量运算是对整个Series进行运算，如求和、求平均值、最大最小值等；向量运算是对Series中的每个元素进行运算，如按元素相乘、求和、逻辑比较等。

```python
my_series1 = pd.Series([1, 2, 3, 4])
my_series2 = pd.Series([-1, -2, -3, -4])

print(my_series1 + my_series2)        # Output: 0    -3
                       1    -4
                       2    -5
                       3    -6
                       dtype: int64

print((my_series1 * my_series2).mean()) # Output: -16.0

print(np.sin(my_series1))              # Output: 0         sin(1)
                         1          sin(2)
                         2          sin(3)
                         3          sin(4)
                         dtype: float64

print(my_series1 > 2)                 # Output: 0    False
                         1     True
                         2     True
                         3    False
                         dtype: bool
```

# 3.DataFrame
DataFrame是一个表格型的数据结构，由多个Series组合而成。每一行代表一个观察对象，每一列代表一个变量，两者都有对应的标签索引。如下图所示：


DataFrame是具有列标签和行标签的二维数据结构，其中可以包含不同类型的数据（数值、字符串、布尔值）。一般情况下，DataFrame由多种不同的Series（观察）组成，同样可以具有标签索引。创建DataFrame的方式有很多，这里举例两种：从Numpy数组创建和读取文件。

```python
import pandas as pd
import numpy as np

# Creating DataFrames from Numpy arrays

data = np.array([[1, 2, 3], [4, 5, 6]])
columns = list('ABC')
rows = ['row1', 'row2']
df = pd.DataFrame(data, columns=columns, index=rows)
print(df)

# Reading CSV files into DataFrames

df = pd.read_csv('file.csv')
print(df)
```

可以通过行标签（index）和列标签（columns）访问和修改DataFrame中的元素。

```python
print(df['A']['row1'])           # Output: 1
print(df.loc['row1']['B'])        # Output: 2
df.iloc[0][1] = 10                # Update element at row 0 col 1 to be 10
```

可以通过`head()`方法查看前几行数据，`tail()`方法查看后几行数据。

```python
print(df.head())                  # View first few rows of the dataset
print(df.tail())                  # View last few rows of the dataset
```

DataFrame的运算包括行列的运算，也可以对DataFrame中的所有元素进行单独运算。

```python
# Row/Column Arithmetic

new_df = df / 2
print(new_df)                     # Divide each element by 2

# Broadcasting arithmetic operations (element-wise addition between two DataFrames)

other_df = df + new_df
print(other_df)                   # Add corresponding elements in both DataFrames

# Single operation across all elements (calculating square root for all elements)

sqrt_df = np.sqrt(df)             # Calculate square roots of all elements
print(sqrt_df)                    # Square Root of Each Element in Original DataFrame
```

DataFrame的合并、分组、排序等常用操作也非常简单。

```python
# Merging DataFrames

merged_df = pd.merge(left_df, right_df, on='key')            # Merge two DataFrames based on common key
grouped_df = df.groupby('column').sum()                      # Group rows by value of specified column
sorted_df = df.sort_values(by='column', ascending=False)     # Sort DataFrame by specific column

# Joining DataFrames

joined_df = left_df.join(right_df, lsuffix='_left', rsuffix='_right') # Join two DataFrames using shared indices or column names

# Concatenating DataFrames

concat_df = pd.concat([df1, df2], axis=1)                        # Concatenate horizontally (column-wise)
concat_df = pd.concat([df1, df2], axis=0)                        # Concatenate vertically (row-wise)
```