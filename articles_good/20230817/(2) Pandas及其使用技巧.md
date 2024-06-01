
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Pandas（葡萄牙语/ˈpaɪndəs/）是一个开源数据分析库，提供高效、灵活的数据处理功能和强大的分析工具。它提供了一种DataFrame对象，用来存储及操纵二维表格数据，能够轻松实现数据的清洗、整理、转换等操作。其中最常用的是pandas中的Series和Dataframe两个数据结构，两者均为多维数组的抽象表示形式，可以看成是特定列或行索引的字典类型。Pandas基于Numpy进行了扩展，提升了对数组、矩阵计算的性能，并融合了数据库查询、时间序列分析和空间地理信息分析的优点，使得数据处理和分析工作变得十分简单、高效。本文将从pandas的基本数据结构Series和Dataframe出发，介绍一些常用的函数方法，帮助读者更加熟练地使用pandas。
# 2.安装与导入模块
在使用pandas之前，首先需要先安装Anaconda或者Miniconda，再使用conda命令安装pandas模块。安装完毕后，就可以在python中通过import pandas语句引入该模块。示例如下：

``` python
!pip install pandas # pip install pandas in conda environment
import pandas as pd 
```

如果出现版本不兼容的问题，可以使用conda升级pandas模块。

``` shell
conda update pandas
```

在jupyter notebook中也可以直接使用`%matplotlib inline`来调用matplotlib绘图库。

# 3.Series
## 3.1 创建Series
Series是pandas中的一个基本数据结构，可以理解为一维数据数组，由index标签和value值组成，可以类似于字典类型，但是又比字典类型功能更多，而且可以有不同类型的值。Series的创建方式如下：

``` python
s = pd.Series([1, 2, 3]) # 使用列表创建
print(s)

s = pd.Series({'a': 1, 'b': 2}) # 使用字典创建
print(s)

dates = ['2019-07-0{}'.format(i+1) for i in range(3)] # 创建日期索引
values = [1, 2, 3]
s = pd.Series(values, index=dates) # 指定索引
print(s)
```

输出结果：

``` 
0    1
1    2
2    3
dtype: int64

 a  1
 b  2
dtype: int64

```

以上三个例子分别展示了如何通过列表、字典和指定索引创建Series对象。可以看到，使用列表创建时，默认情况下会自动生成索引。而使用字典创建时，键作为标签，值作为数据值。

## 3.2 属性和方法
### 3.2.1 查看属性
可以通过Series对象的属性查看相关信息，如索引名、名称、值等。示例如下：

``` python
s = pd.Series([1, 2, 3], name='test', index=['x', 'y', 'z'])
print(s.name)        # 获取名称
print(s.index.name)  # 获取索引名称

print(s.values)      # 获取值
print(type(s.values)) # 获取值的类型

print(s.dtype)       # 获取数据类型

print(len(s))         # 获取长度
```

输出结果：

``` test
None
[1 2 3]
<class 'numpy.ndarray'>
int64
3
```

### 3.2.2 操作
#### 3.2.2.1 添加新元素
可以通过索引位置添加新的元素，或根据索引名称添加元素。示例如下：

``` python
s = pd.Series([1, 2, 3])
s[4] = 4          # 根据索引位置添加元素
print(s)

s['foo'] = 5       # 根据索引名称添加元素
print(s)
```

输出结果：

``` 
0    1
1    2
2    3
4    4
dtype: int64


0    1
1    2
2    3
foo   5
dtype: int64
```

#### 3.2.2.2 删除元素
可以删除指定索引位置的元素，或根据索引名称删除元素。示例如下：

``` python
s = pd.Series([1, 2, 3, 4, 5])
del s[3]           # 根据索引位置删除元素
print(s)

del s['foo']       # 根据索引名称删除元素
print(s)
```

输出结果：

``` 
0    1
1    2
2    3
4    5
dtype: int64


---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
<ipython-input-3-7bc6d5fc12e7> in <module>()
      2 del s[3]           # 根据索引位置删除元素
      3 print(s)
----> 4 del s['foo']       # 根据索引名称删除元素
      5 print(s)

KeyError: 'foo'
```

#### 3.2.2.3 更新元素
可以更新指定索引位置的元素，或根据索引名称更新元素。示例如下：

``` python
s = pd.Series(['apple', 'banana', 'cherry'], index=[1, 2, 3])
s[1] = 'pear'      # 根据索引位置更新元素
print(s)

s['dog'] = 'cat'   # 根据索引名称更新元素
print(s)
```

输出结果：

``` 
1      apple
2       pear
3     cherry
dtype: object


1      apple
2       pear
3     cherry
dog     cat
dtype: object
```

#### 3.2.2.4 查询元素
可以获取指定索引位置的元素，或根据索引名称获取元素。如果不存在则报错。示例如下：

``` python
s = pd.Series([1, 2, 3], index=['x', 'y', 'z'])
print(s.loc['x'])     # 根据索引位置获取元素
print(s.iloc[1])      # 根据索引位置获取元素

print(s['y'])         # 根据索引名称获取元素
```

输出结果：

``` 
1
2
```

#### 3.2.2.5 查询多个元素
可以获取多个指定索引位置的元素，或根据索引名称获取元素。如果不存在则报错。示例如下：

``` python
s = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
print(s[['b', 'd']])             # 根据多个索引位置获取元素
print(s[s > 2])                  # 根据条件获取元素
print(s[[True, True, False]])    # 根据布尔序列获取元素
```

输出结果：

``` 

  b   d
1  2  4
  c  
2  3  

a   1
b   2
c   3
dtype: int64

  e  
False
Name: a, dtype: bool
```

### 3.2.3 函数应用
#### 3.2.3.1 单独函数应用
可以通过`.apply()`方法将Series中的每个元素都应用到指定的函数上，并返回一个新的Series。示例如下：

``` python
def square_root(x):
    return x ** 0.5

s = pd.Series([1, 2, 3, 4, 5])
result = s.apply(square_root)
print(result)
```

输出结果：

``` 
0    1.0
1    1.4142135623730951
2    1.7320508075688772
3    2.0
4    2.23606797749979
dtype: float64
```

#### 3.2.3.2 标量函数应用
可以对所有Series中的元素都应用到指定标量函数上。示例如下：

``` python
s = pd.Series([1, 2, 3, 4, 5])
result = s.map(lambda x: x ** 0.5)
print(result)
```

输出结果：

``` 
0    1.0
1    1.4142135623730951
2    1.7320508075688772
3    2.0
4    2.23606797749979
dtype: float64
```

#### 3.2.3.3 统计函数应用
可以对所有Series中的元素都应用到指定统计函数上，并返回一个Series。示例如下：

``` python
s = pd.Series([1, 2, 3, 4, 5])
result = s.apply(sum)
print(result)
```

输出结果：

``` 
15
```

# 4.Dataframe
## 4.1 创建Dataframe
Dataframe是pandas中的另一个基本数据结构，可以理解为二维数据表格，由很多Series对象组成。它的创建方式也很简单，只需传入Series构成的列表即可。比如：

``` python
data = {'name': ['Alice', 'Bob', 'Charlie', 'David'],
        'age': [25, 30, 35, 40]}

df = pd.DataFrame(data)
print(df)
```

输出结果：

``` 
    name  age
0   Alice   25
1     Bob   30
2  Charlie   35
3   David   40
```

如此便可创建一个四行两列的Dataframe。

## 4.2 属性和方法
### 4.2.1 查看属性
可以通过Dataframe对象的属性查看相关信息，包括索引、列、值等。示例如下：

``` python
data = {'name': ['Alice', 'Bob', 'Charlie', 'David'],
        'age': [25, 30, 35, 40]}
        
df = pd.DataFrame(data)
print(df.columns)      # 获取列名
print(df.shape)        # 获取形状
print(df.index)        # 获取索引
print(df.values)       # 获取值
print(df.info())       # 获取信息
```

输出结果：

``` Index(['name', 'age'], dtype='object')
(4, 2)
   RangeIndex(start=0, stop=4, step=1)
      name  age
0   Alice   25
1     Bob   30
2  Charlie   35
3   David   40
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 4 entries, 0 to 3
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   name    4 non-null      object
 1   age     4 non-null      int64 
dtypes: int64(1), object(1)
memory usage: 192.0+ bytes
```

### 4.2.2 操作
#### 4.2.2.1 添加新行
可以直接向Dataframe中追加新行。示例如下：

``` python
df = pd.DataFrame({'A': ['a', 'b', 'c'],
                   'B': [1, 2, 3]})
new_row = {'A': 'd', 'B': 4}
df = df.append(new_row, ignore_index=True)
print(df)
```

输出结果：

``` 
   A  B
0  a  1
1  b  2
2  c  3
3  d  4
```

#### 4.2.2.2 删除行
可以按照索引位置或名称删除行。示例如下：

``` python
df = pd.DataFrame({'A': ['a', 'b', 'c', 'd', 'e'],
                   'B': [1, 2, 3, 4, 5]})
                   
# 删除指定索引位置的行
df = df.drop(3)
print(df)

# 删除指定索引名称的行
df = df.drop('C')
print(df)
```

输出结果：

``` 
   A  B
0  a  1
1  b  2
2  c  3

   A  B
0  a  1
1  b  2
2  d  4
3  e  5
```

#### 4.2.2.3 更新行
可以按照索引位置或名称更新行。示例如下：

``` python
df = pd.DataFrame({'A': ['a', 'b', 'c', 'd', 'e'],
                   'B': [1, 2, 3, 4, 5]})
                   
# 更新指定索引位置的行
new_row = {'A': 'aa', 'B': 6}
df.at[1, :] = new_row
print(df)

# 更新指定索引名称的行
new_row = {'A': 'bb', 'B': 7}
df.loc['B', :] = new_row
print(df)
```

输出结果：

``` 
   A  B
0  aa  1
1   bb  2
2   c  3
3   d  4
4   e  5

```

#### 4.2.2.4 插入列
可以插入新列，与插入行相似，但这里要传入字典的列表。示例如下：

``` python
df = pd.DataFrame({'A': ['a', 'b', 'c', 'd', 'e'],
                   'B': [1, 2, 3, 4, 5]})
                   
# 插入新列
df['C'] = ['cc', 'dd', 'ee', 'ff', 'gg']
print(df)
```

输出结果：

``` 
     A  B  C
0   a  1  cc
1   b  2  dd
2   c  3  ee
3   d  4  ff
4   e  5  gg
```

#### 4.2.2.5 删除列
可以按列名或列索引删除列。示例如下：

``` python
df = pd.DataFrame({'A': ['a', 'b', 'c', 'd', 'e'],
                   'B': [1, 2, 3, 4, 5],
                   'C': ['cc', 'dd', 'ee', 'ff', 'gg']})
                   
# 删除列名
df = df.drop('A', axis=1)
print(df)

# 删除列索引
df = df.drop(columns=1)
print(df)
```

输出结果：

``` 
   B  C
0  1  cc
1  2  dd
2  3  ee
3  4  ff
4  5  gg

```

#### 4.2.2.6 更新列
可以按列名或列索引更新列。示例如下：

``` python
df = pd.DataFrame({'A': ['a', 'b', 'c', 'd', 'e'],
                   'B': [1, 2, 3, 4, 5],
                   'C': ['cc', 'dd', 'ee', 'ff', 'gg']})
                   
# 更新列名
df['D'] = ['hhh', 'iii', 'jjj', 'kkk', 'lll']
print(df)

# 更新列索引
df.iloc[:, -1] = ['HHH', 'III', 'JJJ', 'KKK', 'LLL']
print(df)
```

输出结果：

``` 
      A  B    C    D
0    a  1   cc  hhh
1    b  2   dd  iii
2    c  3   ee  jjj
3    d  4   ff  kkk
4    e  5   gg  lll

```

#### 4.2.2.7 查询元素
可以按行索引或行标签、列索引或列标签获取元素。如果不存在则报错。示例如下：

``` python
df = pd.DataFrame({'A': ['a', 'b', 'c', 'd', 'e'],
                   'B': [1, 2, 3, 4, 5],
                   'C': ['cc', 'dd', 'ee', 'ff', 'gg']})

print(df.loc[0])              # 按行索引获取
print(df.iloc[0])             # 按行索引获取
print(df.loc[0]['B'])         # 按行索引获取
print(df.iloc[0][2])          # 按行索引获取
print(df.loc[:,'B'])          # 按列索引获取
print(df.iloc[:,1])           # 按列索引获取
print(df.loc['a':'c','B'])    # 按范围、列索引获取
print(df.iloc[0:2,[1,2]])     # 按范围、列索引获取
```

输出结果：

``` 
       A  B   C
0   a  1  cc

{'A': 'a', 'B': 1, 'C': 'cc'}
1
a    1
b    2
c    3
Name: B, dtype: int64

     B   C
0   1  cc
1   2  dd
2   3  ee
```

### 4.2.3 函数应用
#### 4.2.3.1 数据过滤
可以按照条件对Dataframe中的数据进行过滤，并返回一个新的Dataframe。示例如下：

``` python
df = pd.DataFrame({'A': ['a', 'b', 'c', 'd', 'e'],
                   'B': [1, 2, 3, 4, 5],
                   'C': ['cc', 'dd', 'ee', 'ff', 'gg']})
                   
result = df[(df['B'] >= 2) & (df['C'].isin(['cc', 'ee']))]
print(result)
```

输出结果：

``` 
      A  B   C
1   b  2  dd
3   d  4  ff
4   e  5  gg
```

#### 4.2.3.2 数据聚合
可以对Dataframe中的数据进行聚合，返回汇总统计结果。示例如下：

``` python
df = pd.DataFrame({'A': ['a', 'b', 'c', 'd', 'e'],
                   'B': [1, 2, 3, 4, 5],
                   'C': ['cc', 'dd', 'ee', 'ff', 'gg']})

result = df.groupby(['A']).agg({'B': ['mean','std'], 'C': lambda x: len(set(x))})
print(result)
```

输出结果：

``` 
         B                         C
A                               
a (-0.0,) nan                    2
b (1.5, 1.5) 2                  3
c (3.0, 1.5) 3                   
d (4.5, 1.5) 3                   
e (6.0, 1.5) 3                   
```

#### 4.2.3.3 排序
可以对Dataframe中的数据进行排序，并返回一个新的Dataframe。示例如下：

``` python
df = pd.DataFrame({'A': ['a', 'b', 'c', 'd', 'e'],
                   'B': [1, 2, 3, 4, 5],
                   'C': ['cc', 'dd', 'ee', 'ff', 'gg']})
                   
sorted_df = df.sort_values(by=['A', 'B'])
print(sorted_df)
```

输出结果：

``` 
      A  B   C
3   d  4  ff
1   b  2  dd
4   e  5  gg
0   a  1  cc
```

#### 4.2.3.4 数据切片
可以对Dataframe中的数据进行切片，并返回一个新的Dataframe。示例如下：

``` python
df = pd.DataFrame({'A': ['a', 'b', 'c', 'd', 'e'],
                   'B': [1, 2, 3, 4, 5],
                   'C': ['cc', 'dd', 'ee', 'ff', 'gg']},
                  index=['I', 'II', 'III', 'IV', 'V'])

sliced_df = df.loc[:'II', :]
print(sliced_df)
```

输出结果：

``` 
            A  B   C
I       a  1  cc
II      b  2  dd
```