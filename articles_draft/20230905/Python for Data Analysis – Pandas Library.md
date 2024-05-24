
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 1.1 数据分析与Pandas库概述
数据分析(Data Analysis)是指对数据的观察、整理、处理、分析、呈现及其应用的过程。基于数据的分析有助于发现模式、规律和关系，提升决策准确性、改善产品质量等。数据分析通常包括三个阶段：收集数据、处理数据、分析数据。而Pandas库是一个开源的Python数据分析工具包，提供高效率的数据结构和数据操作能力。本文将从基础知识入手，介绍Pandas库的基本功能、用法及一些扩展特性。

## 1.2 安装pandas
```python
pip install pandas
```

## 1.3 数据类型
Pandas支持丰富的数据类型，包括Series（一维数组）、DataFrame（二维表格）、Panel（三维数据）等。其中，最主要的是Series和DataFrame两种数据结构。

### Series
Series是一种一维数组结构，它类似于NumPy中的ndarray对象，但比它更简单灵活。

- 创建Series
  ``` python
  # 创建整数型序列
  s = pd.Series([1, 3, 5, np.nan, 6, 8])
  
  # 创建字符串型序列
  data = ['a', 'b', 'c', None, 'e']
  s = pd.Series(data)

  # 创建带索引的序列
  index = ['a', 'b', 'c', 'd', 'e', 'f']
  s = pd.Series(['A', 'B', 'C', 'D', 'E'], index=index)
  ```
- Series属性
  - values: 获取序列值
  - index: 获取序列索引
  - name: 设置/获取序列名称

- 操作Series
  - 访问元素
    ``` python
    print(s[0])     # 获取第一个元素
    print(s[-1])    # 获取最后一个元素

    # 使用标签访问元素
    print(s['b'])   # 获取索引为'b'的值
    
    # 使用位置访问元素
    print(s[2])     # 获取第3个元素
    ```
  - 查询元素
    ``` python
    mask = (s > 5) & (s < 7)   # 查询序列中大于5且小于7的元素
    s_new = s[mask]            # 返回满足条件的新序列
    print(s_new)               # [6]
    ```
  - 插入元素
    ``` python
    # 在最后插入元素
    s.iloc[-1] = 10
    
    # 在指定位置插入元素
    s.iloc[2] = 20
    
    # 同时插入多个元素
    s[[1, 2]] = [30, 40]
    ```
  
### DataFrame
DataFrame是二维表格结构，可以看做由多个Series组成的字典。

- 创建DataFrame
  ``` python
  # 从列表创建DataFrame
  data = {'name': ['Alice', 'Bob', 'Charlie'],
          'age': [25, 30, 35],
          'city': ['New York', 'San Francisco', 'Chicago']}
  df = pd.DataFrame(data)
  
  # 从字典创建DataFrame
  d = [{'name': 'Alice', 'age': 25, 'city': 'New York'},
       {'name': 'Bob', 'age': 30, 'city': 'San Francisco'}]
  df = pd.DataFrame(d)
  
  # 从Numpy矩阵创建DataFrame
  arr = np.random.rand(5, 3)
  df = pd.DataFrame(arr, columns=['col1', 'col2', 'col3'])
  
  # 从CSV文件读取DataFrame
  df = pd.read_csv('example.csv')
  
  # 从Excel文件读取DataFrame
  df = pd.read_excel('example.xlsx', sheet_name='Sheet1')
  ```
  
- DataFrame属性
  - columns: 获取列名列表
  - index: 获取行索引列表
  - shape: 获取行数、列数

- 操作DataFrame
  - 访问元素
    ``` python
    print(df['name'][1])       # 获取第二行的姓名
    print(df[['name', 'age']][1:])        # 获取第二到倒数第二行的姓名和年龄
    
    # 根据标签访问元素
    print(df.loc[1, 'name'])         # 获取第二行的姓名
    print(df.loc[:, ['name', 'age']][:3])      # 获取前三行的姓名和年龄
    
    # 根据位置访问元素
    print(df.iat[1, 1])             # 获取第二行第二列的值
    ```
  - 查询元素
    ``` python
    # 使用条件表达式查询元素
    mask = (df['age'] >= 28) & (df['gender'] == 'M')
    df_sub = df[mask]
    
    # 对DataFrame进行分组聚合
    grouped = df.groupby('category')['value'].mean()
    ```
  - 插入元素
    ``` python
    # 在最后插入新行
    new_row = {'name': 'Dave', 'age': 40, 'city': 'Los Angeles'}
    df.loc[len(df)] = new_row
    
    # 在指定位置插入新行
    new_rows = [{'name': 'Eve', 'age': 35, 'city': 'Seattle'},
                {'name': 'Frank', 'age': 45, 'city': 'San Diego'}]
    df.loc[2:2, :] = new_rows
    
    # 在最后插入新列
    col_values = [10, 20, 30]
    df['score'] = col_values
    
    # 删除列
    del df['age']
    ```
    
# 2.基本概念术语
## 2.1 轴标签
轴标签即每个条目的名称或编号。在Pandas中，索引、列标签都是轴标签。在不同的情况下，索引与列标签可以是不同的值。

## 2.2 数据框
数据框(dataframe)，是2D结构化数组的名称。数据框中的每一列代表着变量，每一行代表着观测值。数据框除了用来存储数据外，还可以存储相关的描述性信息。

## 2.3 数据类型
数据类型(dtype)表示数据的内部结构和表示方式。在Pandas中，可以通过`dtypes`属性查看数据类型。数据类型有以下几种：

- object: 表示字符数据类型；
- int64: 表示整数数据类型；
- float64: 表示浮点数数据类型；
- datetime64: 表示日期时间数据类型。

## 2.4 缺失值
缺失值(missing value)表示缺少有效值，通常用NaN(Not a Number)表示。