
作者：禅与计算机程序设计艺术                    

# 1.简介
         

NumPy（读音为/nɛmˈpædi/）是一个第三方库，用于科学计算，它提供了多维数组对象ndarray以及相关工具。它的目的是提供高效率的矢量化数组运算能力，主要用于进行矩阵运算、线性代数、统计建模等领域。Numpy也是SciPy（Scientific Python）中的基础库，它同样可以对图像和信号进行处理。

NumPy虽然易于使用，但是熟练掌握它还是非常重要的。因为许多Python的第三方库都依赖于NumPy，比如Pandas、TensorFlow、scikit-learn等。

本文将为您详细介绍NumPy的功能及使用方法。

# 2.环境准备
本文假定读者已经具备以下知识点：

- 安装Python以及Anaconda包管理器；
- 有一定的Python编程经验；
- 对数学和线性代数有一定了解。

首先安装Anaconda或者Miniconda，并在命令行中运行如下指令创建环境：

```bash
conda create -n numpy python=3.7
```

然后激活刚才创建的numpy环境：

```bash
conda activate numpy
```

然后通过pip安装NumPy：

```bash
pip install numpy
```

激活后，命令行输入`python`，进入Python交互模式：

```python
>>> import numpy as np
```

接下来测试一下是否安装成功：

```python
>>> print(np.__version__)
1.19.2
```

如果显示版本号，则安装成功。

# 3.核心概念
## 3.1 Ndarray（多维数组对象）
NumPy最重要的数据结构是ndarray（多维数组），它是一个快速且节省空间的数据类型，支持快速的元素级操作，同时也能够与标准的Python序列兼容。它具有以下几个特征：

1. 数据大小固定：NumPy数组具有固定的内存大小和形状，这就意味着数组的大小不能改变，除非分配新的内存。
2. 元素类型相同：所有元素必须是相同类型的，例如整数、浮点数或复数等。
3. 高效的矢量化运算：NumPy提供的大量数学函数都是矢量化的，也就是说，他们作用到数组中的每一个元素上，使得计算速度更快。
4. 内存连续存储：为了提升性能，NumPy采用了内存连续存储方式。

## 3.2 Array creation functions
### 3.2.1 arange()
arange()函数类似于range()函数，但它会创建一个NumPy数组。

语法：
```python
numpy.arange([start], stop[, step, dtype])
```

参数说明：
- start：起始值，默认值为0。
- stop：终止值（不包含）。
- step：步长，默认为1。
- dtype：数据类型，默认为float。

示例：
```python
>>> np.arange(5)
array([0.,  1.,  2.,  3.,  4.])
```

```python
>>> np.arange(2, 10, 2)
array([2.,  4.,  6.,  8.])
```

### 3.2.2 ones(), zeros() 和 empty()
ones()、zeros()和empty()函数分别用于创建全1、全0和未初始化的数组。

语法：
```python
numpy.ones((r1, r2,..., rn), dtype)
numpy.zeros((r1, r2,..., rn), dtype)
numpy.empty((r1, r2,..., rn))
```

参数说明：
- (r1, r2,..., rn):数组的形状。
- dtype:数据类型。

示例：
```python
>>> np.ones((3, 2))
array([[1., 1.],
[1., 1.],
[1., 1.]])
```

```python
>>> np.zeros((3, 2))
array([[0., 0.],
[0., 0.],
[0., 0.]])
```

```python
>>> np.empty((3, 2)) #返回的数组是未初始化的，里面是一些随机的数据。
array([[ 0.4693118, -0.85579467],
[-0.29072099,  0.92290979],
[ 0.22949029,  0.3023996 ]])
```

注意：empty()函数只分配数组所需的内存，不会对其填充任何值。

### 3.2.3 linspace() 函数
linspace()函数创建了一个等间隔的数组，即把给定的范围分成指定个数的段，然后按指定间隔返回这些段的值。

语法：
```python
numpy.linspace(start, stop, num, endpoint, retstep, dtype)
```

参数说明：
- start：起始值。
- stop：终止值。
- num：要生成的元素数量。
- endpoint：布尔值，表示是否包括终止值。
- retstep：布尔值，表示是否同时返回步长。
- dtype：数据类型。

示例：
```python
>>> np.linspace(2.0, 3.0, num=5)
array([2.       , 2.28571429, 2.57142857, 2.85714286, 3.        ])
```

```python
>>> np.linspace(2.0, 3.0, num=5, endpoint=False)
array([2.      , 2.21428571, 2.42857143, 2.64285714, 2.85714286])
```

```python
>>> a = np.linspace(2.0, 3.0, num=5)
>>> b = np.linspace(2.0, 3.0, num=5, retstep=True)
>>> c = np.linspace(2.0, 3.0, num=5, endpoint=False, retstep=True)
>>> print(a, '\n')
[2.        2.28571429 2.57142857 2.85714286 3.        ] 

>>> print(b, '\n')
(array([2.       , 2.28571429, 2.57142857, 2.85714286, 3.        ]), 0.2857142857142857) 

>>> print(c, '\n')
(array([2.      , 2.21428571, 2.42857143, 2.64285714, 2.85714286]), 0.21428571428571427) 
```

### 3.2.4 array() 函数
array()函数用于将列表转换为NumPy数组。

语法：
```python
numpy.array(object, dtype=None, copy=True, order='K', subok=False, ndmin=0)
```

参数说明：
- object：列表、元组或其他可迭代对象。
- dtype：数据类型，默认为float。
- copy：布尔值，表示是否复制，默认为True。
- order：'C'或'F'，默认为'K'。'C'表示按列优先存储，'F'表示按行优先存储。
- subok：布尔值，表示子类是否要被传递，默认为False。
- ndmin：指定生成数组的最小维度，默认为0。

示例：
```python
>>> lst = [[1, 2], [3, 4]]
>>> arr = np.array(lst)
>>> arr
array([[1, 2],
[3, 4]])
```

```python
>>> tup = ((1, 2), (3, 4))
>>> arr = np.array(tup)
>>> arr
array([[1, 2],
[3, 4]])
```

```python
>>> lis_str = ['apple', 'banana']
>>> arr = np.array(lis_str)
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
ValueError: setting an array element with a sequence.
```

```python
>>> arr = np.array([(1, 2)],dtype=[('x',int), ('y', float)])
>>> arr
array([(1, 2.)],
dtype=[('x', '<i4'), ('y', '<f8')])
```

### 3.2.5 fromfunction() 函数
fromfunction()函数接受一个函数作为参数，根据这个函数来创建数组。

语法：
```python
numpy.fromfunction(function, shape, **kwargs)
```

参数说明：
- function：接收一个坐标元组(或单独的一个坐标)作为参数，并返回该位置上的值。
- shape：数组的形状。
- **kwargs：传递给function函数的其他参数。

示例：
```python
>>> def myfunc(i, j):
return i**2 + j
>>> arr = np.fromfunction(myfunc, (3, 3))
>>> arr
array([[ 0,  1,  2],
[ 3,  4,  5],
[ 6,  7,  8]])
```

```python
>>> def f(x, y):
if x >= 0 and x <= 1 and y >= 0 and y<= 1:
return True
else:
return False
>>> arr = np.fromfunction(f, (4, 4), dtype=int)
>>> arr
array([[0, 0, 0, 0],
[0, 1, 1, 0],
[0, 1, 1, 0],
[0, 0, 0, 0]], dtype=int32)
```

### 3.2.6 random 模块
random模块提供了很多创建随机数的方法。

#### 3.2.6.1 rand() 和 randn() 方法
rand()方法用来产生均匀分布的随机数，randn()方法用来产生正态分布的随机数。

语法：
```python
numpy.random.rand(*shape)
numpy.random.randn(*shape)
```

参数说明：
- *shape：表示输出的数组形状。

示例：
```python
>>> np.random.rand(2, 3)
array([[0.77238934, 0.33821277, 0.4291131 ],
[0.44729893, 0.95482627, 0.05117377]])
```

```python
>>> np.random.randn(2, 3)
array([[ 1.54154342,  0.20824133,  0.33101845],
[ 1.74461259,  0.84289092, -1.47991233]])
```

#### 3.2.6.2 randint() 方法
randint()方法用来产生指定范围内的均匀分布的整数。

语法：
```python
numpy.random.randint(low, high=None, size=None, dtype='l')
```

参数说明：
- low：范围下限，包括此值。
- high：范围上限，不包括此值。
- size：输出的数组形状。
- dtype：输出的数据类型，默认为整型。

示例：
```python
>>> np.random.randint(5, size=(2, 3))
array([[4, 4, 2],
[0, 4, 0]])
```

```python
>>> np.random.randint(-3, 3, size=(2, 3))
array([[-1,  2,  0],
[ 0, -1,  2]])
```

#### 3.2.6.3 choice() 方法
choice()方法用来从一个数组中随机抽取指定个数个元素。

语法：
```python
numpy.random.choice(a, size=None, replace=True, p=None)
```

参数说明：
- a：一个1-D或2-D数组，表示可能选取的元素集合。
- size：输出的数组形状。
- replace：布尔值，表示是否可以重复选择。
- p：一维数组，表示每个元素出现的概率。

示例：
```python
>>> a = np.array([1, 2, 3, 4, 5])
>>> np.random.choice(a, 3)
array([4, 1, 2])
```

```python
>>> a = np.array([[1, 2, 3],[4, 5, 6]])
>>> np.random.choice(a, (2, 3))
array([[2, 6, 3],
[5, 2, 1]])
```

```python
>>> a = np.array([1, 2, 3])
>>> p = np.array([0.1, 0.1, 0.8])
>>> np.random.choice(a, 2, replace=False, p=p)
array([1, 3])
```

### 3.2.7 loadtxt() 和 savetxt() 函数
loadtxt()函数从文件或字符串加载数据，savetxt()函数将数据保存到文件或字符串中。

语法：
```python
numpy.loadtxt(fname, dtype=<class 'float'>, comments='#', delimiter=None, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0)
numpy.savetxt(fname, X, fmt='%.18e', delimiter=' ', newline='\n', header='', footer='')
```

参数说明：
- fname：字符串，表示文件名。
- dtype：<class 'float'>或其他类型，表示数据的类型。
- comments：字符串，表示注释字符。
- delimiter：字符串，表示字段分割符。
- converters：字典，用于自定义数据类型转换函数。
- skiprows：整数，表示跳过前几行。
- usecols：整数或序列，表示读取哪些列。
- unpack：布尔值，表示是否将多个列拆开。
- ndmin：整数，表示数组的最小维度。

示例：
```python
>>> data = '''1 2 3
4 5 6'''
>>> np.loadtxt(StringIO(data))
array([[1., 2., 3.],
[4., 5., 6.]])
```

```python
>>> np.savetxt('/tmp/test.txt', np.arange(5).reshape(5, 1), '%d', '\n', '# ')
```

写入的文件的内容：
```
# 
0
1
2
3
4
```