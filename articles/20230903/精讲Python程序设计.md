
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 1.1 Python简介

Python（英国发音：/paɪθən/）是一种面向对象的高级编程语言，由Guido van Rossum创造于1989年。它最初被称为“胶水语言”，因为它的语法吸收了C、Java和Perl的一些特点。由于其简单易学、直观运行速度快、可移植性强等特点，成为了许多大型机和网络服务端脚本语言的首选。

## 1.2 为什么要精讲Python？

2017年，全球Python社区用户突破3亿，并在软件开发领域广泛应用。不少公司和政府部门都对Python技术积极投入。如今，Python已经成为当今最热门的计算机语言之一，被认为可以用来进行各种机器学习、数据分析、Web开发等任务。

“精讲”Python，即以系统的知识结构和循序渐进的学习方式，探索Python的各项特性、优势、功能及应用，帮助读者更好地理解Python的作用和发展方向。

## 1.3 目标读者

1、具有一定编程经验的人员；
2、了解一些计算机基础知识，包括计算机组成原理、数据结构和算法等；
3、具备独立解决实际问题能力和决心，以及良好的动手能力。


# 2.Python基本概念术语说明
## 2.1 Python变量类型
### 2.1.1 整数型int
整数型int，在Python中用int表示，其值得范围和大小依赖于机器的内存容量。比如一个长整形的int数字的值可能超过了当前平台所能存储的最大整数值。

```python
a = 123     # int (integer) number
b = -456    # negative integer numbers are also supported
c = 0       # zero is an integer too
d = 10**100 # a large integer can be created by using scientific notation
```

### 2.1.2 浮点型float
浮点型float，也叫做浮点数或实数，在Python中用float表示，表示范围比整数型的更大。

```python
e = 3.14      # float (floating-point number)
f = -9.81     # another floating-point number with negative value
g = 4.5e-3    # exponential notation can be used for small floating-point values
h = 1.0 + e   # addition of float and int results in float as the result type
i = g / h     # division always returns a float
j = round(i)  # use built-in function "round" to perform rounding to nearest whole number
k = i // j    # floor division operator "//" rounds down to nearest whole number 
l = abs(-4.5) # absolute value of a float or complex number can be computed using built-in function "abs()" 
m = pow(2,3)  # compute powers of integers using built-in function "pow()"
```

### 2.1.3 复数型complex
复数型complex，在Python中用complex()函数或后缀j或J表示，表示两个浮点数相加得到的结果。

```python
n = 3+4j         # create a complex number
o = n.real        # extract real part of a complex number
p = n.imag        # extract imaginary part of a complex number
q = o*o + p*p    # compute modulus squared of a complex number
r = abs(n)**2     # compute modulus square directly without extracting parts
s = 1j            # the unit imaginary number can be written as j instead of im
t = s**2          # compute the square of the imaginary unit using exponentiation operator "**"
u = complex(5,-3) # create a complex number from its real and imaginary components
v = u + n         # add two complex numbers together
w = u * v         # multiply two complex numbers together
x = u / v         # divide one complex number by another (note: this performs complex conjugate if needed)
y = n ** 3        # raise a complex number to a power (this uses the standard algorithm)
z = z.conjugate() # get the complex conjugate of a complex number
```

### 2.1.4 布尔型bool
布尔型bool，也叫逻辑型boolean，在Python中用True或False表示。

```python
is_raining = True           # boolean variable initialized to True
has_umbrella = False        # another boolean variable initialized to False
can_purchase_car = not has_umbrella # negation operation can be performed on booleans
drink_water = bool(1)       # convert any nonzero value into True, otherwise it's False
```

### 2.1.5 字符串型str
字符串型str，也叫串，用于表示文本数据，用单引号(')或者双引号(")括起来的一系列字符。

```python
name = 'John'              # string variable initialized to John
age = "32"                 # string variable initialized to 32
message = "Hello, World!"  # double quotes can be used interchangeably with single quotes
sentence = name + " is " + str(age) + " years old."  # concatenate strings using operators '+' and ','
```

### 2.1.6 None类型None
None类型，在Python中用关键字None表示，表示空值或缺失值。

```python
nothing = None             # initialize a variable to represent absence of data
result = x if x > y else y # conditional expression can return either x or y based on condition x>y
```

### 2.1.7 数据容器类型list、tuple、set、dict
#### list类型list
列表型list，也叫序列，是一个动态大小的、有序、可变的、元素可以重复的容器。

```python
fruits = ["apple", "banana", "cherry"]  # define a list containing fruits
numbers = [1, 2, 3]                     # define a list containing numbers
empty_list = []                         # empty list initialization
```

#### tuple类型tuple
元组型tuple，也叫有序列表，类似于列表，但是元素不能修改，只能读取。

```python
coordinates = (4, 5)                    # define a tuple containing coordinates
dimensions = (3, 4)                     # define another tuple containing dimensions
empty_tuple = ()                        # empty tuple initialization
```

#### set类型set
集合型set，也叫无序集合，是一个动态大小的、无序、可变的、元素不可重复的容器。

```python
unique_fruits = {"apple", "banana"}     # define a set containing unique fruits
mixed_colors = {"red", "green", "blue"} # define another set containing mixed colors
empty_set = set()                       # empty set initialization
```

#### dict类型dict
字典型dict，也叫关联数组，是一个动态大小的、可变的、无序的、键值对的容器。

```python
person = {'name': 'Alice', 'age': 25}       # define a dictionary containing person details
empty_dict = {}                            # empty dictionary initialization
```

## 2.2 Python运算符
Python支持多种运算符，包括算术运算符、关系运算符、赋值运算符、逻辑运算符、位运算符等。下表列出了Python中主要的运算符。

| 运算符 | 描述 | 示例 |
|---|---|---|
| `=` | 简单的赋值操作符 | c = a + b 将计算结果赋值给c |
| `+=` | 增量赋值运算符 | c += a 等价于 c = c + a |
| `-=` | 减量赋值运算符 | c -= a 等价于 c = c - a |
| `/=` | 除法赋值运算符 | c /= a 等价于 c = c / a |
| `//=` | 整数除法赋值运算符 | c //= a 等价于 c = c // a |
| `%=` | 模ulo赋值运算符 | c %= a 等价于 c = c % a |
| `**=` | 幂赋值运算符 | c **= a 等价于 c = c ** a |
| `&=` | 按位与赋值运算符 | c &= a 等价于 c = c & a |
| `^=` | 按位异或赋值运算符 | c ^= a 等价于 c = c ^ a |
| `<<=` | 左移赋值运算符 | c <<= a 等价于 c = c << a |
| `>>=` | 右移赋值运算符 | c >>= a 等价于 c = c >> a |
| `<` | 小于比较运算符 | a < b 返回True如果a小于b |
| `>` | 大于比较运算符 | a > b 返回True如果a大于b |
| `<=` | 小于等于比较运算符 | a <= b 返回True如果a小于等于b |
| `>=` | 大于等于比较运算符 | a >= b 返回True如果a大于等于b |
| `==` | 等于比较运算符 | a == b 返回True如果a等于b |
| `!=` | 不等于比较运算符 | a!= b 返回True如果a不等于b |
| `not` | 逻辑非运算符 | not a 返回True如果a为False，否则返回False |
| `and` | 逻辑与运算符 | a and b 如果a为True且b为True，则返回True，否则返回False |
| `or` | 逻辑或运算符 | a or b 如果a为True或b为True，则返回True，否则返回False |
| `in` | 成员运算符 | a in b 返回True如果a是b的成员，否则返回False |
| `is` | 对象身份运算符 | a is b 如果a和b引用同一个对象，则返回True，否则返回False |
| `+=` | 增量赋值运算符 | c += a 是先计算a再赋值给c，等价于 c = c + a |
| `-=` | 减量赋值运算符 | c -= a 是先计算a再赋值给c，等价于 c = c - a |
| `*=` | 乘法赋值运算符 | c *= a 是先计算a再赋值给c，等价于 c = c * a |
| `@=` | 矩阵乘法赋值运算符 | c @= a 是先计算a再赋值给c，等价于 c = c @ a |
| `/=` | 除法赋值运算符 | c /= a 是先计算a再赋值给c，等价于 c = c / a |
| `//=` | 整数除法赋值运算符 | c //= a 是先计算a再赋值给c，等价于 c = c // a |
| `%=` | 模ulo赋值运算符 | c %= a 是先计算a再赋值给c，等价于 c = c % a |
| `**=` | 幂赋值运算符 | c **= a 是先计算a再赋值给c，等价于 c = c ** a |
| `&=` | 按位与赋值运算符 | c &= a 是先计算a再赋值给c，等价于 c = c & a |
| `^=` | 按位异或赋值运算符 | c ^= a 是先计算a再赋值给c，等价于 c = c ^ a |
| `|=`| 按位或赋值运算符 | c |= a 是先计算a再赋值给c，等价于 c = c \| a |
| `<<=` | 左移赋值运算符 | c <<= a 是先计算a再赋值给c，等价于 c = c << a |
| `>>=` | 右移赋值运算符 | c >>= a 是先计算a再赋值给c，等价于 c = c >> a |
| `:=` | 定义运算符 | x := 2 在声明语句中将x初始化为2 |