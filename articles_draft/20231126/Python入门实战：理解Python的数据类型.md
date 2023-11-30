                 

# 1.背景介绍


Python数据类型可以分为以下几类：原始类型、容器类型、复合类型、可变类型、不可变类型。下面分别介绍一下每一种类型。
## 1.1 原始类型
原始类型包括数字、字符串、布尔值、空值和特殊值None。
### 1.1.1 数字类型
整数型 int 和浮点型 float 分别用来表示整数和小数。
```python
a = 123
b = -3.14
print(type(a), type(b)) # <class 'int'> <class 'float'>
```

如果希望结果保留精度，可以使用 decimal 模块中的 Decimal() 函数。例如：

```python
from decimal import Decimal

x = Decimal('3.14')
y = Decimal('-123e-2')
z = x * y
print(z)   # -39.75
```

布尔类型 bool 只能取True或False。
```python
flag = True
print(type(flag)) # <class 'bool'>
```

NoneType类型只能代表一个特殊的值None。它用于表示变量缺少有效值或者默认值。

```python
var1 = None
print(var1)       # None
print(type(var1)) # <class 'NoneType'>
```

### 1.1.2 字符类型 str

str类型用于表示字符串，支持多种编码方式。字符串的定义及基本操作如下：

```python
s1 = "Hello World"    # 定义字符串
print(len(s1))        # 获取长度

s2 = s1[::-1]         # 反转字符串

for ch in s1:
    print(ch)
        
if "World" in s1:
    print("Found!")
    
if not len(s1):
    print("Empty string")
    
if len(set(s1)):
    print("No repeated characters")
```

以上例子展示了字符串的一些基本操作方法，包括获取长度、反转、遍历、查找、判空、判断无重复字符等。

### 1.1.3 bytes 类型（原生bytes）

bytes 是 Python3 中新增的数据类型，它是用于存放二进制数据的类型。其基本操作与 str 类似，区别在于不能直接打印，需要通过 decode() 方法转换成对应的字符串后才能打印。示例如下：

```python
import base64

s = b'hello world'
print(s)              # b'hello world'

b_encode = base64.b64encode(s)     # 将bytes进行base64编码
print(b_encode)                   # b'aGVsbG8gd29ybGQ='

b_decode = base64.b64decode(b_encode)      # 将bytes进行base64解码
print(b_decode)                            # b'hello world'
```

### 1.1.4 list 类型

list类型是 Python 中最常用的一种容器类型。列表中的元素都有序排列，可以随时添加或删除元素。列表的基本操作如下：

```python
lst = [1, 2, 3, 4, 5]           # 创建列表

lst.append(6)                    # 添加元素到末尾

lst.insert(0, -1)                # 插入元素到指定位置

lst.pop(-2)                      # 删除指定位置的元素并返回

lst.remove(2)                    # 删除指定元素

lst += [7, 8, 9]                 # 合并两个列表

new_lst = lst + ['ten']          # 拼接列表

sub_lst = lst[:3]                # 切片

sorted_lst = sorted(lst)         # 对列表排序

lst.sort()                       # 不用返回值的排序

count = lst.count(1)             # 统计指定元素出现次数

max_val = max(lst)               # 求最大值

min_val = min(lst)               # 求最小值

sum_val = sum(lst)               # 求和
```

除了基本操作外，还可以使用列表解析语法简化操作，如：

```python
lst2 = [i**2 for i in range(5)]  # 计算列表每个元素的平方
```

### 1.1.5 tuple 类型

tuple类型也是一种容器类型，与列表不同的是，元组中元素是不能修改的。元组的基本操作如下：

```python
tup = (1, 2, 3, 4, 5)            # 创建元组

try:
    tup[1] = 2                     # 尝试修改元素会报错
except TypeError as e:
    print(e)                       # 不允许修改元素
    
    
t = ('apple', 'banana', 'orange') # 定义不定长元组

fruits = t + ('grape', )          # 连接元组


def get_fruit():                  # 生成器函数生成元组
    yield 'pear'
    yield 'watermelon'
    return 
    
fruits = get_fruit()             # 使用生成器函数生成元组
```

### 1.1.6 set 类型

set类型是一个无序集合，集合中的元素没有顺序，元素唯一且不可变。集合的基本操作如下：

```python
st = {1, 2, 3, 2}           # 创建集合

st.add(4)                    # 添加元素到集合

st.discard(2)                # 删除指定元素，若元素不存在则忽略错误

union_st = st | {4, 5, 6}    # 合并两个集合

intersection_st = st & {3, 4} # 交集

difference_st = st - {2, 4}   # 差集

symmetric_diff_st = st ^ {4, 5} # 对称差集

empty_set = set()            # 创建空集合
```

### 1.1.7 dict 类型

dict类型是一个键值对存储结构，键不可变，值可以是任意类型。字典的基本操作如下：

```python
d = {'name': 'Alice', 'age': 20}        # 创建字典

print(d['name'])                         # 通过键访问值

d['email'] = 'alice@example.com'          # 添加键值对

del d['age']                             # 删除指定键值对

key_lst = list(d.keys())                 # 获取所有键

value_lst = list(d.values())             # 获取所有值

item_lst = list(d.items())               # 获取所有键值对

copy_d = d.copy()                        # 深拷贝

inversed_d = dict((v, k) for k, v in copy_d.items())  # 翻转字典

default_val = d.get('phone', '-')         # 根据键获取值，若键不存在返回默认值
```

除此之外，还有 defaultdict 和 Counter 两种高级数据类型。