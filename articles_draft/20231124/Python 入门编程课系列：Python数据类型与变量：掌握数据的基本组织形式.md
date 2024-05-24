                 

# 1.背景介绍


## Python语言简介
Python是一种具有简单性、易用性、丰富库支持、跨平台、开源、可移植等优点的高级脚本语言。它的语法简洁，结构清晰，支持多种编程范式，是一个非常受欢迎的语言。
## 数据类型概览
在Python中，变量的本质就是对象的引用，所有的变量都存储在内存中。每个对象都有自己的类型。Python中的主要数据类型分为以下几类：
- Numbers(数字)
- Strings(字符串)
- Lists(列表)
- Tuples(元组)
- Dictionaries(字典)
- Sets(集合)
- Boolean(布尔值)
其中，数字包括整数型int、浮点型float和复数型complex；字符串是由0个或多个字符组成的序列，可以用单引号('...')或双引号("...")表示；列表是一组按顺序排列的值，可以包含不同的数据类型；元组类似于列表，但元素不能修改；字典是一组键-值对，其中键必须唯一，值可以取任何数据类型；集合是无序不重复元素的集；布尔值只有True和False两种取值。
## 基本变量类型
### 数字类型Number
Python提供了三种数字类型：整数（`int`）、浮点数（`float`）和复数（`complex`）。整数可以表示正负数，也可以表示无限大的整数（大于等于2^31 - 1小于=-2^31），但是不可以表示NaN（Not a Number）和INF（Infinity），即非数字和无穷大。浮点数是带小数点的实数，可以使用科学记数法表示，如1e+2表示100；复数则是由实部和虚部构成的数，可以使用a + bj或者complex(a,b)表示。
```python
# int类型示例
x = 10
y = -2000000000
z = 100000000000000000000000 # 会溢出
print(type(x))   # <class 'int'>
print(type(y))   # <class 'int'>
print(type(z))   # <class 'int'> 

# float类型示例
x = 3.14
y = 7.0/3.0
z = 1e-5     # 表示1*10^(-5)
w = 4E8      # 表示400000000.0
print(type(x))   # <class 'float'>
print(type(y))   # <class 'float'>
print(type(z))   # <class 'float'> 
print(type(w))   # <class 'float'>

# complex类型示例
x = 3 + 5j    # 表示3+5j
y = 4 - 9j
z = x * y     # (3-9j)*(4+5j) = (-7+32j)-(-9+2j) = 6-11j = z
print(type(x))   # <class 'complex'>
print(type(y))   # <class 'complex'>
print(type(z))   # <class 'complex'> 
```
### 字符串类型String
字符串是不可变序列，其中每一个元素都是字符（unicode字符）。字符串可以用单引号('...')或双引号("...")括起来，如果字符串内部有单引号或双引号，可以在字符串前面添加反斜杠转义。字符串的加法运算结果是拼接后的新字符串，乘法运算结果是重复拼接后的字符串。字符串也可以进行索引、切片、成员测试、串联、比较等运算。
```python
# 创建字符串示例
str1 = "Hello World!"
str2 = 'I\'m \"OK\"!'
str3 = '''This is the first line of string.
          This is the second line of string.'''
print(str1)            # Hello World!
print(str2)            # I'm "OK"!
print(str3)            # This is the first line of string.\n          This is the second line of string.

# 获取字符串长度、索引、切片、成员测试、串联、比较运算示例
s = "hello world"
print(len(s))        # 11
print(s[0])          # h
print(s[2:5])        # llo
if s in ["hello", "world"]:
    print("yes")
else:
    print("no")
new_s = " goodbye." + s[:5] + "," + str(3) + "."
print(new_s)         # hello goodbye.,3.
if new_s == "hello goodbye.,3.":
    print("yes")
else:
    print("no")
```
### 列表类型List
列表是一种可变序列，其中每一个元素可以是任意数据类型。列表可以进行索引、切片、追加、删除、插入、排序等操作。列表还可以通过其他序列（如字符串、元组、集合等）进行转换。
```python
# 创建列表示例
list1 = [1, "hello", True, 3.14]
list2 = []
list3 = list((1,2,"three"))
print(list1)           # [1, 'hello', True, 3.14]
print(list2)           # []
print(list3)           # ['one', 'two', 'three']

# 获取列表长度、索引、切片、追加、删除、插入、排序、类型转换等示例
l = [1, 3, 2, 4, 5]
print(len(l))           # 5
print(l[0])             # 1
print(l[-1])            # 5
l[0:2] = [-1,-2]       # 替换原子
print(l)                # [-1, -2, 2, 4, 5]
l.append([1,2,3])       # 添加元素到末尾
print(l)                # [-1, -2, 2, 4, 5, [1, 2, 3]]
l += [[1],[2],[3]]     # 拼接两个列表
print(l)                # [-1, -2, 2, 4, 5, [1, 2, 3], [1], [2], [3]]
l.remove([-1,-2])      # 删除第一个[-1,-2]元素
l.pop()                 # 删除最后一个元素
l.sort(reverse=True)    # 逆序排列
l.insert(1,'ok')       # 插入'ok'到索引1位置
t = tuple(l)            # 转换为元组
print(t)                # ([2, 4, 5, 'ok', [3]], [1], [2], [3])
l2 = set(['hello','world'])
l3 = list(l2)           # 转换为列表
print(l3)               # ['hello', 'world']
```
### 元组类型Tuple
元组是不可变序列，其中每一个元素可以是任意数据类型。元组可以进行索引、切片、连接、复制等操作。
```python
# 创建元组示例
tuple1 = ('apple', 2, 3.14, False)
tuple2 = ()
tuple3 = tuple(('apple', 2, 3.14, False))
print(tuple1)          # ('apple', 2, 3.14, False)
print(tuple2)          # ()
print(tuple3)          # ('apple', 2, 3.14, False)

# 获取元组长度、索引、切片、连接、复制、排序、类型转换等示例
t = (1, "hello", 3.14, True)
print(len(t))           # 4
print(t[0])             # 1
print(t[1:])            # ("hello", 3.14, True)
t = t + (1,)           # 在最后增加元素
t *= 2                  # 复制元组两次
print(t)                # (1, 'hello', 3.14, True, 1, 1, 'hello', 3.14, True)
t2 = sorted(t)[::-1]    # 对元组进行排序并倒叙
print(t2)               # ([1, 1, 3.14, 'hello', 'hello'], True, 1, 'hello', 3.14, 1, 'hello', 3.14, True)
u = ''.join(map(str, t2))   # 用空格拼接元组中的元素并转换为字符串
v = ''.join(["1","2"])      # 通过迭代器和字符串拼接字符串
print(u)                   # 1 1 3.14 hello hello True hello hello 3.14 True
print(v)                   # 12
```
### 字典类型Dictionary
字典是一种无序的键-值对集合，其中每一个键都是唯一的。字典可以进行索引、添加、删除、修改、遍历等操作。字典也是一种映射类型，可以用[]访问键对应的项。
```python
# 创建字典示例
dict1 = {'name': 'Alice', 'age': 25}
dict2 = {}
dict3 = dict([(1, 'one'), (2, 'two')])
print(dict1)           # {'name': 'Alice', 'age': 25}
print(dict2)           # {}
print(dict3)           # {1: 'one', 2: 'two'}

# 获取字典长度、键、值、添加、删除、修改、遍历等示例
d = {'apple': 2, 'banana': 3, 'orange': 4}
print(len(d))           # 3
print(d['apple'])       # 2
for key in d:
    print(key, d[key])   # apple 2 banana 3 orange 4
d['pear'] = 1
del d['banana']
d['grape'] = 2
print(sorted(d.keys()))   # ['apple', 'orange', 'pear', 'grape']
for k, v in sorted(d.items()):
    print(k, v)            # apple 2 grape 2 orange 4 pear 1
```
### 集合类型Set
集合是无序不重复元素的集。集合可以进行并、差、交、并置、子集判断等操作。集合可以用来实现各种关系型数据库中“集合”的概念。
```python
# 创建集合示例
set1 = {1, 2, 3, 4, 5}
set2 = set([1, 2, 2, 3, 4, 4, 4])
set3 = frozenset({1, 2})
print(set1)            # {1, 2, 3, 4, 5}
print(set2)            # {1, 2, 3, 4}
print(set3)            # frozenset({1, 2})

# 集合操作示例
s1 = {1, 2, 3}
s2 = {2, 3, 4}
print(s1 & s2)         # {2, 3}
print(s1 | s2)         # {1, 2, 3, 4}
print(s1 - s2)         # {1}
print(s1 ^ s2)         # {1, 4}
print(s1 <= s2)        # True
print(s1 >= s2)        # False
print({'hello','world'}<{'goodbye','universe'})    # True
```