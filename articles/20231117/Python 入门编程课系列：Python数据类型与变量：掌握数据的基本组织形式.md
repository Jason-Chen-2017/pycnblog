                 

# 1.背景介绍


“数据”这个词在计算机科学领域是一个十分重要的词汇。如何高效地存储、处理和使用数据，是每个工程师都需要熟练掌握的技能之一。而对于数据在计算机内存中的组织形式，很多初级开发者并不知道，更不要说是高级开发者了。这是非常重要的一点知识。

本文主要介绍Python中常用的数据类型（int、float、str、list、tuple、set、dict）及其变量。了解这些数据类型的不同存储方式，能够帮助读者更好地理解和应用它们。同时，还将结合编程语言的一些特点，介绍如何应用这些数据类型来解决实际问题。
# 2.核心概念与联系
## 2.1 Python 数据类型概述
Python 是一种具有动态类型特征的高级编程语言，它支持多种数据类型。以下是 Python 中常用的几种数据类型：

1. int （整数）
2. float （浮点数）
3. str （字符串）
4. list （列表）
5. tuple （元组）
6. set （集合）
7. dict （字典） 

## 2.2 Python 数据类型之间有什么区别？
Python 中的数据类型并不是一成不变的，每一种数据类型都有自己的特性、功能以及应用场景。下面对几个数据类型进行详细介绍。

### 2.2.1 int (整数)
int 表示整数，也称为整型。整数可以表示正数、负数、零等。例如：1、-9、0等。Python 中的整数类型包括 bool （布尔值），它的取值为 True 或 False。

语法如下：

```python
num = 1   # 赋值整数值 1 
print(type(num))    # <class 'int'>
```

#### 整数运算

整数的四则运算支持加减乘除法，以及取模（余数）。

| 运算符 | 描述       | 实例           | 结果     |
| ------ | ---------- | -------------- | -------- |
| +      | 加法       | x + y          | 相加两个对象，如果有任意一个对象是字符串，那么另外一个对象也要转换成字符串 |
| -      | 减法       | x - y          | 从第一个对象中减去第二个对象，如果有任意一个对象是字符串，那么另外一个对象也要转换成字符串 |
| *      | 乘法       | x * y          | 返回给定参数的乘积 |
| /      | 除法       | x / y          | 返回商的整数部分 |
| %      | 求模运算符 | x % y          | 返回除法运算后的余数，即除法运算的模 |
| **     | 幂         | pow(x,y)       | 返回 x 的 y 次方 |

#### 其他函数

可以使用内置函数 `abs()` 来获取数字的绝对值：

```python
a = abs(-3)    # a 为 3
b = abs(4.5)   # b 为 4.5
c = abs("abc") # c 为报错
```

还可以使用 `divmod()` 函数来计算两个数的商和余数：

```python
quotient, remainder = divmod(17, 3)    # quotient 为 5，remainder 为 2
```

### 2.2.2 float (浮点数)
float 表示小数或复数，也称为浮点型。浮点数用于表示无法精确表示整数的小数。例如：3.14、1.5、-2.5等。Python 中的浮点类型可以表示无穷大、NaN（Not A Number，非数值）等特殊数值。

语法如下：

```python
num_float = 3.14   # 赋值浮点值 3.14
print(type(num_float))    # <class 'float'>
```

#### 浮点数运算

浮点数的四则运算与整数相同，但存在一些差异：

1. 当遇到除法运算时，如果除数为零，会出现 ValueError；
2. 如果分母很小，则可能会丢失有效数字。

#### 其他函数

除了一般的四则运算，还有一些其他函数可供使用。如：`round()` 函数用于返回浮点数舍入到指定位数的值：

```python
x = round(3.1415926, 2)    # x 为 3.14
```

`isfinite()`、`isnan()` 和 `isinf()` 函数用来判断浮点数是否为无穷大、NaN 或正无穷大/负无穷大：

```python
import math

a = float('nan')   # 定义 NaN 值
b = math.inf        # 获取正无穷大值
c = -math.inf       # 获取负无穷大值

if math.isfinite(a):
    print("a is finite.")
elif math.isnan(a):
    print("a is nan.")
    
if math.isinf(b):
    print("b is positive infinity.")
else:
    print("b is not positive infinity.")
    
if math.isinf(c):
    print("c is negative infinity.")
else:
    print("c is not negative infinity.")
```

### 2.2.3 str (字符串)
str 表示字符串，是由单引号（'）或双引号（"）括起来的任意文本，比如："hello world" 。字符串是不可更改的，因此不能对字符串的单个字符进行修改。

#### 创建字符串

创建字符串的方法有两种：第一种是直接在引号之间输入字符串内容；第二种是通过调用内置函数 `join()` 将序列中的元素以指定的字符连接起来。

```python
s = "Hello World!"
t = "-".join([i for i in range(1, 6)])
u = ", ".join(["apple", "banana", "orange"])
v = "".join(["Hi", " ", "there!"])
w = "{} {} {}".format("Welcome", "to", "our course!")
```

#### 字符串运算

字符串支持拼接（+）、重复（*）和切片（[]）运算。但是，对于长字符串来说，切片操作会比较耗时。

#### 查找子串

查找子串的最简单方法是使用 `find()` 方法。该方法从字符串中寻找指定子串的第一次出现的位置。如果没有找到子串，则返回 `-1`。

```python
s = "hello python"
pos = s.find("p")    # pos 为 6
sub_s = s[pos:]     # sub_s 为 "python"
```

#### 替换子串

替换子串最简单的方法是使用 `replace()` 方法。该方法将指定子串替换为另一个子串。

```python
s = "hello python"
new_s = s.replace("p", "*")    # new_s 为 "hell*ython"
```

#### 删除空白字符

删除空白字符最简单的方法是使用 `strip()` 方法。该方法会移除字符串两端的所有空白字符。

```python
s = "\n\r hello \t there \n "
s = s.strip()    # s 为 "hello \t there"
```

#### 分割字符串

分割字符串最简单的方法是使用 `split()` 方法。该方法根据指定字符将字符串分割成多个子串。

```python
s = "apple, banana, orange"
fruits = s.split(", ")    # fruits 为 ["apple", "banana", "orange"]
```

#### 对齐字符串

对齐字符串最简单的方法是使用 `ljust()`, `center()`, `rjust()` 方法。

```python
name = "John Doe"
left = name.ljust(10)    # left 为 "John Doe     "
right = name.rjust(10)   # right 为 "     John Doe"
centered = name.center(10)    # centered 为 "  John Doe  "
```

### 2.2.4 list (列表)
list 是 Python 中最常用的一类数据结构。它可以存储多个数据项，并且可以按索引访问各个数据项。列表中的所有数据项必须具有相同的类型，通常情况下都是同一类型的元素。

#### 创建列表

创建列表的最简单的方式是直接在方括号 [] 中输入元素，或者通过调用内置函数 `list()` 将序列转换成列表。

```python
lst = [1, 2, 3]   # 直接创建列表
lst2 = list(range(1, 6))   # 通过 range 生成列表
```

#### 添加元素

向列表添加元素有两种方式：第一种是直接使用 `append()` 方法，该方法在列表末尾添加一个元素；第二种是使用赋值语句来添加一个元素至指定位置。

```python
lst = [1, 2, 3]
lst.append(4)    # lst 为 [1, 2, 3, 4]
lst[len(lst)] = 5    # lst 为 [1, 2, 3, 4, 5]
```

#### 访问元素

列表中的元素可以通过索引访问，其中索引从 0 开始，从左往右依次递增。也可以使用负索引从右往左访问。

```python
lst = [1, 2, 3, 4, 5]
first = lst[0]    # first 为 1
last = lst[-1]    # last 为 5
third = lst[2]    # third 为 3
```

#### 修改元素

列表中的元素也可以被修改。修改某个元素的语法类似于访问某个元素的语法。

```python
lst = [1, 2, 3, 4, 5]
lst[2] = 6    # lst 为 [1, 2, 6, 4, 5]
```

#### 删除元素

从列表中删除元素有两种方式：第一种是使用 `del` 关键字，该关键字根据索引删除元素；第二种是使用 `remove()` 方法，该方法根据值的前后顺序删除首个匹配的元素。

```python
lst = [1, 2, 3, 4, 5]
del lst[2]    # lst 为 [1, 2, 4, 5]
lst.remove(2)    # lst 为 [1, 4, 5]
```

#### 排序

列表可以排序，可以通过 `sort()` 方法实现。`reverse=True` 参数用来倒序排序。

```python
lst = [3, 1, 4, 2, 5]
lst.sort()    # lst 为 [1, 2, 3, 4, 5]
lst.sort(reverse=True)    # lst 为 [5, 4, 3, 2, 1]
```

#### 合并列表

列表可以被合并，可以使用 `extend()` 方法或 `+` 操作符。

```python
lst1 = [1, 2, 3]
lst2 = [4, 5, 6]

lst1.extend(lst2)    # lst1 为 [1, 2, 3, 4, 5, 6]
lst3 = lst1 + lst2    # lst3 为 [1, 2, 3, 4, 5, 6]
```

#### 遍历列表

列表可以被迭代，可以使用 `for...in` 循环。也可以使用 `enumerate()` 函数获得索引和对应值。

```python
lst = [1, 2, 3, 4, 5]
for num in lst:
    print(num)
    
for idx, val in enumerate(lst):
    print(idx, val)
```

#### 列表切片

列表可以通过切片操作来得到子列表。

```python
lst = [1, 2, 3, 4, 5]
sub_lst = lst[:3]    # sub_lst 为 [1, 2, 3]
```

#### 列表相关函数

列表还提供了一些函数用于操纵列表，如 `count()`、`index()` 和 `pop()`。

```python
lst = [1, 2, 3, 2, 4, 5, 2]
count = lst.count(2)    # count 为 3
idx = lst.index(2)    # idx 为 1
lst.pop(2)    # pop 掉索引为 2 的元素
```

### 2.2.5 tuple (元组)
tuple 是另一种不可变容器数据类型，与列表类似，但是它是只读的。元组中的元素可以是任何类型，且元组在创建之后就不能再改变。

#### 创建元组

创建元组的最简单的方式是直接在圆括号 () 中输入元素，或者通过调用内置函数 `tuple()` 将序列转换成元组。

```python
tup = (1, 2, 3)   # 直接创建元组
tup2 = tuple(range(1, 6))   # 通过 range 生成元组
```

#### 访问元素

元组中的元素可以通过索引访问，与列表类似，索引也是从 0 开始，从左往右依次递增。

```python
tup = (1, 2, 3, 4, 5)
first = tup[0]    # first 为 1
second = tup[1]    # second 为 2
```

#### 修改元组

尝试修改元组会导致错误，因为元组是不可变的。但是，可以将元组转换成列表来修改。

#### 删除元素

元组中不能删除元素，但可以重新创建一个新的元组。

#### 元组相关函数

元组还提供了一些函数用于操纵元组，如 `count()`、`index()`。

```python
tup = (1, 2, 3, 2, 4, 5, 2)
count = tup.count(2)    # count 为 3
idx = tup.index(2)    # idx 为 1
```

### 2.2.6 set (集合)
set 是 Python 中另一种基础数据类型，与列表和元组不同，它是无序的和不可重复的。集合中的元素可以是任何类型，而且元素不能重复。

#### 创建集合

创建集合的最简单的方式是直接在花括号 {} 中输入元素，或者通过调用内置函数 `set()` 将序列转换成集合。

```python
st = {1, 2, 3}   # 直接创建集合
st2 = set(range(1, 6))   # 通过 range 生成集合
```

#### 添加元素

向集合添加元素有两种方式：第一种是直接使用 `add()` 方法，该方法将元素加入到集合中；第二种是使用赋值语句来将元素赋给集合。

```python
st = {1, 2, 3}
st.add(4)    # st 为 {1, 2, 3, 4}
```

#### 访问元素

集合中的元素只能通过循环来访问，由于集合是无序的，所以每次输出的顺序可能不同。

```python
st = {1, 2, 3, 4, 5}
for elem in st:
    print(elem)
```

#### 修改集合

集合不能修改元素，但是可以通过增加、删除元素来修改集合的内容。

#### 删除元素

从集合中删除元素有两种方式：第一种是使用 `discard()` 方法，该方法删除指定元素；第二种是使用 `remove()` 方法，该方法根据值的前后顺序删除首个匹配的元素。

```python
st = {1, 2, 3, 4, 5}
st.discard(2)    # st 为 {1, 3, 4, 5}
st.remove(5)    # st 为 {1, 3, 4}
```

#### 集合相关函数

集合还提供了一些函数用于操纵集合，如 `union()`、`intersection()`、`difference()` 和 `symmetric_difference()`。

```python
st1 = {1, 2, 3}
st2 = {2, 3, 4}

union = st1.union(st2)    # union 为 {1, 2, 3, 4}
intersecion = st1.intersection(st2)    # intersecion 为 {2, 3}
diff = st1.difference(st2)    # diff 为 {1}
sym_diff = st1.symmetric_difference(st2)    # sym_diff 为 {1, 4}
```

### 2.2.7 dict (字典)
dict 是 Python 中唯一的映射类型，可以把键映射到值上。字典中的键必须是独一无二的，但值可以重复。

#### 创建字典

创建字典的最简单的方式是直接在花括号 {} 中输入键值对，或者通过调用内置函数 `dict()` 将序列转换成字典。

```python
dct = {"one": 1, "two": 2, "three": 3}   # 直接创建字典
lst = [(1, 2), (3, 4), (5, 6)]
dct2 = dict(lst)    # 通过序列生成字典
```

#### 添加键值对

向字典添加键值对有两种方式：第一种是直接使用 `update()` 方法，该方法更新字典中的元素；第二种是使用赋值语句来给字典新增元素。

```python
dct = {"one": 1, "two": 2, "three": 3}
dct["four"] = 4    # dct 为 {"one": 1, "two": 2, "three": 3, "four": 4}
dct = {"one": 1, "two": 2}.update({"three": 3})    # 更新字典元素
```

#### 访问值

字典中的值可以通过键来访问。如果字典中不存在所查询的键，就会返回 KeyError。

```python
dct = {"one": 1, "two": 2, "three": 3}
val = dct["two"]    # val 为 2
val2 = dct.get("four", None)    # val2 为 None
```

#### 修改值

字典中的值可以通过键来修改。

```python
dct = {"one": 1, "two": 2, "three": 3}
dct["two"] = 4    # dct 为 {"one": 1, "two": 4, "three": 3}
```

#### 删除键值对

从字典中删除键值对有两种方式：第一种是使用 `pop()` 方法，该方法删除指定键值对；第二种是使用 `del` 关键字，该关键字根据键删除键值对。

```python
dct = {"one": 1, "two": 2, "three": 3}
dct.pop("two")    # dct 为 {"one": 1, "three": 3}
del dct["three"]    # dct 为 {"one": 1}
```

#### 字典相关函数

字典还提供了一些函数用于操纵字典，如 `keys()`、`values()` 和 `items()`。

```python
dct = {"one": 1, "two": 2, "three": 3}

kys = dct.keys()    # kys 为 dict_keys(['one', 'two', 'three'])
vals = dct.values()    # vals 为 dict_values([1, 2, 3])
item = dct.items()    # item 为 dict_items([('one', 1), ('two', 2), ('three', 3)])
```

## 2.3 总结
本文介绍了 Python 中常用的几种数据类型及其存储方式，并以列表、元组、集合、字典作为案例，讲解了其之间的关系、区别和应用。通过对数据类型的理解，读者能够准确理解应用场景，灵活选择合适的数据类型，提升代码质量和效率。