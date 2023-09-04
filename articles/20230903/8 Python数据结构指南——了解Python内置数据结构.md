
作者：禅与计算机程序设计艺术                    

# 1.简介
  


在编程语言中，数据结构可以分为以下五种：

1）基础数据类型（primitive data types）:整数、浮点数、布尔值等不可变的数据类型；

2）容器数据类型（container data types）：列表、元组、集合、字典等可变的数据类型，能够容纳多个值的容器。

3）序列（sequence）数据类型：列表、字符串、元组都属于序列数据类型。

4）映射（mapping）数据类型：字典属于映射数据类型，它是一个键-值对的无序表。

5）文件对象（file object）：用于操作文件或网络连接的对象。

本文以Python为例，介绍Python数据结构的使用方法和用法示例。文章主要内容包括：

第一节 Python中的基本数据类型及其容器特性；

第二节 Python中常用的容器数据类型，比如列表、元组、集合、字典等；

第三节 利用Python的语法特性实现一些具体功能；

第四节 Python的文件读写操作。

# 2.基本数据类型

Python中有七种基本数据类型：整数、浮点数、布尔值、字符串、None、空值、函数。每一种基本数据类型都有自己的特点，下面我们将详细介绍它们之间的区别和联系。

## 2.1 整数

整数(int) 是 Python 中最基本的数据类型之一，表示整数。整型可以存储整个数字范围内的值，而且也支持数字运算符(+,-,*,/,//,%,**,<<,>>,&,^,|,~)。除法默认采用“地板除”的方式，即不论除数是否有余数都会执行。

```python
a = 7
b = -3
c = a + b # c 的值为 4
d = a * b # d 的值为 -21
e = a / b # e 的值为 -2 （注意这里是浮点型）
f = a // b # f 的值为 2
g = a % b # g 的值为 4
h = a ** b # h 的值为 343 (a的b次方)
i = 1 << 3 # i 的值为 8 (左移3位相当于乘以2的3次方)
j = 9 >> 1 # j 的值为 4 (右移1位相当于除以2的1次方)
k = ~a # k 的值为 -8 (-a-1)
l = a & b # l 的值为 3 (取a和b的补码按位与的结果)
m = a ^ b # m 的值为 4 (取a和b的补码按位异或的结果)
n = a | b # n 的值为 4 (取a和b的补码按位或的结果)
print(type(a))   # <class 'int'>
``` 

上面展示了整数类型的常用操作，如+、-、*、/、//、%、**、<<、>>、&、^、|和~。其中，`print(type(a))`用于输出变量a的类型。也可以用内建函数`isinstance()`判断一个对象是否为整数类型。

```python
if isinstance(a, int):
    print('a is an integer')
else:
    print('a is not an integer')
```

## 2.2 浮点数

浮点数(float) 是 Python 中的一种实数数据类型，它用来存储带小数的数字。浮点数有两种精度：单精度和双精度。对于较大的浮点数，双精度精度更高。

```python
a = 3.14
b = -1.5
c = a + b
d = a * b
e = a / b
f = abs(a)
g = round(a, 1) # 把 a 保留到小数点后两位
h = pow(a, b)
print(type(a))    # <class 'float'>
print(type(f))    # <class 'float'>
``` 

浮点数的运算规则与整数相同。另外，浮点数还支持 `abs()` 函数返回绝对值，`round()` 函数对浮点数进行四舍五入，`pow()` 函数计算 x 的 y 次幂。

## 2.3 布尔值

布尔值(bool) 是 Python 中一种逻辑数据类型，只有两个值 True 和 False。布尔值经常作为条件语句的结果，在控制流中起着重要作用。

```python
a = True
b = False
c = a and b     # 短路与，若 a 为 False，则返回 False;否则返回 b 的值
d = a or b      # 短路或，若 a 为 True，则返回 True;否则返回 b 的值
e = not a       # 返回 False 或 True
f = bool('')    # 返回 False
print(type(a))    # <class 'bool'>
``` 

布尔值的运算规则非常简单，只有与、或、非三种运算。其中，短路与（and）和短路或（or）运算会根据第一个运算数的真假决定第二个运算数是否需要求值，从而避免不必要的计算开销。

还有一点要注意的是，布尔值也支持布尔运算符，但它并不是直接对应的，而是在一定条件下自动转换为相应的布尔值。例如，表达式 `not ''`，如果''不为空字符串，则结果为True；如果为空字符串，则结果为False。

## 2.4 字符串

字符串(str) 是 Python 中最常用的文本数据类型，用来存储一系列字符。字符串可以使用单引号(')或者双引号(")，并可以跨越多行。

```python
a = 'Hello World'
b = "I'm John"
c = '''This is the first line.
       This is the second line.'''
d = """This is the third line.
        This is the fourth line."""
e = len(a)         # 获取字符串长度
f = 'a' in a        # 判断字符串中是否存在子串 'a'
g = a[-1]           # 访问最后一个元素
h = a[:5]           # 截取前5个元素
i = a.upper()       # 将所有字符转为大写
j = a.lower()       # 将所有字符转为小写
k = a.split(' ')    # 以空格分割字符串成一个列表
l = a.replace('H', 'J') # 替换所有的 'H' 为 'J'
m = '{} {} {}'.format('Hi', ',', 'how are you?') # 使用 format 方法格式化字符串
n = '%s %s %s'%('hi','what','is your name?')   # 使用 % 操作符格式化字符串
o = str(3.14)            # 将数字转为字符串
p = chr(65)              # 从 ASCII 编码获取对应字符
q = ord('A')             # 获取字符的 ASCII 编码
r = list('hello world')  # 将字符串转换为列表
s = ''.join(r)           # 用指定字符连接列表元素
t = r'\n'                # 转义字符
u = u'\ua000'            # Unicode字符
v = iter([1, 2, 3])      # 创建迭代器对象
w = next(v)              # 返回迭代器对象的下一个元素
x = sum(range(1, 5))     # 求和
y = any(['', [], {}, None]) # 判断是否存在真值
z = all([1, 2, '', []])  # 判断是否全为真值
``` 

除了上面提到的常用操作外，还有很多字符串相关的方法，如 startswith(), endswith(), find(), index(), count() 等。

## 2.5 None

None 是 Python 中一种特殊的空值，在条件语句中或其他地方被赋值时表示一个缺省值。None 可以看作是一个空引用，不指向任何有效内存地址。

```python
a = None
if a == None:
   pass
``` 

## 2.6 空值

空值(empty value) 是 Python 中没有明确含义的特殊值。对于某些特殊情形，比如定义了一个变量但是没有对它赋初始值，此时这个变量就是空值。

```python
a = None      # 空值
b = ""        # 空字符串
c = 0         # 0
d = float('nan')   # Not a Number
e = ()        # 空元组
f = []        # 空列表
g = {}        # 空字典
h = set()     # 空集合
i =...       # Ellipsis 对象（省略号对象）
j = NotImplemented  # NotImplemented 关键字，用于与 NotImplemented 比较
k = eval('None')    # 执行字符串，得到 None
``` 

## 2.7 函数

函数(function) 是 Python 中用于组织代码块和数据的一种机制。它通过给定输入参数，生成输出结果。在 Python 中，函数由 def 关键词声明，函数名后跟圆括号()，括号内部可以传递任意数量的参数，函数体则用冒号(:)和缩进来标识代码块。

```python
def add(x, y):   # 添加函数
    return x + y
    
result = add(2, 3) # 调用 add 函数，返回结果为 5
``` 

函数的参数可以是任意数据类型，也可以包含多个参数。可以定义匿名函数，也就是没有函数名的函数。

```python
add_lambda = lambda x, y : x + y   # 匿名函数，添加两个数字
result = add_lambda(2, 3)          # 调用匿名函数，返回结果为 5
``` 

虽然定义匿名函数有时候比较方便，但是命名函数通常是更好的做法。