                 

# 1.背景介绍



首先，关于什么是Python变量，这个问题我们可以从计算机内存管理的角度来看待。简单来说，变量就是在内存中存放数据的存储器单元，每个变量都有一个名称、数据类型（整数、浮点数、字符串等）、值。它可以被赋予新值，也可以计算得到一个结果。

另外，变量还具有四种作用域，分别是全局作用域、局部作用域、嵌套作用域和内置作用域。一般情况下，变量的作用域从内到外依次是全局作用域、当前函数作用域、当前模块作用域以及内置作用域。

接着，Python中的变量分为动态语言和静态语言。对于动态语言，程序在运行时才确定变量的数据类型；而对于静态语言，编译器会预先进行类型检查，确保变量的数据类型正确。

除了上面提到的这些基础知识外，我们还需要了解一些Python特有的变量类型，比如元组、列表、集合、字典等。不同类型的变量之间也存在一些共同之处和区别，如元组不可修改、列表可修改、集合和字典之间的映射关系等。最后，我们还要了解变量的声明方式，包括全局变量、局部变量和参数传递等。

# 2.核心概念与联系

## 2.1 变量的定义

变量的定义语法如下所示：

```python
variable_name = value
```

其中，`variable_name`表示变量名，它是一个合法的标识符；`value`表示变量的值。一般来说，赋值语句右边的值可以是任意的表达式或值。

## 2.2 变量类型

### 2.2.1 数字类型

Python中支持的数字类型有整数、布尔型、浮点数和复数。整数类型可以使用`int()`函数创建，例如：

```python
 x = int(1)    # 创建整型变量x并赋值为1
```

布尔型变量只能取值为True或者False，可以使用`bool()`函数创建，例如：

```python
flag = bool(True)   # 创建布尔型变量flag并赋值为True
```

浮点数类型可以使用`float()`函数创建，例如：

```python
y = float(3.14)     # 创建浮点型变量y并赋值为3.14
```

复数类型可以使用`complex()`函数创建，例如：

```python
z = complex(1j)      # 创建复数型变量z并赋值为1j
```

除此之外，Python还支持用十六进制、八进制和二进制表示整数：

```python
a = 0xAF             # 175 in hexadecimal notation
b = 0o17             # 15 in octal notation
c = 0b1101           # 13 in binary notation
```

注意，整数类型不支持使用科学计数法表示。

### 2.2.2 字符串类型

Python中字符串由单引号`'`或双引号`"`括起来的若干字符组成。字符串可以使用`str()`函数创建，例如：

```python
s = str('hello')        # 创建字符串变量s并赋值为'hello'
t = "world"            # 创建另一个字符串变量t并赋值为'world'
u = '''Hello World!'''  # 使用三个单引号或三个双引号创建多行字符串
v = """This is a long string that spans multiple lines."""
```

字符串的索引从0开始，长度可以通过`len()`函数获取：

```python
print("The first character of s is:", s[0])       # The first character of s is: h
print("The last character of t is:", t[-1])        # The last character of t is: d
print("The length of u is:", len(u))               # The length of u is: 15
```

### 2.2.3 列表类型

Python中列表是一个有序序列，其元素可以重复。列表可以使用方括号`[]`创建，并且可以使用索引访问元素，索引从0开始，可以使用切片操作访问子列表。列表还提供了一些内置方法，包括`append()`、`insert()`、`pop()`、`remove()`、`clear()`、`sort()`、`reverse()`等。

```python
lst = [1, 'two', True]                 # 创建列表
print(lst)                             # Output: [1, 'two', True]
lst.append([1, 2, 3])                  # 在末尾添加一个列表
print(lst)                             # Output: [1, 'two', True, [1, 2, 3]]
lst[2] = False                         # 替换列表中的第二个元素
print(lst)                             # Output: [1, 'two', False, [1, 2, 3]]
sublist = lst[2:]                      # 切片操作生成子列表
print(sublist)                         # Output: ['two', False, [1, 2, 3]]
```

### 2.2.4 元组类型

元组也是有序序列，但它的元素不能修改。元组可以使用圆括号`()`创建，其元素也可以通过索引访问。元组比较少使用。

```python
tpl = (1, 2, 3)                        # 创建元组
print(type(tpl), tpl)                   # Output: <class 'tuple'> (1, 2, 3)
try:
    tpl[0] = 0                          # 不允许修改元组元素
except TypeError as e:
    print(e)                            # Output: 'tuple' object does not support item assignment
```

### 2.2.5 集合类型

集合是一个无序的、不重复元素集。集合可以使用花括号`{}`创建，并支持一些基本操作，包括`add()`、`remove()`、`union()`、`intersection()`等。

```python
set1 = {1, 2, 3}                       # 创建集合
set2 = {'apple', 'banana'}
set3 = set()                           # 创建空集合
print(set1.union(set2).union(set3))     # Output: {1, 2, 3, 'apple', 'banana'}
print(set1 & set2)                     # Output: {}
print(set1 | set2)                     # Output: {1, 2, 3, 'apple', 'banana'}
```

### 2.2.6 字典类型

字典是一种键-值对的集合。字典可以使用花括号`{}`创建，并通过键访问值。字典提供一些内置方法，包括`keys()`、`values()`、`items()`、`get()`、`update()`等。

```python
dic = {'name': 'Alice', 'age': 20}          # 创建字典
print(dic['name'])                           # Output: Alice
for key, val in dic.items():
    print(key + ':'+ str(val))              # Output: name: Alice age: 20
if 'gender' in dic:
    print(dic['gender'])                    # Output: None
else:
    print("'gender' doesn't exist")         # Output: 'gender' doesn't exist
dic.setdefault('gender', 'female')
print(dic['gender'])                        # Output: female
```

### 2.2.7 类型转换

Python中可以把不同类型的值转换成相同类型的值。可以使用`type()`函数查看类型，`isinstance()`函数判断对象是否属于某个类型。

```python
num = 10                                    # 将整型变量num转换成浮点型变量f
f = float(num)                              # num为10，f等于10.0
s = str(f)                                  # f为10.0，s等于'10.0'
lst = list(range(1, 6))                     # 将范围1~5转换成列表lst
lst.append('six')                           # 添加元素'six'到lst中
tup = tuple(lst)                            # 把lst转换成元组tup
dict_from_tup = dict(enumerate(tup))         # 用enumerate函数将tup转换成字典
print(type(num), type(f), type(s), lst[:], tup[:], dict_from_tup)
                                                    # Output: <class 'int'> <class 'float'> <class'str'> 
                                                    #         [1, 2, 3, 4, 5,'six'] (1, 2, 3, 4, 5,'six')
                                                    #         {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5:'six'}
```

## 2.3 作用域

Python变量的作用域决定了变量的生命周期及其可访问性。Python的作用域包括全局作用域、局部作用域、嵌套作用域和内置作用域。

### 2.3.1 全局作用域

全局作用域指的是在函数体之外定义的变量。它可以在整个脚本中使用。全局作用域的变量可以在任何位置访问，包括函数体内部。

```python
total = 0                                # 全局变量

def add(n):                               # 函数体
    global total                          # 使用global关键字声明变量为全局变量
    total += n
    return total

print(add(1))                             # Output: 1
print(add(2))                             # Output: 3
```

### 2.3.2 局部作用域

局部作用域指的是在函数体内定义的变量，只在该函数内有效。它可以访问父函数的局部变量。局部变量在函数调用完成后就失效。

```python
def show():
    local = 1                            # 局部变量
    def inner():
        nonlocal local                     # 通过nonlocal关键字声明变量为非局部变量
        local = 2                         # 修改局部变量值
        return local
    
    return inner()
    
print(show())                             # Output: 2
```

### 2.3.3 嵌套作用域

嵌套作用域指的是在函数体内又嵌套了一个函数，这样子函数就可以访问父函数的变量。但是，它只能访问直接父函数的局部变量，而不能访问间接父函数的变量。

```python
count = 0                                 # 全局变量

def outer():
    count = 1                             # 重新赋值为局部变量
    def inner():
        nonlocal count                     # 修改全局变量值
        count += 1                         # 修改全局变量值
        
    inner()                                # 执行inner函数
    return count

print(outer())                            # Output: 2
```

### 2.3.4 内置作用域

内置作用域指的是Python预先定义的命名空间。它包括很多内置函数和变量。你可以使用`dir(__builtins__)`函数查看所有的内置变量。

```python
print(dir(__builtins__))
```

## 2.4 变量的声明方式

### 2.4.1 参数传递

在Python中，函数的参数传递遵循“引用”传参的规则。如果函数需要修改传入的参数，则应该使用可变数据类型（例如列表、字典）作为参数类型。

```python
def modify_arg(arg):
    arg = 'new value'
    print(arg)
    
my_list = ['old value']
modify_arg(my_list)                          # Output: new value
print(my_list)                               # Output: ['new value']
```

### 2.4.2 返回值

函数返回值可以有多个返回值，可以使用tuple或者list作为返回值的容器。但是，返回值仍然遵循“引用”传参的规则，即改变返回值本身不会影响函数外部变量的值。

```python
def sum_and_product(a, b):
    c = a + b
    d = a * b
    return (c, d)
    
result = sum_and_product(2, 3)                # result包含两个值
print(result[0], result[1])                   # Output: 5 6
```