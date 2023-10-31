
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


在计算机编程中，程序的执行流程由控制结构（如if-else、for、while等）和数据结构（如数组、链表、栈、队列、散列表等）组成。其中控制结构用于控制程序的执行路径，主要用来实现条件判断、循环控制、跳转等功能；而数据结构则负责保存和处理数据，包括变量、常量、函数调用的参数及返回值等信息。本文将阐述Python语言中的条件控制结构（if-else语句）和循环控制结构（for-while语句），并介绍一些常用的算法原理和例子。
# 2.核心概念与联系
## Python条件语句
Python支持如下几种条件语句:

1. if 语句
```python
if condition1:
    # do something
    
elif condition2:
    # do something else

else:
    # no condition is true, do this instead
```
2. while 语句
```python
while expression:
    # do something repeatedly until the expression becomes false
```
3. for 语句
```python
for variable in iterable:
    # do something with each item in the iterable object
```
上述三种条件语句都可嵌套使用，比如可以有多重if-else嵌套或者for-while循环组合。
## Python循环语句
### range() 函数
range() 函数用于生成一个整数序列。语法格式如下：

```python
range(start, stop, step)
```
- start 表示起始值，默认为 0 。
- stop 表示结束值，该值是范围内最后的值。
- step 表示步长，默认为 1 ，表示每次迭代后增加 1 。

举例如下：

```python
>>> range(5)   # 生成一个从 0 到 4 的整数序列
range(0, 5)

>>> list(range(5))    # 将整数序列转换为列表
[0, 1, 2, 3, 4]

>>> range(1, 5)      # 从 1 到 4 （不包括 5）的整数序列
range(1, 5)

>>> range(1, 9, 2)   # 从 1 到 8（不包括 9）的奇数序列
range(1, 9, 2)
```

### 通用循环结构——for...in...
for...in...语句用于遍历任意类型的可迭代对象（如list、tuple、set、dict、str等）。其基本语法格式如下：

```python
for var_name in iter_obj:
   # 这里放要对var_name做的事情
```
实际上，该语句相当于在每次循环时，依次对iter_obj中的每个元素进行赋值给var_name。

比如下面的例子，将一个数字列表乘以2：

```python
numbers = [1, 2, 3, 4, 5]
result = []
for num in numbers:
    result.append(num * 2)
print(result)  # Output: [2, 4, 6, 8, 10]
```

另一种常见的用法是将字典中的key和value分别取出来：

```python
my_dict = {'a': 1, 'b': 2, 'c': 3}
keys = []
values = []
for key, value in my_dict.items():
    keys.append(key)
    values.append(value)
print(keys)     # Output: ['a', 'b', 'c']
print(values)   # Output: [1, 2, 3]
```

### break 和 continue 语句
break和continue语句主要用于跳出当前循环或单次循环。

- break：终止当前循环，立即开始执行紧接着的语句，常用于提前退出循环。
- continue：终止当前循环的本次迭代，并直接开始下一次迭代。

例如：

```python
for i in range(10):
    if i == 5:
        print('Found it!')
        break
    elif i % 2!= 0:
        print("Skipping odd number:", i)
        continue
    print('Current number:', i)
```

输出结果：

```
Current number: 0
Skipping odd number: 1
Current number: 2
Skipping odd number: 3
Current number: 4
Found it!
```

### pass 语句
pass 语句什么都不做，它一般作为占位符，用来补全代码块。例如：

```python
def function_without_body():
    pass  # 没有任何作用，只是为了完整性
```