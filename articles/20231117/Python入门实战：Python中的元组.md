                 

# 1.背景介绍


元组（tuple）是一个不可变序列类型，由若干个元素组成，元素间用逗号隔开，有序可迭代对象。元组可以进行索引、切片、加法运算等操作。元组的特点是不能修改其中的元素值，只能重新赋值创建新的元组。元组通常用于函数返回多个值时，或者作为集合中元素的唯一标识符。

在Python语言中，元组的定义格式如下:

```python
tup = (item1, item2,..., itemN)
```

其中`item1`，`item2`，...，`itemN`都是有效的Python表达式，用来填充元组的值。例如：

```python
>>> t = ('apple', 'banana', 'cherry')
>>> type(t)
<class 'tuple'>
>>> t[0]
'apple'
>>> len(t)
3
```

# 2.核心概念与联系

## 2.1 什么是元组？

元组是一个有序列表，它可以存储多个数据项。元组属于不可变序列类型。它的每个元素都有一个编号，可以通过索引访问到相应的数据。元组中的元素类型可以不同，而长度也是不固定的。元组可以被索引、切片、遍历、比较、合并等操作。

## 2.2 为什么要使用元组？

1. 函数返回多值的情况

   在Python中，函数可以返回一个或多个值，但是只有一种方式可以将这些值传递给函数的调用者。也就是说，调用者无法通过参数名来指定某个值是否是返回值。因此，如果需要从函数返回多个值，就应该使用元组类型。
   
   ```python
   def get_person():
       name = "John"
       age = 30
       return name, age
   
   person = get_person()
   print(type(person))    # <class 'tuple'>
   print(person)          # ('John', 30)
   ```

2. 作为集合中元素的唯一标识符

   在Python中，集合是一种无序且元素不可重复的容器。因此，使用元组作为集合的元素唯一标识符可以防止集合中出现相同的元素。
   
   ```python
   >>> fruits = {'apple', 'banana', 'cherry'}
   >>> fruits       # {('c', 'h', 'e', 'r'), ('a', 'n'), ('b', 'l'), ('p')}
   >>> 
   >>> my_fruits = [('apple',), ('banana',), ('cherry',)]   # 使用元组作为集合元素唯一标识符
   >>> my_fruits     # [(), (), ()]
   ```

   
## 2.3 有哪些使用场景？

### 2.3.1 函数参数

函数的参数是位置参数，是按照顺序依次传入。当函数需要接收多个值，而且这些值之间没有任何先后关系时，可以使用元组作为参数的形式。

```python
def add_numbers(num1, num2):
    return num1 + num2
    
result = add_numbers(5, 7)   # 结果是12
```

也可以传入多个值，将它们放在元组中作为参数传入函数。

```python
nums = (5, 7)
result = add_numbers(*nums)   # 结果也是12
```

当函数内部需要对元组的所有值进行处理时，可以逐个访问元组中的元素。

```python
def multiply_tuples(tup1, tup2):
    result = []
    for i in range(len(tup1)):
        result.append(tup1[i] * tup2[i])
    return tuple(result)
        
t1 = (2, 4, 6)
t2 = (3, 2, 1)
result = multiply_tuples(t1, t2)   # 结果是(6, 8, 6)
```

### 2.3.2 返回值的赋值

函数的返回值可以赋值给变量。如果函数的返回值为单一值，则直接将这个值赋给变量即可。

```python
value = square_root(9)   # value是3
```

但如果函数的返回值为多个值，那么将会得到一个元组。此时需要对该元组中的元素逐个赋值。

```python
x, y = coordinates()   # x,y分别赋值为坐标轴的两点
```

### 2.3.3 数据存取

对于多维数组或矩阵，可以通过索引或切片的方式获取数据，这种情况下，可以使用元组作为索引。

```python
matrix = ((1, 2, 3),
          (4, 5, 6),
          (7, 8, 9))
          
row1 = matrix[0]   # 获取第1行
col2 = row1[1]      # 获取第1列的第二个元素
```

### 2.3.4 数据结构嵌套

元组还可以嵌套，形成更复杂的结构。

```python
nested_tuple = ((1, 2, 3),
                (4, 5, (6, 7)),
                8)
                 
print(nested_tuple[2][1][1])   # 输出7
```