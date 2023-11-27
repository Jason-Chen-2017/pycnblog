                 

# 1.背景介绍


“字典”是编程语言中一个非常重要的数据类型，它提供了一种灵活、方便的方式存储和访问数据。本文将从基本知识出发，带领读者用较简单的方式理解字典的工作原理。

首先，让我们先了解一些相关的术语：

1. key-value pair：字典中的每个元素都是一个key-value pair，也就是键值对。
2. keys：字典的关键字，即键。在字典中，所有的keys必须是独一无二的，不能重复。
3. values：字典的值。
4. item：字典中一个键值对称为一个item。
5. length：字典中的元素个数。

# 2.核心概念与联系
## 2.1 字典的创建与初始化
字典的创建方式有两种，第一种是通过dict()函数，第二种是通过{}符号。 

下面展示如何创建一个空字典，然后逐步添加元素：

```python
# 创建一个空字典
my_dict = {}

# 通过[]索引来设置键值对
my_dict['name'] = 'Alice'
my_dict[1] = [2, 3, 4]
print(my_dict)   # {'name': 'Alice', 1: [2, 3, 4]}

# 使用dict()函数创建字典
new_dict = dict([('name', 'Bob'), (2, 5), ('age', 27)])
print(new_dict)   # {'name': 'Bob', 2: 5, 'age': 27}
```

字典中还可以包括其他各种数据结构，比如列表、元组等。如果多个键对应同一个值，则最后出现的键-值对会覆盖之前的键-值对。

## 2.2 获取和修改字典中的元素
获取字典中的元素有两种方式：

1. 通过键索引获取：`my_dict[key]`；
2. 通过get方法获取：`my_dict.get(key)`。

修改字典中的元素也有两种方式：

1. 修改已存在的键值对：`my_dict[key] = value`;
2. 添加新的键值对：`my_dict[new_key] = new_value`。

下面演示一下这两种方法：

```python
my_dict = {'a': 1, 'b': 2, 'c': 3}

# 方法1：通过键索引获取元素并赋值
my_dict['b'] = 4

# 方法2：通过get方法获取元素并赋值
my_dict.setdefault('d', 4)

print(my_dict)    # {'a': 1, 'b': 4, 'c': 3, 'd': 4}
```

通过`.get()`或`.setdefault()`方法设置默认值的目的是为了避免KeyError。当尝试访问不存在的键时，如果不设置默认值，就会引发KeyError。

## 2.3 删除字典中的元素
删除字典中的元素有三种方式：

1. `del my_dict[key]`：通过键删除对应的键值对；
2. `.pop(key)`：通过键删除对应的键值对，并返回对应的值；
3. `.clear()`：清除字典的所有元素。

下面演示一下这三种方法：

```python
my_dict = {
    'a': 1, 
    'b': 2, 
    'c': 3
}

# 方法1：通过del删除键值对
del my_dict['b']

print(my_dict)   # {'a': 1, 'c': 3}

# 方法2：通过pop删除键值对并返回值
val = my_dict.pop('a')
print(val)       # 1
print(my_dict)   # {'c': 3}

# 方法3：通过clear清空字典
my_dict.clear()
print(my_dict)   # {}
```

注意：以上方法只是删除了键值对，不会释放内存空间，如果有大的对象作为值，还是建议使用del方法手动删除。


## 2.4 对字典进行遍历
对字典进行遍历有两种方式：

1. for循环遍历：`for key in my_dict:`或者`for key, value in my_dict.items():`，其中后者可以同时获得键和值；
2. `iter()`方法：`it = iter(my_dict)`，然后用`next(it)`遍历字典。

下面演示一下这两种方法：

```python
my_dict = {
    'a': 1, 
    'b': 2, 
    'c': 3
}

# 方法1：for循环遍历
for k in my_dict:
    print(k)
    
for k, v in my_dict.items():
    print(k, v)    

# 方法2：iter()方法
it = iter(my_dict)
while True:
    try:
        key = next(it)
        val = my_dict[key]
        print(key, val)
    except StopIteration:
        break     
```

输出结果如下所示：

```python
a b c
a 1 b 2 c 3
a 1
b 2
c 3
```

## 2.5 判断字典是否为空或非空
判断字典是否为空可以使用`len()`方法。但是更推荐的方法是通过`bool()`函数判断，因为空字典就是False，非空字典就是True。

下面举例说明：

```python
my_dict = {}

if not bool(my_dict):
    print("my_dict is empty")   # my_dict is empty
    

my_dict = {'a': 1, 'b': 2}

if bool(my_dict):
    print("my_dict is not empty")   # my_dict is not empty
```

## 2.6 深拷贝和浅拷贝
由于字典是可变类型，所以当给一个新变量赋值时，其实是引用了原来的字典。因此，对新变量进行修改也会影响到原字典。为了防止这种情况发生，需要拷贝一个新的字典。

### 浅拷贝（shallow copy）
使用`copy()`方法，将原字典拷贝成一个新字典，新字典与原字典共享内存，任何修改都会反映到两个字典。

```python
original_dict = {"a": 1, "b": 2}
new_dict = original_dict.copy()
new_dict["b"] = 3

print(original_dict)  # {"a": 1, "b": 3}
```

上面的例子中，当修改新字典`new_dict`的`"b"`值时，原字典`original_dict`也跟着变化。这是因为两个字典共享内存，它们指向相同的地址。所以，修改任意一个字典，另一个字典也会受到影响。

### 深拷贝（deep copy）
使用`deepcopy()`方法，将原字典拷贝成一个新字典，新字典与原字典完全独立，任何修改不会影响到另一个字典。

```python
import copy

original_dict = {"a": 1, "b": [{"c": 3}]}
new_dict = copy.deepcopy(original_dict)
new_dict["b"][0]["c"] = 4

print(original_dict)  # {"a": 1, "b": [{"c": 4}]}
print(new_dict)        # {"a": 1, "b": [{"c": 4}]}
```

上面的例子中，当修改新字典`new_dict`的嵌套字典的`"c"`值时，原字典`original_dict`没有变化。这是因为复制了一个新的字典，原字典与新字典之间没有关系。