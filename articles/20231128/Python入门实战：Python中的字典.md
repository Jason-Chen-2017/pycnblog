                 

# 1.背景介绍


在Python中，字典(Dictionary)是一种内置的数据类型，它是一个无序的键值对集合。字典可以存储任意类型的对象，而且通过键可以检索到对应的值。因此，字典在Python中扮演着非常重要的作用。本文将从以下两个方面介绍Python中的字典：

⒈ 字典的定义及其特性；

⒉ 如何创建、访问、修改和删除字典中的元素。

首先，让我们看一下字典的定义及其特性。

# 2.字典的定义及其特性
## 2.1 字典的定义
字典是另一个高级数据类型，它以键-值（key-value）对形式存储。字典的特点是:

1. 每个键都是独一无二的，不能重复；
2. 可以通过键获取对应的值；
3. 如果键不存在，则返回None或报错；
4. 字典中的所有键必须是不可变对象，如字符串、数字或元组；
5. 键必须是可散列的，即哈希值不一样；
6. 可以通过键迭代字典中的所有键值对。

## 2.2 字典的特性
### 2.2.1 创建字典
```python
# 方式1：使用 {} 创建空字典
d = {}

# 方式2：使用 dict() 函数创建字典
e = dict({'name': 'John', 'age': 30}) # 使用关键字参数初始化字典

# 方式3：使用 zip() 将两个序列组合成字典
keys = ['apple', 'banana', 'orange']
values = [3, 5, 7]
f = dict(zip(keys, values))

print(d, e, f) 
#{} {'name': 'John', 'age': 30} {'apple': 3, 'banana': 5, 'orange': 7}
```

### 2.2.2 访问字典元素
```python
d = {
    "a": 1, 
    "b": 2, 
    "c": 3
}

# 通过键访问字典元素
print("Accessing dictionary elements using keys:")
for key in d:
    print("{} -> {}".format(key, d[key]))
    
# Accessing dictionary elements using keys:
# a -> 1
# b -> 2
# c -> 3

print("")

# 使用 get() 方法访问字典元素
print("Accessing dictionary elements using the get() method:")
print(d.get("b"))    # Output: 2
print(d.get("x", -1))   # Output: -1 (if x not found in the dictionary)

# Using the get() method to access missing elements with a default value of -1:
# Output: 2
# Output: -1
```

### 2.2.3 修改字典元素
```python
# 添加/更新元素
print("Adding or updating an element in the dictionary")
d["d"] = 4        # Add a new key-value pair to the dictionary
d["c"] = 9        # Update the existing value for key "c" to 9
print(d)          # Output: {"a": 1, "b": 2, "c": 9, "d": 4}

print("")

# 删除元素
print("Deleting an element from the dictionary")
del d['b']       # Remove the key-value pair for key 'b' from the dictionary
print(d)          # Output: {"a": 1, "c": 9, "d": 4}
```

### 2.2.4 更新字典
```python
dict_1 = {'a': 1, 'b': 2}
dict_2 = {'c': 3, 'd': 4}

dict_1.update(dict_2)     # update dict_1 with items from dict_2 

print(dict_1)             # Output: {'a': 1, 'b': 2, 'c': 3, 'd': 4}

list_of_tuples = [('e', 5), ('f', 6)]

dict_1.update(list_of_tuples)      # update dict_1 with list of tuples as argument 

print(dict_1)                     # Output: {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6}
```