                 

# 1.背景介绍


在Python中，字典(Dictionary)是一种可变容器类型，它是一个键值对的集合。其中，每个键都对应一个值，字典中的值可以取任何数据类型。字典具有以下几个特点:

1. 查找速度快，通过键值快速找到对应的元素。
2. 有序性，字典内部存放的数据项会按照添加顺序排列。
3. 可变性，字典中的元素是可修改的，可以动态添加、删除或更改。
4. 支持多种访问方式，字典支持键、值、键值对三种访问方式。

集合（Set）是另一种类似于字典的数据结构。集合也是由一组元素组成，不同的是集合不存储元素的值，而是只存储元素本身。集合中的元素不能重复，而且元素无先后顺序。集合具有以下特点：

1. 唯一性，集合中的元素不允许重复。
2. 互异性，集合中任意两个元素必定互斥，不存在相同元素。
3. 支持交集、并集等运算，集合支持交集、并集、差集等基本的数学运算。

因此，字典和集合在实际应用中的广泛应用。对于Python初学者来说，掌握字典和集合的用法，能够更好地理解数据结构和算法。下面我们就从这里面入手，带领大家了解Python字典和集合的基本使用方法。
# 2.核心概念与联系
## 2.1 字典
### 2.1.1 创建字典
创建一个空的字典可以用花括号{}表示：

```python
my_dict = {}
print(type(my_dict)) # <class 'dict'>
```

也可以创建包含初始值的字典，如下所示：

```python
my_dict = {'name': 'Alice', 'age': 25}
print(my_dict['name'])   # Alice
print(my_dict['age'])    # 25
```

注意：字典中的键必须是不可变对象，比如数字、字符串和元组等。列表、字典和集合属于可变对象，所以不能作为键值。

### 2.1.2 添加、修改和删除元素
向字典中添加新的键值对的方法是直接赋值给键名即可，如：

```python
my_dict['gender'] = 'female'
print(my_dict)          # {'name': 'Alice', 'age': 25, 'gender': 'female'}
```

如果某个键已存在，则修改该键对应的值；如果某个键不存在，则添加新键值对。另外，还可以通过字典的update()方法更新字典中的多个键值对。

要删除字典中的元素，有两种办法：

1. 通过del语句删除键值对，如：

   ```python
   del my_dict['gender']
   print(my_dict)      # {'name': 'Alice', 'age': 25}
   ```
   
2. 清空整个字典，再重新添加键值对。如：

   ```python
   my_dict.clear()
   my_dict = {'city': 'Beijing', 'country': 'China'}
   print(my_dict)      # {'city': 'Beijing', 'country': 'China'}
   ```

### 2.1.3 字典的遍历
字典提供了四种不同的遍历方式，分别为：

1. keys(): 返回所有的键。
2. values(): 返回所有的值。
3. items(): 以元组形式返回所有的键值对。
4. iterkeys(): 返回一个迭代器，用于遍历所有键。
5. itervalues(): 返回一个迭代器，用于遍历所有值。
6. iteritems(): 返回一个迭代器，用于遍历所有键值对。

例如，下面的代码展示了字典的遍历方法：

```python
my_dict = {'name': 'Alice', 'age': 25, 'gender': 'female'}

for key in my_dict.keys():
    print(key)         # name age gender
    
for value in my_dict.values():
    print(value)       # Alice 25 female

for item in my_dict.items():
    print(item[0], item[1])     # name Alice
                                  # age 25
                                  # gender female

# 使用iterkeys()函数遍历字典的所有键
it = my_dict.iterkeys()
while True:
    try:
        key = next(it)
        print(key)                 # name age gender...
    except StopIteration:        # 当迭代到最后一个元素时，抛出StopIteration异常
        break
    
# 使用iteritems()函数遍历字典的所有键值对
it = my_dict.iteritems()
while True:
    try:
        item = next(it)
        print(item[0], item[1])     # name Alice
                                    # age 25
                                    # gender female...
    except StopIteration:
        break
```

需要注意的是，遍历字典时，不要对字典进行添加、删除或修改操作，否则可能会导致迭代过程中出现错误。如果需要修改字典，建议将其转换为列表或集合之后再遍历。