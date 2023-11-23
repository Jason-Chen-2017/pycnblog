                 

# 1.背景介绍


## 什么是列表？
列表（list）是一种数据结构，它可以存储多个值，每个值都具有索引值，可以通过索引访问其中的元素。列表可以用来存储多种类型的数据，如字符串、数字、布尔值等。
## 为什么要用列表？
在编程中经常会遇到需要处理多个值的场景。比如，当用户输入一些信息时，可能希望将这些信息保存起来，方便后续读取和分析。又如，网站的后台管理系统，为了更好地展示数据，通常需要进行分页显示。对于这样的需求，列表是一个很好的选择。
## 列表的特点
### 可变性
列表的值可以动态变化，即可以在创建列表的时候不指定初始容量大小，随着添加新的元素而自动扩充。
```python
my_list = []

for i in range(10):
    my_list.append(i)
    
print(len(my_list)) # Output: 10
```
### 有序性
列表中的元素顺序与插入的先后顺序一致，可以通过索引访问列表中的元素，也可通过切片方法来获取子序列。因此，列表具备高效随机访问的特性。
### 可切片
列表可以通过切片操作来提取子列表或复制列表，对子列表进行修改也不会影响原始列表。
```python
my_list = [1, 2, 3, 4, 5]

sub_list = my_list[1:3]    # 提取子列表
print(sub_list)           # Output: [2, 3]

new_list = sub_list + [6, 7]   # 对子列表进行扩展
print(new_list)              # Output: [2, 3, 6, 7]

my_list += new_list          # 在原始列表中追加新元素
print(my_list)               # Output: [1, 2, 3, 4, 5, 2, 3, 6, 7]
```
### 支持嵌套
列表还支持多维数组的形式，允许嵌套其他的列表。通过嵌套列表，可以实现更加复杂的应用场景。
```python
my_list = [[1, 2], [3, 4]]

print(my_list[0])            # Output: [1, 2]
print(my_list[0][1])         # Output: 2
```

# 2.核心概念与联系
## 定义
 - 序列（sequence），数据结构中的一个术语，指的是一组按照特定顺序排列的元素，其中每一个元素都有其唯一的编号，称为下标（index）。

 - 列表（list）：它是一种有序集合数据类型，类似于数组。它的元素是按次序排列的，可以重复。列表用方括号[]括起来，元素之间用逗号隔开。

 - 下标（index）：列表中的每一个元素都有一个编号，叫做下标。列表的第一个元素的下标是0，第二个元素的下标是1，依此类推。

 - 切片（slice）：指的是从列表中提取子序列的操作。

## 操作

- 创建列表
   - 方法1：使用[]构造器创建列表

   ```python
   lst = ['apple', 'banana', 'orange']
   ```

   - 方法2：使用list()函数创建列表

   ```python
   nums = list(range(10))
   print(nums)
   >>> [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
   letters = list("hello world")
   print(letters)
   >>> ["h", "e", "l", "l", "o", " ", "w", "o", "r", "l", "d"]
   ```

 
- 添加元素
 
 ```python
 lst.append('grape')     # 向列表末尾添加元素
 lst += ['peach']        # 使用+=运算符来拼接列表
 ```
 
- 删除元素
 
 ```python
 del lst[-1]             # 通过索引删除元素
 lst.remove('banana')    # 根据值删除元素
 ```
 
- 修改元素
 
 ```python
 lst[0] = 'pear'          # 修改指定位置上的元素
 lst[:3] = [1, 2, 3]      # 用切片语法替换列表中的一部分元素
 lst[::2] = [-1, -2]      # 用步长参数来替换列表中的某些元素
 ```
 
- 查询元素
 
 ```python
 if 'banana' in lst:      # 判断是否存在某个元素
     index = lst.index('banana')
     print('The index of the element is:', index)
 else:
     print('Element not found.')
 ```
 
- 获取子集
 
 ```python
 subset = lst[1:-1]       # 获取中间的三个元素作为子列表
 superset = sorted(lst)   # 将列表转换成有序的集合
 reversed_lst = lst[::-1] # 反转整个列表
 ```
 
- 计算长度
 
 ```python
 len(lst)                 # 返回列表的长度
 ```
 
- 排序
 
 ```python
 sorted_lst = sorted(lst) # 对列表进行排序
 sorted_lst.reverse()     # 对列表进行反转
 ```
 
- 查找最大最小值
 
 ```python
 max(lst), min(lst)       # 查找列表中最大和最小值
 ```
 
- 分组
 
 ```python
 grouped_lst = [(x, y) for x, y in zip([1, 2, 3], ['a', 'b', 'c'])] 
 grouped_lst.sort()
 print(grouped_lst)
 >>> [('a', 1), ('b', 2), ('c', 3)]
 ```
 
- 拆分
 
 ```python
 first_half = lst[:len(lst)//2]    # 以列表的左半边作为第一部分
 second_half = lst[len(lst)//2:]   # 以列表的右半边作为第二部分
 odds = filter(lambda x: x % 2 == 1, lst)  # 从列表中过滤奇数元素
 evens = list(filterfalse(lambda x: x % 2 == 1, lst))  # 从列表中过滤偶数元素
 ```