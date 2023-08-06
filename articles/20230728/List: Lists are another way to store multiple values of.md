
作者：禅与计算机程序设计艺术                    

# 1.简介
         
 Python 中的列表（list）是一个非常有用的内置数据结构，可以存储多个不同类型的变量值，并且可以通过索引来访问元素。列表也是一种动态的数据类型，可以随着数据的添加而自动扩容。下面是一些关于列表的基本信息：
          - 创建一个空列表： `my_list = []`；
          - 创建一个包含三个元素的列表：`my_list = [1, "hello", True]`；
          - 获取列表中的元素数量：`len(my_list)` 或 `print(len(my_list))`。
          - 使用索引来访问列表元素：`print(my_list[0])`，表示获取第一个元素的值。
          - 对列表进行切片：`new_list = my_list[1:]` 表示从第二个元素开始切割整个列表，得到新的列表。
          - 在列表中添加新元素：`my_list.append("world")`，表示在末尾添加一个“world”字符串。
          - 在列表中插入元素：`my_list.insert(1, "python")`，表示在位置1处插入“python”。
          - 删除列表中的元素：`del my_list[0]`，表示删除第一个元素。
          - 更新列表中的元素：`my_list[0] = False`，表示更新第一个元素值为False。
          - 判断某个元素是否在列表中：`"hello" in my_list` 返回True。

          除了这些基础的操作外，列表还有一些高级功能需要熟悉一下。以下是一些常用方法：

          - sort() 方法用来对列表排序，默认为升序排列：`my_list.sort()`
          - reverse() 方法用来反转列表中的元素顺序：`my_list.reverse()`
          - copy() 方法用来创建当前列表的一个浅拷贝：`new_list = my_list.copy()`
          - count() 方法用来统计某元素在列表中出现的次数：`count = my_list.count("hello")`
          - extend() 方法用来在列表末尾追加另一个列表：`my_list.extend(["happy", "birthday"])`
          - pop() 方法用来弹出指定位置的元素，并返回该元素的值：`popped_element = my_list.pop(0)`
          - remove() 方法用来移除列表中指定的元素：`my_list.remove("happy")`
          上述方法都可以在列表上进行调用，也可以使用对应的缩写形式，如：`lst.sort()` 可以简写成 `lst.s()` 。

          通过以上介绍，你可以快速了解到如何创建、访问和操作列表，以及一些高级功能的用法。
          # 2.基本概念术语说明
          1. List
             A list is an ordered sequence of items enclosed by square brackets and separated by commas. The individual items can be of any data type such as strings, numbers or even other lists. In order for the indexing operation to work correctly, all the elements of the list must have a unique identifier that starts at zero. We access the elements inside a list using their indexes, which start from zero.

             To create an empty list, we use either the following commands:
              ```python
                my_empty_list = []      # creates an empty list
                print(my_empty_list)    # prints '[]'

                my_empty_list2 = list() # equivalently creates an empty list
              ```
           
             To create a list containing three elements, we use one of these commands:

              ```python
                my_list = ["apple", "banana", "cherry"]     # creates a list containing three string elements
                my_list2 = [1, 2, 3]                     # creates a list containing three integer elements
                my_list3 = [[1, 2], [3, 4]]               # creates a list containing two sub-lists
              ```
           
             Once we have created our list, we can perform various operations on it using the above mentioned methods or functions.

          2. Indexing/Slicing
             List indexing and slicing allow us to access specific elements of a list and extract slices out of them. Slicing means taking only part of a larger list and creating a new smaller list based on the specified range. The syntax for slicing looks like this: 

               `[start:end:step]`

               Here, `start` specifies the beginning index (inclusive), `end` specifies the ending index (exclusive), and `step` specifies the step size between successive elements of the slice. If we do not specify `start`, it defaults to zero, if we don't specify `end`, it defaults to the length of the list, and if we don't specify `step`, it defaults to one. It's important to note that when we take a slice of a list, the resulting slice is also a list, and we cannot assign its value back to the original list directly. Instead, we should update the entire slice at once using assignment statements.

            For example, suppose we want to extract the last two elements of a given list called `my_list`:

              ```python
                >>> my_list = ['apple', 'banana', 'cherry', 'orange']
                
                >>> print(my_list[-2:])   # outputs ['cherry', 'orange']
              ```

            Similarly, we can extract all odd numbered elements of a list using negative steps and slicing notation:

              ```python
                >>> my_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]
                
                >>> odd_numbers = my_list[::2][::-1]
                
                >>> print(odd_numbers)    # outputs [9, 7, 5, 3, 1]
              ```

          3. Mutable vs Immutable objects
             In Python, there are two categories of objects: mutable and immutable. An object whose value can change is considered mutable, while those whose value stays constant throughout their lifetime are considered immutable. Examples of mutable objects include lists, dictionaries, sets, etc., whereas examples of immutable objects include integers, floats, complex numbers, tuples, frozensets, booleans, etc.