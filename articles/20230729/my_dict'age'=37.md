
作者：禅与计算机程序设计艺术                    

# 1.简介
         
 在这个快速上手系列教程中，我会给大家带来一些最基础的Python知识。本文为第一章节，主要内容是字典的基础知识。

         # 2.字典(Dictionary)
          Python中的字典(Dictionary)是一种可变容器类型，它存储了一个 key-value 对。你可以通过键获取对应的值。一个字典可以看作是一个无序的键值对集合，其中每个键都是独一无二的。字典中的键可以是任意不可变类型，通常用字符串或数字类型表示。而值的类型则不做限制。

          在python中，字典的创建和访问方式如下所示：

          ``` python
            my_dict = {}    # 创建空字典

            my_dict[key] = value     # 添加元素
            print(my_dict[key])      # 获取元素

            del my_dict[key]        # 删除元素
            len(my_dict)            # 获取字典长度
          ```

          上述代码创建了一个空字典 `my_dict`，并向其中添加了键值对，之后通过键`key`获得了对应的值`value`。你也可以通过`del`语句删除某个键值对，也可以通过`len()`函数获取字典长度。

         # 3.字典方法
         Python 的字典还提供了一些内置的方法用于操作字典对象。这些方法包括：

          - clear(): 清空字典的所有项。

          - copy(): 返回当前字典的一个浅拷贝。

          - fromkeys(): 从指定序列创建一个新字典。

          - get(): 获取指定键的值，如果不存在则返回默认值。

          - items(): 以列表返回可遍历的（键、值）元组数组。

          - keys(): 以列表返回字典所有的键。

          - pop(): 根据键移除并返回对应的值，如果键不存在则返回默认值。

          - popitem(): 随机移除并返回字典中的一对键值对。

          - setdefault(): 如果键不存在于字典中，则将其添加并设置值为默认值。

          - update(): 更新现有字典或添加新键值对到字典中。

          - values(): 以列表返回字典所有的值。

          下面我们将对字典方法进行详细介绍。

         ### 3.1 dict.clear() 方法
         该方法用于清空字典的所有项。

         ### 3.2 dict.copy() 方法
         该方法用于返回当前字典的一个浅拷贝。

         ### 3.3 dict.fromkeys() 方法
         该方法用于从指定序列创建一个新字典。

         ### 3.4 dict.get() 方法
         该方法用于获取指定键的值，如果不存在则返回默认值。例如：

         ``` python
           my_dict = {'name': 'Alice', 'age': 25}
           print(my_dict.get('name'))           # output: Alice
           print(my_dict.get('gender','male'))  # output: male
         ```

         ### 3.5 dict.items() 方法
         该方法用于以列表返回可遍历的（键、值）元组数组。例如：

         ``` python
           my_dict = {'name': 'Alice', 'age': 25}
           for item in my_dict.items():
             print(item)   # output: ('name', 'Alice')
                            #          ('age', 25)
         ```

         ### 3.6 dict.keys() 方法
         该方法用于以列表返回字典所有的键。例如：

         ``` python
           my_dict = {'name': 'Alice', 'age': 25}
           for key in my_dict.keys():
             print(key)   # output: name
                          #       age
         ```

         ### 3.7 dict.pop() 方法
         该方法用于根据键移除并返回对应的值，如果键不存在则返回默认值。例如：

         ``` python
           my_dict = {'name': 'Alice', 'age': 25}
           print(my_dict.pop('name'))           # output: Alice
           print(my_dict.pop('gender','male'))  # output: male
         ```

         ### 3.8 dict.popitem() 方法
         该方法用于随机移除并返回字典中的一对键值对。

         ### 3.9 dict.setdefault() 方法
         该方法用于如果键不存在于字典中，则将其添加并设置值为默认值。例如：

         ``` python
           my_dict = {'name': 'Alice', 'age': 25}
           print(my_dict.setdefault('age', 30))   # output: 25
                                                # because the key exists and its value is 25
                                               # so we can't add it again with default value of 30

           print(my_dict.setdefault('gender', 'female'))  # output: female
                                                          # because the key doesn't exist yet
                                                               # so we create a new key-value pair with gender as key
         ```

         ### 3.10 dict.update() 方法
         该方法用于更新现有字典或添加新键值对到字典中。

         ### 3.11 dict.values() 方法
         该方法用于以列表返回字典所有的值。例如：

         ``` python
           my_dict = {'name': 'Alice', 'age': 25}
           for val in my_dict.values():
             print(val)   # output: Alice
                          #       25
         ```

         通过以上方法，我们对字典的相关方法有了一定的了解。

         # 4.应用场景
         有了字典的基础概念，我们现在来看看字典在哪些地方能够派上用场。

         ## 4.1 字典模拟枚举
         使用字典模拟枚举的方式非常简单。首先，定义一个字典，然后通过循环遍历字典的所有键及其对应的值。例如，假设有一个班级有三名学生：

         ``` python
           class_dict = {
             'A': ['John', 'Jack'],
             'B': ['Peter', 'Paul'],
             'C': ['Mike', 'Sarah']
           }
         ```

         可以这样遍历字典：

         ``` python
           for grade in class_dict:
             print("Grade:", grade)
             students = class_dict[grade]
             for student in students:
               print("-", student)
         ```

         执行输出结果：

         ``` python
           Grade: A
           - John
           - Jack
           Grade: B
           - Peter
           - Paul
           Grade: C
           - Mike
           - Sarah
         ```

         ## 4.2 文件映射
         将文件路径映射到数据结构，也就是将文件名作为字典的键，将文件的内容作为字典的值。下面的例子演示了如何实现这一功能：

         ``` python
           import os

           file_mapping = {}
           path = '/path/to/files/'

           files = os.listdir(path)
           for filename in files:
              full_filename = os.path.join(path, filename)

              if not os.path.isfile(full_filename):
                continue

              with open(full_filename, 'r') as f:
                  contents = f.read()

              file_mapping[filename] = contents
       ```

       此处，字典 `file_mapping` 记录了所有文件的名称和对应的内容。注意，此处忽略掉非文件类型的文件。


       ## 4.3 数据查询
       字典既可以用来保存数据，也可用来进行数据的查询。对于数据的查询，字典提供了几个比较方便的方法：

       1. 检测键是否存在
       2. 查询键对应的值
       3. 查询多个键对应的值

       下面展示几个字典查询的方法：

       ``` python
         my_dict = {'a': 1, 'b': 2, 'c': 3}

         # 判断键是否存在
         if 'd' in my_dict:
            print('Key "d" exists.')
         else:
            print('Key "d" does not exist.')

         # 查询键对应的值
         print(my_dict.get('b'))    # output: 2

         # 查询多个键对应的值
         result = [my_dict.get(k) for k in ('a', 'b', 'd')]
         print(result)              # output: [1, 2, None]
                                   # Note that when querying nonexistent key 'd', it returns None instead of raising KeyError
     ```

     上述代码展示了字典查询的几种方法。

