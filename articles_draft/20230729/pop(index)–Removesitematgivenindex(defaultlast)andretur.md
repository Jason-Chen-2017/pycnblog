
作者：禅与计算机程序设计艺术                    

# 1.简介
         
 在编程语言中，列表（List）、集合（Set）及字典（Dictionary）等数据结构都有着自己的一些方法。其中，列表最常用的方法之一是`pop()`方法，该方法用于删除并返回列表中的某个元素或索引位置处的元素，默认是最后一个元素。
          
          `pop()` 方法接收两个参数：
          1. 要删除的元素的索引值(index)，或者
          2. 不传入任何参数时，表示删除列表末尾元素。
          
          例如，下面的代码删除列表中的第2个元素:
          
          ```python
          my_list = [1, 'a', True, False]
          removed_element = my_list.pop(1)
          print("Removed Element:", removed_element)
          print("New List:", my_list)
          ```
          
          上述代码将输出：
          
          ```python
          Removed Element: a
          New List: [1, True, False]
          ```
          
        # 2. 基本概念术语说明
        
        ## 描述
        
        Python 语言中的 `pop()` 方法可以用来从列表中删除指定索引处的元素，并且可以选择是否返回被删除的元素。当不传参给 `pop()` 方法时，会默认删除列表末尾元素。如果索引超出了范围，则抛出 `IndexError` 异常。
        
        下面让我们具体地了解一下这个方法的使用方式、参数含义和函数签名。
        
        
        ## 使用方式
        ### pop()
        
        `pop(index)`

        Remove the item at the specified position in the list, and return it. If no index is specified, remove and return the last item. Raises IndexError if list is empty or index is out of range.
        
        |   Argument    |                           Description                             | Default Value | Example           |
        |:-------------:|:-----------------------------------------------------------------:|:--------------|:------------------|
        |     index     | Index of the element to be deleted.<br>可以是一个整数也可以是切片对象，此时将返回被删除元素组成的列表。|      -1       |<ul><li>`pop(2)`</li><li>`pop(-1)`</li><li>`pop([1,2])`</li></ul>|
        
        Returns the removed element.

        ### 函数签名

        def pop(self, __index=None):
            """Removes and returns item at index (default last). Raises IndexError if list is empty or index is out of range."""
            pass
        
        
        
    # 3.核心算法原理和具体操作步骤
    ## 基本思路
    
    根据上面的描述，我们知道 `pop()` 方法有两种不同的使用方式，一种是通过索引删除元素，另一种是删除列表末尾元素。为了实现这两种功能，我们需要考虑以下几点：
    
    1. 如果没有指定索引值，则删除列表末尾元素；
    2. 如果指定的索引值越界，则抛出 `IndexError` 异常；
    3. 删除元素后，应该返回被删除的元素。
    
    通过以上分析，我们得到如下的基本思路：
    
    1. 检查列表是否为空，如果为空，则抛出 `IndexError` 异常；
    2. 判断索引值是否有效，如果索引值有效，则删除对应元素并返回；否则，删除列表末尾元素；
    3. 返回被删除的元素。
    
	## 具体实现
    
    源码文件为`/Lib/collections/__init__.py`，具体实现如下：
    
    
    class MutableSequence(object):
    
       ...
        
        def pop(self, *args):
            """Remove and return item at index (default last).
            
            Raises IndexError if list is empty or index is out of range.
            """
            l = len(self)
            i = args[0] if args else -1
        
            if not isinstance(i, int):
                raise TypeError('indices must be integers')

            if i < 0:
                i += l
                if i < 0:
                    raise IndexError('pop index out of range')
            elif i >= l:
                raise IndexError('pop index out of range')

            value = self[i]
            del self[i]
            return value
            
    此外，还可以用切片语法直接删除多个元素，示例如下：

    
    >>> nums = [1, 2, 3, 4, 5]
    >>> del nums[:2]
    >>> print(nums)
    [3, 4, 5]

    可以看到，上述代码删除了列表中的前两项，并将其赋值给新的变量。

