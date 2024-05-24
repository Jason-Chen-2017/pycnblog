
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 什么是排序？
数据处理中经常要对数据进行排序，这样才能方便地分析、统计、搜索等，排序算法一般分为内部排序和外部排序。
- 内部排序：指的是在内存中完成排序，不需要借助外存，直接在数组上进行操作即可；如直接插入排序、冒泡排序、选择排序、堆排序、归并排序、基数排序等。
- 外部排序：指的是需要先将大文件（例如超大型数据库表）读入内存，然后再进行排序，因为无法一次性将整个文件读入内存，因此需要通过一些操作手段分批次对其进行排序。如归并排序、快速排序、外部合并排序等。
### 为什么要用排序算法？
排序算法的应用非常广泛，比如：
- 电话簿排序，对联系人按姓名、电话号码排序。
- 文件夹整理，按照文件名、大小或时间顺序对文件夹中的文件进行分类。
- 数据库查询结果排序，对查询结果集按某些字段排序。
- 文本搜索引擎，对文档或网页按相关性排序。
- 游戏玩法设计，按照人气、难度、时间复杂度排序游戏角色。
所以，对数据的各种不同类型、各种规模进行排序，就是每种场景下都需要用到的算法。
### 内部排序算法有哪些？
按照排序工作原理可以分为两类：稳定排序和不稳定排序。
#### 不稳定的排序算法
- 插入排序 Insertion sort
- 希尔排序 Shell sort
- 简单选择排序 Selection sort
- 堆排序 Heap sort
#### 稳定的排序算法
- 冒泡排序 Bubble sort
- 计数排序 Counting sort
- 桶排序 Bucket sort
- 基数排序 Radix sort
### 外部排序算法有哪些？
- 归并排序 Merge sort
- 快速排序 Quick sort
- 外部合并排序 External merge sort

# 2.基本概念术语说明
## 数据结构
计算机存储数据的形式称为数据结构。常用的数据结构有：
- 线性结构：包括单链表、双链表、栈、队列、数组、串等。
- 树形结构：包括二叉树、红黑树、AVL树等。
- 图形结构：包括邻接矩阵、邻接表、十字链表、散列表等。
## 分治策略
分治策略是一种处理复杂计算任务的方式，它将一个大的问题划分成两个或多个小的子问题，递归地解决这些子问题，最后再合并得到完整的解。分治算法通常分为两类：一类是在空间复杂度方面分治，另一类是在时间复杂度方面分治。
- 归并排序算法在空间复杂度上没有任何代价，主要开销在于内存分配和复制操作，所以只适用于少量元素的排序。
- 快速排序算法在空间复杂度上要高于归并排序，但是由于在每个层级上交换数据的次数更少，所以它适合于排序海量元素。
## 工作原理
### 冒泡排序
冒泡排序是一种简单的排序算法，它的原理是依次比较相邻的元素，如果逆序则交换位置，直到没有逆序的元素为止。它的平均时间复杂度为Θ(n^2)，最坏情况的时间复杂度为Θ(n^2)。
```python
def bubble_sort(arr):
    n = len(arr)
    # Traverse through all array elements
    for i in range(n):
        # Last i elements are already sorted
        for j in range(0, n - i - 1):
            # Swap if the element found is greater than the next element
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]

```
### 插入排序
插入排序也是一种简单排序算法，它的原理是构建有序序列，对于第i个待排序记录，在已排序序列中从后向前扫描，找到相应位置并插入。插入排序的时间复杂度为Θ(n^2)。
```python
def insertion_sort(arr):
    n = len(arr)
    
    # Traverse through 1 to len(arr)
    for i in range(1, n):
        
        key = arr[i]
        
        # Move elements of arr[0..i-1], that are
        # greater than key, to one position ahead
        # of their current position
        j = i - 1
        while j >= 0 and key < arr[j] :
                arr[j + 1] = arr[j]
                j -= 1
        arr[j + 1] = key
        
```
### 选择排序
选择排序也是一种简单排序算法，它的原理是通过遍历数组，找到最小值或者最大值，然后放置到左侧。选择排序的时间复杂度为Θ(n^2)。
```python
def selection_sort(arr):
    n = len(arr)
    
    # Traverse through all array elements
    for i in range(n):
        
        # Find the minimum element in remaining unsorted array
        min_idx = i
        for j in range(i+1, n):
            if arr[min_idx] > arr[j]:
                min_idx = j
                
        # Swap the found minimum element with the first element         
        arr[i], arr[min_idx] = arr[min_idx], arr[i]
        
```
### 快速排序
快速排序是一种基于分治策略的排序算法，它的原理是选取一个枢纽元（pivot），把数组分成两个子数组，左边的子数组元素都比枢纽元小，右边的子数组元素都比枢纽元大。递归地对两个子数组进行相同的操作。快速排序的时间复杂度为Θ(nlogn)。
```python
def partition(arr, low, high):
    i = (low - 1)         # index of smaller element
    pivot = arr[high]      # pivot
 
    for j in range(low, high):
 
        # If current element is smaller than or
        # equal to pivot
        if   arr[j] <= pivot:
 
            # increment index of smaller element
            i = i+1
            arr[i],arr[j] = arr[j],arr[i]
 
    arr[i+1],arr[high] = arr[high],arr[i+1]
    return i+1
 
def quickSort(arr, low, high):
    if low < high:
 
        # pi is partitioning index, arr[p] is now
        # at right place
        pi = partition(arr, low, high)
 
        # Separately sort elements before
        # partition and after partition
        quickSort(arr, low, pi-1)
        quickSort(arr, pi+1, high)
    
```