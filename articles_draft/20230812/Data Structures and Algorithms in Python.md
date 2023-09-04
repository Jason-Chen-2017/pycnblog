
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Python作为一门高级语言,拥有丰富的数据结构和算法库,在科学计算、数据分析领域具有重要作用.本文将从浅到深,以最通俗易懂的方式阐述常用的Python数据结构和算法.
# 2.基本概念术语说明
## 2.1 数据结构
- List(列表): 是一种有序的集合,其元素可以重复,允许同名元素存在.可以使用方括号[]表示.
- Tuple(元组): 和List类似,但是Tuple是不可变的.可以使用圆括号()表示.
- Set(集合): 是一种无序的集合,其元素没有重复性,不允许同名元素存在.可以使用花括号{}表示.
- Dictionary(字典): 是一种键值对的存储方式,其中每个键对应一个值,多个键可以对应相同的值.可以使用大括号{}表示.
## 2.2 算法
- Searching: 包括二分查找、顺序查找、线性搜索.
- Sorting: 包括快速排序、冒泡排序、归并排序、堆排序等多种排序算法.
- Trees: 包括树的遍历(先序、中序、后序)及广度优先/深度优先搜索.
- Graphs: 包括图的深度优先搜索、广度优先搜索、最短路径搜索等算法.
## 3.核心算法原理
### 3.1 搜索算法(Searching)
#### 3.1.1 二分查找(Binary search)
二分查找法是指在升序排列的数组中,假设它中间的元素就是目标值,则只需比较左半边或右半边即可确定该值是否存在,缩小范围直至找到或者不存在.其时间复杂度为O(log n).
```python
def binary_search(arr, x):
    low = 0
    high = len(arr)-1
    while low <= high:
        mid = (low + high)//2
        if arr[mid] == x:
            return mid
        elif arr[mid] < x:
            low = mid + 1
        else:
            high = mid - 1
    return -1 # element not found
```
#### 3.1.2 顺序查找(Sequential search)
顺序查找法也是通过依次访问数组中的元素,判断是否有目标元素.其时间复杂度为O(n),适用于较小规模的数据集.
```python
def sequential_search(arr, x):
    for i in range(len(arr)):
        if arr[i] == x:
            return i
    return -1 #element not found
```
#### 3.1.3 线性搜索(Linear search)
线性搜索法也称单步搜索,即一次检查整个数组,直到找到或查找完毕.其时间复杂度为O(n),一般用在数据量较少且关键字分布随机的情况下.
```python
def linear_search(arr, x):
    for i in range(len(arr)):
        if arr[i] == x:
            return i
    return -1 #element not found
```
### 3.2 排序算法(Sorting)
#### 3.2.1 快速排序(Quick sort)
快速排序是指对数组进行递归地分割,并使得左侧比Pivot小,右侧比Pivot大.然后再分别对左右两边的数组继续上面的过程,最后得到排好序的数组.其时间复杂度为O(n log n).
```python
def quicksort(arr, low, high):
    if low < high:
        pi = partition(arr, low, high)
        quicksort(arr, low, pi-1)
        quicksort(arr, pi+1, high)
        
def partition(arr, low, high):
    pivot = arr[high] # pivot can be any value from the array
    i = low-1 
    for j in range(low, high): 
        if arr[j] < pivot: 
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i+1], arr[high] = arr[high], arr[i+1]
    return i+1
```
#### 3.2.2 冒泡排序(Bubble sort)
冒泡排序是通过相邻元素的比较和交换,进行无限循环直至数组完全排序.其时间复杂度为O(n^2),非常低效,但简单易于实现.
```python
def bubbleSort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
```
#### 3.2.3 归并排序(Merge sort)
归并排序是指将两个有序数组合并成一个大的有序数组.其时间复杂度为O(n log n),比快速排序稍慢但是效率更高.
```python
def mergeSort(arr):
    if len(arr)>1:
        mid = len(arr)//2
        L = arr[:mid]
        R = arr[mid:]

        mergeSort(L)
        mergeSort(R)

        i=j=k=0

        while i<len(L) and j<len(R):
            if L[i]<R[j]:
                arr[k]=L[i]
                i+=1
            else:
                arr[k]=R[j]
                j+=1
            k+=1

        while i<len(L):
            arr[k]=L[i]
            i+=1
            k+=1

        while j<len(R):
            arr[k]=R[j]
            j+=1
            k+=1
```
#### 3.2.4 堆排序(Heap sort)
堆排序是指先构建一个最大堆或最小堆,然后将堆顶元素和末尾元素互换,再逐渐减少堆的大小,直到只剩下一个元素为止.其时间复杂度为O(n log n).
```python
import heapq as h
def heapify(arr, n, i):
    largest = i # Initialize largest as root
    l = 2 * i + 1     # left = 2*i + 1
    r = 2 * i + 2     # right = 2*i + 2
 
    # If left child is larger than root
    if l < n and arr[l][1] > arr[largest][1]:
        largest = l
 
    # If right child is larger than largest so far
    if r < n and arr[r][1] > arr[largest][1]:
        largest = r
 
    # Change root, if needed
    if largest!= i:
        arr[i],arr[largest] = arr[largest],arr[i]  # swap
 
        # Heapify the root.
        heapify(arr, n, largest)
 
def heapSort(arr):
    n = len(arr)
     
    # Build a maxheap.
    for i in range(n//2 - 1, -1, -1):
        heapify(arr, n, i)
 
    # One by one extract elements
    for i in range(n-1, 0, -1):
        arr[i], arr[0] = arr[0], arr[i]   # swap
        heapify(arr, i, 0)
```