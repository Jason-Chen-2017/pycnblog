
作者：禅与计算机程序设计艺术                    

# 1.简介
  

最大堆是一个数组，满足对于任意节点i，其父亲节点的索引值等于（i-1)//2，并且该节点的值不小于其子节点的值。用数组表示最大堆时，数组第一个元素的索引值就是0，而最后一个元素的索引值等于（n-1)//2。通过树状数组的形式也可以方便地实现最大堆结构。最大堆的操作主要包括插入、删除、取最大值操作。
在很多数据结构中都可以看到最大堆的应用，如排序、优先队列等。本文首先会对最大堆的定义及性质进行阐述，然后介绍对最大堆进行插入和删除操作的方法，最后通过堆排序给出堆排序的具体步骤。另外，文章还将会给出最大堆排序算法的python实现。
# 2.定义及性质
## 2.1 最大堆定义
设有一个数组arr[0..n-1],n>=1。若满足如下条件的数组arr称为最大堆，那么它有如下性质：

1）最大堆的根节点必为最大值；

2）以根节点为最高点的子树为最大堆；

3）arr[i] <= arr[(i+1)//2] 所有 i=0~n-1；

4）从索引 (n/2)-1 到 n-1 的节点构成了最大堆。

## 2.2 最大堆操作
### 2.2.1 插入操作
插入操作是在最大堆末尾插入一个新节点之后，对其重新调整使之成为最大堆的过程。
假定已经有一个最大堆A的根节点的值为v，新的节点的值为w，则可以按照以下方法更新最大堆A:

1. 将w添加到最大堆A的末尾;

2. 如果w比v小，则不用做任何事情，否则执行以下步骤:

    a. 将w值赋值给当前节点的父节点的值，并将当前节点索引赋值给变量x;
    
    b. 重复以下步骤直到x的值等于0或w比v大:
    
        i. 判断父节点的值是否小于w，如果是，则交换w和父节点的值;
        
        ii. 计算新的父节点索引：x = (x-1)/2;
        
        iii. 回到第b步继续执行循环操作。
        
3. 此时最大堆A中的根节点一定是最大值，无需再做调整。因此，当某个元素被加入到最大堆A后，其必然是一个最大值。

### 2.2.2 删除操作
删除操作需要两个步骤，即找到要删除的元素，然后进行调整使得堆保持完整性。

#### 2.2.2.1 找出最大元素
首先找到最大元素，然后把堆最后一个元素放到这个位置，然后进行调整使得堆仍然保持最大堆性质。

#### 2.2.2.2 求得父节点
求得父节点比较简单，只需要注意的是父节点索引要向下取整除2即可。

#### 2.2.2.3 交换元素
因为找到最大元素和它的父节点之后，就可以直接交换两者的值。然后对交换后的元素执行调整操作，保证它仍然是一个最大堆。

重复以上三个步骤，直到堆的大小减少至1，此时堆就为空，其定义为根节点为-1，其他元素也没有意义。

### 2.2.3 堆排序
堆排序是将一个无序数组构建为最大堆，再将根节点放到数组的最后，最后一次进行删除操作。完成一次堆排序之后，数组中的最后一个元素就是最大值，因此只需将堆尾元素与数组最后元素交换，然后在重复上述步骤，即可实现另一个删除操作，一直重复，直到堆为空。

## 2.3 Python实现
为了便于理解，我会分别使用数组表示最大堆，使用列表表示堆。同时，为了验证算法的正确性，我会提供两种堆排序算法：一种是迭代版本，采用while循环；另一种是递归版本，采用函数调用。

```python
def build_max_heap(arr):
    '''建立最大堆'''
    for i in range((len(arr)+1)//2-1,-1,-1):
        _max_heapify(arr,i,len(arr))
            
def insert(arr,val):
    '''插入新元素'''
    # 在数组尾部增加新元素
    arr.append(-1)
    # 对新增元素执行插入操作
    __insert(arr,val,len(arr)-1)
    
def delete(arr):
    '''删除最大元素'''
    if len(arr)>1:
        val = arr[0]
        # 把根节点和数组尾元素交换
        arr[0],arr[-1] = arr[-1],arr[0]
        del arr[-1]
        # 执行删除操作
        __delete(arr,0,len(arr)-1)
        return val
    else:
        raise Exception('Heap underflow')
        
def _max_heapify(arr,root,size):
    '''迭代版本的建立最大堆'''
    left = root*2+1
    right = root*2+2
    largest = root
    while left < size and right < size:
        if arr[left]>arr[right]:
            largest = left
        elif arr[right]>arr[left]:
            largest = right
        else:
            break
        if arr[largest]>arr[root]:
            arr[largest],arr[root] = arr[root],arr[largest]
            root = largest
        else:
            break
        left = root*2+1
        right = root*2+2
        
def _build_max_heap(lst):
    '''迭代版本的建立最大堆'''
    lstlen = len(lst)
    lastparent = (lstlen - 2) // 2
    for node in range(lastparent, -1, -1):
        _siftup(lst, node, lstlen)
                
def _insert(lst,val):
    '''迭代版本的插入元素'''
    lst.append(-1)
    pos = len(lst) - 1
    parent = (pos - 1) // 2
    while pos > 0 and lst[parent] < val:
        lst[pos] = lst[parent]
        pos = parent
        parent = (pos - 1) // 2
    lst[pos] = val
                            
def _delete(lst):
    '''迭代版本的删除最大元素'''
    if len(lst)<2:
        raise Exception('Heap Underflow')
    ret = lst[0]
    lst[0] = lst[-1]
    lst.pop()
    siftdown(lst,0,len(lst))
    return ret

def _siftup(lst, pos, lstlen):
    '''迭代版本的堆调整'''
    endpos = lstlen - 1
    startpos = pos
    newitem = lst[pos]
    while pos > 0:
        parentpos = (pos - 1) // 2
        parent = lst[parentpos]
        if parent >= newitem:
            break
        lst[pos] = parent
        pos = parentpos
    lst[pos] = newitem
                 
def sort_iteratively(arr):
    '''迭代版本的堆排序'''
    n = len(arr)
    # 建堆
    build_max_heap(arr)
    # 遍历堆，把最大元素放到数组尾部
    for i in range(n-1,0,-1):
        arr[0],arr[i] = arr[i],arr[0]
        _delete(arr[:i])

def siftup(lst, start, end):
    """递归版本的堆调整"""
    root = start
    while True:
        child = root * 2 + 1
        if child > end or lst[child] > lst[root]:
            swap(lst, child, root)
            root = child
        else:
            break
                     
def heapsort_recursive(lst):
    """递归版本的堆排序"""
    def _heapsift(start, end):
        nonlocal lst
        mid = (start + end) // 2
        siftup(lst, start, mid)
        siftup(lst, mid + 1, end)
        if start < mid and lst[mid] < lst[mid + 1]:
            swap(lst, mid, mid + 1)
        _heapsift(start, mid)
        _heapsift(mid + 1, end)
        
    def _heapsort(start, end):
        nonlocal lst
        if start == end:
            pass
        else:
            mid = (start + end) // 2
            _heapsort(start, mid)
            _heapsort(mid + 1, end)
            merge(lst, start, mid, end)
            
    n = len(lst)
    _heapsift(0, n - 1)
    _heapsort(0, n - 1)
        
def main():
    import random
    arr = [random.randint(1, 100) for i in range(10)]
    print("原始数组:", arr)
    build_max_heap(arr)
    print("建立最大堆后的数组:", arr)
    insert(arr,99)
    print("插入新元素后的数组:", arr)
    val = delete(arr)
    print("删除最大元素后剩余的元素:", val)
    print("删除最大元素后的数组:", arr)
    arr = list(range(10, 0, -1))
    print("待排序数组:", arr)
    sort_iteratively(arr)
    print("迭代版堆排序后的数组:", arr)
    arr = list(range(10, 0, -1))
    print("待排序数组:", arr)
    heapsort_recursive(arr)
    print("递归版堆排序后的数组:", arr)
    
if __name__=="__main__":
    main()
```