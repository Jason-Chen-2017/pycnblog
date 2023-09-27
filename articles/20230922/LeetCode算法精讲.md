
作者：禅与计算机程序设计艺术                    

# 1.简介
  

LeetCode是一个帮助训练力求编程能力的平台。通过提交题目并通过测试用例的形式反馈自己的解决方案，参与到高频面试中。本文将详细介绍LeetCode上经典算法的设计思路、主要的应用场景、关键数据结构的实现等。希望能够给读者提供一些思路和参考。
# 2.基本概念术语说明
## 2.1. 数组
数组（Array）是一种线性表数据结构，其中的元素按一定顺序排列，每个元素都有一个唯一的索引，可以随机访问。在计算机科学中，数组广泛应用于计算机程序的方方面面。包括内存分配，数据存储，多维数组处理等领域。数组有两个重要的特征：有序性和元素唯一性。数组的有序性意味着可以通过下标索引找到对应的元素；元素唯一性表示相同元素只能出现一次。数组的定义如下: 

`array[n] = {value_1, value_2,..., value_m}` 

其中 n 表示数组的长度，即数组中的元素个数。对于连续的整数范围，通常可以使用一个整型变量作为数组的索引，从 0 开始计数。数组元素的最大值和最小值可以通过数组的头尾指针计算出来。比如，数组 a 的第一个元素 a[0] 和最后一个元素 a[n-1] 。

数组的两种主要操作：

1. 插入操作：向数组插入一个新的元素。
2. 删除操作：删除数组中的某个元素。

数组的容量（capacity）就是指数组能够存放的元素个数。当数组的元素个数超过它的容量时，需要扩充它的容量。数组的扩充方式一般有三种：

1. 自动扩充：当数组元素个数超过当前的容量时，自动创建一个新的数组，将老数组的所有元素拷贝到新数组中，然后释放老数组的空间。这种扩充方式非常简单，无需额外的存储开销。但它存在时间复杂度的问题，可能会导致数组扩充的时间过长。
2. 分配更多空间：将现有的数组的容量乘以 2 或 其他增长因子，创建出新的数组，将老数组的元素拷贝到新数组中，然后释放老数组的空间。这种扩充方式不需要移动数据，只需要重新分配内存。时间复杂度为 O(n)。
3. 动态分配：在运行过程中，根据需要动态分配数组的容量。这种扩充方式适用于数组大小不固定的情况，可避免产生大量的小对象。时间复杂度为 O(1)。

## 2.2. 链表
链表（Linked List）也是一种线性表数据结构，但是它不同于一般的数组，因为链表中的元素不仅记录了元素的值还记录了指向下一个元素的位置。链表最重要的特征是每个节点都有一个指针，这个指针指向下一个节点，从而构成一个链式结构。链表的定义如下:

```cpp
struct Node{
    int data; // 数据域
    struct Node *next; // 指针域
};
```

链表由多个结点组成，每个结点除保存数据之外，还有两个指针域，一个指向下一个结点，另一个指向任意结点的前驱结点。链表的特点是元素零散地分布在内存中，靠近一起的结点通常会被放在同一个物理页或缓存行中。链表相比于数组更加灵活，对数据的添加和删除操作都可以在头部进行，但对中间的数据进行操作效率较低。由于每个结点都有一个指针域，因此链表具有更好的灵活性，在很多情况下都可以取代数组。

## 2.3. 栈
栈（Stack）是一种线性数据结构，只有两个操作：压栈和弹栈。栈顶元素是栈的“最近”元素，也就是说最新添加的元素。当栈为空时，不能再弹栈。栈的基本操作是 Push() 入栈和 Pop() 出栈。栈的应用很广泛，如函数调用栈，表达式运算栈， undo/redo 操作栈，浏览器的前进后退栈，打印机的打印队列等。栈的定义如下：

```cpp
template <typename T> class Stack {
  private:
    vector<T> stack;
  public:
    void push(T x) {
        stack.push_back(x);
    }

    T pop() {
        if (stack.empty()) throw "pop from empty stack";
        return stack.back();
    }
    
    bool isEmpty() {
        return stack.empty();
    }
}
```

其中 `vector<T>` 是 STL 中的模板类，用来保存栈内元素。`push()` 方法向栈顶增加元素，`pop()` 方法删除栈顶元素，若栈为空则抛出异常。`isEmpty()` 方法判断栈是否为空。

## 2.4. 队列
队列（Queue）是一种先进先出的线性数据结构。队列中的元素按照先进先出的次序进行排序。队首元素（队头）在队列中排在第一位，队尾元素（队尾）在队列中排在最后一位。只允许在队尾添加元素，只允许在队首删除元素，队尾进队，队头出队。队列的应用也十分广泛，如 CPU 中断请求队列，磁盘 IO 请求队列，页面置换算法使用的页面换出请求队列等。队列的定义如下：

```cpp
template <typename T> class Queue {
  private:
    queue<T> q;
  public:
    void enqueue(T x) {
        q.push(x);
    }

    T dequeue() {
        if (q.empty()) throw "dequeue an empty queue";
        auto x = q.front();
        q.pop();
        return x;
    }
    
    bool isEmpty() {
        return q.empty();
    }
}
```

其中 `queue<T>` 是 STL 中的模板类，用来保存队列的元素。`enqueue()` 方法向队尾加入元素，`dequeue()` 方法删除队首元素，若队列为空则抛出异常。`isEmpty()` 方法判断队列是否为空。

## 2.5. 哈希表
哈希表（Hash Table）是一个基于键值对的无序集合。在哈希表中，每个键都对应一个值。键的作用是查找值，值可以是任何类型。哈希表通过把键映射到数组的位置来快速查找。哈希表的性能依赖于哈希函数，如果选择不好哈希函数，就会造成冲突，使得查询变慢。所以，哈希表应尽可能保证均匀分布，而且哈希函数应快速、散列地址应该尽量减少碰撞。哈希表的定义如下：

```cpp
class HashTable {
  private:
    unordered_map<string, int> table_;

  public:
    void insert(const string& key, const int val) {
      table_[key] = val;
    }

    int search(const string& key) {
      if (!table_.count(key)) return -1;
      return table_[key];
    }
};
```

其中 `unordered_map<K, V>` 是 C++ 中的 unordered_map 类，用来保存键值对。`insert()` 方法向哈希表插入键值对，`search()` 方法搜索指定键对应的值。

## 2.6. 二叉树
二叉树（Binary Tree）是一种树形数据结构，它的每个节点最多有两个子节点，分别是左孩子和右孩子。二叉树的高度定义为树中所有叶子结点的最长距离。在二叉树中，每个节点的左、右子树也是二叉树。二叉树的应用很广泛，如语法分析器、压缩算法、二分查找树、平衡二叉树、BST等。二叉树的定义如下：

```cpp
struct TreeNode {
    int val;
    TreeNode* left;
    TreeNode* right;
    TreeNode(int x):val(x),left(NULL),right(NULL){}
};
```

其中 `TreeNode` 是二叉树的结点结构体，`val` 属性保存结点值，`left` 和 `right` 属性分别指向左右孩子结点。

## 2.7. 堆
堆（Heap）是一种特殊的二叉树，堆中每个结点的值都要满足特定关系（小根堆或者大根堆）。对于小根堆来说，父节点的值永远小于或等于它的两个子节点的值；对于大根堆来说，父节点的值永远大于或等于它的两个子节点的值。堆的应用十分广泛，如优先级队列、调度算法、快速排序中的堆排序、图论中的堆算法、优化问题中的堆优化、搜索引擎中的倒排索引、股票交易中的最佳买卖点问题等。堆的定义如下：

```cpp
class MaxHeap {
  private:
    vector<int> heap;

  public:
    MaxHeap():heap({0}) {}

    void push(int val) {
        heap.push_back(val);
        swimUp(heap.size()-1);
    }

    int peek() {
        return heap[1];
    }

    int poll() {
        swap(heap[1], heap[heap.size()-1]);
        auto res = heap.back();
        heap.pop_back();
        sinkDown(1);
        return res;
    }

    size_t size() const {
        return heap.size()-1;
    }

    bool empty() const {
        return heap.size() == 1;
    }

  private:
    static int parentIdx(int idx) {
        return idx / 2;
    }

    static int leftChildIdx(int idx) {
        return idx * 2;
    }

    static int rightChildIdx(int idx) {
        return idx * 2 + 1;
    }

    void swimUp(int k) {
        while (k > 1 && heap[parentIdx(k)] < heap[k]) {
            swap(heap[parentIdx(k)], heap[k]);
            k = parentIdx(k);
        }
    }

    void sinkDown(int k) {
        while (leftChildIdx(k) <= size()) {
            int j = leftChildIdx(k);
            if (j+1 <= size() && heap[j+1] > heap[j])
                j++;
            if (heap[j] >= heap[k]) break;
            swap(heap[j], heap[k]);
            k = j;
        }
    }
};
```

其中 `MaxHeap` 是大根堆的实现。`push()` 方法向堆中添加元素，`peek()` 方法查看堆顶元素，`poll()` 方法删除堆顶元素，`size()` 方法返回堆中元素个数，`empty()` 方法判断堆是否为空。堆的实现采用父节点与子节点交换的方式保持堆有序，且每次操作都是 O(log N) 的时间复杂度。

# 3. 基本算法
## 3.1. 搜索（Search）
### 3.1.1. 顺序搜索（Sequential Search）
顺序搜索（Sequential Search）是最简单的搜索算法，其工作原理是从列表的起始位置开始，依次比较每一个元素，直到找到目标元素或搜索完整个列表。顺序搜索算法的平均时间复杂度是 O(N)，最坏情况下时间复杂度是 O(N)。

代码如下：

```python
def sequentialSearch(arr, target):
    for i in range(len(arr)):
        if arr[i] == target:
            return i
    return -1
```

### 3.1.2. 有序数组搜索（Sorted Array Search）
有序数组搜索（Sorted Array Search）是顺序搜索的一个特殊情况。如果待查序列已经升序或降序排好序，并且数组大小不会太大，则可以使用有序数组搜索算法，其时间复杂度是 O(log N)。

代码如下：

```python
def sortedArraySearch(arr, target):
    start, end = 0, len(arr)-1
    while start <= end:
        mid = (start + end) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            start = mid + 1
        else:
            end = mid - 1
    return -1
```

该算法通过二分法搜索，每次缩小待搜索区间的半径，这样在平均时间复杂度 O(log N) 下可确保时间复杂度最优。

### 3.1.3. 查找第一个和最后一个Occurrence of Element in Sorted Array
查找第一个和最后一个Occurrence of Element in Sorted Array 是为了解决在有序数组中寻找第一个和最后一个元素出现的位置。在有序数组中寻找一个元素，一般都采用二分法搜索，但是此处采用顺序搜索的方式来寻找。

代码如下：

```python
def findFirstAndLastOccurrences(arr, target):
    firstIndex = lastIndex = -1
    for i in range(len(arr)):
        if arr[i] == target and firstIndex == -1:
            # Find the First Occurrence
            firstIndex = i
        elif arr[i] == target and firstIndex!= -1:
            # Find the Last Occurrence
            lastIndex = i
    return [firstIndex, lastIndex]
```

该算法首先初始化 firstIndex 和 lastIndex 为 -1，然后遍历整个数组，当遇到等于target的元素时，判断firstIndex是否为-1，如果是，则记录当前元素的索引值。否则，记录lastIndex。最后返回结果。

### 3.1.4. 二分查找
二分查找（Binary Search）是一种在有序数组中查找特定元素的算法。二分查找算法要求数组必须是有序的。数组的中间元素确定了搜索范围。若数组中间元素正好是要查找的元素，则搜索结束；否则，如果中间元素大于要查找的元素，则搜索左侧部分；否则，搜索右侧部分。重复以上过程，直至找到目标元素或区间不存在目标元素为止。二分查找算法的平均时间复杂度是 O(log N)，最坏情况下时间复杂度是 O(N)。

代码如下：

```python
def binarySearch(arr, low, high, target):
    """
    :type nums: List[int]
    :type target: int
    :rtype: int
    """
    if high < low:
        return -1
    mid = (high + low) // 2
    if arr[mid] == target:
        return mid
    elif arr[mid] > target:
        return binarySearch(arr, low, mid - 1, target)
    else:
        return binarySearch(arr, mid + 1, high, target)
```

该算法使用递归的方法，以数组中间元素为界限，把搜索空间缩小为原来的一半。如果中间元素正好是目标元素，则停止搜索，返回索引值；否则，继续搜索左侧还是右侧区域，直到找到目标元素或区间不存在目标元素为止。

## 3.2. 插入（Insertion）
### 3.2.1. 插入排序
插入排序（Insertion Sort）是一种简单排序算法。它的工作原理是构建有序序列，对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入。插入排序在实现上，通常采用直接插入方式，即插到已排序序列的适当位置。

代码如下：

```python
def insertionSort(arr):
    for i in range(1, len(arr)):
        temp = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > temp:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = temp
```

该算法通过不断迭代逐个将无序序列中的元素插入到前面已排序序列的适当位置，最终使整个序列变为有序序列。

### 3.2.2. 希尔排序
希尔排序（Shell Sort）是插入排序的一种更高效的改进版本。希尔排序通过对插入排序的折叠操作来提高效率。

代码如下：

```python
def shellSort(arr):
    sublistCount = len(arr)//2
    while sublistCount > 0:

        for startIndx in range(sublistCount):

            gapInsertionSort(arr, startIndx, sublistCount)

        print("After increments of size", sublistCount, "The list is ", arr)

        sublistCount = sublistCount // 2

def gapInsertionSort(arr, start, gap):
    for i in range(start+gap, len(arr), gap):

        currentValue = arr[i]
        position = i

        while position>=gap and arr[position-gap]>currentValue:
            arr[position]=arr[position-gap]
            position=position-gap

        arr[position]=currentValue
```

该算法在插入排序的基础上，通过步长逐渐缩短分组距离来将列表划分成若干个子列表，然后使用插入排序来对各个子列表独立排序，最后合并得到整个有序序列。