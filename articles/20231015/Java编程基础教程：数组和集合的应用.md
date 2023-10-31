
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


## 一、什么是数组？
数组（Array）是一种数据结构，它用于存储同类型元素的集合，它可以根据需要自动增长或收缩其容量，也能通过下标访问其中的元素。例如，如果需要保存学生信息，我们可以用一个学生数组来存储这些信息，其中每个元素代表一个学生对象。如此一来，无论有多少学生，都只需定义一个数组即可。当然，数组也有一些缺点，比如固定大小的限制、效率低下等。除此之外，数组还支持多维数组，这样就可以将不同类型的数据存储在一个二维数组中。

## 二、什么是集合？
集合（Collection）是指能够存放多个元素的容器。在面向对象编程领域，集合是用来管理对象集合的接口，常用的集合类有List、Set、Queue和Map等。

## 三、为什么要使用数组和集合？
在实际开发过程中，经常会遇到需要处理大量数据的情况，对于大量数据的操作，数组和集合都会起到比较大的作用。以下几种场景可以考虑使用数组和集合：
1、储存大量相同类型的数据。由于数组和集合都是对内存空间进行分配，所以对于相同类型的数据，用数组或者集合去存储会比用其他方式节省大量的内存空间。

2、随机访问。对于需要随机访问大量数据的场合，数组和集合都会更快捷、高效。如排序算法、查找算法等都依赖于数组和集合的快速访问能力。

3、增删改查操作。当需要频繁的添加、删除和修改数组或者集合中的元素时，数组和集合的优势就会体现出来。如游戏中的物品装备、热门搜索排行榜等。

以上就是数组和集合的一些基本概念。接下来我们一起学习一下数组和集合的一些基本操作。

# 2.核心概念与联系
## 一、数组
### （1）概述
数组是计算机编程语言中的重要组成部分。它是一个固定长度的顺序容器，可以存储多个相同类型的值。数组中的每一项称为元素，可以通过索引访问数组的元素。
### （2）特点
- 数组拥有固定的长度；
- 可以通过索引直接访问数组的元素；
- 数组中的元素是相同的数据类型；
### （3）声明数组
```java
dataType[] arrayName = new dataType[arrayLength]; // int[] arr = new int[10]; // string[] strArr = new string[10];
```
- 数据类型：表示数组中所存储数据的类型，只能是整数型、浮点型、字符型、布尔型或引用型；
- 数组名：数组名称，以数组元素的类型作为前缀，一般采用小写字母开头，如int[] scores;
- 数组长度：数组中可存储元素的个数，必须是大于零的正整数。
### （4）初始化数组
可以在创建数组之后立即给其赋值，也可以使用一个循环语句依次赋值。以下示例展示了两种初始化方法：
```java
// 方法一
int[] nums = {1, 2, 3};
String[] fruits = {"apple", "banana"};

// 方法二
int size = 5;
double[] prices = new double[size];
for (int i = 0; i < size; i++) {
    prices[i] = (double) (Math.random() * 100);
}
```
- 初始化数组元素：可以在创建数组的同时，直接指定数组的元素值。上面的代码中，分别初始化了一个int数组和一个String数组。注意：数组元素的个数和数组长度不一致时，会导致编译出错。
- 使用循环语句赋值：可以使用循环语句逐个赋值给数组元素，这样可以设置不同的初始值。

### （5）读取数组元素
```java
arr[index]
```
- index：索引从0开始。若超出数组范围，则会出现ArrayIndexOutOfBoundsException异常。
- 返回数组元素：可以通过数组下标进行访问，返回对应位置的数组元素值。

### （6）改变数组元素
```java
arr[index] = value
```
- index：索引从0开始。若超出数组范围，则会出现ArrayIndexOutOfBoundsException异常。
- 修改数组元素：可以通过数组下标修改数组元素的值。

### （7）空元素占据的空间
数组是由连续的内存单元组成，因此空元素并不会真正占据额外的内存。数组的长度只是表明该数组可以存储的元素的个数，并不是实际占用的内存空间大小。

### （8）多维数组
Java允许创建多维数组，也就是数组的数组。多维数组中的各个元素也是按行优先顺序排列。

```java
dataType[][] multiDimArray = new dataType[row][column];
```
- row: 行数。
- column: 列数。

### （9）数组排序
Java提供了Arrays.sort()方法来对数组排序。该方法利用了快速排序算法实现对数组的排序，时间复杂度为O(nlogn)。

```java
import java.util.Arrays;

public class ArrayDemo {

    public static void main(String[] args) {

        int[] numbers = {4, 2, 1, 6, 8, 9, 3, 5};
        System.out.println("Original array:");
        Arrays.stream(numbers).forEach(System.out::print);
        
        Arrays.sort(numbers);
        System.out.println("\nSorted array:");
        Arrays.stream(numbers).forEach(System.out::print);
    }
}
```

输出结果：

```
Original array:
42168935 
Sorted array:
123456789
```

### （10）线性探测法
如果要确定某个数组是否存在某个元素，通常会遍历整个数组，直到找到该元素或者发现不存在该元素为止。这种查找方式叫做线性探测法，平均时间复杂度为O(n)。但是，假设数组满负载，所有的元素都被其他元素填充，这时候使用线性探测法查找会非常慢。为了避免线性探测法的时间复杂度过高，可以考虑使用散列表来代替数组。

### （11）交换两个变量的值
一般情况下，我们可以通过如下的方式交换两个变量的值：

```java
int temp = a;
a = b;
b = temp;
```

不过，由于Java提供的内置类型变量之间的转换器，变量的交换操作可以写得更加简洁：

```java
a = a + b;
b = a - b;
a = a - b;
```

这段代码首先将a和b相加得到c，然后将a减去b得到d，再将d赋值给a，最后将c赋值给b，完成了变量值的交换。

### （12）克隆数组
Java提供了clone()方法来克隆数组。通过这个方法，可以创建一个新的数组，但这个新数组和原始数组共享相同的元素。

```java
int[] original = {1, 2, 3, 4, 5};
int[] cloned = original.clone();
cloned[0] = 0;
System.out.println(original[0]);   // 打印结果为1
System.out.println(cloned[0]);     // 打印结果为0
```

克隆后的数组与原数组共享数据，改变克隆后的数组的内容会影响到原数组的内容。所以应当注意克隆后的数组的修改不要影响到原数组的内容。

## 二、集合
### （1）概述
集合是用来管理对象的集合，比如数组一样，集合也有一些优点，比如：
1、具有较好的灵活性；
2、方便有效地存储、查找、删除对象；
3、提供了许多实用且高效的方法。
### （2）特点
- 集合元素可以重复；
- 每个集合都有一个迭代器（Iterator）；
- 支持各种类型的集合。
### （3）主要接口及其实现类
- Collection接口：该接口包含了用来操作集合的通用方法。
  - List接口：存储有序的元素，元素可以重复，可以通过索引访问元素。
    - ArrayList类：ArrayList继承自AbstractList类，是一个动态数组，可以根据需要自动增加和减少容量，ArrayList的实现依赖于数组。
      ```java
      List<Integer> list = new ArrayList<>();
      for (int i = 0; i < 10; i++) {
          list.add(i);
      }
      
      Integer num = list.get(0);    // 通过索引获取元素
      Iterator iterator = list.iterator();    // 获取迭代器
      while (iterator.hasNext()) {
          System.out.println(iterator.next());
      }
      ```
    - LinkedList类：LinkedList继承自AbstractSequentialList类，是一个双向链表，可以从两端插入和删除元素，LinkedList的实现依赖于双向链表。
      ```java
      List<Integer> linkedList = new LinkedList<>();
      linkedList.addFirst(1);
      linkedList.addLast(2);
      linkedList.removeFirst();
      linkedList.removeLast();
      ```
  - Set接口：存储无序的元素，元素不能重复。
    - HashSet类：HashSet继承自AbstractSet类，是一个哈希表，无序且唯一的元素集。
      ```java
      Set<Integer> set = new HashSet<>();
      set.add(1);
      boolean contains = set.contains(1);   // 判断元素是否存在
      ```
    - TreeSet类：TreeSet继承自AbstractSet类，是一个基于红黑树的集合，保证元素处于有序状态。
      ```java
      Set<Integer> treeSet = new TreeSet<>();
      treeSet.add(1);
      Integer firstElement = treeSet.first();
      Integer lastElement = treeSet.last();
      ```
  - Map接口：存储键-值对的集合，类似于Properties文件。
    - HashMap类：HashMap继承自AbstractMap类，是一个哈希表，存储键-值对映射。
      ```java
      Map<String, String> map = new HashMap<>();
      map.put("name", "Tom");
      String name = map.get("name");
      ```
    - TreeMap类：TreeMap继承自NavigableMap类，是一个基于红黑树的有序的键-值对映射。
      ```java
      Map<String, String> treemap = new TreeMap<>(Collections.reverseOrder());
      treemap.put("age", "18");
      treemap.putIfAbsent("gender", "male");
      treemap.computeIfPresent("age", (k, v) -> Integer.parseInt(v) + 1);
      ```
### （4）遍历集合元素
集合可以通过迭代器（Iterator）进行遍历，包括ArrayList、LinkedList、HashSet、TreeSet、HashMap、TreeMap等。

```java
List<Integer> list = new ArrayList<>();
list.addAll(Arrays.asList(1, 2, 3));

Iterator<Integer> it = list.iterator();
while (it.hasNext()) {
    System.out.println(it.next());
}
```

输出结果：

```
1
2
3
```

### （5）集合转换为数组
Java提供了toArray()方法把集合转换为数组。

```java
List<Integer> list = new ArrayList<>();
list.addAll(Arrays.asList(1, 2, 3));

Object[] arr = list.toArray();
Integer[] intArray = list.toArray(new Integer[0]);
```

- toArray(): 把集合转换为Object类型的数组。
- toArray(T[]): 把集合转换为指定的数组类型。

### （6）集合的选择
在实际项目中，应该根据业务需求选择合适的集合类型。比如，如果元素是不变的，并且要求集合具有顺序，则使用ArrayList类；如果元素不必保持顺序，而且元素可能重复，则使用HashSet类；如果元素可以按照任意顺序排列，而且希望实现查找功能，则使用TreeMap类。

# 3.核心算法原理和具体操作步骤以及数学模型公式详细讲解
## 一、数组排序
数组排序算法有多种，这里只讨论快速排序算法。快速排序的基本思想是选定一个基准元素，分割待排序元素，左边的小于基准元素，右边的大于基准元素，递归排序左右子序列，直到每个序列只有一个元素为止。
### （1）算法过程描述
- 选择一个基准元素pivot；
- 分割待排序元素，左边的小于基准元素，右边的大于基准元素；
- 递归排序左右子序列；
- 当待排序序列为空或者只有一个元素时，结束。
### （2）算法步骤
- 1. 从数列中取出一个元素，称为“基准”（pivot），这个元素要么在低位，要么在高位；
- 2. 重新排序数列，所有元素比基准值小的摆放在基准左边，所有元素比基准值大的摆放在基准右边；
- 3. 对低位子数组重复第二步，直到子数组长度为1；
- 4. 对高位子数组重复第一步，直到子数组长度为1；
- 5. 合并两个子数组，产生一个有序的数组。
### （3）算法分析
- 最坏时间复杂度：O($n^2$)
- 平均时间复杂度：O($nlogn$)
- 最好情况时间复杂度：O($nlogn$)
- 不稳定排序
- 空间复杂度：O(1)
### （4）算法实现
```java
public static void quickSort(int[] arr) {
    if (arr == null || arr.length <= 1) return;
    quickSortHelper(arr, 0, arr.length - 1);
}

private static void quickSortHelper(int[] arr, int left, int right) {
    if (left >= right) return;
    
    int pivotIndex = partition(arr, left, right);
    quickSortHelper(arr, left, pivotIndex - 1);
    quickSortHelper(arr, pivotIndex + 1, right);
}

private static int partition(int[] arr, int left, int right) {
    int pivotValue = arr[right];
    int i = left - 1;
    for (int j = left; j < right; j++) {
        if (arr[j] < pivotValue) {
            i++;
            swap(arr, i, j);
        }
    }
    swap(arr, i + 1, right);
    return i + 1;
}

private static void swap(int[] arr, int i, int j) {
    int tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
}
```
### （5）数学模型公式
## 二、数组搜索
数组搜索算法是指在数组中寻找一个或几个元素，数组搜索的算法有多种，这里只讨论线性搜索和二分搜索。
### （1）算法过程描述
- 从第一个元素开始，依次与目标元素进行比较，直至找到匹配元素或遍历完整个数组。
### （2）算法步骤
- 在数组的第一个元素和数组的最后一个元素之间进行查找，如果第一个元素等于目标元素，则查找成功；否则移动指针指向中间元素，并重复第一次查找过程，直到查找成功或指针到达数组的另一端。
- 设置两个指针low和high分别指向数组的第一个元素和数组的最后一个元素，通过判断中间元素与目标元素的大小关系，依次移动low和high指针，直至找到目标元素或low大于high。
### （3）算法分析
- 查找成功的条件是目标元素出现在数组的某个位置；
- 如果数组是有序的，那么二分查找的时间复杂度是$O(\log n)$；
- 如果数组是倒序的，那么二分查找的时间复杂度是$O(n)$；
- 如果数组是随机分布的，那么二分查找的时间复杂度介于$O(\log n)$与$O(n)$之间；
### （4）算法实现
```java
public static int linearSearch(int[] arr, int target) {
    if (arr == null || arr.length == 0) throw new IllegalArgumentException("Array is empty!");
    
    for (int i = 0; i < arr.length; i++) {
        if (arr[i] == target) return i;
    }
    return -1;
}

public static int binarySearch(int[] arr, int target) {
    if (arr == null || arr.length == 0) throw new IllegalArgumentException("Array is empty!");
    
    int low = 0;
    int high = arr.length - 1;
    while (low <= high) {
        int mid = (low + high) / 2;
        if (target == arr[mid]) {
            return mid;
        } else if (target > arr[mid]) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return -1;
}
```
### （5）数学模型公式
- 插值查找：使用斐波那契数列作为增量，使得区间范围最终缩小到1。
- 斐波那契查找：使用斐波那契数列作为增量，使得区间范围最终缩小到1。
- 斐波那契查找的变种：二分查找最坏情况区间调整规则，减小查找次数。
  
# 4.具体代码实例和详细解释说明
## 一、数组排序
### （1）选择排序
选择排序（Selection sort）是一种简单直观的排序算法。它的工作原理是每一次从待排序的数据元素中选出最小（最大）的一个元素，存放在序列的起始位置，直到全部待排序的数据元素排完。
```java
public static void selectionSort(int[] arr){
    int len = arr.length;
    for(int i=0;i<len-1;i++){
        int minIndex = i; // 当前最小值元素的索引
        for(int j=i+1;j<len;j++){
            if(arr[j]<arr[minIndex]){
                minIndex = j; // 更新当前最小值元素的索引
            }
        }
        if(minIndex!=i){
            int temp = arr[i]; // 将当前最小值元素赋值给temp
            arr[i]=arr[minIndex]; // 将最小值元素放入已排好序的序列
            arr[minIndex]=temp; // 将之前保存的temp值赋值给当前最小值元素
        }
    }
}
```
#### 执行流程
1. 以第一个元素为主元，遍历余下的数组，记录最小元素的索引位置minIndex为第一个元素的索引号。
2. 遍历后面的元素，如果发现比主元元素小的元素，则更新minIndex为该元素的索引号。
3. 将主元元素与minIndex所指元素的值进行交换，即将当前主元元素放置到正确的位置。
4. 重复步骤1~3，直至遍历所有元素。

#### 时间复杂度
选择排序的平均时间复杂度和最坏时间复杂度都为$O(n^2)$，原因如下：
- 任何一次排序都会使剩余未排序元素的数目减少一半，因此，每一次选择操作的时间消耗和减少的元素数目呈阶乘关系。
- 次数越多，每次选择操作的时间消耗就越多，所以总时间消耗的趋势也是以$n^2$的速率倾斜的。
- 选择排序的最好情况时间复杂度是O(n)，这是因为，待排序的数组已经是排好序的，无需再排序。
#### 空间复杂度
选择排序只涉及辅助变量的操作，故空间复杂度为$O(1)$。

### （2）冒泡排序
冒泡排序（Bubble Sort）也是一种简单直观的排序算法。它的工作原理是比较相邻的元素，如果他们的顺序错误就把他们交换过来。
```java
public static void bubbleSort(int[] arr){
    int len = arr.length;
    for(int i=0;i<len-1;i++){
        for(int j=0;j<len-i-1;j++){
            if(arr[j]>arr[j+1]){
                int temp = arr[j]; // 交换两者的值
                arr[j]=arr[j+1];
                arr[j+1]=temp;
            }
        }
    }
}
```
#### 执行流程
1. 比较相邻的元素。
2. 如果第一个比第二个大，就交换他们两个。
3. 对每一对相邻元素作同样的工作，除了最后一个。
4. 持续每次对越来越少的元素重复上面的步骤，直到没有任何变化发生。

#### 时间复杂度
冒泡排序的平均时间复杂度和最坏时间复杂度都为$O(n^2)$。原因如下：
- 任何一次排序都会使剩余未排序元素的数目减少一半，因此，每一次交换操作的时间消耗和减少的元素数目呈阶乘关系。
- 次数越多，每次交换操作的时间消耗就越多，所以总时间消耗的趋势也是以$n^2$的速率倾斜的。
- 冒泡排序的最好情况时间复杂度是O(n)，这是因为，待排序的数组已经是排好序的，无需再排序。
#### 空间复杂度
冒泡排序只涉及辅助变量的操作，故空间复杂度为$O(1)$。

### （3）插入排序
插入排序（Insertion sort）是一种简单直观的排序算法。它的工作原理是通过构建有序序列，对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入。
```java
public static void insertionSort(int[] arr){
    int len = arr.length;
    for(int i=1;i<len;i++){
        int currentVal = arr[i];
        int preIndex = i-1;
        while((preIndex>=0) && (arr[preIndex]>currentVal)){
            arr[preIndex+1] = arr[preIndex];
            preIndex--;
        }
        arr[preIndex+1] = currentVal;
    }
}
```
#### 执行流程
1. 从第一个元素开始，该元素可认为已经被排序。
2. 取出下一个元素，在已经排序的元素序列中从后向前扫描。
3. 如果该元素（已排序）大于新元素，将该元素移到下一位置；
4. 重复步骤3，直到找到已排序的元素小于或者等于新元素的位置。
5. 将新元素插入到该位置后。

#### 时间复杂度
插入排序的平均时间复杂度和最坏时间复杂度都为$O(n^2)$，原因如下：
- 待排序元素个数为n，第i次插入需要i次比较和复制操作，故共需要$\frac{1}{2}(n^2)$次操作。
- 插入排序的最好情况时间复杂度是O(n)，这是因为，待排序的数组已经是排好序的，无需再排序。
#### 空间复杂度
插入排序只涉及辅助变量的操作，故空间复杂度为$O(1)$。

### （4）希尔排序
希尔排序（Shell Sort）也是插入排序的一种。它的工作原理是先将整个待排序的记录序列分割成为若干子序列，分别对子序列进行插入排序，待整个序列中的记录“基本有序”时，再对全体记录进行依次直接插入排序。
```java
public static void shellSort(int[] arr){
    int len = arr.length;
    int gap = len/2;
    while(gap>0){
        for(int i=gap;i<len;i++){
            int temp = arr[i];
            int preIndex = i-gap;
            while ((preIndex>=0) && (arr[preIndex]>temp)){
                arr[preIndex+gap] = arr[preIndex];
                preIndex -= gap;
            }
            arr[preIndex+gap] = temp;
        }
        gap /= 2;
    }
}
```
#### 执行流程
1. 设置一个整数d，称为步长，初始值为n/2。
2. 按步长划分序列。
3. 对每个子序列进行插入排序。
4. 重复步骤2、3，直至步长为1。

#### 时间复杂度
希尔排序的时间复杂度受步长影响，具体来说：
- 步长为1时，希尔排序的性能就退化成插入排序了，仍然为O(n^2)；
- 步长为n/2时，希尔排序的时间复杂度降为O(nlogn)，这是希尔排序的最优步长。
#### 空间复杂度
希尔排序不使用任何辅助变量，故空间复杂度为$O(1)$。

### （5）归并排序
归并排序（Merge Sort）是建立在归并操作上的一种有效的排序算法。该算法是采用分治法（Divide and Conquer）的一个非常典型的应用。将已有序的子序列合并，得到完全有序的序列；即先使每个子序列有序，再使子序列段间有序。
```java
public static void mergeSort(int[] arr){
    if(arr==null||arr.length<=1) return;
    int mid = arr.length/2;
    int[] left = Arrays.copyOfRange(arr,0,mid);
    int[] right = Arrays.copyOfRange(arr,mid,arr.length);
    mergeSort(left);
    mergeSort(right);
    merge(arr,left,right);
}

private static void merge(int[] arr, int[] left, int[] right){
    int i = 0, j = 0, k = 0;
    while(i<left.length&&j<right.length){
        if(left[i]<=right[j]){
            arr[k++] = left[i++];
        }else{
            arr[k++] = right[j++];
        }
    }
    while(i<left.length){
        arr[k++] = left[i++];
    }
    while(j<right.length){
        arr[k++] = right[j++];
    }
}
```
#### 执行流程
1. 递归地将输入数组拆分成两个半部。
2. 两半各自调用mergeSort函数，递归地进行拆分。
3. 拼接两个已经有序的序列，合并成一个完整的有序序列。

#### 时间复杂度
归并排序的时间复杂度为$O(nlogn)$，原因如下：
- 递归的层数为logn；
- 数组拆分、合并的次数为logn，每次操作需要O(n)的时间；
- 每次操作的时间复杂度是O(n)，所以整个过程的时间复杂度就是$O(nlogn)$。
#### 空间复杂度
归并排序需要申请临时的数组空间，故空间复杂度为$O(n)$。

### （6）计数排序
计数排序（Counting Sort）是一种非比较排序算法。其核心在于将输入的数据值转化为键存储在额外开辟的数组空间中。然后根据键值，将数据存储在正确的位置。
```java
public static void countingSort(int[] arr){
    if(arr==null||arr.length<=1) return;
    int max = arr[0], min = arr[0];
    for(int i=1;i<arr.length;i++){
        if(max<arr[i]) max = arr[i];
        if(min>arr[i]) min = arr[i];
    }
    int[] count = new int[max-min+1]; // 创建长度为(max-min+1)的计数数组
    for(int i=0;i<arr.length;i++){
        count[arr[i]-min]++; // 计算每个元素出现的次数
    }
    for(int i=1;i<count.length;i++){
        count[i]+=count[i-1]; // 更新每个元素的排序位置
    }
    int[] result = new int[arr.length];
    for(int i=arr.length-1;i>=0;i--){
        result[count[arr[i]-min]-1] = arr[i]; // 根据每个元素的排序位置，写入结果数组
        count[arr[i]-min]--; // 已经写入的元素，对应位置减1
    }
    for(int i=0;i<result.length;i++){
        arr[i] = result[i]; // 将结果数组的值赋回原数组
    }
}
```
#### 执行流程
1. 找到数组中的最大值和最小值，以确定计数数组的长度。
2. 创建计数数组，并统计每个数字出现的次数。
3. 根据计数数组生成索引数组。
4. 用索引数组重建原始数组。

#### 时间复杂度
计数排序的时间复杂度为$O(n+k)$，其中k是输入的 range。原因如下：
- 需要遍历数组两次：第一次计算每个元素出现的次数，第二次生成排序位置。
- 时间复杂度主要取决于计数数组的大小，而计数数组的大小等于输入范围的大小。
#### 空间复杂度
计数排序需要开辟计数数组，故空间复杂度为$O(k)$。

### （7）桶排序
桶排序（Bucket Sort）也属于非比较排序算法。它的基本思路是：假设输入数据服从均匀分布，将数据分到有限数量的桶里，对每个桶进行排序。桶排序的速度和空间花销都远远超过了快速排序。
```java
public static void bucketSort(int[] arr){
    if(arr==null||arr.length<=1) return;
    int max = arr[0], min = arr[0];
    for(int i=1;i<arr.length;i++){
        if(max<arr[i]) max = arr[i];
        if(min>arr[i]) min = arr[i];
    }
    int bucketNum = (max-min)/arr.length+1;
    ArrayList<ArrayList<Integer>> buckets = new ArrayList<>(); // 创建桶
    for(int i=0;i<bucketNum;i++){
        buckets.add(new ArrayList<>()); // 添加桶
    }
    for(int i=0;i<arr.length;i++){
        int index = (arr[i]-min)/arr.length; // 计算元素对应的桶编号
        buckets.get(index).add(arr[i]); // 将元素放入桶中
    }
    for(int i=0;i<buckets.size()-1;i++){
        Collections.sort(buckets.get(i)); // 对每个桶进行排序
    }
    int index = 0;
    for(int i=0;i<buckets.size();i++){
        for(int j=0;j<buckets.get(i).size();j++){
            arr[index++] = buckets.get(i).get(j); // 将桶中的元素写入原始数组
        }
    }
}
```
#### 执行流程
1. 找到数组中的最大值和最小值，并算出桶的数量。
2. 创建多个桶，将元素分到不同的桶中。
3. 对每个桶进行排序。
4. 将桶中的元素逐个写入原始数组。

#### 时间复杂度
桶排序的时间复杂度取决于分桶的数量，如果桶的数量过多，那么最坏情况下的时间复杂度将变成 O(n^2)。
#### 空间复杂度
桶排序不需要任何辅助变量，仅需要几个桶即可，故空间复杂度为$O(n+k)$。