                 

# **基于Java的智能家居设计：构建基于Java的智能环境监控系统**

## **一、面试题库**

### **1. Java中的多线程如何实现？如何保证线程安全？**

**答案：** Java中多线程可以通过继承Thread类或实现Runnable接口来创建。为了保证线程安全，可以采用以下方法：

- **使用同步方法（synchronized）：** 通过synchronized关键字对方法进行同步，保证同一时间只有一个线程能够访问该方法。
- **使用同步代码块（synchronized）：** 通过synchronized关键字对代码块进行同步，可以更加细粒度地控制同步范围。
- **使用ReentrantLock等高级锁：** Java中的ReentrantLock、ReentrantReadWriteLock等提供了更加灵活的锁机制，如可中断锁、公平锁等。
- **使用线程安全的数据结构：** 如java.util.concurrent包中的ConcurrentHashMap、CopyOnWriteArrayList等。

### **2. 什么是死锁？如何避免死锁？**

**答案：** 死锁是指多个进程在运行过程中，因争夺资源而造成的一种僵持状态，每个进程都在等待其他进程释放资源。

避免死锁的方法：

- **资源有序分配策略：** 通过预先定义资源分配顺序，避免进程同时争夺多个资源。
- **锁超时机制：** 在尝试获取锁时，设置一个超时时间，如果超时则放弃当前操作，尝试其他方法。
- **银行家算法：** 通过预分配资源并确保每个进程都处于安全状态，避免死锁发生。

### **3. 如何实现生产者消费者问题？**

**答案：** 生产者消费者问题可以通过以下方法实现：

1. 创建一个共享缓冲区，用于存储产品。
2. 生产者线程负责生产产品，并将其放入缓冲区。
3. 消费者线程负责从缓冲区中取出产品进行消费。

使用Java中的条件变量（java.util.concurrent.locks.Condition）可以实现更灵活的控制。

### **4. 简述Java内存模型？**

**答案：** Java内存模型定义了Java程序中各种变量（线程共享变量）的访问规则，包括主内存和工作内存。

Java内存模型主要包括以下内容：

- 主内存：存储所有线程共享的变量。
- 工作内存：每个线程有一块独立的工作内存，用于存储该线程使用的变量的副本。
- 同步原则：当线程读取共享变量时，需要从主内存读取到工作内存；当线程写入共享变量时，需要从工作内存写入到主内存。

### **5. Java中的四种访问控制符有什么区别？**

**答案：** Java中的四种访问控制符分别是public、protected、default（默认）和private。

- **public：** 可以被任何其他类访问。
- **protected：** 可以被同一个包中的类或其他继承自该类的子类访问。
- **default（默认）：** 只能被同一个包中的类访问。
- **private：** 只能被当前类访问。

### **6. 简述Java中的垃圾回收机制。**

**答案：** Java中的垃圾回收（GC）是一种自动内存管理机制，用于回收不再使用的对象占用的内存。

Java中的垃圾回收机制主要包括以下步骤：

- **标记-清除：** 首先标记所有需要回收的对象，然后清除这些被标记的对象。
- **引用计数：** 通过记录对象的引用次数来回收不再使用的对象，当引用次数变为0时，说明该对象不再被引用，可以被回收。
- **复制算法：** 将内存分为两个相等的区域，每次只使用一个区域。垃圾回收时，将存活的对象复制到另一个区域，然后清空当前区域。

### **7. 简述Java中的四种基本类型（原始数据类型）。**

**答案：** Java中的四种基本类型分别是：

- **整数类型（byte、short、int、long）：** 用于表示整数。
- **浮点类型（float、double）：** 用于表示浮点数。
- **字符类型（char）：** 用于表示单个字符。
- **布尔类型（boolean）：** 用于表示布尔值。

### **8. 简述Java中的四种访问修饰符（public、protected、default、private）的作用。**

**答案：** Java中的四种访问修饰符用于控制类、方法、变量等的访问权限。

- **public：** 可以被任何其他类访问。
- **protected：** 可以被同一个包中的类或其他继承自该类的子类访问。
- **default（默认）：** 只能被同一个包中的类访问。
- **private：** 只能被当前类访问。

### **9. 什么是泛型？泛型有什么作用？**

**答案：** 泛型是一种允许在代码中重复使用的类型参数，用于解决类型安全问题和代码复用问题。

泛型的作用包括：

- **类型安全：** 通过泛型可以确保在编译时检查类型错误，避免运行时出现类型不匹配的错误。
- **代码复用：** 通过泛型可以将通用代码抽象出来，减少重复代码，提高代码可读性和可维护性。

### **10. 简述Java中的面向对象编程的基本特征。**

**答案：** Java中的面向对象编程具有以下基本特征：

- **封装：** 将数据和操作数据的方法封装在一起，隐藏内部细节，只暴露必要的接口。
- **继承：** 通过继承关系，可以复用父类的属性和方法，实现代码的复用。
- **多态：** 通过方法重载和方法重写，可以实现不同的对象在接收到相同的消息时产生不同的行为。

### **11. 什么是反射（Reflection）？反射在Java编程中有什么应用？**

**答案：** 反射是Java语言提供的一种基础功能，允许在运行时动态地获取程序中的各种信息，并操纵这些信息。

反射的应用包括：

- **动态创建对象：** 通过反射可以在运行时创建任意类的对象。
- **访问和修改字段、方法：** 通过反射可以访问和修改类中的字段和方法。
- **类型检查和转换：** 通过反射可以检查对象的类型，并在需要时进行类型转换。
- **实现AOP（面向切面编程）：** 通过反射可以实现动态代理，用于实现日志记录、事务管理等功能。

### **12. 什么是Java异常处理？Java中异常处理的机制是怎样的？**

**答案：** Java异常处理是一种机制，用于处理程序运行时可能出现的错误情况。

Java中异常处理的机制包括：

- **异常捕获（try-catch）：** 通过try块捕获可能抛出的异常，并在catch块中处理异常。
- **异常抛出（throws）：** 在方法签名中声明可能抛出的异常，由调用者处理。
- **异常传递（throws）：** 异常可以在方法之间传递，直到被捕获和处理。
- **自定义异常：** 可以通过继承Exception类来创建自定义异常。

### **13. 简述Java中的四种访问控制符（public、protected、default、private）的作用。**

**答案：** Java中的四种访问控制符用于控制类、方法、变量等的访问权限。

- **public：** 可以被任何其他类访问。
- **protected：** 可以被同一个包中的类或其他继承自该类的子类访问。
- **default（默认）：** 只能被同一个包中的类访问。
- **private：** 只能被当前类访问。

### **14. 简述Java中的四种基本类型（原始数据类型）。**

**答案：** Java中的四种基本类型分别是：

- **整数类型（byte、short、int、long）：** 用于表示整数。
- **浮点类型（float、double）：** 用于表示浮点数。
- **字符类型（char）：** 用于表示单个字符。
- **布尔类型（boolean）：** 用于表示布尔值。

### **15. 什么是Java中的泛型？泛型有什么作用？**

**答案：** 泛型是一种允许在代码中重复使用的类型参数，用于解决类型安全问题和代码复用问题。

泛型的作用包括：

- **类型安全：** 通过泛型可以确保在编译时检查类型错误，避免运行时出现类型不匹配的错误。
- **代码复用：** 通过泛型可以将通用代码抽象出来，减少重复代码，提高代码可读性和可维护性。

### **16. 简述Java中的面向对象编程的基本特征。**

**答案：** Java中的面向对象编程具有以下基本特征：

- **封装：** 将数据和操作数据的方法封装在一起，隐藏内部细节，只暴露必要的接口。
- **继承：** 通过继承关系，可以复用父类的属性和方法，实现代码的复用。
- **多态：** 通过方法重载和方法重写，可以实现不同的对象在接收到相同的消息时产生不同的行为。

### **17. 什么是Java反射（Reflection）？反射在Java编程中有什么应用？**

**答案：** 反射是Java语言提供的一种基础功能，允许在运行时动态地获取程序中的各种信息，并操纵这些信息。

反射的应用包括：

- **动态创建对象：** 通过反射可以在运行时创建任意类的对象。
- **访问和修改字段、方法：** 通过反射可以访问和修改类中的字段和方法。
- **类型检查和转换：** 通过反射可以检查对象的类型，并在需要时进行类型转换。
- **实现AOP（面向切面编程）：** 通过反射可以实现动态代理，用于实现日志记录、事务管理等功能。

### **18. 什么是Java异常处理？Java中异常处理的机制是怎样的？**

**答案：** Java异常处理是一种机制，用于处理程序运行时可能出现的错误情况。

Java中异常处理的机制包括：

- **异常捕获（try-catch）：** 通过try块捕获可能抛出的异常，并在catch块中处理异常。
- **异常抛出（throws）：** 在方法签名中声明可能抛出的异常，由调用者处理。
- **异常传递（throws）：** 异常可以在方法之间传递，直到被捕获和处理。
- **自定义异常：** 可以通过继承Exception类来创建自定义异常。

### **19. 简述Java中的四种访问修饰符（public、protected、default、private）的作用。**

**答案：** Java中的四种访问修饰符用于控制类、方法、变量等的访问权限。

- **public：** 可以被任何其他类访问。
- **protected：** 可以被同一个包中的类或其他继承自该类的子类访问。
- **default（默认）：** 只能被同一个包中的类访问。
- **private：** 只能被当前类访问。

### **20. 简述Java中的四种基本类型（原始数据类型）。**

**答案：** Java中的四种基本类型分别是：

- **整数类型（byte、short、int、long）：** 用于表示整数。
- **浮点类型（float、double）：** 用于表示浮点数。
- **字符类型（char）：** 用于表示单个字符。
- **布尔类型（boolean）：** 用于表示布尔值。


## **二、算法编程题库**

### **1. 如何实现二分查找算法？**

**题目：** 给定一个排序好的数组arr，和一个要查找的目标值target，实现一个函数来查找target是否存在于数组中，如果存在返回其索引，否则返回-1。

```java
public int search(int[] arr, int target) {
    // 请在此处实现代码
}
```

**答案：** 

```java
public int search(int[] arr, int target) {
    int left = 0;
    int right = arr.length - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;

        if (arr[mid] == target) {
            return mid;
        } else if (arr[mid] < target) {
            left = mid + 1;
        } else {
            right = mid - 1;
        }
    }

    return -1;
}
```

### **2. 如何实现归并排序算法？**

**题目：** 实现归并排序，对给定的数组arr进行排序。

```java
public void mergeSort(int[] arr) {
    // 请在此处实现代码
}
```

**答案：** 

```java
public void mergeSort(int[] arr) {
    if (arr == null || arr.length < 2) {
        return;
    }

    process(arr, 0, arr.length - 1);
}

private void process(int[] arr, int l, int r) {
    if (l == r) {
        return;
    }

    int mid = l + (r - l) / 2;
    process(arr, l, mid);
    process(arr, mid + 1, r);
    merge(arr, l, mid, r);
}

private void merge(int[] arr, int l, int mid, int r) {
    int[] temp = new int[r - l + 1];
    int i = l, j = mid + 1, k = 0;

    while (i <= mid && j <= r) {
        if (arr[i] < arr[j]) {
            temp[k++] = arr[i++];
        } else {
            temp[k++] = arr[j++];
        }
    }

    while (i <= mid) {
        temp[k++] = arr[i++];
    }

    while (j <= r) {
        temp[k++] = arr[j++];
    }

    for (int p = 0; p < temp.length; p++) {
        arr[l + p] = temp[p];
    }
}
```

### **3. 如何实现快速排序算法？**

**题目：** 实现快速排序，对给定的数组arr进行排序。

```java
public void quickSort(int[] arr) {
    // 请在此处实现代码
}
```

**答案：**

```java
public void quickSort(int[] arr) {
    if (arr == null || arr.length < 2) {
        return;
    }

    process(arr, 0, arr.length - 1);
}

private void process(int[] arr, int l, int r) {
    if (l == r) {
        return;
    }

    int pivot = arr[l + (r - l) / 2];
    int i = l, j = r;

    while (i <= j) {
        if (arr[i] < pivot) {
            i++;
        } else if (arr[j] > pivot) {
            j--;
        } else {
            swap(arr, i, j);
            i++;
            j--;
        }
    }

    process(arr, l, j);
    process(arr, i, r);
}

private void swap(int[] arr, int i, int j) {
    int temp = arr[i];
    arr[i] = arr[j];
    arr[j] = temp;
}
```

### **4. 如何实现快速选择算法（QuickSelect）？**

**题目：** 实现快速选择算法，在未排序的数组arr中找出第k小的元素。

```java
public int quickSelect(int[] arr, int k) {
    // 请在此处实现代码
}
```

**答案：**

```java
public int quickSelect(int[] arr, int k) {
    if (arr == null || arr.length < k) {
        throw new IllegalArgumentException("k is larger than array length");
    }

    int left = 0;
    int right = arr.length - 1;

    while (left <= right) {
        int pivot = partition(arr, left, right);
        if (pivot == k - 1) {
            return arr[pivot];
        } else if (pivot > k - 1) {
            right = pivot - 1;
        } else {
            left = pivot + 1;
        }
    }

    throw new IllegalArgumentException("k is out of range");
}

private int partition(int[] arr, int left, int right) {
    int pivot = arr[right];
    int i = left;

    for (int j = left; j < right; j++) {
        if (arr[j] < pivot) {
            swap(arr, i, j);
            i++;
        }
    }

    swap(arr, i, right);
    return i;
}
```

### **5. 如何实现链表反转？**

**题目：** 给定一个链表的头节点head，实现一个函数来反转链表。

```java
public ListNode reverseList(ListNode head) {
    // 请在此处实现代码
}
```

**答案：**

```java
public ListNode reverseList(ListNode head) {
    ListNode prev = null;
    ListNode curr = head;

    while (curr != null) {
        ListNode nextTemp = curr.next;
        curr.next = prev;
        prev = curr;
        curr = nextTemp;
    }

    return prev;
}
```

### **6. 如何实现两个有序链表的合并？**

**题目：** 给定两个有序链表head1和head2，实现一个函数来合并它们，并返回合并后链表的头节点。

```java
public ListNode mergeTwoLists(ListNode head1, ListNode head2) {
    // 请在此处实现代码
}
```

**答案：**

```java
public ListNode mergeTwoLists(ListNode head1, ListNode head2) {
    if (head1 == null) {
        return head2;
    }
    if (head2 == null) {
        return head1;
    }

    ListNode dummy = new ListNode(0);
    ListNode curr = dummy;

    while (head1 != null && head2 != null) {
        if (head1.val < head2.val) {
            curr.next = head1;
            head1 = head1.next;
        } else {
            curr.next = head2;
            head2 = head2.next;
        }
        curr = curr.next;
    }

    if (head1 != null) {
        curr.next = head1;
    }
    if (head2 != null) {
        curr.next = head2;
    }

    return dummy.next;
}
```

### **7. 如何实现两个数组的交集？**

**题目：** 给定两个整数数组nums1和nums2，实现一个函数来找出它们的交集。

```java
public int[] intersection(int[] nums1, int[] nums2) {
    // 请在此处实现代码
}
```

**答案：**

```java
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;

public int[] intersection(int[] nums1, int[] nums2) {
    Map<Integer, Integer> count1 = new HashMap<>();
    for (int num : nums1) {
        count1.put(num, count1.getOrDefault(num, 0) + 1);
    }

    Set<Integer> result = new HashSet<>();
    for (int num : nums2) {
        if (count1.containsKey(num) && count1.get(num) > 0) {
            result.add(num);
            count1.put(num, count1.get(num) - 1);
        }
    }

    int[] arr = new int[result.size()];
    int i = 0;
    for (int num : result) {
        arr[i++] = num;
    }

    return arr;
}
```

### **8. 如何实现两个有序数组的合并？**

**题目：** 给定两个有序整数数组nums1和nums2，实现一个函数来合并它们，并将结果存储在nums1中。

```java
public void merge(int[] nums1, int m, int[] nums2, int n) {
    // 请在此处实现代码
}
```

**答案：**

```java
public void merge(int[] nums1, int m, int[] nums2, int n) {
    int index1 = m - 1;
    int index2 = n - 1;
    int mergeIndex = m + n - 1;

    while (index1 >= 0 && index2 >= 0) {
        if (nums1[index1] > nums2[index2]) {
            nums1[mergeIndex--] = nums1[index1--];
        } else {
            nums1[mergeIndex--] = nums2[index2--];
        }
    }

    while (index2 >= 0) {
        nums1[mergeIndex--] = nums2[index2--];
    }
}
```

### **9. 如何实现栈的最大值？**

**题目：** 实现一个包含栈功能的类MaxStack，支持push、pop、top操作，同时能够获取当前栈的最大值。

```java
public class MaxStack {
    Stack<Integer> stack;
    Stack<Integer> maxStack;

    public MaxStack() {
        stack = new Stack<>();
        maxStack = new Stack<>();
    }

    public void push(int x) {
        stack.push(x);
        if (maxStack.isEmpty() || x >= maxStack.peek()) {
            maxStack.push(x);
        }
    }

    public void pop() {
        if (stack.pop().equals(maxStack.peek())) {
            maxStack.pop();
        }
    }

    public int top() {
        return stack.peek();
    }

    public int max() {
        return maxStack.peek();
    }
}
```

### **10. 如何实现队列的最大值？**

**题目：** 实现一个包含队列功能的类MaxQueue，支持enqueue、dequeue、max操作，同时能够获取当前队列的最大值。

```java
public class MaxQueue {
    Queue<Integer> queue;
    Deque<Integer> maxQueue;

    public MaxQueue() {
        queue = new LinkedList<>();
        maxQueue = new LinkedList<>();
    }

    public void enqueue(int val) {
        queue.offer(val);
        while (!maxQueue.isEmpty() && val > maxQueue.getLast()) {
            maxQueue.pollLast();
        }
        maxQueue.offerLast(val);
    }

    public int dequeue() {
        if (queue.isEmpty()) {
            throw new RuntimeException("Queue is empty");
        }
        int val = queue.poll();
        if (val == maxQueue.getFirst()) {
            maxQueue.pollFirst();
        }
        return val;
    }

    public int max() {
        if (maxQueue.isEmpty()) {
            throw new RuntimeException("Queue is empty");
        }
        return maxQueue.getFirst();
    }
}
```

### **11. 如何实现最小栈？**

**题目：** 实现一个包含栈功能的类MinStack，支持push、pop、top、getMin操作，同时能够获取当前栈的最小值。

```java
public class MinStack {
    Stack<Integer> stack;
    Stack<Integer> minStack;

    public MinStack() {
        stack = new Stack<>();
        minStack = new Stack<>();
    }

    public void push(int x) {
        stack.push(x);
        if (minStack.isEmpty() || x <= minStack.peek()) {
            minStack.push(x);
        }
    }

    public void pop() {
        if (stack.pop().equals(minStack.peek())) {
            minStack.pop();
        }
    }

    public int top() {
        return stack.peek();
    }

    public int getMin() {
        return minStack.peek();
    }
}
```

### **12. 如何实现滑动窗口的最大值？**

**题目：** 给定一个整数数组nums和一个整数k，实现一个滑动窗口，计算窗口中元素的最大值。

```java
public int[] maxSlidingWindow(int[] nums, int k) {
    // 请在此处实现代码
}
```

**答案：**

```java
public int[] maxSlidingWindow(int[] nums, int k) {
    if (nums == null || nums.length == 0 || k <= 0) {
        return new int[0];
    }

    LinkedList<Integer> queue = new LinkedList<>();
    int[] result = new int[nums.length - k + 1];

    for (int i = 0; i < nums.length; i++) {
        if (!queue.isEmpty() && queue.peek() == nums[i - k]) {
            queue.poll();
        }

        while (!queue.isEmpty() && queue.getLast() < nums[i]) {
            queue.pollLast();
        }

        queue.offer(nums[i]);

        if (i >= k - 1) {
            result[i - k + 1] = queue.peek();
        }
    }

    return result;
}
```

### **13. 如何实现字符串的搜索算法（KMP）？**

**题目：** 给定两个字符串s和p，实现一个函数来计算字符串p在字符串s中出现的次数。

```java
public int strStr(String s, String p) {
    // 请在此处实现代码
}
```

**答案：**

```java
public int strStr(String s, String p) {
    if (s == null || p == null || p.length() > s.length()) {
        return -1;
    }

    int[] next = new int[p.length()];
    getNext(p, next);

    int i = 0;
    int j = 0;
    while (i < s.length() && j < p.length()) {
        if (j == -1 || s.charAt(i) == p.charAt(j)) {
            i++;
            j++;
        } else {
            j = next[j];
        }
    }

    if (j == p.length()) {
        return i - j;
    } else {
        return -1;
    }
}

private void getNext(String p, int[] next) {
    int length = p.length();
    next[0] = -1;
    int k = -1;
    int i = 0;
    while (i < length) {
        if (k == -1 || p.charAt(i) == p.charAt(k)) {
            k++;
            i++;
            next[i] = k;
        } else {
            k = next[k];
        }
    }
}
```

### **14. 如何实现回文数验证？**

**题目：** 给定一个整数num，实现一个函数来判断它是否是回文数。

```java
public boolean isPalindrome(int num) {
    // 请在此处实现代码
}
```

**答案：**

```java
public boolean isPalindrome(int num) {
    if (num < 0) {
        return false;
    }

    int reversed = 0;
    int original = num;
    while (num > 0) {
        reversed = reversed * 10 + num % 10;
        num /= 10;
    }

    return original == reversed;
}
```

### **15. 如何实现链表的回文判断？**

**题目：** 给定一个链表的头节点head，实现一个函数来判断链表是否是回文。

```java
public boolean isPalindrome(ListNode head) {
    // 请在此处实现代码
}
```

**答案：**

```java
public boolean isPalindrome(ListNode head) {
    if (head == null || head.next == null) {
        return true;
    }

    ListNode slow = head;
    ListNode fast = head;
    ListNode prevSlow = null;
    while (fast != null && fast.next != null) {
        fast = fast.next.next;
        prevSlow = slow;
        slow = slow.next;
    }

    if (fast != null) {
        slow = slow.next;
    }

    prevSlow.next = null;

    ListNode reversed = reverseList(slow);
    ListNode p1 = head;
    ListNode p2 = reversed;

    while (p1 != null && p2 != null) {
        if (p1.val != p2.val) {
            return false;
        }
        p1 = p1.next;
        p2 = p2.next;
    }

    return true;
}

private ListNode reverseList(ListNode head) {
    ListNode prev = null;
    ListNode curr = head;
    while (curr != null) {
        ListNode nextTemp = curr.next;
        curr.next = prev;
        prev = curr;
        curr = nextTemp;
    }
    return prev;
}
```

### **16. 如何实现排序链表？**

**题目：** 给定一个链表的头节点head，实现一个函数来对链表进行排序。

```java
public ListNode sortList(ListNode head) {
    // 请在此处实现代码
}
```

**答案：**

```java
public ListNode sortList(ListNode head) {
    if (head == null || head.next == null) {
        return head;
    }

    ListNode middle = getMiddle(head);
    ListNode nextOfMiddle = middle.next;
    middle.next = null;

    ListNode left = sortList(head);
    ListNode right = sortList(nextOfMiddle);

    return merge(left, right);
}

private ListNode getMiddle(ListNode head) {
    if (head == null) {
        return head;
    }

    ListNode slow = head;
    ListNode fast = head;

    while (fast.next != null && fast.next.next != null) {
        slow = slow.next;
        fast = fast.next.next;
    }

    return slow;
}

private ListNode merge(ListNode left, ListNode right) {
    ListNode result = new ListNode(0);
    ListNode curr = result;

    while (left != null && right != null) {
        if (left.val < right.val) {
            curr.next = left;
            left = left.next;
        } else {
            curr.next = right;
            right = right.next;
        }
        curr = curr.next;
    }

    if (left != null) {
        curr.next = left;
    }
    if (right != null) {
        curr.next = right;
    }

    return result.next;
}
```

### **17. 如何实现链表环检测？**

**题目：** 给定一个链表的头节点head，实现一个函数来检测链表中是否有环。

```java
public boolean hasCycle(ListNode head) {
    // 请在此处实现代码
}
```

**答案：**

```java
public boolean hasCycle(ListNode head) {
    ListNode slow = head;
    ListNode fast = head;

    while (fast != null && fast.next != null) {
        slow = slow.next;
        fast = fast.next.next;

        if (slow == fast) {
            return true;
        }
    }

    return false;
}
```

### **18. 如何实现排序数组的中位数？**

**题目：** 给定一个整数数组nums，实现一个函数来找到数组的中位数。

```java
public double findMedianSortedArrays(int[] nums1, int[] nums2) {
    // 请在此处实现代码
}
```

**答案：**

```java
public double findMedianSortedArrays(int[] nums1, int[] nums2) {
    int totalLength = nums1.length + nums2.length;
    if (totalLength % 2 == 0) {
        return (findKth(nums1, 0, nums1.length - 1, nums2, 0, nums2.length - 1, totalLength / 2) + findKth(nums1, 0, nums1.length - 1, nums2, 0, nums2.length - 1, totalLength / 2 + 1)) / 2.0;
    } else {
        return findKth(nums1, 0, nums1.length - 1, nums2, 0, nums2.length - 1, totalLength / 2 + 1);
    }
}

private int findKth(int[] nums1, int start1, int end1, int[] nums2, int start2, int end2, int k) {
    int len1 = end1 - start1 + 1;
    int len2 = end2 - start2 + 1;
    if (len1 > len2) {
        return findKth(nums2, start2, end2, nums1, start1, end1, k);
    }

    if (len1 == 0) {
        return nums2[start2 + k - 1];
    }

    if (k == 1) {
        return Math.min(nums1[start1], nums2[start2]);
    }

    int i = Math.min(k / 2, len1);
    int j = Math.min(k / 2, len2);

    if (nums1[start1 + i - 1] > nums2[start2 + j - 1]) {
        return findKth(nums1, start1 + i, end1, nums2, start2, start2 + j - 1, k - j);
    } else {
        return findKth(nums1, start1, start1 + i - 1, nums2, start2 + j, end2, k - i);
    }
}
```

### **19. 如何实现无重复字符的最长子串？**

**题目：** 给定一个字符串s，实现一个函数来找到其包含重复字符的最长子串的长度。

```java
public int lengthOfLongestSubstring(String s) {
    // 请在此处实现代码
}
```

**答案：**

```java
public int lengthOfLongestSubstring(String s) {
    int n = s.length();
    int ans = 0;
    Map<Character, Integer> map = new HashMap<>();

    for (int j = 0, i = 0; j < n; j++) {
        if (map.containsKey(s.charAt(j))) {
            i = Math.max(map.get(s.charAt(j)) + 1, i);
        }
        ans = Math.max(ans, j - i + 1);
        map.put(s.charAt(j), j);
    }

    return ans;
}
```

### **20. 如何实现最小覆盖子串？**

**题目：** 给定一个字符串s和一个字符集合t，实现一个函数来找到s中涵盖t所有字符的最小子串。

```java
public String minWindow(String s, String t) {
    // 请在此处实现代码
}
```

**答案：**

```java
public String minWindow(String s, String t) {
    int[] need = new int[128];
    int[] window = new int[128];

    for (char c : t.toCharArray()) {
        need[c]++;
    }

    int left = 0;
    int right = 0;
    int valid = 0;
    String ans = "";

    while (right < s.length()) {
        char c = s.charAt(right);
        right++;

        if (need[c] > 0) {
            window[c]++;
            if (window[c] <= need[c]) {
                valid++;
            }
        }

        while (valid == t.length()) {
            if (right - left < ans.length() || ans.length() == 0) {
                ans = s.substring(left, right);
            }

            char d = s.charAt(left);
            left++;

            if (need[d] > 0) {
                if (window[d] <= need[d]) {
                    valid--;
                }
                window[d]--;
            }
        }
    }

    return ans;
}
```

