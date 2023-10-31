
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


Collections Framework 是 Java 中提供的非常重要的一个类库，它提供了许多用来操作集合或数组等容器对象的通用接口和方法，帮助开发人员更方便、高效地处理数据的各种操作。它包含了 List、Set 和 Map 等最常用的集合体系，其中包括 ArrayList、LinkedList、HashSet、TreeSet、HashMap、TreeMap、LinkedHashMap 等常用集合类，还包括 Collections 提供的排序工具类。另外，除了基础的 Collection 和 Map 接口之外，还有一些额外的子类（如 Properties）来存储键值对、日期时间、本地化信息等。在学习 Java 的过程中，掌握 Collections 框架是十分必要的。本文将从多个视角介绍 Java Collections 框架，详细阐述其中的核心概念与联系，并通过案例分析展示如何使用这些集合类的基本功能。
# 2.核心概念与联系
## （1）Collection 集合
`java.util.Collection<E>` 是一个接口，它代表一个元素集合。此接口定义了一组用于对集合进行添加、删除、检索、遍历的基本方法。Collection 是 List、Set 和 Map 的父接口。因此，List、Set 和 Map 本质上都是 Collection 的实现类。例如：ArrayList implements List；HashSet implements Set；HashMap implements Map。

Collection 中的主要方法如下：
- add(E e)：添加指定元素到集合中，如果成功返回 true，否则返回 false。
- remove(Object o)：移除指定元素，如果存在于集合中则返回 true，否则返回 false。
- contains(Object o)：检查指定的元素是否存在于集合中。
- clear()：清空集合中的所有元素。
- size()：返回集合中元素个数。
- isEmpty()：判断集合是否为空。
- iterator()：返回一个迭代器对象，可以用来遍历集合中的元素。

## （2）List 列表
`java.util.List<E>` 继承自 Collection 接口，表示一个有序的元素序列，可以通过元素索引来访问元素。List 提供了动态增长的能力，可以增加或减少元素。

List 中的主要方法如下：
- add(int index, E element)：在指定位置插入指定元素。
- remove(int index)：移除指定位置上的元素。
- get(int index)：获取指定位置上的元素。
- set(int index, E element)：替换指定位置上的元素。
- indexOf(Object o)：查找第一个匹配给定对象的元素索引。
- lastIndexOf(Object o)：查找最后一个匹配给定对象的元素索引。
- listIterator()：返回一个 ListIterator 对象，可用来按顺序或逆序遍历 List 中的元素。

## （3）Set 集
`java.util.Set<E>` 继承自 Collection 接口，表示一个不包含重复元素的集合。Set 不允许出现重复的元素，也就是说，集合中不可能有两个相同的元素。

Set 中的主要方法如下：
- add(E e)：添加指定元素到集合中，如果该元素已经存在，不会再次加入，而是直接返回 false。
- remove(Object o)：移除指定元素，如果存在于集合中，才会移除，否则不会做任何事情。
- containsAll(Collection<?> c)：检查指定的集合中是否全部都存在于当前集合中。
- equals(Object obj)：比较两个 Set 是否相等，当且仅当两个 Set 具有相同的元素，顺序也相同。
- hashCode()：计算 Set 的哈希码值。
- retainAll(Collection<?>)：保留当前 Set 中存在于指定集合中的元素，其他元素将被去除。
- toArray()：转换成数组形式。

## （4）Map 映射
`java.util.Map<K, V>` 继承自 Collection 接口，表示一个键-值映射关系的集合。其中每个元素由一个键 K 和一个值 V 构成，映射关系保证每个键对应的值恒唯一。

Map 中的主要方法如下：
- put(K key, V value)：向 Map 中添加新的键值对。
- remove(Object key)：根据键移除对应的键值对。
- get(Object key)：根据键获取对应的键值。
- containsKey(Object key)：检查是否存在指定键。
- containsValue(Object value)：检查是否存在指定值。
- size()：返回 Map 中键值对的数量。
- isEmpty()：判断 Map 是否为空。
- keys()：返回所有键的集合视图。
- values()：返回所有值的集合视图。
- entrySet()：返回所有键值对的集合视图。

## （5）常用子类
### Properties
`java.util.Properties extends Hashtable<Object, Object> implements Map<Object, Object>, Cloneable, Serializable` ，是一个包含 key-value 对属性的集合。其中的 key 和 value 可以是任意对象，但一般情况下，key 只能是字符串类型，value 可以是字符串、整数、浮点数或者布尔类型。Properties 可用于加载配置文件，或者保存应用运行期间的配置信息。

Properties 中的主要方法如下：
- load(InputStream inStream)：从输入流中加载属性。
- getProperty(String key)：根据键获取对应的属性值。
- setProperty(String key, String value)：设置或修改属性。
- store(OutputStream out, String comments)：将属性写入输出流中。

### Vector
`java.util.Vector<E> extends AbstractList<E> implements List<E>, RandomAccess, Cloneable, Serializble` ，是一个动态大小的、线程安全的集合类。Vector 提供了访问、搜索及修改元素的快速随机访问方式。Vector 实现了 java.util.List 接口，并提供了所有继承自 List 接口的方法。Vector 是旧版本的 ArrayList 替代者。

Vector 中的主要方法如下：
- add(int index, E element)：在指定位置插入指定元素。
- remove(int index)：移除指定位置上的元素。
- get(int index)：获取指定位置上的元素。
- set(int index, E element)：替换指定位置上的元素。
- size()：返回 Vector 中元素个数。
- capacity()：返回 Vector 的容量。
- trimToSize()：缩小 Vector 的容量至实际大小。
- ensureCapacity(int minCapacity)：确保 Vector 的容量足够大。
- subList(int fromIndex, int toIndex)：返回一个子列表。

### LinkedList
`java.util.LinkedList<E> extends AbstractSequentialList<E> implements List<E>, Deque<E>, Cloneable, Serializable` ，是一个双端队列，类似于栈或队列的数据结构。其元素支持 FIFO（先进先出）和 LIFO（后进先出）两种方式。链表中的每一个节点都包含一个引用（指针）指向前驱结点，同时也指向后续结点。另外，链表还实现了 Deque 接口，提供了栈、队列两端的入队和出队操作。LinkedList 实现了 java.util.List 接口，并提供了所有继承自 List 接口的方法。

LinkedList 中的主要方法如下：
- addFirst(E element)：将指定的元素作为首个元素进入队列。
- addLast(E element)：将指定的元素作为末尾元素进入队列。
- offerFirst(E element)：将指定的元素作为首个元素进入队列。
- offerLast(E element)：将指定的元素作为末尾元素进入队列。
- pollFirst()：移出队列头部的元素，并将其返回。
- pollLast()：移出队列末尾的元素，并将其返回。
- peek()：返回队列头部的元素，但不删除它。
- peekFirst()：返回队列头部的元素，但不删除它。
- peekLast()：返回队列末尾的元素，但不删除它。
- push(E element)：将指定的元素压入栈顶。
- pop()：将栈顶元素弹出。