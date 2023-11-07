
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


Java是一个非常流行的面向对象的语言，具有高效率、跨平台、可靠性等优点。在其标准库中提供了强大的集合类，可以方便地完成集合数据结构的各种操作，包括元素的增删查改、遍历、合并、交集、差集等。Java集合框架包含了Map接口和Collection接口两个主要部分，其中Map用于存储键值对映射关系的数据结构，而Collection则用于存储单列对象集合。同时，还提供了一些常用的迭代器工具类及多线程并发容器类。本文将以浅显易懂的方式讲解Java集合框架的各个知识点。
# 2.核心概念与联系
## 2.1 Map接口
Map接口表示一种存储键值对的数据结构，其中每个键都是唯一的，所对应的值可以是任何类型，如字符串、整数、自定义对象等。Java中的Map接口包括三种具体实现类：HashMap、TreeMap、LinkedHashMap。


### HashMap
HashMap继承于AbstractMap类，内部通过哈希表（Hashtable）实现，故具有较高的查询速度。但是它不保证元素顺序，即使相同的输入，得到的输出也可能不同。HashMap最常用的是存储键值对映射关系的数据结构，它的构造方法如下：

```java
public HashMap(int initialCapacity, float loadFactor);
```

- `initialCapacity`：初始容量大小。默认为16。
- `loadFactor`：负载因子，它决定了链表和红黑树的填充程度。默认值为0.75。

HashMap中的重要方法：

| 方法                         | 描述                             | 返回值                                    |
| ---------------------------- | -------------------------------- | ----------------------------------------- |
| put(key, value)              | 将指定键-值对添加到映射中       | 如果该键之前不存在，返回null；否则返回旧值 |
| get(key)                     | 根据键获取对应的值               | 没有对应的键时返回null                    |
| remove(key)                  | 根据键移除键值对                 | 删除成功时返回被删除的值，否则返回null    |
| containsKey(key)             | 是否包含指定的键                 | true或false                               |
| size()                       | 获取映射中键-值对数量           | map的元素个数                              |
| clear()                      | 清空映射中的所有键值对           | null                                      |
| keySet()                     | 返回映射中所有键的集合           | Set<K>                                    |
| values()                     | 返回映射中所有值的集合           | Collection<V>                             |
| entrySet()                   | 返回映射中所有键-值对的集合     | Set<Map.Entry<K, V>>                      |
| isEmpty()                    | 判断映射是否为空                 | true或false                               |
| equals(Object obj)           | 比较两个映射是否相等             | true或false                               |
| hashCode()                   | 返回该映射的哈希码               | int                                       |
| clone()                      | 克隆该映射对象                   | Object                                    |


### TreeMap
TreeMap继承于AbstractMap类，是一个有序的Map实现类，它的entrySet()方法会按照Map.Entry的排序方式，返回一个排好序的集合。TreeMap的构造方法如下：

```java
public TreeMap();
public TreeMap(Comparator<? super K> comparator);
public TreeMap(SortedMap<K,? extends V> m);
```

- `comparator`：用于比较元素的排序规则，当两个元素的排序顺序不一致时，需要提供比较器。如果不提供，则使用自然排序或升序排序。
- `m`：一个SortedMap对象，用来初始化该TreeMap对象。

TreeMap中的重要方法：

| 方法                        | 描述                                                         | 返回值          |
| --------------------------- | ------------------------------------------------------------ | --------------- |
| firstKey()                  | 返回第一个键（最小的键）                                     | Key type        |
| lastKey()                   | 返回最后一个键（最大的键）                                   | Key type        |
| lowerKey(K key)             | 返回小于等于给定键的最大键                                   | Key type        |
| floorKey(K key)             | 返回小于给定键的最大键                                       | Key type        |
| ceilingKey(K key)           | 返回大于等于给定键的最小键                                   | Key type        |
| higherKey(K key)            | 返回大于给定键的最小键                                       | Key type        |
| pollFirstEntry()            | 返回并删除第一个映射项                                       | Map.Entry<K, V> |
| pollLastEntry()             | 返回并删除最后一个映射项                                     | Map.Entry<K, V> |
| subMap(K fromKey, boolean fromInclusive, K toKey, boolean toInclusive) | 返回某一范围内的子映射                                       | SortedMap<K, V> |
| headMap(K toKey, boolean inclusive) | 返回小于给定键的映射                                         | SortedMap<K, V> |
| tailMap(K fromKey, boolean inclusive) | 返回大于等于给定键的映射                                     | SortedMap<K, V> |
| firstEntry()                | 返回第一个映射项                                             | Map.Entry<K, V> |
| lastEntry()                 | 返回最后一个映射项                                           | Map.Entry<K, V> |
| comparator()                | 返回用来比较元素的比较器，如果没有设置比较器，则返回null         | Comparator      |
| clear()                     | 清除映射中的所有元素                                         | void            |
| clone()                     | 生成该对象的副本                                             | Object          |
| containsKey(Object key)     | 是否包含指定的键                                             | boolean         |
| get(Object key)             | 获取指定键对应的值                                           | Value type      |
| isDescendingOrder()         | 判断TreeMap是否采用降序排序                                  | boolean         |
| isEmpty()                   | 检测TreeMap是否为空                                           | boolean         |
| keySet()                    | 返回TreeMap中所有的键                                        | Set<K>          |
| navigableKeySet()           | 返回TreeMap中所有的键，包括边界键                            | NavigableSet<K> |
| descendingKeySet()          | 返回TreeMap中所有降序排序的键                                | NavigableSet<K> |
| values()                    | 返回TreeMap中所有的值                                        | Collection<V>   |
| entrySet()                  | 返回TreeMap中所有的映射项                                    | Set<Map.Entry<K, V>> |
| equals(Object obj)          | 判断两个TreeMap是否相等                                       | boolean         |
| put(K key, V value)         | 添加一个新的映射项                                            | null            |
| putAll(Map<? extends K,? extends V> map) | 从给定的Map对象中添加映射                                    | null            |
| replace(K key, V value)     | 替换某个键的映射值，如果不存在该键，则添加新映射                | V               |
| remove(Object key)          | 删除某个键的映射                                              | V               |
| size()                      | 获取TreeMap中映射项的数量                                    | int             |
| subMap(K fromKey, K toKey)   | 返回某一范围内的子映射                                       | SortedMap<K, V> |
| tailMap(K fromKey)          | 返回大于等于给定键的映射                                     | SortedMap<K, V> |

### LinkedHashMap
LinkedHashMap继承于HashMap，是一个带链接关系的Map实现类。 LinkedHashMap按插入顺序保存了映射关系。 LinkedHashMap的构造方法如下：

```java
public LinkedHashMap(int initialCapacity, float loadFactor, boolean accessOrder);
```

- `initialCapacity`：初始容量大小。默认为16。
- `loadFactor`：负载因子，它决定了链表和红黑树的填充程度。默认值为0.75。
- `accessOrder`：当设置为true的时候，则每次访问都会记录最近访问的顺序。默认为false。

LinkedHashMap中的重要方法：

| 方法                            | 描述                                                      | 返回值          |
| ------------------------------- | ---------------------------------------------------------| -------------- |
| removeEldestEntry(Map.Entry<K, V> eldest) | 当 LinkedHashMap 中元素个数超过 LinkedHashMap 的最大值（default 16），就会调用这个方法来判断是否要移除 eldest 条目。如果返回 null ，则保留 eldest 条目，如果返回非 null 值，则 eldest 条目将被替换为新条目。 | Map.Entry<K, V>|
| headMap(K toKey)                 | 获得此映射中键值比指定键小的所有键（key 小于等于toKey）。    | LinkedHashMap |
| tailMap(K fromKey)               | 获得此映射中键值比指定键大的所有键（key 大于等于fromKey）。  | LinkedHashMap |
| descendingMap()                  | 获得此映射根据键的逆序视图。                                | LinkedHashMap |
| getOrDefault(Object key, V defaultValue) | 返回指定的键所映射的值，如果该键没有对应的值，则返回defaultValue。 | V |