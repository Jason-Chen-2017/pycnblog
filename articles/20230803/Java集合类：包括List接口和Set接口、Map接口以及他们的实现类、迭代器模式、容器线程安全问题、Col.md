
作者：禅与计算机程序设计艺术                    

# 1.简介
         
　　Java集合类（Collection）是一种对多个数据进行集合处理的通用数据结构。在面向对象编程中，集合被用来存储、组织和管理元素。而Java集合类为我们提供了多种形式的集合，比如List、Set、Queue、Map等。
         　　Java集合类通过统一化的方式对数据进行管理，使得开发者可以更轻松地访问到需要的数据，并且可以使用各种集合进行灵活组合，以满足不同的应用场景需求。本文将详细介绍Java集合类及其相关接口以及实现类的特性和用法。
         　　阅读本文，你将学习到以下内容：
         　　1) Java集合类都有哪些？它们分别有什么特点？
         　　2) List接口的特点和功能，以及如何使用它？
         　　3) Set接口的特点和功能，以及如何使用它？
         　　4) Map接口的特点和功能，以及如何使用它？
         　　5) 在集合类中如何添加、删除元素、获取元素、遍历集合元素？
         　　6) 为什么要使用Iterator迭代器？Iterator的特点和作用？
         　　7) 在并发情况下，集合类是否线程安全？如果不是，如何解决线程安全问题？
         　　8) Collections工具类的常用方法有哪些？它们都有什么作用？
         　　9) Java泛型的概述及如何使用它？注解又是什么？
          
         　　本文假定读者已经掌握了基本的Java语言语法和面向对象的编程知识。
         # 2. 基本概念术语说明
         ## Collection 接口
        `java.util`包下的一个顶层接口，表示一个集合，该接口提供统一的集合视图，定义了对集合的基本操作。它包含四个子接口：List，Set，Queue 和 Map。Collection接口是所有集合类的父接口，它提供了集合所需的方法，如：添加，删除，判断是否为空，返回大小等。
        ```java
        public interface Collection<E> extends Iterable<E> {
            int size(); // 返回集合的大小

            boolean isEmpty(); // 判断集合是否为空

            boolean contains(Object o); // 判断集合是否包含某个元素

            @Override
            Iterator<E> iterator(); // 获取集合中的元素的迭代器
            
            Object[] toArray(); // 将集合转换成数组
            
            <T> T[] toArray(T[] a); // 将集合转换成指定类型的数组
            
           boolean add(E e); // 添加元素到集合

           boolean remove(Object o); // 从集合中删除元素

           boolean containsAll(Collection<?> c); // 判断集合是否包含另外一个集合的所有元素

           boolean addAll(Collection<? extends E> c); // 将另一个集合中的所有元素添加到当前集合

           boolean retainAll(Collection<?> c); // 只保留当前集合中存在于另外一个集合中的元素

           boolean removeAll(Collection<?> c); // 删除当前集合中存在于另外一个集合中的元素
        }
        ```
        Collection接口是一个抽象类，提供了一些基础的方法，如size()、isEmpty()、contains()等。它的子接口List、Set、Queue、Map继承了Collection接口，并实现了List、Set、Queue、Map的具体逻辑。
        ### List接口
        List接口是一个有序集合，元素按照插入顺序排序。List的主要特点是在索引位置上进行元素存取，元素可以通过整数值进行快速定位。List接口定义如下：
        ```java
        public interface List<E> extends Collection<E> {
            void add(int index, E element); // 在指定位置处添加元素

            E get(int index); // 根据索引获取元素

            E set(int index, E element); // 修改指定索引处的元素

            void clear(); // 清空列表
            
            ListIterator<E> listIterator(); // 获取List迭代器
            
            ListIterator<E> listIterator(int index); // 以指定索引获取List迭代器

            List<E> subList(int fromIndex, int toIndex); // 提取子列表
            
           void replaceAll(UnaryOperator<E> operator); // 使用函数运算符替换元素
           
           default void sort(Comparator<? super E> c) {} // 对列表排序
          
        }
        ```
        其中，add()方法可以在指定的位置添加新的元素，get()方法根据索引获取元素，set()方法修改指定索引处的元素，clear()方法清空列表，listIterator()方法获取List迭代器，subList()方法提取子列表。addAll()方法已过时，不建议使用。replaceAl()l方法接收一个UnaryOperator<E>参数，该参数是一个函数对象，用于替换列表中每一个元素。sort()方法采用比较器进行排序。
        ### Set接口
        Set接口是一个无序集合，元素不能重复。Set的主要特点是元素不可重复，且没有索引。Set接口定义如下：
        ```java
        public interface Set<E> extends Collection<E> {
            boolean equals(Object obj); // 判断两个集合是否相等

           int hashCode(); // 返回集合的哈希码

            boolean add(E e); // 添加元素到集合

            boolean remove(Object o); // 从集合中删除元素

            boolean containsAll(Collection<?> c); // 判断集合是否包含另外一个集合的所有元素

            boolean addAll(Collection<? extends E> c); // 将另一个集合中的所有元素添加到当前集合

            boolean retainAll(Collection<?> c); // 只保留当前集合中存在于另外一个集合中的元素

            boolean removeAll(Collection<?> c); // 删除当前集合中存在于另外一个集合中的元素

            void clear(); // 清空集合

        }
        ```
        其中，equals()方法用来判断两个集合是否相等，hashCode()方法返回集合的哈希码。addAll()方法已过时，不建议使用。addAll()方法将一个元素添加到Set中，remove()方法用来从集合中删除元素，retainAll()方法只保留当前集合中存在于另外一个集合中的元素，removeAll()方法删除当前集合中存在于另外一个集合中的元素，clear()方法清空集合。
        ### Queue接口
        Queue接口是一个先入先出（FIFO）队列，它只能在队尾添加元素，从队头删除元素。Queue接口定义如下：
        ```java
        public interface Queue<E> extends Collection<E> {
           boolean offer(E e); // 添加元素到队列

           E poll(); // 从队列中取出首部元素

           E peek(); // 查看队列首部元素，不弹出

        }
        ```
        其中，offer()方法可以向队列中添加一个元素，poll()方法可以从队列中取出首部的元素，peek()方法查看队列首部元素但不弹出。
        ### Map接口
        Map接口是一个键值对集合，存储着一个映射关系，每个键值对都由一个键和一个值组成。Map的主要特点是通过键来查找对应的值。Map接口定义如下：
        ```java
        public interface Map<K,V> extends Collection<Map.Entry<K,V>> {
           int size(); // 获取map中key-value对数量

           boolean isEmpty(); // 判断map是否为空

           boolean containsKey(Object key); // 判断map中是否包含指定key的映射关系

           boolean containsValue(Object value); // 判断map中是否包含指定value的值

           V get(Object key); // 根据key获取对应的value值

           V put(K key, V value); // 添加或更新key-value对

           V remove(Object key); // 删除指定key的映射关系

           void putAll(Map<? extends K,? extends V> m); // 批量添加或更新key-value对

           void clear(); // 清空map

           Set<K> keySet(); // 获取map中所有的key值集合

           Collection<V> values(); // 获取map中所有的value值集合

           Set<Map.Entry<K,V>> entrySet(); // 获取map中所有的entry集合，entry代表一个key-value对

            default void forEach(BiConsumer<? super K,? super V> action) {
                Objects.requireNonNull(action);
                for (Map.Entry<K,V> entry : entrySet())
                    action.accept(entry.getKey(), entry.getValue());
            }

           static <K,V> Map<K,V> ofEntries(Map.Entry<K,V>... entries) {
               return new SimpleImmutableEntry<>(entries);
           }

        }
        ```
        其中，size()方法获取map中key-value对的数量，isEmpty()方法判断map是否为空，containsKey()方法判断map中是否包含指定key的映射关系，containsValue()方法判断map中是否包含指定value的值，get()方法根据key获取对应的value值，put()方法添加或更新key-value对，remove()方法删除指定key的映射关系，putAll()方法批量添加或更新key-value对，clear()方法清空map，keySet()方法获取map中所有的key值集合，values()方法获取map中所有的value值集合，entrySet()方法获取map中所有的entry集合。forEach()方法遍历map中的所有key-value对。ofEntries()静态方法接受一个Map.Entry<K,V>数组作为参数，生成一个不可变的SimpleImmutableEntry。
        ## Iterator迭代器
        Iterator接口是一个接口，它提供了遍历集合元素的方法，Iterator对象可以保存遍历过程中的状态，支持hasNext()方法检测是否还有下一个元素，next()方法获取下一个元素，移除当前元素。
        ```java
        public interface Iterator<E> {
           boolean hasNext(); // 是否还有下一个元素

           E next(); // 获取下一个元素

           default void remove() {
               throw new UnsupportedOperationException("remove");
           }

           default void forEachRemaining(Consumer<? super E> action) {
               Objects.requireNonNull(action);
               while (hasNext())
                   action.accept(next());
           }
        }
        ```
        Iterator接口的主要方法是hasNext()和next()。hasNext()方法用于判断是否还有下一个元素，如果有则返回true，否则返回false；next()方法用于获取下一个元素，只有调用hasNext()返回true后才能调用此方法。默认情况下，Iterator接口没有任何实现，因此子类必须实现此接口。但是，Iterator有一个default的方法remove()，用于在迭代期间移除当前元素。对于Collection接口来说，iterator()方法是获取Iterator对象的方法。
        ## Collections工具类
        Collections工具类提供了许多用于操作集合的静态方法，这些方法不需要创建集合对象就可以实现相应的操作。
        ### 概述
        - emptyList()：返回一个空的列表对象，该列表对象不包含任何元素。
        - singletonList()：创建一个只包含单个元素的列表对象。
        - emptySet()：返回一个空的集合对象，该集合对象不包含任何元素。
        - singletonSet()：创建一个只包含单个元素的集合对象。
        - emptyMap()：返回一个空的Map对象，该Map对象不包含任何键值对。
        - singletonMap()：创建一个只包含单个键值对的Map对象。
        - unmodifiableCollection()：返回一个不可修改的集合对象，即只能读取，不能修改。
        - synchronizedCollection()：返回一个同步的集合对象，可以线程安全地访问集合对象。
        - checkedCollection()：返回一个类型安全的集合对象，只能存储指定类型的元素。
        - min()：返回集合中的最小元素。
        - max()：返回集合中的最大元素。
        - frequency()：统计集合中指定元素出现的次数。
        - reverse()：反转集合元素的顺序。
        - copy()：返回一个副本，即浅拷贝。
        - fill()：填充集合中的元素。
        - rotate()：将集合中的元素循环移动。
        - shuffle()：随机打乱集合中的元素顺序。
        - swap()：交换两个集合中的元素。
        - nCopies()：创建指定个数的相同元素的集合。
        ### 常用方法
        - sort()：对集合按自然顺序排序。
        - binarySearch()：对有序集合进行二分搜索。
        - swap()：交换两个集合中的元素。
        - replaceAll()：使用函数运算符替换集合中的元素。
# 3. List接口详解
　　List接口代表有序的集合，元素可以重复。List接口继承了Collection接口，扩展了add()、remove()等方法。
　　List接口定义如下：
　　```java
    public interface List<E> extends Collection<E> {
       boolean add(E e); // 在列表末尾增加元素

       void add(int index, E element); // 在指定位置增加元素

       E get(int index); // 获取指定位置上的元素

       E set(int index, E element); // 替换指定位置上的元素

       E remove(int index); // 删除指定位置上的元素

       int indexOf(Object o); // 查询指定元素所在位置

       int lastIndexOf(Object o); // 查询指定元素最后一次出现的位置

       ListIterator<E> listIterator(); // 返回列表迭代器

       ListIterator<E> listIterator(int index); // 返回指定位置上的列表迭代器

       List<E> subList(int fromIndex, int toIndex); // 提取子列表

       void sort(Comparator<? super E> c); // 对列表元素按指定比较器排序

    }
   ```
　　List接口主要定义了7个方法：

　　1. `boolean add(E e)`：在列表末尾增加元素e，若成功，返回true；否则，抛出UnsupportedOperationException异常。

　　2. `void add(int index, E element)`：在指定位置index处增加元素element，若成功，返回；否则，抛出IndexOutOfBoundsException异常。

　　3. `E get(int index)`：获取指定位置index处上的元素，若不存在，抛出IndexOutOfBoundsException异常。

　　4. `E set(int index, E element)`：将指定位置index处上的元素设置为element，若成功，返回设置前的元素；否则，抛出IndexOutOfBoundsException异常。

　　5. `E remove(int index)`：删除指定位置index处上的元素，若成功，返回删除的元素；否则，抛出IndexOutOfBoundsException异常。

　　6. `int indexOf(Object o)`：查询指定元素o第一次出现的位置，若成功，返回该位置；否则，返回-1。

　　7. `int lastIndexOf(Object o)`：查询指定元素o最后一次出现的位置，若成功，返回该位置；否则，返回-1。


　　为了实现List接口，通常会使用ArrayList或者LinkedList。ArrayList是动态数组，LinkedList是链表。区别是ArrayList可以在中间增删元素，而LinkedList只能在表头和表尾增删元素。由于ArrayList线程不安全，所以推荐使用LinkedList。

　　List接口还提供了三个方法：

　　1.`ListIterator<E> listIterator()`：返回列表迭代器，用于遍历列表。

　　2.`ListIterator<E> listIterator(int index)`：返回指定位置index处的列表迭代器，用于遍历列表。

　　3.`List<E> subList(int fromIndex, int toIndex)`：返回子列表，包含fromIndex到toIndex之间的内容。


　　ListIterator接口的主要方法有：

　　1.`boolean hasPrevious()`：如果当前位置有上一个元素，则返回true；否则，返回false。

　　2.`E previous()`：返回当前位置上一个元素，并使当前位置前移一位。

　　3.`int nextIndex()`：返回当前位置之后的第一个元素的索引位置。

　　4.`int previousIndex()`：返回当前位置之前的最后一个元素的索引位置。

　　5.`void add(E e)`：在当前位置添加一个元素e。

　　6.`void set(E e)`：将当前位置元素替换为元素e。

　　7.`void remove()`：删除当前位置上的元素。


　　ListIterator用于遍历列表，一般配合for-each循环一起使用。例如：
```java
List<String> names = Arrays.asList("Alice", "Bob", "Charlie");
ListIterator<String> it = names.listIterator();
while (it.hasNext()){
    String name = it.next();
    System.out.println(name);
}
// Output: Alice
//          Bob
//          Charlie
```

## ArrayList类
```java
public class ArrayList<E> extends AbstractList<E> implements List<E>, RandomAccess, Cloneable, Serializable
```
ArrayList是一个动态数组，基于Object数组实现。ArrayList允许duplicate元素，提供了随机访问功能。
　　ArrayList构造方法：
　　1.`ArrayList()`：创建一个空的ArrayList对象。

　　2.`ArrayList(Collection<? extends E> c)`：创建一个包含c元素的ArrayList对象。

　　3.`ArrayList(int initialCapacity)`：创建一个初始容量为initialCapacity的ArrayList对象。

　　ArrayList提供了3个额外的方法：

　　1.`void trimToSize()`：当ArrayList的大小小于等于1/2时，调用该方法会减少ArrayList的容量，使ArrayList的内存使用量降低。

　　2.`void ensureCapacity(int minCapacity)`：保证ArrayList至少具有minCapacity个元素的容量。

　　3.`void set(int index, E element)`：替换指定位置index的元素。

　　ArrayList支持序列化，在序列化时，会把ArrayList中的元素逐个写入到流中，使得能够被反序列化。

## LinkedList类
```java
public class LinkedList<E> extends AbstractSequentialList<E> implements List<E>, Deque<E>, Cloneable, Serializable
```
LinkedList是一个双向链表，实现了Deque接口。LinkedList允许duplicate元素，不允许null元素。
　　LinkedList构造方法：

　　　　1.`LinkedList()`：创建一个空的LinkedList对象。

　　　　2.`LinkedList(Collection<? extends E> c)`：创建一个包含c元素的LinkedList对象。

　　LinkedList提供了4个额外的方法：

　　　　1.`void addFirst(E e)`：在链表首端增加一个元素。

　　　　2.`void addLast(E e)`：在链表末端增加一个元素。

　　　　3.`E removeFirst()`：删除链表首端的一个元素。

　　　　4.`E removeLast()`：删除链表末端的一个元素。

　　LinkedList支持序列化，在序列化时，会把LinkedList中的元素逐个写入到流中，使得能够被反序列化。

## Vector类
```java
public class Vector<E> extends AbstractList<E> implements List<E>, RandomAccess, Cloneable, Serializable
```
Vector是ArrayList的线程安全版本，但是效率比ArrayList差很多。
　　Vector和ArrayList类似，也有三种构造方法：

　　　　1.`Vector()`：创建一个空的Vector对象。

　　　　2.`Vector(int capacity)`：创建一个初始容量为capacity的Vector对象。

　　　　3.`Vector(Collection<? extends E> c)`：创建一个包含c元素的Vector对象。

　　除了ArrayList和Vector，LinkedList也是常用的实现List接口的类。两者之间的选择主要取决于不同的使用场景，比如在单线程环境中，ArrayList的效率高于LinkedList，而在多线程环境中，应优先选择Vector。