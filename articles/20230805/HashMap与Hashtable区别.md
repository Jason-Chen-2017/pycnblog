
作者：禅与计算机程序设计艺术                    

# 1.简介
         
　　HashMap和Hashtable都是java的重要的内置类，两者的实现方式不同，但是它们最主要的区别就是：HashMap线程不安全、Hashtable线程安全。HashMap是非 synchronized 的类，可以让多个线程同时写入同一个HashMap，而Hashtable则是synchronized 的类，只能让单个线程同时写入。另外 Hashtable 在效率上要高于 HashMap。
         　　在java中，Hashtable 是遗留类，因为为了兼容旧版本 JDK ，Hashtable 提供了类似 Dictionary 的映射接口。从 JDK 1.2 开始，Hashtable 已被完全废弃，用 ConcurrentHashMap 来替代它。
         # 2.概念及术语
         　　HashMap是基于哈希表的 Map 接口的非同步实现。哈希表的工作原理是将键值对映射到数组索引上，通过hash函数将键映射到数组中的一个位置，如果出现冲突（两个键值对应同一个索引），则通过链表或红黑树的方式解决。Hashtable继承自Dictionary类，其存储的数据和HashMap类似，不同的是它是同步的，即每次只能有一个线程访问Hashtable对象，其他线程都要排队等待。所以，当多个线程同时操作Hashtable时，可能会造成线程死锁或者性能低下。
        # 3.HashMap操作原理
         （1）HashMap的结构

         HashMap 是一个无序的、密钥-值对映射容器，内部实现是一个哈希表。哈希表的底层是一个数组，数组中每个元素是一个链表，用来保存拥有相同 key 的键值对。每个链表上的元素是一个 Node 对象，Node 中保存着键、值、前驱节点和后继节点等信息。

         （2）HashMap 的构造方法

         默认构造方法：创建一个空的 HashMap，默认初始容量大小为 16，负载因子为 0.75f，且允许自动扩容；

         指定初始容量和负载因子的构造方法：创建一个指定初始容量和负载因子的 HashMap，初始容量默认为 16，负载因子默认为 0.75f，且允许自动扩容。

         指定初始容量、负载因子和加载因子的构造方法：创建一个指定初始容量、负载因子和加载因子的 HashMap，允许设置是否自动扩容。

         （3）HashMap 的增删改查操作

         根据 HashMap 的特性，它的 put() 方法允许添加键值对到 HashMap 中，而 get() 方法则用于根据键获取对应的 value。

         put() 方法：

            public V put(K key, V value) {
                return putVal(hash(key), key, value, false, true);
            }

            ① hash(key): 将键 key 通过 hashCode() 函数计算出 hashCode 值，再经过一些处理得到可以作为数组索引的实际 hash 值。
            ② putVal(): 将键值对存入 HashMap 的逻辑分三步：
              a. 判断键 key 是否已经存在于 HashMap 中：
               i. 如果键 key 存在，则先调用 removeNode() 方法删除旧节点，再插入新节点。
               ii. 如果键不存在，则直接插入新的节点。
              b. 对比计算当前容量是否超过最大容量 threshold，如果超过则进行扩容操作。
              c. 返回插入的值。

         get() 方法：

            public V get(Object key) {
                Node<K,V> e;
                return (e = getNode(hash(key), key)) == null? null : e.value;
            }

           与 put() 方法的过程相似，只是没有修改 HashMap 的逻辑，只查找键 key 是否存在并返回相应的值。

        （4）HashMap 的扩容操作

          当 HashMap 中的元素个数超过阈值 (capacity * load factor)，HashMap 会自动扩容。扩容操作包括重新计算 capacity 和 rehash 操作：

           a. 计算新的 capacity: capacity = newCapacity = oldCapacity << 1 + 1 （最小容量是 4）。
           b. 创建新的桶数组 newTable[]。
           c. 把原 table 里的元素转移到新的桶数组中。

           private void resize(int newCapacity) {
                Entry[] oldTable = table;
                int oldCapacity = oldTable.length;
                if (oldCapacity == MAXIMUM_CAPACITY) {
                    threshold = Integer.MAX_VALUE;
                    return;
                }

                Entry[] newTable = new Entry[newCapacity];
                transfer(newTable); // 把旧桶数组的元素迁移到新桶数组中
                table = newTable;
                threshold = (int) Math.min(newCapacity * loadFactor, MAX_ARRAY_SIZE + 1); // 设置新的扩容阈值
            }

           transfer() 方法是在扩容过程中执行的关键操作。遍历旧桶数组中的每个元素，判断它的 hash 值对新的桶数组的哪个索引位置应该放置。如果索引位置为空，则直接把该元素放进去；否则，就新建一个链表，把该元素加到链表头部。

           rehash() 方法完成了实际的重哈希操作，它在 put() 方法中调用，目的是计算新的 index，并检查旧索引位置是否为空，为空的话才可以把新的 entry 添加进去。

            private int hash(Object key) {
                int h;
                return (key == null)? 0 : (h = key.hashCode()) ^ (h >>> 16);
            }

            /**
             * Returns the node for the given key, or {@code null} if it doesn't exist.
             */
            final Node<K,V> getNode(int hash, Object key) {
                Node<K,V>[] tab;
                Node<K,V> first;
                int n;
                K k;
                if ((tab = table)!= null && (n = tab.length) > 0 &&
                        (first = tab[(n - 1) & hash])!= null) {
                    if (first.hash == hash &&
                            ((k = first.key) == key || (key!= null && key.equals(k))))
                        return first;
                    if ((e = first.next)!= null) {
                        do {
                            if (e.hash == hash &&
                                    ((k = e.key) == key || (key!= null && key.equals(k))))
                                return e;
                        } while ((e = e.next)!= null);
                    }
                }
                return null;
            }

        （5）HashMap 并发控制

         Hashmap 在多线程环境下可能出现数据不一致的问题，这需要通过加锁或同步机制来解决。在 put() 方法中调用 getLock() 方法获取一个可重入的锁，然后再进行 put 操作。当发生冲突时，则会阻塞其他线程，直到这个锁释放。

          private final Lock getLock() {
              if (!nonfairlocks)
                  return lock;
              if (sync == null)
                  sync = new NonfairSync();
              return sync;
          }

          static class NonfairSync extends Sync {}

          abstract static class Sync extends AbstractQueuedSynchronizer {

              private static final long serialVersionUID = -7793576445616618765L;

              protected final boolean tryAcquire(int arg) {
                  throw new UnsupportedOperationException();
              }

              protected final boolean tryRelease(int arg) {
                  throw new UnsupportedOperationException();
              }

              Condition newCondition() {
                  throw new UnsupportedOperationException();
              }

              final boolean isHeldExclusively() {
                  return getState() == 0;
              }
          }


        # 4. Hashtable 操作原理

        （1）Hashtable的结构

         Hashtable 是 Map 接口的同步实现，不同之处在于 Hashtable 继承自 Dictionary 类而不是 Map 接口，而且 Hashtable 不允许 null 作为键，也不允许有重复的键。Hashtable 采用开放寻址法来解决冲突，即当某个桶中发生冲突时，就在该桶之后的所有桶中查找，直到找到一个空桶为止。Hashtable 使用 synchronized 来实现线程同步，因此同一时间只有一个线程能够访问Hashtable。

         （2）Hashtable的构造方法

         Hashtable 有两种构造方法：默认构造方法和带初始容量参数的构造方法。

         默认构造方法：创建一个空的 Hashtable。

         指定初始容量参数的构造方法：创建一个指定初始容量的参数的 Hashtable，初始容量默认是 11。

         （3）Hashtable 的增删改查操作

         Hashtable 与 HashMap 之间的唯一区别就是线程安全的问题，所有方法都需要进行同步。由于 Hashtable 使用 synchronized 来实现线程同步，所以同一时间只有一个线程能够访问Hashtable。

          put() 方法：将键值对存入 Hashtable 的逻辑分三步：

          a. 如果键 key 为 null 或 Hashtable 已满，则抛出 NullPointerException 或 IllegalStateException。
          b. 获取 Hashtable 的所有锁，以便其他线程不能同时修改此 Hashtable。
          c. 查找键 key 是否已经存在于 Hashtable 中。
          d. 如果键不存在，则将该键值对插入 Hashtable 。
          e. 释放 Hashtable 的所有锁。

          get() 方法：将键 key 从 Hashtable 中查询，逻辑也分三步：

          a. 如果 key 为 null，则抛出 NullPointerException。
          b. 获取 Hashtable 的所有锁。
          c. 查询 key 是否存在于 Hashtable 中。
          d. 如果 key 存在，则返回相应的值。
          e. 释放 Hashtable 的所有锁。

        （4）Hashtable 的扩容操作

          Hashtable 没有扩容操作，原因是 Hashtable 不是基于数组的哈希表，它是基于链表的哈希表。当 Hashtable 中的元素个数超过一定比例 (load factor) 时，Hashtable 就会重新计算哈希码，使得 Hashtable 的性能尽量提高。如果 Hashtable 的元素很少，那么 Hashtable 的扩容操作就浪费时间，反而影响效率。

        （5）Hashtable 的优缺点

         总体来说，Hashtable 比较适合于小型集合，同时线程安全性也比较好，不会像 Vector 那样占用过多的内存资源。但 Hashtable 是一个古老的类库，效率慢，一般用作数据缓存用途。

    	# 5.Hashtable与HashMap的应用场景
    	# 6.未来发展趋势与挑战
    	   Hashtable的发展趋势是趋向淘汰，原因有二：
   		   一是 Hashtable 基于 synchronize 关键字进行了同步，导致效率比较低，在高并发场景中，Hashtable 有一定的性能损失；
   		   二是 Hashtable 实现起来比较复杂，并且在迭代的时候需要锁住整个表，有可能会产生死锁。
   	　　   
  	   可以预计 Hashtable 在 JDK 1.8 以后的版本中会被彻底淘汰。

  	   HashMap 的发展趋势更加积极，在 JDK 1.8 以后，HashMap 借鉴了 LinkedHashMap 以及 ConcurrentHashMap 的特点，提供了一些新的特性。比如：
   		   一是改善了扩容策略，它增加了红黑树的结构，降低了哈希碰撞的概率，减少了 resize 操作；
   		   二是新增了 computeIfAbsent() 和 merge() 方法，允许在并发环境下对数据进行安全地并发操作。


  	   