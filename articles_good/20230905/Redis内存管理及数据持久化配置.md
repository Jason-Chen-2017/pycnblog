
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Redis是一个开源的高性能的key-value存储数据库。它的优点很多，既提供快速读写性能，又支持丰富的数据结构，包括string、hashmap、list、set等。它的客户端语言也非常丰富，包括C、C++、Java、Python等。除了这些特性之外，Redis还有几个比较重要的特点：数据持久化功能，能够将内存中的数据在磁盘上进行持久化，可以很好的支持集群部署；内存管理功能，它对内存的分配和回收都有一套机制保障系统高效运行；事务处理功能，Redis提供了一套事务机制，通过它可以一次执行多个命令，保证数据的完整性；复制功能，可以实现主从节点的数据自动同步，提高可用性；多种访问协议，如Redis支持的多种访问协议，如标准协议、发布订阅协议、Redis哨兵、Redis集群等，使得Redis可以应用于不同的场景下。本文主要讨论Redis的内存管理及数据持久化配置相关的内容。
# 2.基本概念术语说明
## 2.1 Redis内存管理概述
Redis的内存管理与操作系统中内存管理的原理大体相同。Redis申请和释放内存的过程如下图所示：

1.Redis初始化时会预先申请一定的内存空间作为缓存区，用来存放数据。
2.当需要存储新数据时，Redis会先检查是否存在空闲缓存空间，如果有的话，就直接使用该空间；否则，就会根据一定的淘汰策略（比如LRU）选择一些将要被删除的数据释放掉，然后再使用新的空闲空间存放数据。
3.当缓存空间用完时，Redis也不会真正地立即抛弃数据，而是采用某种“惰性删除”策略，等待后续访问到数据时再进行真正的删除。
4.为了避免浪费内存资源，Redis会定期进行内存碎片整理，将相邻的内存块进行合并，达到释放更多内存的目的。

总的来说，Redis的内存管理可以分为以下几个方面：
1.申请释放内存：Redis采用先申请后释放的模式，通过`malloc()`函数向操作系统申请内存，并使用`free()`函数释放；因此，系统自身会跟踪并管理Redis使用的内存。
2.缓存空间预留：Redis在初始化时会预先分配一定的内存空间作为缓存区。
3.缓存淘汰策略：当缓存空间用完时，Redis会根据一定策略（比如LRU）选择一些数据进行删除，确保Redis的内存利用率始终保持在一个合理的水平。
4.内存碎片整理：Redis通过定期进行内存碎片整理，将相邻的内存块进行合并，达到释放更多内存的目的。

## 2.2 数据类型
Redis支持五种数据类型，包括String、Hash、List、Set和Sorted Set。其中，String是最基础的数据类型，其他四种都是其上的抽象。各个数据类型的内部结构与实现方式不同，但是它们都遵循着共同的规则：
1.所有的键都是字符串类型。
2.所有的键值对都可以设置过期时间。
3.所有的键值对都可以设置生存时间（TTL）。
4.Redis只会为已经过期或者即将过期的键值对分配额外的空间来保存数据，而不是像Memcached一样完全忽略已经过期的数据。
5.对于每一种数据类型，Redis都提供了一组命令用于对其进行操作。

## 2.3 Redis持久化
Redis的持久化功能是指把当前内存的数据存储在磁盘上。通过持久化功能，可以将内存中的数据保留在磁盘上，防止因服务重启或者宕机导致的数据丢失。Redis提供了两种持久化方案：RDB和AOF。
### RDB（Redis DataBase）
RDB持久化方案是指每次服务器退出时，都会将当前内存中的所有数据保存到硬盘的一个独立文件中。当服务器重新启动时，Redis首先会使用RDB持久化方案恢复其状态，也就是从最后一次持久化时保存的快照开始，读取对应的指令，逐条执行，最终达到最新的数据状态。这种持久化方式虽然短暂，但是具有较高的安全性，能够保护数据不丢失。此外，RDB还可以在灾难备份和数据迁移时提供帮助。
### AOF（Append Only File）
AOF持久化方案是指除了记录每个写操作（执行一个命令），Redis还会记录当前已执行的所有写操作命令序列，并在写入日志文件的末尾追加。当服务器重新启动时，Redis会按照顺序执行日志文件中的所有指令，从而达到恢复数据的目的。与RDB持久化方案相比，AOF持久化方案的最大优势在于：当Redis发生故障切换时，AOF持久化方案可以确保数据的完整性。此外，由于AOF持久化会记录所有执行过的写操作命令，因此，它可以提供更加细粒度的数据冗余。

# 3.核心算法原理和具体操作步骤
## 3.1 Redis内存分配器
Redis底层采用的是内存池技术，所有的内存管理都围绕着内存池实现的。内存池分配器负责维护整个内存池的大小，并确保分配出的内存块都是连续的，并且分配出去的内存块都被记录在相应的Bitmap中，以便内存释放的时候，能够正确地回收内存。

内存池划分如下图所示：

- `volatile-lru`: 从上次访问的时间里随机采样出一部分数据，然后将这些数据缓存在LRU列表中，防止这些数据被回收掉。
- `allkeys-lfu`: 不做任何淘汰策略，将所有键都缓存在LFU队列中。
- `volatile-random`: 从所有缓存的数据里随机选择一部分数据，将这些数据缓存在LRU列表中，防止这些数据被回收掉。
- `volatile-ttl`: 将所有带有TTL的键值对的过期时间记录在优先级队列里，优先处理已经即将过期的键值对。

当需要新的数据插入时，Redis首先会尝试从内存池中分配一块连续的内存块，并返回给调用者，如果分配失败，则会触发内存淘汰策略，对现有的一些内存进行回收，直到找到足够大的内存块分配出来，然后才进行插入操作。

## 3.2 Redis的淘汰策略
当Redis存储的Key数量超过了其可用内存容量时，就会出现内存溢出的问题。Redis采用一种淘汰策略（eviction policy）解决这一问题。淘汰策略的目的是为了回收部分或全部过期或低频的缓存数据，以腾出足够的空间容纳新的数据。

Redis提供了六种不同的淘汰策略：
- volatile-lru：从已设置过期时间的数据集(server.db[i].expires)中挑选最近最少使用的数据淘汰。
- allkeys-lru：从所有数据集(databases)中挑选最近最少使用的数据淘汰。
- volatile-lfu：从已设置过期时间的数据集(server.db[i].expires)中挑选最不经常使用的数据淘汰。
- allkeys-lfu：从所有数据集(databases)中挑选最不经常使用的数据淘汰。
- volatile-rand：从已设置过期时间的数据集(server.db[i].expires)中随机选择数据淘汰。
- no-eviction：当内存不足时，新写入的数据不会被缓存。

一般情况下，Redis会同时采用多个淘汰策略，这样可以同时满足不同级别的数据的缓存需求。

## 3.3 Redis的持久化
Redis通过持久化可以实现将内存中的数据持久化到磁盘，防止因服务停止或者宕机导致数据丢失。Redis提供了两种持久化方案：RDB和AOF。

### 3.3.1 RDB（Redis DataBase）
RDB持久化是指，Redis根据配置选项定时（默认是隔1小时）保存快照文件到磁盘，这个时候的文件可以用来重建整个Redis的数据。

开启RDB持久化的方法是添加一条SAVE配置指令到Redis的配置文件redis.conf中。SAVE指令的语法如下：
```
save <seconds> <changes>
```
- seconds: 表示多少秒钟之后执行一次持久化操作。
- changes: 表示多少次写入操作之后执行一次持久化操作。

示例：
```
save 60 1000
```
表示在1分钟内写入1000次数据之后，执行一次持久化操作。

Redis在持久化过程中，fork()了一个子进程，让子进程进行持久化操作，父进程继续处理命令请求。这样做的好处是，Redis不会因为持久化操作而影响到性能，而且可以保证数据持久化的完整性。

当Redis以RDB方式持久化数据时，它会单独创建（fork()）一个子进程来进行持久化操作，而且子进程刚开始执行持久化操作时，内存里的数据是一致的。在持久化期间，父进程会一直处于休眠状态，以便让子进程尽可能长的时间执行完任务。

RDB持久化的缺陷是它是单线程、全量生成的持久化方案，不能解决实时的写入问题，延迟太高。所以一般情况下，建议配合AOF持久化一起使用，组合使用RDB+AOF持久化的方式可以有效的降低持久化的延迟。

### 3.3.2 AOF（Append Only File）
AOF（Append Only File）持久化是指，Redis的写入操作只追加到文件末尾，不断的将命令追加到日志文件中，但是对于相同的数据集来说，AOF会比RDB更为高效。AOF持久化日志文件是一个文本文件，内容就是对于Redis服务器执行的所有写入操作的脚本。

AOF持久化可以使用appendfsync参数来控制日志同步策略。appendfsync选项共有三种可取值：
- always：表示每次写入操作都将强制调用一次fsync操作，强制将数据同步到磁盘。
- everysec：表示每秒钟执行一次fsync操作，将数据同步到磁盘。
- no：表示从不执行fsync操作，操作系统负责确保数据安全，不会丢失任何数据。

示例：
```
appendonly yes   # 是否打开AOF持久化
appendfilename "appendonly.aof"    # AOF持久化日志文件名
appendfsync everysec    # 每秒钟执行一次fsync操作
no-appendfsync-on-rewrite no    # 在RDB持久化时是否执行fsync操作
auto-aof-rewrite-percentage 100   # AOF重写的触发条件
auto-aof-rewrite-min-size 64mb     # AOF重写的最小阈值
```

AOF持久化默认情况是不开启的，要想启用AOF持久化，需要将配置文件中的appendonly设置为yes。AOF持久化日志文件默认的文件名是appendonly.aof，可以通过修改配置文件中的appendfilename来自定义。

AOF持久化的优势在于，它支持实时的数据写入，保证数据完整性，延迟低，更适合用于高性能写入且要求严格的数据完整性的场合。但同时，AOF持久化也有自己的一些缺点：
- 由于AOF的写入操作是同步的，速度慢，占用过多的磁盘IO。
- 如果由于某些原因造成了AOF持久化文件的损坏，可能会导致服务异常退出或者数据丢失。
- 对于AOF持久化的分析、修复和缩减都十分困难。

# 4.具体代码实例和解释说明

## 4.1 Redis内存分配器
首先引入辅助函数，创建一个redisMemoryPool的结构体，声明三个指针变量：firstblock，lastblock，currentblock。初始化的时候，设置firstblock，lastblock为NULL，currentblock指向firstblock。
```c++
typedef struct redisMemoryBlock {
    void *ptr;       // 指向内存块首地址
    size_t allocated; // 当前内存块已分配的字节数
    size_t used;      // 当前内存块已使用的字节数
    int prev;         // 指向前驱块的偏移量
    int next;         // 指向后继块的偏移量
} redisMemoryBlock;
 
typedef struct redisMemoryPool {
    uint64_t total_allocated;        // 内存池总的分配字节数
    uint64_t memory_limit;           // 内存限制字节数
    redisMemoryBlock* firstblock;    // 第一个内存块的起始位置
    redisMemoryBlock* lastblock;     // 最后一个内存块的结束位置
    redisMemoryBlock* currentblock;  // 当前分配内存的内存块
} redisMemoryPool;
 
void initMemoryPool(redisMemoryPool* pool, size_t limit) {
    memset(pool, 0, sizeof(*pool));
    pool->memory_limit = limit;
    pool->total_allocated = POOL_OVERHEAD;
 
    /* 分配第一个内存块 */
    redisMemoryBlock* block = (redisMemoryBlock*)zmalloc(sizeof(redisMemoryBlock)+MEMORY_BLOCK_SIZE);
    block->ptr = (void*)(block + 1);
    block->allocated = MEMORY_BLOCK_SIZE - sizeof(redisMemoryBlock);
    block->used = sizeof(redisMemoryBlock);
    block->prev = -1;
    block->next = -1;
    if (pool->firstblock == NULL) {
        pool->firstblock = block;
        pool->lastblock = block;
    } else {
        /* 插入到链表头部 */
        block->next = pool->firstblock->offset;
        *(redisMemoryBlock**)((char*)block->ptr-sizeof(int)) = pool->firstblock;
        pool->firstblock->prev = block->offset;
        pool->firstblock = block;
    }
 
    /* 初始化分配指针 */
    pool->currentblock = block;
    pool->total_allocated += POOL_OVERHEAD;
}
 
static inline void releaseCurrentMemoryChunk(redisMemoryPool* pool) {
    assert(pool->currentblock!= NULL &&
           pool->currentblock!= pool->firstblock &&
           pool->currentblock!= pool->lastblock);
    redisMemoryBlock* prev = pool->currentblock->prev?
                            ((redisMemoryBlock*)((char*)pool->currentblock->ptr - pool->currentblock->prev)):
                            NULL;
    redisMemoryBlock* next = *(redisMemoryBlock**)((char*)pool->currentblock->ptr + pool->currentblock->used -
                                                 sizeof(int));
 
    zfree(pool->currentblock);
    if (prev!= NULL) {
        next->prev = prev->offset;
        memcpy(prev->ptr + prev->used, &next, sizeof(int));
        prev->used += sizeof(int);
    } else {
        pool->firstblock = next;
        next->prev = -1;
        memcpy(next->ptr, &pool->firstblock, sizeof(int));
        next->used += sizeof(int);
    }
 
    pool->currentblock = next;
}
 
void releaseMemoryPool(redisMemoryPool* pool) {
    while (pool->firstblock!= NULL) {
        redisMemoryBlock* block = pool->firstblock;
        pool->firstblock = pool->firstblock->next!= -1?
                           (redisMemoryBlock*)(((char*)block) + block->next):NULL;
        if (block!= pool->lastblock)
            free(block);
        else
            break;
    }
    zfree(pool);
}
```

然后创建redisAlloc结构体，定义两个成员变量：redisMemoryPool pool和void* ptr。redisAlloc结构体的主要作用是在Redis申请内存的时候，分配的内存首先会在内存池中申请，申请成功后返回指针；申请失败的时候，会调用zmalloc函数进行实际的内存分配。

```c++
struct redisAlloc {
    redisMemoryPool pool;
    void* (*zmalloc)(size_t size);
    
    redisAlloc(size_t limit, void* (*f)(size_t)) : pool(), zmalloc(f) {
        initMemoryPool(&pool, limit);
        this->zmalloc = f;
    }
    
    ~redisAlloc() {
        releaseMemoryPool(&pool);
    }
    
    void* allocate(size_t size) {
        void* result = NULL;
 
        /* 根据申请的内存大小进行大小分类 */
        if (size <= SMALL_REQUEST_THRESHOLD) {
            result = malloc(size);
        } else if (size <= MEDIUM_REQUEST_THRESHOLD) {
            return do_small_alloc(&pool, size);
        } else if (size > LARGE_REQUEST_THRESHOLD) {
            result = zmalloc(size+LARGEST_BLOCK);
        } else {
            result = do_medium_alloc(&pool, size);
        }
 
        return result;
    }
 
    static void deallocate(void* p) {
        if (p!= nullptr) {
            if (((long)p % REDIS_MEMORY_ALIGNMENT)!= 0)
                abort();
            
            /* 检查指针指向的区域是否是属于当前内存池 */
            auto& head = *((std::atomic<redisMemoryPool*>*)((char*)p - LARGEST_BLOCK / 2));
            if (&head.load(std::memory_order_relaxed) ==
                 std::addressof(*(redisMemoryPool**)p)) {
                size_t s = *(size_t*)((char*)p - LARGEST_BLOCK / 2 + sizeof(std::atomic<redisMemoryPool*>));
                releaseMemoryChunk(&*(redisMemoryPool**)p, (char*)p - s, s+LARGEST_BLOCK);
            } else {
                free(p);
            }
        }
    }
 
    static void releaseMemoryChunk(redisMemoryPool* pool, char* ptr, size_t size) {
        assert(((long)ptr % REDIS_MEMORY_ALIGNMENT) == 0);
        assert((((char*)&pool) - ptr) >= (POOL_OVERHEAD + LARGEST_BLOCK) ||
               ((char*)&pool) - ptr == POOL_OVERHEAD);
        
        if (ptr == (char*)pool->lastblock) {
            *(int*)((char*)ptr + size - sizeof(int)) = 0;
            pool->lastblock = *(redisMemoryBlock**)((char*)ptr + size - sizeof(int));
        } else {
            *(int*)((char*)ptr + size - sizeof(int)) = (*(redisMemoryBlock**)((char*)ptr + size -
                                                                              sizeof(int)))->offset;
            (*(redisMemoryBlock**)((char*)ptr + size - sizeof(int)))->prev =
                    (*(redisMemoryBlock**)((char*)ptr + size -
                                             sizeof(int)))->offset;
            *(redisMemoryBlock**)((char*)ptr - POOL_OVERHEAD)->next =
                        (*(redisMemoryBlock**)((char*)ptr + size - sizeof(int)))->offset;
        }
        
        while (true) {
            bool success = false;
            for (redisMemoryBlock* b = pool->firstblock;!success && b!= pool->lastblock; ) {
                char* bp = (char*)b->ptr;
                
                if (!REDIS_IN_RANGE(bp, ptr, ptr+size))
                    continue;
                
                if (bp + b->used == ptr) {
                    b->used -= size;
                    
                    if (b->used == 0) {
                        if (b->prev!= -1) {
                            *(redisMemoryBlock**)((char*)bp - b->prev)->next =
                                    (*(redisMemoryBlock**)((char*)bp + size))->offset;
                            (*(redisMemoryBlock**)((char*)bp + size))->prev = b->prev;
                        } else {
                            pool->firstblock = (*(redisMemoryBlock**)((char*)bp + size));
                            (*(redisMemoryBlock**)((char*)bp + size))->prev = -1;
                        }
                        
                        zfree(b);
                        success = true;
                    } else {
                        *(size_t*)((char*)bp + size - POOL_OVERHEAD) = b->used;
                    }
                } else {
                    b->used -= size;
                    *(size_t*)((char*)bp + size - POOL_OVERHEAD) = b->used;
                    
                    if (b->used == 0) {
                        if (b->prev!= -1) {
                            *(redisMemoryBlock**)((char*)bp - b->prev)->next =
                                (*(redisMemoryBlock**)((char*)bp + size))->offset;
                            (*(redisMemoryBlock**)((char*)bp + size))->prev = b->prev;
                        } else {
                            pool->firstblock = (*(redisMemoryBlock**)((char*)bp + size));
                            (*(redisMemoryBlock**)((char*)bp + size))->prev = -1;
                        }
                        
                        zfree(b);
                        success = true;
                    }
                }
            }
            
            if (success)
                break;
            
            /* 清理内存碎片 */
            int freed_blocks = 0;
            for (redisMemoryBlock* b = pool->firstblock; b!= pool->lastblock;) {
                if ((char*)b + b->used == (char*)b->ptr) {
                    redisMemoryBlock* nxt = (*(redisMemoryBlock**)((char*)b->ptr +
                                                                  b->used - sizeof(int)));
                    zfree(b);
                    b = nxt;
                    freed_blocks++;
                } else {
                    b = *(redisMemoryBlock**)((char*)b->ptr + b->used - sizeof(int));
                }
            }
            if (freed_blocks == 0)
                break;
        }
    }
};
 
 
/* 对内存池进行内存分配 */
static void* do_small_alloc(redisMemoryPool* pool, size_t size) {
    redisMemoryBlock* curr = pool->currentblock;
 
    /* 判断是否有足够的剩余空间 */
    if (curr->allocated - curr->used >= size) {
        curr->used += size;
        return curr->ptr + curr->used - size;
    }
 
    /* 没有足够的剩余空间，尝试寻找合适的内存块 */
    int pos = find_best_fit_pos(pool, size);
    if (pos!= -1) {
        curr = (redisMemoryBlock*)((char*)curr->ptr + pos);
        curr->allocated = POOL_CHUNK_SIZE;
        curr->used = size;
        pool->total_allocated += POOL_OVERHEAD;
        return curr->ptr;
    }
 
    /* 没有合适的内存块，尝试增加新的内存块 */
    redisMemoryBlock* newblock = (redisMemoryBlock*)zmalloc(sizeof(redisMemoryBlock)+
                                                             POOL_CHUNK_SIZE+SMALL_FREELISTS_SIZE);
    newblock->allocated = POOL_CHUNK_SIZE + SMALL_FREELISTS_SIZE;
    newblock->used = size;
    newblock->prev = -1;
    newblock->next = -1;
    if (pool->lastblock!= NULL) {
        newblock->prev = pool->lastblock->offset;
        *(redisMemoryBlock**)((char*)newblock->ptr +
                              newblock->allocated -
                              sizeof(int)) = pool->lastblock;
        pool->lastblock->next = newblock->offset;
    }
    pool->lastblock = newblock;
    if (pool->firstblock == NULL)
        pool->firstblock = newblock;
 
    return newblock->ptr;
}
 
 
/* 对内存池进行内存分配 */
static void* do_medium_alloc(redisMemoryPool* pool, size_t size) {
    int best_fit_index = find_best_fit_index(MEDIUM_FREELISTS_SIZE, size);
    size_t alloc_size = MEGA * pow(2, best_fit_index);
    size_t overhead = POOL_OVERHEAD + large_overhead(size);
    
    redisMemoryBlock* curr = pool->currentblock;
    while (curr->next!= -1) {
        if (curr->allocated - curr->used >= alloc_size) {
            curr->used += alloc_size;
            break;
        }
        
        curr = (redisMemoryBlock*)((char*)curr->ptr + curr->next);
    }
    
    if (curr->allocated - curr->used >= alloc_size) {
        curr->used += alloc_size;
        pool->total_allocated += POOL_OVERHEAD;
        return curr->ptr + curr->used - alloc_size;
    }
    
    /* 没有足够的空间，尝试增加新的内存块 */
    redisMemoryBlock* newblock = (redisMemoryBlock*)zmalloc(sizeof(redisMemoryBlock)+
                                                             POOL_CHUNK_SIZE+LARGE_FREELISTS_SIZE);
    newblock->allocated = POOL_CHUNK_SIZE + LARGE_FREELISTS_SIZE;
    newblock->used = alloc_size + overhead;
    newblock->prev = -1;
    newblock->next = -1;
    if (pool->lastblock!= NULL) {
        newblock->prev = pool->lastblock->offset;
        *(redisMemoryBlock**)((char*)newblock->ptr +
                              newblock->allocated -
                              sizeof(int)) = pool->lastblock;
        pool->lastblock->next = newblock->offset;
    }
    pool->lastblock = newblock;
    if (pool->firstblock == NULL)
        pool->firstblock = newblock;

    return newblock->ptr + LARGE_FREELISTS_SIZE + overhead;
}
 
 
/* 查找合适的内存块 */
static int find_best_fit_pos(redisMemoryPool* pool, size_t size) {
    const size_t sizes[] = {
        16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176,
        192, 208, 224, 240, 256, 288, 320, 352, 384, 416,
        448, 480, 512, 576, 640, 704, 768, 832, 896, 960,
        1024, 1152, 1280, 1408, 1536, 1664, 1792, 1920, 2048,
        2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4608,
        5120, 5632, 6144, 6656, 7168, 7680, 8192, 9216, 10240,
        11264, 12288, 13312, 14336, 15360, 16384, 18432, 20480,
        22528, 24576, 26624, 28672, 30720, 32768, 36864, 40960,
        45056, 49152, 53248, 57344, 61440, 65536, 73728, 81920,
        90112, 98304, 106496, 114688, 122880, 131072, 147456,
        163840, 180224, 196608, 212992, 229376, 245760, 262144
    };
 
    int i, j;
    int start = floor_log2(size);
    int end = ceil_log2(size);
    size_t candidate_size;
 
    for (j = start; j <= end; ++j) {
        for (i = 0; i < POOL_NUM_CHUNKS; ++i) {
            candidate_size = MIN(sizes[(start+j)/2], POOL_CHUNK_SIZE);
            if (candidate_size >= size) {
                return i*POOL_CHUNK_SIZE + (start+j)/2;
            }
        }
    }
 
    return -1;
}
 
 
/* 返回最接近size的2的整数次幂 */
static inline int floor_log2(size_t size) {
    int log2 = 0;
    if (size & 0xffff0000) { log2 += 16; size >>= 16; }
    if (size & 0xff00) { log2 += 8; size >>= 8; }
    if (size & 0xf0) { log2 += 4; size >>= 4; }
    if (size & 0xc) { log2 += 2; size >>= 2; }
    if (size & 0x2) { log2++; }
    return log2;
}
 
static inline int ceil_log2(size_t size) {
    int log2 = floor_log2(size);
    return (size >> log2) << log2;
}
 
 
/* 返回申请的内存块的地址 */
static char* get_mem_addr(redisMemoryPool* pool, void* p) {
    return ((char*)p)-((char*)pool)-(POOL_OVERHEAD+LARGEST_BLOCK/2);
}
 
 
/* 获取某个申请的内存块的大小 */
static size_t get_mem_size(redisMemoryPool* pool, void* p) {
    long offset = (char*)get_mem_addr(pool, p)-(char*)pool;
    size_t size = *(size_t*)(offset+(LARGEST_BLOCK+POOL_OVERHEAD)*2);
    if (offset%REDIS_MEMORY_ALIGNMENT!=0 ||
        ((size_t)(get_mem_addr(pool,p)+size)%REDIS_MEMORY_ALIGNMENT)!=0) {
        printf("invalid pointer or size\n");
        exit(1);
    }
    return size-(LARGEST_BLOCK+POOL_OVERHEAD)*2;
}
 
#define is_aligned(p) (((long)(p) % REDIS_MEMORY_ALIGNMENT)==0)
````