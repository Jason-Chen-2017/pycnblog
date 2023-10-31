
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


近几年来，随着微服务架构的兴起，分布式系统架构和高并发场景的普及，互联网应用的业务需求也日渐复杂化，使得应用的开发、部署、运维等都面临越来越多的挑战。如今，基于Spring Boot的开源框架正在成为应用开发者不可或缺的一环，它能让开发者用最小的代价快速搭建出一个可独立运行的微服务系统。

本文将讨论Spring Boot中关于缓存的相关知识点，包括Spring Cache和Redis的整合、缓存雪崩和击穿、缓存穿透、缓存预热、缓存更新机制、缓存降级策略、缓存一致性解决方案、缓存监控工具等。另外，本文还将讲解一些关于性能优化方面的技巧，比如JVM调优、接口限流、数据分片、异步处理、响应式编程等。最后，还会介绍一些开源组件的使用方法。读者可以通过阅读本文获取到Spring Boot中缓存、性能优化的相关知识，并掌握如何在实际工作中运用这些知识。
# 2.核心概念与联系
## 2.1什么是缓存？
缓存（Cache）是一种提高资源利用率的方法，通过将频繁访问的数据（数据源）暂存于易失性存储器（如磁盘），可以避免直接访问原始数据而减少延迟，提升用户体验。通常情况下，缓存可以降低计算、网络开销、数据库查询次数，从而提升应用程序的响应速度和吞吐量。

## 2.2缓存分类
缓存可以分为强制缓存和协商缓存：
- 强制缓存（也称“命中缓存”）指当请求的资源在缓存中存在时，直接从缓存中返回资源，不再向源服务器发送请求；
- 协商缓存（也称“未命中缓存”）指当请求的资源不存在于缓存中时，需要先到源服务器进行请求，然后再把请求结果存入缓存中，供后续请求使用。

除了强制缓存和协商缓存之外，还有第三种缓存模式——终端缓存。终端缓存是浏览器自身实现的，也就是当我们在浏览器上打开某个网站页面时，浏览器会将网站的静态资源（如HTML文件、CSS样式表、JavaScript脚本、图片等）缓存在本地磁盘上，下一次访问相同页面时就不需要再次请求了。当然，终端缓存也是可以设置过期时间的，防止因资源过期而造成的问题。

## 2.3为什么要用缓存？
- 提升应用性能：缓存能够极大地提升应用程序的响应速度和吞吐量，对于那些计算密集型或者网络IO类型的应用尤其有效；
- 节省带宽：减少对源服务器的请求次数，节省了带宽，提升用户体验；
- 分担服务器压力：由于缓存一般在本地，因此不占用任何服务器资源，降低了服务器负载，提升了服务器的可用性；
- 提升用户体验：缓存能够降低延迟，让用户感觉到应用更快、更 responsive，从而提升用户体验。

## 2.4什么是Spring Cache？
Spring Framework是一个开源Java开发框架，提供了缓存抽象层，使得开发者可以便捷地集成各种缓存产品，Spring Cache是其中的一个重要模块。Spring Cache的主要作用就是减少对数据源的重复查询，提升系统的响应能力。Spring Cache包含两类基本功能：
- Spring Cache注解：通过使用注解，可以很方便地在方法上添加缓存逻辑；
- Spring Cache SPI：提供自定义缓存管理器，自定义配置，以及与其他缓存技术的集成。

## 2.5什么是Redis？
Redis 是完全开源免费的，遵守BSD协议，是一个高性能的key-value数据库。它支持数据持久化。高性能使得Redis相比memcached来说占有一席之地。它是当前最热门的NoSQL技术之一。Redis的主要特征是速度快，支持丰富的数据类型，实时的发布/订阅，以及命令支持丰富。目前，Redis已被广泛应用于缓存、消息队列、排行榜、计数器等领域。

## 2.6Redis的特性
- 数据类型丰富：Redis支持五种数据类型，包括字符串、散列、列表、集合、有序集合。Redis使用简单动态字符串表示法，同时能支持二进制数据值。
- 支持事务：Redis的所有操作都是原子性的，同时Redis还支持对几个操作全包办，即multi/exec事务。
- 高速数据结构：Redis内部采用自己构建的数据结构，其中hash table用于存储对象，skip list用于排序，整数集合用于基数统计。
- 复制与容灾：Redis支持主从同步，即master可以执行写操作，slave可以执行读操作。这样既可以提高Redis的读写性能，又可以减轻master服务器的压力。Redis还支持Sentinel，可以自动检测master服务器是否出现故障，从而可以通知其他slave切换新的master服务器。
- 高可用性：Redis支持分布式集群架构，多个节点可以组成集群。若有某个节点发生故障，其他节点仍然可以继续提供服务。

# 3.核心算法原理和具体操作步骤以及数学模型公式详细讲解
## 3.1缓存雪崩、击穿和穿透
### （一）缓存雪崩
缓存雪崩（Cache Avalanche）是指，缓存集中过期导致某一时刻，整个缓存空间处于混乱状态，所有请求直接访问数据库，导致数据库连接瘫痪甚至超时，严重危害数据库的正常服务。

假设有一个秒杀系统，系统设计时按照道具的ID作为缓存键，将道具信息（如名称、描述、库存数量等）缓存到内存中。缓存预热之后，此时如果某些道具的库存数量突然变化，可能会引起缓存雪崩。举例如下：
1. 首先，某些道具的库存数突然从100个增加到了150个，而其它道具的库存保持不变。
2. 此时，所有的请求都会命中缓存，直接返回道具库存为150的库存数量。
3. 当大量请求同时到来时，发现缓存中缓存的是每件道具库存数都是150，导致大量请求直接访问数据库，造成数据库连接池饱和，严重影响数据库的正常服务。

为了防止缓存雪崩，通常可以在缓存预热的时候，加载一定量的冷数据，并且为缓存设置一个较短的过期时间。同时，可以设置不同的过期策略，给不同的键设置不同的过期时间，避免不同类型数据的缓存同时过期，进一步提升缓存命中率。

### （二）缓存击穿
缓存击穿（Cache Storm）是指，某个热点Key在某一时刻失效，大量请求也同时命中该Key，导致数据库压力激增。这种现象往往伴随着错误日志的打击波。

为了解决缓存击穿，可以将热点Key设置时间更长的过期时间。不过，对于一些比较重要的Key，可以使用互斥锁（Mutex Lock）的方式，保证只有第一个请求到来时才去数据库查询，其余等待锁的请求则直接命中缓存。

### （三）缓存穿透
缓存穿透（Cache Penetration）是指，攻击者故意放入虚假数据，或者制造缓存穿透攻击，直接查询一个根本不存在的Key，导致所有请求都直接访问数据库。这种现象会造成数据库压力剧增，甚至宕机。

为了防止缓存穿透，可以在系统初始化时，预先填充一定量的缓存，避免空指针异常等问题。也可以设置布隆过滤器（Bloom Filter）或者采用同样的方案来限制查询Key的范围。

## 3.2缓存预热
缓存预热（Cache Warming）是指，把缓存中的数据进行提前的初始化加载，从而保证系统在刚启动时就拥有比较完整的数据。

缓存预热的方法有两种：
- 手动预热：管理员人工操作预先加载缓存数据，一般是从数据库中批量读取一批数据进行缓存；
- 自动预热：后台定时任务定期刷新缓存数据，根据一定规则自动刷新缓存，一般是每隔半小时左右加载一次缓存。

缓存预热的目的就是尽可能保证应用的启动速度，避免因为缓存没有完全加载，而造成的较长的等待时间。

## 3.3缓存更新机制
### （一）缓存更新方案
#### 方案一：完全更新模式
完全更新模式（Full Update Mode）指的是当更新缓存数据时，删除缓存数据；

#### 方案二：增量更新模式
增量更新模式（Increment Update Mode）指的是当更新缓存数据时，只更新有变化的数据项；

#### 方案三：异步更新模式
异步更新模式（Asynchronous Update Mode）指的是当更新缓存数据时，触发后台线程更新缓存，返回客户端时才进行展示。

### （二）缓存更新流程
1. 请求者（Client）向服务端发起请求；
2. 服务端接收到请求，调用远程服务（Remote Service）来获取数据；
3. 远程服务向数据库（Database）查询数据；
4. 查询成功后，数据被写入到数据库；
5. 服务端将数据缓存到缓存服务器（Cache Server），同时设置缓存有效期（Expire Time）；
6. 下次请求者访问相同的Key时，服务端会检查缓存有效期（Expire Time），判断缓存是否已经过期；
7. 如果缓存没有过期，服务端直接返回缓存数据，否则，会继续执行第3步，重新从数据库查询数据；
8. 更新缓存数据的同时，还需要更新缓存更新时间戳（Last Updated Timestamp）。

## 3.4缓存降级策略
缓存降级（Cache Downgrade）是指，系统默认开启缓存，但是当数据发生变化时，根据一定的策略把缓存设置为空，或者做缓存预热。例如，电商网站首页商品列表，如果缓存时间设置的过短，当商品价格修改时，缓存不会及时更新，这就会造成商品信息展示滞后。所以，需要设置一定的降级策略。

降级策略一般分为两种：
- 把缓存数据暂停使用：当缓存数据不准确时，比如老产品的促销信息，可以暂停使用缓存数据；
- 清除缓存数据：当数据发生变化时，清除缓存数据，重新加载缓存。

## 3.5缓存一致性问题
缓存一致性（Cache Consistency）问题是指，当缓存更新后，马上要访问数据库，由于缓存数据与数据库数据不一致，导致数据异常。常见的缓存一致性问题包括：
- 时延性问题：当缓存服务器和数据库之间存在网络延迟时，会造成数据不一致；
- 顺序性问题：当两个并发请求同时更新缓存时，也会出现数据不一致；
- 丢失更新问题：缓存更新之后，可能会因为网络波动或其他原因，丢失部分更新的数据；
- 缓存穿透问题：当缓存中没有对应的数据时，所有请求都直接访问数据库，致使数据库压力加大。

解决缓存一致性问题的方法有两种：
- 异步刷新机制：将缓存更新操作设置为异步操作，避免用户等待；
- 使用锁机制：使用分布式锁（Distributed Lock）来确保缓存数据的一致性，比如Redis的setnx命令。

## 3.6缓存监控工具
缓存监控（Cache Monitoring）工具的作用是实时监控缓存的命中率、 miss率、 平均访问时间等。常用的监控工具有：
- RedisInsight：RedisInsight是一个开源的Redis可视化工具，能够直观地看到Redis中数据的详细信息，并提供丰富的图形化界面，支持多种数据分析，如数据访问统计，缓存键值对数量，缓存命中率等；
- Redis Commander：Redis Commander是一个基于Web的开源工具，能够管理和监控Redis，并提供友好的图形化界面，支持Windows、Mac和Linux平台；
- RedisLive：RedisLive是一个基于Web的开源工具，能够实时显示Redis服务器的性能指标，并提供丰富的图表查看，非常适合作为Redis服务器的实时监控工具；
- Redis Monitor：Redis Monitor是一个监控工具，安装在Redis服务器上，能够实时显示Redis服务器的性能指标，并提供丰富的图表查看，能够更好地了解Redis的性能瓶颈。

# 4.具体代码实例和详细解释说明
本文将通过具体的代码示例，帮助读者理解Spring Boot中缓存、性能优化的相关知识。具体操作步骤如下所示：
## 4.1 Spring Cache注解使用
### （一）使用@Cacheable注解
```java
import org.springframework.cache.annotation.CacheConfig;
import org.springframework.cache.annotation.Cacheable;

@Service
@CacheConfig(cacheNames = "books") // 配置缓存名称
public class BookService {
    
    @Cacheable(key="#isbn", unless="#result==null") // 缓存配置
    public Book findBookByIsbn(String isbn){
        return bookDao.findByIsbn(isbn);
    }
    
}
```
### （二）使用@CacheEvict注解
```java
import org.springframework.cache.annotation.CacheConfig;
import org.springframework.cache.annotation.CacheEvict;

@Service
@CacheConfig(cacheNames = "books") // 配置缓存名称
public class BookService {
    
    @CacheEvict(allEntries = true) // 清除全部缓存
    public void clearAllBooks(){
        
    }
    
    @CacheEvict(key="'book:'+#isbn") // 根据isbn清除指定缓存
    public void deleteBookByIsbn(String isbn){
        bookDao.deleteByIsbn(isbn);
    }
    
}
```
### （三）使用@Caching注解
```java
import org.springframework.cache.annotation.CacheConfig;
import org.springframework.cache.annotation.CacheEvict;
import org.springframework.cache.annotation.Caching;

@Service
@CacheConfig(cacheNames = {"books"}) // 配置缓存名称
public class BookService {
    
    @Caching(evict = {
            @CacheEvict(allEntries = true),
            @CacheEvict(cacheNames = "users", allEntries = true)}) // 清除所有缓存和users缓存
    public void refreshData(){
        
    }
    
}
```
## 4.2 Redis整合
### （一）配置文件配置
```yaml
spring:
  redis:
    host: xxx.xxx.xx.xx # Redis IP地址
    port: xxxx          # Redis端口号
    database: 0         # Redis数据库索引（默认为0）
    password: xxxxx     # Redis密码，没有就不用填写
    timeout: 500ms      # 连接超时时间（毫秒）
    jedis:
      pool:
        max-active: 50   # 连接池最大连接数（默认值：8）
        max-idle: 10     # 连接池最大空闲连接数（默认值：8）
        min-idle: 0      # 连接池最小空闲连接数（默认值：0）
        max-wait: -1     # 连接池最大阻塞等待时间（默认值：-1，表示永不超时）
```
### （二）RedisTemplate配置
```java
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.redis.connection.lettuce.LettuceConnectionFactory;
import org.springframework.data.redis.core.RedisTemplate;
import org.springframework.data.redis.serializer.Jackson2JsonRedisSerializer;

import com.fasterxml.jackson.databind.ObjectMapper;

@Configuration
public class RedisConfig {

    @Autowired
    private LettuceConnectionFactory lettuceConnectionFactory;

    @Bean
    public RedisTemplate<Object, Object> redisTemplate() {
        Jackson2JsonRedisSerializer jackson2JsonRedisSerializer = new Jackson2JsonRedisSerializer(Object.class);

        ObjectMapper om = new ObjectMapper();

        // 设置 ObjectMapper 的序列化内容类型为 json
        om.setVisibility(PropertyAccessor.ALL, JsonAutoDetect.Visibility.ANY);
        om.enableDefaultTyping(ObjectMapper.DefaultTyping.NON_FINAL);

        jackson2JsonRedisSerializer.setObjectMapper(om);

        RedisTemplate<Object, Object> template = new RedisTemplate<>();
        template.setConnectionFactory(lettuceConnectionFactory);
        template.setValueSerializer(jackson2JsonRedisSerializer);
        template.afterPropertiesSet();
        return template;
    }

}
```
### （三）Redis操作
```java
@Component
public class RedisUtil {

    @Autowired
    private RedisTemplate<String, Object> redisTemplate;

    /**
     * 插入缓存
     */
    public boolean setCache(String key, String value) {
        try {
            if (StringUtils.isNotBlank(value)) {
                redisTemplate.opsForValue().set(key, value);
                return true;
            } else {
                throw new Exception("缓存的值不能为空");
            }
        } catch (Exception e) {
            log.error("【Redis】插入缓存失败：{}", e);
            return false;
        }
    }

    /**
     * 获取缓存
     */
    public Object getCache(String key) {
        try {
            return redisTemplate.opsForValue().get(key);
        } catch (Exception e) {
            log.error("【Redis】获取缓存失败：{}", e);
            return null;
        }
    }

}
```
## 4.3 JVM调优
### （一）堆大小配置
一般来说，堆大小应该根据系统内存、CPU核数、应用的并发量来确定。推荐配置总内存的`80%`作为堆大小，以保证系统运行稳定。

### （二）垃圾回收算法配置
由于垃圾回收器存在不同的特性，如stw算法、分代收集算法等，根据应用场景选择合适的垃圾回收算法能够提升应用的性能。

推荐选择`-XX:+UseParallelGC`，它采用多线程进行垃圾回收，有效减少gc停顿的时间。

### （三）GC日志分析
如果启用了GC日志，那么日志记录级别需要设置为`-verbose:gc`。通过分析日志，可以找到应用的gc频率、停顿时间、老生代占比、新生代占比等。

## 4.4 接口限流
### （一）接口限流算法
限流算法的核心是控制流量，一般采用漏桶算法来实现。令牌桶算法由若干个同心球组成，第一个球填满后，后面的球就会倒向漏出来，形成有一定的概率让流经，以平滑流量。

漏桶算法中，设定一个固定容量的桶，只允许请求的一定数量通过，超过这个数量的请求将被丢弃，以此来控制流量。在API限流中，将每个IP地址的访问频率限制在规定的时间内，并对超出的访问请求进行拒绝。

### （二）Guava RateLimiter实现限流
```java
import com.google.common.util.concurrent.RateLimiter;

public class ApiRateLimit {

    private static final Map<String, RateLimiter> rateLimiters = new ConcurrentHashMap<>();

    public synchronized static double acquireToken(String ipAddress) {
        long currentTimeMillis = System.currentTimeMillis();
        RateLimiter rateLimiter = rateLimiters.computeIfAbsent(ipAddress, k -> createRateLimiter());
        double availableTokens = rateLimiter.getAvailableTokens(TimeUnit.MILLISECONDS) / 1000.0;
        double timeDiffInSeconds = Math.max((currentTimeMillis - getLastUpdatedTimeMillis(ipAddress)), 1) / 1000.0;
        double tokensToAdd = Math.min(availableTokens + timeDiffInSeconds * RATE_LIMITING_REQUESTS_PER_SECOND, MAXIMUM_TOKENS);
        addTokens(rateLimiter, tokensToAdd - availableTokens);
        setLastUpdatedTimeMillis(ipAddress, currentTimeMillis);
        return tokensToAdd;
    }

    private static void addTokens(RateLimiter rateLimiter, double tokensToAdd) {
        for (; tokensToAdd > 0; ) {
            double acquiredTokens = rateLimiter.tryAcquire(Math.min(tokensToAdd, Double.MAX_VALUE));
            tokensToAdd -= acquiredTokens;
        }
    }

    private static long getLastUpdatedTimeMillis(String ipAddress) {
        return redisUtil.getCache(LAST_UPDATED_TIME_KEY_PREFIX + ipAddress) == null? 0 : Long.parseLong(redisUtil.getCache(LAST_UPDATED_TIME_KEY_PREFIX + ipAddress).toString());
    }

    private static void setLastUpdatedTimeMillis(String ipAddress, long lastUpdatedTimeMillis) {
        redisUtil.setCache(LAST_UPDATED_TIME_KEY_PREFIX + ipAddress, String.valueOf(lastUpdatedTimeMillis));
    }

    private static RateLimiter createRateLimiter() {
        int numPermitsPerSecond = 10; // 每秒允许请求次数
        return RateLimiter.create(numPermitsPerSecond);
    }

    private static final Integer RATE_LIMITING_REQUESTS_PER_SECOND = 10;
    private static final Integer MAXIMUM_TOKENS = 10;
    private static final String LAST_UPDATED_TIME_KEY_PREFIX = "api_limit:";

}
```
## 4.5 数据分片
数据分片（Data Sharding）是指将数据分布到不同的数据库或表中，以达到水平扩展的目的。在分布式缓存中，数据分片可以帮助降低单个缓存服务器的压力，提高缓存的可用性。

数据分片的原理是将数据按照业务规则，分配到不同的数据库或表中。比如，将用户信息放在user数据库的user表中，将订单信息放在order数据库的order表中，从而实现了水平扩展。

### （一）ShardingJDBC实现数据分片
#### 1. 创建分库分表的数据库表结构
创建分库分表的数据库表，并在表中加上主键id字段。
```sql
CREATE TABLE user (
  id BIGINT NOT NULL AUTO_INCREMENT COMMENT '主键',
  name VARCHAR(50) NOT NULL DEFAULT '' COMMENT '姓名',
  age INT NOT NULL DEFAULT 0 COMMENT '年龄',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE utf8mb4_general_ci;

CREATE TABLE order (
  id BIGINT NOT NULL AUTO_INCREMENT COMMENT '主键',
  user_id BIGINT NOT NULL DEFAULT 0 COMMENT '用户ID',
  amount DECIMAL(10,2) NOT NULL DEFAULT 0.00 COMMENT '金额',
  status TINYINT NOT NULL DEFAULT 0 COMMENT '状态',
  PRIMARY KEY (`id`),
  INDEX `idx_userId`(`user_id`) USING BTREE 
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE utf8mb4_general_ci;
```
#### 2. 修改配置文件
```yaml
spring:
  shardingsphere:
    datasource:
      names: master,slave0,slave1
      master:
        type: com.zaxxer.hikari.HikariDataSource
        driverClassName: com.mysql.cj.jdbc.Driver
        jdbcUrl: jdbc:mysql://localhost:3306/demo_ds0?serverTimezone=UTC&useUnicode=true&characterEncoding=UTF-8&useSSL=false&allowPublicKeyRetrieval=true
        username: root
        password: <PASSWORD>
      slave0:
        type: com.zaxxer.hikari.HikariDataSource
        driverClassName: com.mysql.cj.jdbc.Driver
        jdbcUrl: jdbc:mysql://localhost:3306/demo_ds1?serverTimezone=UTC&useUnicode=true&characterEncoding=UTF-8&useSSL=false&allowPublicKeyRetrieval=true
        username: root
        password: <PASSWORD>
      slave1:
        type: com.zaxxer.hikari.HikariDataSource
        driverClassName: com.mysql.cj.jdbc.Driver
        jdbcUrl: jdbc:mysql://localhost:3306/demo_ds2?serverTimezone=UTC&useUnicode=true&characterEncoding=UTF-8&useSSL=false&allowPublicKeyRetrieval=true
        username: root
        password: <PASSWORD>
    rules:
      sharding:
        tables:
          user:
            actual-data-nodes: demo_ds$->{0..1}.user
            table-strategy:
              complex:
                shardings:
                  ds-inline:
                    algorithm-expression: ds_$[shardingValue % 3]
                key-generators:
                  snowflake:
                    type: SNOWFLAKE
                    props:
                      worker-id: 123
      default-database-strategy:
        inline:
          sharding-column: user_id
          algorithm-expression: ds_${user_id % 3}
    props:
      sql-show: true
```
#### 3. 使用ShardingJDBC操作数据
```java
@Service
public class UserService {
    
    @Autowired
    private DataSource dataSource;
    
    @Autowired
    private NamedParameterJdbcTemplate namedParameterJdbcTemplate;
    
    public List<User> getAllUsers() {
        String sql = "SELECT id,name,age FROM user";
        return namedParameterJdbcTemplate.query(sql, BeanPropertyRowMapper.newInstance(User.class));
    }
    
    public User getUserById(long userId) {
        String sql = "SELECT id,name,age FROM user WHERE id=?";
        User user = namedParameterJdbcTemplate.queryForObject(sql, BeanPropertyRowMapper.newInstance(User.class), userId);
        return user;
    }
    
    public boolean addUser(User user) throws SQLException {
        Connection conn = dataSource.getConnection();
        PreparedStatement ps = conn.prepareStatement("INSERT INTO user (name, age) VALUES (?,?)");
        ps.setString(1, user.getName());
        ps.setInt(2, user.getAge());
        int count = ps.executeUpdate();
        conn.close();
        return count > 0;
    }
    
    public boolean updateUser(User user) throws SQLException {
        Connection conn = dataSource.getConnection();
        PreparedStatement ps = conn.prepareStatement("UPDATE user SET name=?, age=? WHERE id=?");
        ps.setString(1, user.getName());
        ps.setInt(2, user.getAge());
        ps.setLong(3, user.getId());
        int count = ps.executeUpdate();
        conn.close();
        return count > 0;
    }
    
    public boolean deleteUser(long userId) throws SQLException {
        Connection conn = dataSource.getConnection();
        PreparedStatement ps = conn.prepareStatement("DELETE FROM user WHERE id=?");
        ps.setLong(1, userId);
        int count = ps.executeUpdate();
        conn.close();
        return count > 0;
    }
    
}
```
#### 4. 测试
```java
@SpringBootTest
public class DemoApplicationTests {

    @Autowired
    private UserService userService;

    @Test
    public void testAddUser() throws SQLException {
        User user = new User();
        user.setName("zhangsan");
        user.setAge(20);
        assertTrue(userService.addUser(user));
    }

    @Test
    public void testGetAllUsers() {
        List<User> users = userService.getAllUsers();
        assertEquals(2, users.size());
    }

    @Test
    public void testGetUserById() {
        User user = userService.getUserById(1);
        assertEquals(1, user.getId());
        assertEquals("zhangsan", user.getName());
        assertEquals(20, user.getAge());
    }

    @Test
    public void testUpdateUser() throws SQLException {
        User user = new User();
        user.setId(1);
        user.setName("lisi");
        user.setAge(25);
        assertTrue(userService.updateUser(user));
    }

    @Test
    public void testDeleteUser() throws SQLException {
        assertTrue(userService.deleteUser(1));
    }

}
```