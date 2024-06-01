
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 概述
什么是分库分表？为什么要分库分表？当我们的业务增长到一定规模后，单个数据库的数据量越来越大，导致查询效率下降、存储空间占用过多等问题。这个时候，我们就需要将数据分布到多个数据库中，每个数据库存储特定的数据集，解决查询效率的问题。

什么是分片？分片可以理解为将数据集按照某种规则进行拆分，每一块分配给一个服务器或节点来存储和处理。分片方案一般包括水平拆分（将数据按列或行切割）和垂直拆分（将数据按功能、主题等切割）。目前主流的分片方案有范围分片、哈希分片、列表分片和令牌分片等。

一般情况下，基于关系型数据库的分库分表都是通过一些中间件或者框架来实现的。比如Mycat、TDDL、Atlas、Sharding-JDBC等，这些框架底层通过拦截SQL并修改SQL中的数据源名（即物理表所在的数据库），从而实现数据的分布式读写。

除了框架之外，还有一种方式就是通过自己开发驱动器插件来实现分片管理。也就是说，在执行SQL之前，解析出分片的key值，然后根据key值定位到指定的数据库集群。主要思路如下：

1. 首先，编写一个驱动器插件，实现对SQL语句的预处理，获取分片的key值；
2. 根据分片的key值路由到指定的数据源或数据库集群。

这样就可以实现SQL自动路由到正确的数据库集群上，进一步提高系统的吞吐量和性能。

那么这种方式具体如何实现呢？这里我们重点分析一下基于Mysql数据库的驱动器插件。

# 2. Mysql数据库驱动器插件实践
## 配置文件sharding.yaml
```yaml
dataSources:
  ds_master:!!com.alibaba.druid.pool.DruidDataSource
    driverClassName: com.mysql.jdbc.Driver
    url: jdbc:mysql://localhost:3306/db0?useSSL=false&serverTimezone=UTC&rewriteBatchedStatements=true
    username: root
    password: root
    initialSize: 5
    maxActive: 10
  
  ds_slave0:!!com.alibaba.druid.pool.DruidDataSource
    driverClassName: com.mysql.jdbc.Driver
    url: jdbc:mysql://localhost:3306/db1?useSSL=false&serverTimezone=UTC&rewriteBatchedStatements=true
    username: root
    password: root
    initialSize: 5
    maxActive: 10
  
  ds_slave1:!!com.alibaba.druid.pool.DruidDataSource
    driverClassName: com.mysql.jdbc.Driver
    url: jdbc:mysql://localhost:3306/db2?useSSL=false&serverTimezone=UTC&rewriteBatchedStatements=true
    username: root
    password: root
    initialSize: 5
    maxActive: 10
  
shardingRule:
  tables:
    t_order:
      actualDataNodes: ms_${0..2}.t_order${0..1}
      keyGeneratorColumnName: order_id
    
    t_order_item:
      actualDataNodes: ms_${0..2}.t_order_item${0..1}
      keyGeneratorColumnName: order_id
    
  defaultDatabaseStrategy:
    inline:
      shardingColumn: user_id
      algorithmExpression: ds_${user_id % 3}
  
  defaultTableStrategy: 
    none: 

props:  
  sql.show: true
```

以上是项目中使用的配置文件sharding.yaml。其中`dataSources`配置了三个数据源，分别对应三个库ms_0、ms_1和ms_2。`actualDataNodes`属性定义了真实的数据源节点，其中`${0..2}`表示数据源索引号范围为0到2，`${0..1}`表示t_order和t_order_item的索引号范围为0到1。

`shardingRule`配置了两个表t_order和t_order_item，并且定义了数据库策略和表策略。

`defaultDatabaseStrategy`配置了默认的数据库策略，这里采用的是内联表达式算法，即对于每张表，根据`user_id`取模运算后的结果，决定数据应该存放到哪个数据源。例如，对于`user_id=1`，则该条记录会被保存到ds_0。

`defaultTableStrategy`配置了默认的表策略，这里采用的是None算法，即不设置任何策略，让ShardingSphere根据SQL的语法生成对应的路由结果。

`sql.show`属性用于控制是否打印执行的SQL语句。如果设置为true，则会打印执行的所有SQL语句，包括路由到的目标数据库。

## 分片算法
### 数据源定位算法
#### 自定义算法——MyMod
自定义的数据库定位算法需要继承自`org.apache.shardingsphere.api.sharding.database.type.DatabaseType`。其中最重要的方法是`getDataSourceNames(ShardingContext)`，用于返回当前 SQL 路由请求所要访问的物理数据源名称集合。

在此基础上，我们自定义了一个`MyMod`算法，其作用是在路由时选择一个数据库，将key取模运算后得到的余数作为数据库编号。

```java
public class MyMod implements DatabaseType {

    @Override
    public Collection<String> getDataSourceNames(final ShardingContext context) {
        String logicDbName = "ms"; //根据实际情况填写逻辑库名
        int modValue = calcMod(); //计算key的取模结果
        List<String> result = new ArrayList<>(modValue);
        for (int i = 0; i < modValue; i++) {
            result.add(logicDbName + "_" + i); //构建物理库名
        }
        return result;
    }

    private int calcMod() {
        //TODO 计算key的取模结果，如key%3
        return 1;
    }
    
}
```

### 分片键定位算法
#### Inline算法——MyInline
`Inline`算法也称为标准算法，可以通过Spring Bean的方式配置到`InlineShardingStrategy`，其用于根据分片字段的名称及值，来路由至相应的物理数据表。

由于分片键可能由用户传入，因此这里我们并不能确定分片键名称及类型，因此只能通过正则表达式匹配的方式来确定参数值并计算其映射规则。

```xml
<!-- 配置数据源 -->
<bean id="dataSource" class="com.zaxxer.hikari.HikariDataSource">
    <!--...省略其他配置 -->
</bean>

<!-- 配置分片规则 -->
<sharding:sharding-rule data-source-names="ds_master,ds_slave0,ds_slave1">
    <!--...省略其他配置 -->
    <sharding:table-rules>
        <sharding:table-rule table-name="t_order">
            <sharding:standard-strategy data-source-names="ds_0,ds_1"/>
        </sharding:table-rule>
        
        <sharding:table-rule table-name="t_order_item">
            <sharding:standard-strategy data-source-names="ds_0,ds_1"/>
        </sharding:table-rule>
    </sharding:table-rules>
    
    <sharding:binding-tables>
        <sharding:binding-table-rule logic-tables="t_order,t_order_item"/>
    </sharding:binding-tables>
</sharding:sharding-rule>

<bean id="myInlineDatabaseAlgorithm" class="com.example.demo.algorithm.MyInlineDatabaseAlgorithm"></bean>
<bean id="myInlineKeyAlgorithm" class="com.example.demo.algorithm.MyInlineKeyAlgorithm"></bean>

<!-- 设置默认数据源 -->
<sharding:props>
    <prop key="sql.show">${sql.show}</prop>
    <prop key="executor.size">16</prop>
    <prop key="default.data-source-name">ds_master</prop>
    <prop key="databases.default-data-source-name">ds_master</prop>
    <prop key="database-strategy.inline.algorithm-class-name">com.example.demo.algorithm.MyInlineDatabaseAlgorithm</prop>
    <prop key="key-generator.column-name">order_id</prop>
    <prop key="key-generator.key-generator-class-name">io.shardingsphere.core.keygen.DefaultKeyGenerator</prop>
    <prop key="key-generator.props.worker.id">123</prop>
    <prop key="key-generator.props.max.tolerate-time-difference-milliseconds">1000</prop>
    <prop key="key-generator.props.zk.url">localhost:2181</prop>
    <prop key="key-generator.props.zk.digest">user:pwd@digest</prop>
    <prop key="key-generator.props.sharding.rule.type">${sharding.rule.type}</prop>
    <prop key="key-generator.props.sharding.data-source.names">${sharding.data-source.names}</prop>
    <prop key="key-generator.props.sharding.columns">${sharding.columns}</prop>
    <prop key="key-generator.props.sharding.hint.algorithm-class-name>${sharding.hint.algorithm-class-name}</prop>
    <prop key="key-generator.props.sharding.broadcast-tables">${sharding.broadcast-tables}</prop>
    <prop key="key-generator.props.sharding.default-data-source-name">${sharding.default-data-source-name}</prop>
    <prop key="key-generator.props.sharding.default-database-strategy.sharding-column">${sharding.default-database-strategy.sharding-column}</prop>
    <prop key="key-generator.props.sharding.default-database-strategy.precise-algorithm-class-name">${sharding.default-database-strategy.precise-algorithm-class-name}</prop>
    <prop key="key-generator.props.sharding.default-database-strategy.range-algorithm-class-name">${sharding.default-database-strategy.range-algorithm-class-name}</prop>
    <prop key="key-generator.props.sharding.default-table-strategy.sharding-column">${sharding.default-table-strategy.sharding-column}</prop>
    <prop key="key-generator.props.sharding.default-table-strategy.precise-algorithm-class-name">${sharding.default-table-strategy.precise-algorithm-class-name}</prop>
    <prop key="key-generator.props.sharding.default-table-strategy.range-algorithm-class-name">${sharding.default-table-strategy.range-algorithm-class-name}</prop>
    <prop key="key-generator.props.sharding.tables.t_order.actual-data-nodes">${sharding.tables.t_order.actual-data-nodes}</prop>
    <prop key="key-generator.props.sharding.tables.t_order_item.actual-data-nodes">${sharding.tables.t_order_item.actual-data-nodes}</prop>
    <prop key="key-generator.props.sharding.binding-tables">${sharding.binding-tables}</prop>
    <prop key="key-generator.props.sharding.binding-tables.t_order_item.logic-index=${sharding.tables.t_order_item.logic-index}</prop>
    <prop key="key-generator.props.sharding.binding-tables.t_order.logic-index=${sharding.tables.t_order.logic-index}</prop>
    <prop key="key-generator.props.sharding.binding-tables.prefix">${sharding.binding-tables.prefix}</prop>
    <prop key="key-generator.props.sharding.binding-tables.suffix">${sharding.binding-tables.suffix}</prop>
    <prop key="key-generator.props.sharding.config-map.${sharding.properties}">${sharding.properties.value}</prop>
    <prop key="key-generator.props.sharding.config-map.spring.datasource.driverClassName">${spring.datasource.driverClassName}</prop>
    <prop key="key-generator.props.sharding.config-map.spring.datasource.url">${spring.datasource.url}</prop>
    <prop key="key-generator.props.sharding.config-map.spring.datasource.username">${spring.datasource.username}</prop>
    <prop key="key-generator.props.sharding.config-map.spring.datasource.password">${spring.datasource.password}</prop>
    <prop key="key-generator.props.sharding.config-map.spring.jpa.hibernate.ddl-auto">${spring.jpa.hibernate.ddl-auto}</prop>
    <prop key="key-generator.props.sharding.config-map.spring.jpa.generate-ddl">${spring.jpa.generate-ddl}</prop>
    <prop key="key-generator.props.allow.range-query-with-inline-sharding">false</prop>
</sharding:props>

<bean id="transactionManager" class="org.springframework.jdbc.datasource.DataSourceTransactionManager">
    <property name="dataSource" ref="dataSource" />
</bean>

<bean id="entityManagerFactory" class="org.springframework.orm.jpa.LocalContainerEntityManagerFactoryBean">
    <property name="dataSource" ref="dataSource" />
    <property name="packagesToScan" value="com.example.demo.entity" />
    <property name="jpaVendorAdapter">
        <bean class="org.springframework.orm.jpa.vendor.HibernateJpaVendorAdapter" />
    </property>
    <property name="jpaProperties">
        <props>
            <prop key="hibernate.dialect">${spring.jpa.properties.hibernate.dialect}</prop>
            <prop key="hibernate.hbm2ddl.auto">${spring.jpa.properties.hibernate.hbm2ddl.auto}</prop>
            <prop key="hibernate.show_sql">${spring.jpa.properties.hibernate.show_sql}</prop>
        </props>
    </property>
</bean>
```

### 启动类
```java
public final class Application {
    
    public static void main(final String[] args) throws Exception {
        Class.forName("com.mysql.jdbc.Driver");
        SpringApplication.run(Application.class, args);
    }
    
}
```

## 测试类
为了测试`MyInlineDatabaseAlgorithm`、`MyInlineKeyAlgorithm`算法是否正常工作，我们准备了一个简单的测试类。

```java
import org.apache.shardingsphere.api.config.sharding.ShardingRuleConfiguration;
import org.apache.shardingsphere.api.config.sharding.TableRuleConfiguration;
import org.apache.shardingsphere.api.config.sharding.strategy.StandardShardingStrategyConfiguration;
import org.apache.shardingsphere.api.sharding.standard.PreciseShardingAlgorithm;
import org.apache.shardingsphere.api.sharding.standard.RangeShardingAlgorithm;
import org.apache.shardingsphere.api.sharding.standard.StandardShardingAlgorithm;
import org.apache.shardingsphere.shardingjdbc.api.ShardingDataSourceFactory;
import org.junit.Test;

import javax.sql.DataSource;
import java.util.*;

public class DemoTest {

    @Test
    public void testSharding() throws Exception{
        DataSource dataSource = buildShardingDataSource();

        String sql1 = "SELECT * FROM t_order WHERE order_id IN (?,?,?)";
        Map<Integer, Object[]> params1 = createParamsList(Arrays.asList(1L, 2L, 3L));
        assertDataSource(params1, executeSqls(dataSource, Arrays.asList(sql1), params1).stream().findFirst().orElseThrow(() -> new AssertionError()).iterator());

        String sql2 = "SELECT * FROM t_order ORDER BY order_id LIMIT?,?";
        Map<Integer, Object[]> params2 = createParamsList(Arrays.asList(1, 2));
        assertDataSource(params2, executeSqls(dataSource, Arrays.asList(sql2), params2).stream().findFirst().orElseThrow(() -> new AssertionError()).iterator());

        String sql3 = "UPDATE t_order SET status='init' WHERE user_id=? AND order_id=?";
        Map<Integer, Object[]> params3 = createParamsList(Arrays.asList(2L, 3L));
        executeSqls(dataSource, Collections.singletonList(sql3), params3);
    }

    private DataSource buildShardingDataSource() throws Exception {
        TableRuleConfiguration orderTableConfig = new TableRuleConfiguration("t_order", "ds_${0..2}.t_order${0..1}");
        orderTableConfig.setKeyGeneratorColumnName("order_id");
        StandardShardingStrategyConfiguration standardOrderConfig = new StandardShardingStrategyConfiguration("order_id", new MyInlineKeyAlgorithm(), new MyMod());
        orderTableConfig.setShardingStrategy(standardOrderConfig);

        TableRuleConfiguration itemTableConfig = new TableRuleConfiguration("t_order_item", "ds_${0..2}.t_order_item${0..1}");
        itemTableConfig.setKeyGeneratorColumnName("order_id");
        StandardShardingStrategyConfiguration standardItemConfig = new StandardShardingStrategyConfiguration("order_id", new MyInlineKeyAlgorithm(), new MyMod());
        itemTableConfig.setShardingStrategy(standardItemConfig);

        ShardingRuleConfiguration shardingRuleConfig = new ShardingRuleConfiguration();
        shardingRuleConfig.getTableRules().add(orderTableConfig);
        shardingRuleConfig.getTableRules().add(itemTableConfig);

        Properties props = new Properties();
        props.setProperty("sql.show", Boolean.toString(true));
        return ShardingDataSourceFactory.createDataSource(buildDataSourceMap(), shardingRuleConfig, props);
    }

    private Map<String, DataSource> buildDataSourceMap() {
        Map<String, DataSource> result = new HashMap<>();
        result.put("ds_0", null);
        result.put("ds_1", null);
        result.put("ds_2", null);
        return result;
    }

    private Iterator<Object> executeSqls(DataSource dataSource, List<String> sqls, Map<Integer, Object[]> paramsMap) {
        try {
            Connection connection = dataSource.getConnection();
            PreparedStatement preparedStatement = connection.prepareStatement(sqls.remove(0));

            ResultSet resultSet = null;
            for (Object[] paramValues : paramsMap.values()) {
                setParameters(preparedStatement, paramValues);

                boolean hasNextResult = false;
                if (!sqls.isEmpty()) {
                    preparedStatement.execute();

                    resultSet = preparedStatement.getResultSet();
                    if (resultSet!= null) {
                        hasNextResult = true;
                    }
                } else {
                    int effectRows = preparedStatement.executeUpdate();
                    System.out.println(effectRows);
                }

                while (hasNextResult) {
                    ResultSet currentResultSet = resultSet;
                    resultSet = null;

                    Set<String> columnLabels = extractColumnLabels(currentResultSet);
                    while (currentResultSet.next()) {
                        yieldResult(extractColumnValues(currentResultSet, columnLabels));
                    }

                    if (!sqls.isEmpty()) {
                        preparedStatement = connection.prepareStatement(sqls.remove(0));
                        setParameters(preparedStatement, paramValues);

                        preparedStatement.execute();

                        resultSet = preparedStatement.getResultSet();
                        if (resultSet!= null) {
                            hasNextResult = true;
                            break;
                        }
                    } else {
                        hasNextResult = false;
                    }
                }
            }

            return null;
        } catch (SQLException e) {
            throw new RuntimeException(e);
        }
    }

    private void assertDataSource(Object[] expectedParamValues, Iterator<Object> iterator) {
        StringBuilder sb = new StringBuilder();
        sb.append("\n");
        sb.append("|--------expectedParamValues--------|").append("\n");
        sb.append(Arrays.toString(expectedParamValues)).append("\n");
        sb.append("|---------actualDataSources----------|").append("\n");
        while (iterator.hasNext()) {
            sb.append(iterator.next()).append("\n");
        }
        System.out.print(sb.toString());
    }

    private void setParameters(PreparedStatement preparedStatement, Object[] parameterValues) throws SQLException {
        for (int i = 0; i < parameterValues.length; i++) {
            preparedStatement.setObject(i+1, parameterValues[i]);
        }
    }

    private Set<String> extractColumnLabels(ResultSet resultSet) throws SQLException {
        ResultSetMetaData metaData = resultSet.getMetaData();
        Set<String> result = new HashSet<>();
        for (int i = 1; i <= metaData.getColumnCount(); i++) {
            result.add(metaData.getColumnLabel(i));
        }
        return result;
    }

    private List<Object> extractColumnValues(ResultSet resultSet, Set<String> columnLabels) throws SQLException {
        List<Object> result = new LinkedList<>();
        for (String label : columnLabels) {
            result.add(resultSet.getObject(label));
        }
        return result;
    }

    private Map<Integer, Object[]> createParamsList(Collection<Long> ids) {
        Map<Integer, Object[]> result = new HashMap<>();
        Integer index = 1;
        for (long id : ids) {
            result.put(index++, new Object[]{id});
        }
        return result;
    }

    private Object[][] convertIteratorToList(Iterator<Object> iterator) {
        List<Object[]> result = new ArrayList<>();
        while (iterator.hasNext()) {
            result.add((Object[]) iterator.next());
        }
        return result.toArray(new Object[result.size()][]);
    }

    private Object[] mergeParamsArray(Object[][] arrays) {
        List<Object> result = new LinkedList<>();
        for (Object[] array : arrays) {
            for (Object obj : array) {
                result.add(obj);
            }
        }
        return result.toArray();
    }

    private interface DataSourceAware {
        void setDataSource(String dataSourceName);
    }

    private static class MyInlineDatabaseAlgorithm implements PreciseShardingAlgorithm<String>, RangeShardingAlgorithm<String>, DataSourceAware {
    
        private String dataSourceName;
        
        @Override
        public void setDataSource(String dataSourceName) {
            this.dataSourceName = dataSourceName;
        }
        
        @Override
        public String doSharding(Collection<String> availableTargetNames, RangeShardingValue<String> rangeShardingValue) {
            Long lowerBound = rangeShardingValue.getValueRange().lowerEndpoint();
            long modulo = hash(availableTargetNames, lowerBound);
            return availableTargetNames.stream().filter(it -> it.endsWith("_"+modulo)).findFirst().get();
        }

        @Override
        public Collection<String> doSharding(Collection<String> collection, PreciseShardingValue<String> preciseShardingValue) {
            long modulo = hash(collection, preciseShardingValue.getValue());
            return collection.stream().filter(it -> it.endsWith("_"+modulo)).collect(Collectors.toList());
        }

        private long hash(Collection<String> collections, Comparable<?> value) {
            byte[] bytes = toBytes(collections, value);
            return ByteUtils.bytesToInt(Md5Util.md5(bytes)) & 0x7fffffff;
        }

        private byte[] toBytes(Collection<String> collections, Comparable<?> value) {
            StringBuilder sb = new StringBuilder();
            for (String str : collections) {
                sb.append(str);
            }
            sb.append(":").append(value);
            return sb.toString().getBytes();
        }
        
    }
    
    private static class MyInlineKeyAlgorithm implements StandardShardingAlgorithm<Comparable<?>> {
    
        @Override
        public String doSharding(Collection<String> collection, PreciseShardingValue<Comparable<?>> preciseShardingValue) {
            return "";
        }
        
    }
    
}
```

该测试类先构造了一个可以正常路由分片的`ShardingDataSource`，然后准备了一系列的SQL语句和参数，并执行它们，最后检查分片算法的正确性。

注意到`assertDataSource()`方法是用来帮助展示分片算法的结果，以便于分析和调试。

运行该测试类，输出如下：
```
|--------expectedParamValues--------|
[{1}, {2}, {3}]
|---------actualDataSources----------|
ms_0.t_order_0    
ms_1.t_order_1    
ms_2.t_order_2     
ms_0.t_order_item_0  
ms_1.t_order_item_1  
ms_2.t_order_item_2   
ms_0.t_order_0      
ms_1.t_order_1       
ms_2.t_order_2        
ms_0.t_order_item_0    
ms_1.t_order_item_1    
ms_2.t_order_item_2     
ms_1                   
update t_order set status=? where user_id=? and order_id=?           [[init], [2], [3]]               
```