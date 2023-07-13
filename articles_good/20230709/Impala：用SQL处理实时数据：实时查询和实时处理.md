
作者：禅与计算机程序设计艺术                    
                
                
《50. Impala：用 SQL 处理实时数据：实时查询和实时处理》
==========================

# 50. Impala：用 SQL 处理实时数据：实时查询和实时处理

# 1. 引言

## 1.1. 背景介绍

Impala是一款由Alphabet公司开发的开源分布式SQL查询引擎，专为大数据和云计算构建。通过使用Impala，用户可以轻松地在分布式环境中运行高效、可扩展的SQL查询，实现快速查询大量数据并返回实时结果。

## 1.2. 文章目的

本文旨在通过Impala的实时查询和实时处理功能，向读者介绍如何使用SQL语言处理实时数据，提高数据处理效率。

## 1.3. 目标受众

本文主要针对以下目标受众：

- 有一定SQL基础的程序员和软件架构师
- 希望利用SQL语言处理实时数据的开发者和数据分析人员
- 有兴趣了解大数据和云计算技术的读者

# 2. 技术原理及概念

## 2.1. 基本概念解释

- SQL：结构化查询语言，用于操作关系型数据库
- Impala：支持SQL查询的分布式计算引擎，专为大数据和云计算构建
- 实时数据：即 real-time data，指在数据产生时即存在，可以进行实时查询的数据
- 实时查询：在数据产生时进行查询，以获取实时数据的应用场景
- 实时处理：在数据产生时进行处理，以实现实时结果的应用场景

## 2.2. 技术原理介绍: 算法原理，具体操作步骤，数学公式，代码实例和解释说明

- SQL查询：使用 SELECT 语句从关系型数据库中查询数据
- Impala查询：使用 Impala SQL 语句在分布式计算环境中查询实时数据
- 分布式计算：在多个计算节点上并行执行 SQL 查询，以提高查询效率
- 实时数据处理：使用 Java 和 Impala SQL 包将实时数据流处理为实时结果，并将其返回

## 2.3. 相关技术比较

- SQL：用于操作关系型数据库的基本语言，支持复杂数据结构和函数
- Python：一种通用脚本语言，广泛应用于数据处理和人工智能领域
- Java：一种通用编程语言，广泛应用于企业级应用和大数据领域
- Spark：一种大数据处理引擎，支持实时数据处理和机器学习

# 3. 实现步骤与流程

## 3.1. 准备工作：环境配置与依赖安装

要在计算机上安装和配置Impala，请参考官方文档：https://impala.readthedocs.io/en/latest/quickstart/。

## 3.2. 核心模块实现

- 在项目中创建一个Impala驱动的类
- 使用 Impala SQL 语句从数据库中查询实时数据
- 使用 Java 和 Impala SQL 包将实时数据流处理为实时结果，并将其返回

## 3.3. 集成与测试

- 在项目中集成Impala驱动，并运行测试用例

# 4. 应用示例与代码实现讲解

## 4.1. 应用场景介绍

以一个简单的实时数据处理应用为例，介绍如何使用Impala处理实时数据

### 4.1.1. 数据产生

假设我们有一个实时数据源，如Kafka或Hadoop等，其中每秒产生一个包含实时数据的批次数据。

### 4.1.2. 数据查询

在Impala中，可以使用以下 SQL语句查询实时数据：
```sql
SELECT * FROM <table_name>;
```
查询结果包括表中所有字段的数据。

### 4.1.3. 数据处理

在Impala中，可以使用Java代码将实时数据流处理为实时结果。以下是一个简单的Java代码示例：
```java
import java.util.Properties;
import org.apache.impala.sql.SaveMode;
import org.apache.impala.sql.SQLInputFormat;
import org.apache.impala.sql.SQLOutputFormat;
import org.apache.impala.sql.RealTime;
import org.apache.impala.sql.Save;
import org.apache.impala.sql.SqlException;
import org.apache.impala.sql.collection.InstantCollection;
import org.apache.impala.sql.collection.InstantCollection<?>;
import org.apache.impala.sql.util.SQLPromotingFunctions;
import org.apache.impala.sql.util.SQLWrapper;
import org.apache.impala.sql.ast.CompileContext;
import org.apache.impala.sql.ast.tree.Expression;
import org.apache.impala.sql.ast.tree.Table;
import org.apache.impala.sql.descriptors.SqlDescriptor;
import org.apache.impala.sql.descriptors.SqlFieldDescriptor;
import org.apache.impala.sql.descriptors.SqlGreaterThan;
import org.apache.impala.sql.descriptors.SqlLessThan;
import org.apache.impala.sql.descriptors.SqlLike;
import org.apache.impala.sql.descriptors.SqlNamed;
import org.apache.impala.sql.descriptors.SqlNotNamed;
import org.apache.impala.sql.descriptors.SqlOfType;
import org.apache.impala.sql.descriptors.SqlParam;
import org.apache.impala.sql.descriptors.SqlTable;
import org.apache.impala.sql.fieldtypes.SqlDoubleField;
import org.apache.impala.sql.fieldtypes.SqlFloatField;
import org.apache.impala.sql.fieldtypes.SqlIntegerField;
import org.apache.impala.sql.fieldtypes.SqlStringField;
import org.apache.impala.sql.fields.BytesRef;
import org.apache.impala.sql.functions.SqlDateFunction;
import org.apache.impala.sql.functions.SqlFunction;
import org.apache.impala.sql.functions.SqlProcedure;
import org.apache.impala.sql.functions.SqlStaticFunction;
import org.apache.impala.sql.functions.SqlTerm;
import org.apache.impala.sql.绝缘层.SqlBlockingStore;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.Event;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.EventType;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.StoredEvent;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.StoredEventType;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.StoredProcedure;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.StoredProcedureCall;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.StoredTable;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.StoredTableCall;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.Table;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.TableCall;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.TableStaticFunction;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.TableStaticProcedure;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.TableStaticTable;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.TableStaticTableCall;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.TableStaticTableStoredProcedure;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.TableStaticTableStoredProcedureCall;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.TableGreaterThan;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.TableLessThan;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.TableLike;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.TableNamed;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.TableNotNamed;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.TableOfType;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.TableStaticField;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.TableStaticFunction;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.TableStaticProcedure;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.TableStaticTable;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.TableStaticTableCall;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.TableStaticTableStoredProcedure;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.TableStaticTableStoredProcedureCall;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.TableTable;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.TableTableCall;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.TableStaticColumn;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.TableStaticColumnInfo;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.TableStaticTable;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.TableStaticTableCall;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.StoredTable;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.StoredTableCall;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.StoredTableStoredProcedure;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.StoredTableStoredProcedureCall;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.StoredTableTable;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.StoredTableTableCall;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.SqlBlockingStore;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.SqlBlockingStoreCall;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.SqlBlockingStoreTable;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.SqlBlockingStoreTableCall;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.SqlBlockingStoreStoredProcedure;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.SqlBlockingStoreStoredProcedureCall;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.SqlBlockingStoreTable;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.SqlBlockingStoreTableCall;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.SqlBlockingStoreStoredProcedure;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.SqlBlockingStoreStoredProcedureCall;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.Table;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.TableCall;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.StoredTable;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.StoredTableCall;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.SqlBlockingStoreTable;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.SqlBlockingStoreTableCall;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.SqlBlockingStoreStoredProcedure;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.SqlBlockingStoreStoredProcedureCall;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.SqlBlockingStoreTable;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.SqlBlockingStoreTableCall;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.StoredTable;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.StoredTableCall;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.SqlBlockingStoreTable;
import org.apache.impala.sql.绝缘层.SqlBlockingStore.SqlBlockingStoreTableCall;
import org.apache.impala.sql.table.api.Table;
import org.apache.impala.sql.table.api.TableBlockingStore;
import org.apache.impala.sql.table.api.TableBlockingStoreCall;
import org.apache.impala.sql.table.api.TableStaticTable;
import org.apache.impala.sql.table.api.TableStaticTableCall;
import org.apache.impala.sql.table.api.TableTable;
import org.apache.impala.sql.table.api.TableTableCall;
import org.apache.impala.sql.util.SQLPromotingFunctions;
import org.apache.impala.sql.util.SQLWrapper;
import org.apache.impala.sql.ast.CompileContext;
import org.apache.impala.sql.ast.tree.Expression;
import org.apache.impala.sql.ast.tree.Table;
import org.apache.impala.sql.ast.tree.TableColumn;
import org.apache.impala.sql.ast.tree.TableRecord;
import org.apache.impala.sql.descriptors.SqlBlockingStoreDescriptors;
import org.apache.impala.sql.descriptors.SqlTableDescriptor;
import org.apache.impala.sql.fieldtypes.SqlDoubleField;
import org.apache.impala.sql.fieldtypes.SqlFloatField;
import org.apache.impala.sql.fieldtypes.SqlIntegerField;
import org.apache.impala.sql.fieldtypes.SqlStringField;
import org.apache.impala.sql.fieldtypes.SqlBooleanField;
import org.apache.impala.sql.fieldtypes.SqlDateField;
import org.apache.impala.sql.fieldtypes.SqlNamed;
import org.apache.impala.sql.fieldtypes.SqlNotNamed;
import org.apache.impala.sql.descriptors.SqlTable;
import org.apache.impala.sql.descriptors.SqlTableDescriptor;
import org.apache.impala.sql.descriptors.SqlTableFieldDescriptor;
import org.apache.impala.sql.descriptors.SqlTableFieldName;
import org.apache.impala.sql.descriptors.SqlTableGreaterThan;
import org.apache.impala.sql.descriptors.SqlTableLessThan;
import org.apache.impala.sql.descriptors.SqlTableLike;
import org.apache.impala.sql.descriptors.SqlTableNamed;
import org.apache.impala.sql.descriptors.SqlTableNotNamed;
import org.apache.impala.sql.descriptors.SqlTableOfType;
import org.apache.impala.sql.descriptors.SqlNamed;
import org.apache.impala.sql.descriptors.SqlTableDescriptorBuilder;
import org.apache.impala.sql.descriptors.SqlTableField;
import org.apache.impala.sql.descriptors.SqlTableFieldName;
import org.apache.impala.sql.descriptors.SqlTableGreaterThanOrEqual;
import org.apache.impala.sql.descriptors.SqlTableLessThanOrEqual;
import org.apache.impala.sql.descriptors.SqlTableLikeOrEqual;
import org.apache.impala.sql.descriptors.SqlTableNamedGreaterThan;
import org.apache.impala.sql.descriptors.SqlTableNamedLessThan;
import org.apache.impala.sql.descriptors.SqlTableOfType;
import org.apache.impala.sql.descriptors.SqlTableStaticFunction;
import org.apache.impala.sql.descriptors.SqlTableStaticProcedure;
import org.apache.impala.sql.descriptors.SqlTableStaticTable;
import org.apache.impala.sql.descriptors.SqlTableTable;
import org.apache.impala.sql.descriptors.SqlTableTableCall;
import org.apache.impala.sql.descriptors.SqlTableStaticTableCall;
import org.apache.impala.sql.descriptors.SqlTableTableStaticFunction;
import org.apache.impala.sql.descriptors.SqlTableTableStaticProcedure;
import org.apache.impala.sql.descriptors.SqlTableTableStaticTable;
import org.apache.impala.sql.fieldtypes.SqlDoubleField;
import org.apache.impala.sql.fieldtypes.SqlFloatField;
import org.apache.impala.sql.fieldtypes.SqlIntegerField;
import org.apache.impala.sql.fieldtypes.SqlStringField;
import org.apache.impala.sql.fieldtypes.SqlBooleanField;
import org.apache.impala.sql.fieldtypes.SqlDateField;
import org.apache.impala.sql.fieldtypes.SqlNamed;
import org.apache.impala.sql.fieldtypes.SqlNotNamed;

import java.util.Map;

public class SQL50 {
  // SQL语句
  public static String sql50() {
    return "SELECT * FROMimpala.table.table_name WHEREcolumn_name = 1";
  }

  // 获取表名
  public static String getTableName(String columnName) {
    return "table_name";
  }

  // 判断两个表达式是否相等
  public static boolean isGreaterThan(String column1, String column2) {
    // 解析表达式
    Expression left = new Expression(column1);
    Expression right = new expression(column2);

    // 比较大小
    return left.compareTo(right) > 0;
  }

  // 判断两个表达式是否小于
  public static boolean isLessThan(String column1, String column2) {
    // 解析表达式
    Expression left = new expression(column1);
    Expression right = new expression(column2);

    // 比较大小
    return left.compareTo(right) < 0;
  }

  // 判断一个日期是否大于某个日期
  public static boolean isBefore(Date date1, Date date2) {
    // 解析日期
    return date1.before(date2);
  }

  // 判断一个日期是否小于某个日期
  public static boolean isAfter(Date date1, Date date2) {
    // 解析日期
    return date1.after(date2);
  }

  // 判断两个日期是否在某个范围内
  public static boolean isWithinRange(Date startDate, Date endDate) {
    // 解析日期
    return startDate.isBefore(endDate) && endDate.isAfter(startDate);
  }

  // 判断一个数是否大于某个数
  public static boolean isGreaterThan(int num1, int num2) {
    // 解析整数
    int result = num1 - num2;

    // 判断大小
    return result > 0;
  }

  // 判断一个数是否小于某个数
  public static boolean isLessThan(int num1, int num2) {
    // 解析整数
    int result = num1 - num2;

    // 判断大小
    return result < 0;
  }

  // 判断两个数是否在某个范围内
  public static boolean isWithinRange(int start, int end) {
    // 解析整数
    return start <= end;
  }

  // 将字符串解析为整数
  public static int parseInt(String value) {
    // 解析整数
    return Integer.parseInt(value);
  }

  // 将字符串解析为浮点数
  public static double parseDouble(String value) {
    // 解析浮点数
    return Double.parseDouble(value);
  }

  // 将字符串解析为布尔值
  public static boolean isTrue(String value) {
    // 解析布尔值
    return Boolean.parseBoolean(value);
  }

  // 将字符串解析为字符串
  public static String parseString(String value) {
    // 解析字符串
    return value;
  }

  // 将字符串解析为数组
  public static Array<String> parseArray(String value) {
    // 解析数组
    return value.split(",");
  }

  // 将字符串解析为日期
  public static Date parseDate(String value) {
    // 解析日期
    return new Date(parseString(value));
  }

  // 将日期解析为字符串
  public static String parseStringDate(Date value) {
    // 解析日期
    return value.toLocaleString();
  }

  // 将日期解析为Date
  public static Date parseDate(String value) {
    // 解析日期
    return new Date(parseString(value));
  }

  // 将字符串解析为数组
  public static Array<Date> parseArrayDate(String value) {
    // 解析数组
    return value.split(",");
  }

  // 将字符串解析为布尔值
  public static boolean isBoolean(String value) {
    // 解析布尔值
    return Boolean.parseBoolean(value);
  }

  // 将字符串解析为整数
  public static int parseInt(String value, int radix) {
    // 解析整数
    int result = Integer.parseInt(value);

    // 根据进制转换字符数组
    result = result % radix;

    return result;
  }

  // 将字符串解析为浮点数
  public static double parseDouble(String value, int radix) {
    // 解析浮点数
    double result = Double.parseDouble(value);

    // 根据进制转换字符数组
    result = result % radix;

    return result;
  }

  // 将字符串解析为字符串
  public static String parseString(String value, int radix) {
    // 解析字符串
    String result = value;

    // 根据进制转换字符数组
    result = result.replaceAll("" + Integer.toString(radix), Integer.toString(radix));

    return result;
  }

  // 将字符串解析为数组
  public static Array<String> parseArray(String value, int radix) {
    // 解析数组
    Array<String> result = new ArrayList<>();

    // 解析字符串
    result = result.addAll(value.split(","));

    // 根据进制转换字符数组
    result = result.replaceAll("" + Integer.toString(radix), Integer.toString(radix));

    return result;
  }

  // 将日期格式化为字符串
  public static String formatDate(Date date, String format) {
    // 解析日期
    long timestamp = date.toInstant().toEpochMilli();

    // 解析字符串格式
    String formatted = format.replaceAll("(?<=\\S)", "").replaceAll("(?<!\\S)", ""));

    // 格式化日期
    return new SimpleDateFormat(formatted).format(timestamp);
  }

  // 将日期格式化为日期对象
  public static Date parseDate(String value, String format) {
    // 解析日期
    long timestamp = parseStringDate(value);

    // 解析字符串格式
    SimpleDateFormat formatted = new SimpleDateFormat(format);

    // 格式化日期
    return new Date(timestamp);
  }

  // 判断两个日期是否在某个范围内
  public static boolean isDateInRange(Date startDate, Date endDate, int radix) {
    // 解析日期
    long startTimestamp = parseDate(startDate.toString(), "yyyy-MM-dd");
    long endTimestamp = parseDate(endDate.toString(), "yyyy-MM-dd");

    // 判断是否在指定范围内
    return startTimestamp <= endTimestamp;
  }

  // 获取指定列中的数据
  public static List<String> getColumnData(Table table, String columnName) {
    // 解析列名
    List<TableFieldDescriptor> fields = table.getTableDescriptor().getFieldInfos(columnName);

    // 获取数据
    List<String> data = new ArrayList<>();

    // 遍历所有行
    for (TableFieldDescriptor field : fields) {
      // 解析字段值
      String value = field.getValue(null);

      // 判断数据是否为空
      if (value == null) {
        data.add(null);
      } else if (isDateInRange(new Date(value), new Date(value.length))) {
        data.add(value);
      }
    }

    return data;
  }

  // 获取指定列中的数据
  public static List<Date> getColumnDateData(Table table, String columnName) {
    // 解析列名
    List<TableFieldDescriptor> fields = table.getTableDescriptor().getFieldInfos(columnName);

    // 获取数据
    List<Date> data = new ArrayList<>();

    // 遍历所有行
    for (TableFieldDescriptor field : fields) {
      // 解析字段值
      String value = field.getValue(null);

      // 判断数据是否为空
      if (value == null) {
        data.add(null);
      } else if (isDateInRange(new Date(value), new Date(value.length))) {
        data.add(new Date(value));
      }
    }

    return data;
  }

  // 解析日期字段
  public static Date parseDateString(String value) {
    // 解析日期
    long timestamp = parseStringDate(value);

    // 解析日期格式
    return new Date(timestamp);
  }

  // 解析日期字段
  public static Date parseDate(String value) {
    // 解析日期
    long timestamp = parseString(value);

    // 解析日期格式
    return new Date(timestamp);
  }

  // 获取指定列中的数据
  public static List<String> getColumnData(Table table, String columnName) {
    // 解析列名
    List<TableFieldDescriptor> fields = table.getTableDescriptor().getFieldInfos(columnName);

    // 获取数据
    List<String> data = new ArrayList<>();

    // 遍历所有行
    for (TableFieldDescriptor field : fields) {
      // 解析字段值
      String value = field.getValue(null);

      // 判断数据是否为空
      if (value == null) {
        data.add(null);
      } else if (isDateInRange(new Date(value), new Date(value.length))) {
        data.add(value);
      }
    }

    return data;
  }

  // 获取指定列中的数据
  public static List<Date> getColumnDateData(Table table, String columnName) {
    // 解析列名
    List<TableFieldDescriptor> fields = table.getTableDescriptor().getFieldInfos(columnName);

    // 获取数据
    List<Date> data = new ArrayList<>();

    // 遍历所有行
    for (TableFieldDescriptor field : fields) {
      // 解析字段值
      String value = field.getValue(null);

      // 判断数据是否为空
      if (value == null) {
        data.add(null);
      } else if (isDateInRange(new Date(value), new Date(value.length))) {
        data.add(new Date(value));
      }
    }

    return data;
  }

  // 根据日期范围查询数据
  public static List<String> queryDateData(Table table, String columnName, int startDate, int endDate) {
    // 解析列名
    List<TableFieldDescriptor> fields = table.getTableDescriptor().getFieldInfos(columnName);

    // 获取数据
    List<String> data = new ArrayList<>();

    // 遍历所有行
    for (TableFieldDescriptor field : fields) {
      // 解析字段值
      String value = field.getValue(null);

      // 判断数据是否为空
      if (value == null) {
        data.add(null);
      } else if (isDateInRange(new Date(value), new Date(endDate))) {
        data.add(value);
      }
    }

    return data;
  }

  // 根据日期范围查询数据
  public static List<Date> queryDate(Table table, String columnName, int startDate, int endDate) {
    // 解析列名
    List<TableFieldDescriptor> fields = table.getTableDescriptor().getFieldInfos(columnName);

    // 获取数据
    List<Date> data = new ArrayList<>();

    // 遍历所有行
    for (TableFieldDescriptor field : fields) {
      // 解析字段值
      String value = field.getValue(null);

      // 判断数据是否为空
      if (value == null) {
        data.add(null);
      } else if (isDateInRange(new Date(value), new Date(endDate))) {
        data.add(new Date(value));
      }
    }

    return data;
  }

  // 根据条件查询数据
  public static List<String> queryDateData(Table table, String columnName, String condition) {
    // 解析列名
    List<TableFieldDescriptor> fields = table.getTableDescriptor().getFieldInfos(columnName);

    // 获取数据
    List<String> data = new ArrayList<>();

    // 遍历所有行
    for (TableFieldDescriptor field : fields) {
      // 解析字段值
      String value = field.getValue(null);

      // 判断数据是否为空
      if (value == null) {
        data.add(null);
      } else if (isDateInRange(new Date(value), new Date(endDate))) {
        data.add(value);
      }

      // 判断条件是否成立
      if (condition.startsWith(">")) {
        data.add(value);
      } else if (condition.startsWith("<")) {
        data.add(new Date(value));
      }

