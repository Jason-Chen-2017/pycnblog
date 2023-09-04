
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 什么是DynamoDB？
DynamoDB 是 Amazon Web Services (AWS) 提供的无服务器的 NoSQL 数据库服务。它提供了快速、可扩展且高度可用的数据库存储，是构建高并发、实时应用的理想选择。
## 为什么要创建DynamoDB全局二级索引？
由于 DynamoDB 的设计目标之一就是具备低延迟（低于10毫秒）的访问时间，所以在某些情况下需要创建全局二级索引来满足查询性能要求。
举个例子，如果你的应用中存在一个查询条件是基于某个特定字段进行范围查询的场景，例如根据用户的年龄段来过滤数据，那么可以考虑创建一个全局二级索引来加速这个查询操作。
另外，如果你的数据库中存在一个大表，但有多个需要按频率排序或者搜索的数据字段，那么可以考虑创建全局二级索引来提升相应查询的效率。

本文将详细阐述如何在 DynamoDB 中创建全局二级索引。

# 2. 基本概念与术语
## 全局二级索引(Global secondary index)
DynamoDB 支持全局二级索引(GSI)，允许对同一张 DynamoDB 表中的多列创建不同的索引。全局二级索引一般都有两个属性：主键和范围键。主键只能是一个哈希键或分片键，而范围键则可以是任意类型的属性。全局二级索引通过建模不同粒度的关键子集来有效地支持复杂查询，从而优化 DynamoDB 表的查询性能。


图1: DynamoDB GSI 示例

## DynamoDB API
DynamoDB 提供了丰富的 API 来管理表、插入/删除记录、执行查询等操作。其中主要使用的 API 有：

1. CreateTable()
2. PutItem()
3. Query()
4. Scan()
5. UpdateItem()
6. DeleteItem()
7. DescribeTable()

## 属性类型
DynamoDB 中的属性类型包括字符串、数字、二进制、列表、字典和分片键。

### 字符串(String)
字符串属性用于存储 UTF-8 编码文本。字符串属性可以在存储前进行自动压缩以节省存储空间。

### 数字(Number)
数字属性用于存储整数和浮点数值。

### 二进制(Binary)
二进制属性用于存储任意字节流数据。

### 列表(List)
列表属性用于存储同一种类型的元素集合。列表元素可嵌套其他属性类型。

### 字典(Map)
字典属性用于存储任意数量的名称-值对。字典元素的值可以是任何属性类型，包括嵌套列表和字典。

### 分片键(Partition key)
分片键是 DynamoDB 表的主健，也是全局二级索引的主键。分片键的作用类似于关系型数据库中的主键，并且必须唯一标识每条记录。每个表只能有一个分片键，而且不能修改。

### 排序键(Sort key)
排序键是全局二级索引的范围键，用于定位记录的具体位置。它通常是根据应用需求来定义的，可以用于组合分片键以便在 DynamoDB 中建立多级索引。

## 流(Stream)
DynamoDB 支持发布订阅模式，当 DynamoDB 表发生变更时，会将变更记录发送到指定的流中。

## TTL
TTL 可以用于在指定的时间间隔后过期掉未被更新的记录，减轻了维护数据的成本。

# 3. 核心算法原理及操作步骤

## 创建全局二级索引
全局二级索引的创建非常简单。只需调用 DynamoDB API 的 `CreateIndex()` 方法即可完成全局二级索引的创建。

```python
response = client.create_table(
    TableName='test_table',
    KeySchema=[
        {
            'AttributeName': 'partition_key',
            'KeyType': 'HASH'
        },
        {
            'AttributeName':'sort_key',
            'KeyType': 'RANGE'
        }
    ],
    AttributeDefinitions=[
        {
            'AttributeName': 'partition_key',
            'AttributeType': 'S'
        },
        {
            'AttributeName':'sort_key',
            'AttributeType': 'N'
        }
    ],
    ProvisionedThroughput={
        'ReadCapacityUnits': 10,
        'WriteCapacityUnits': 10
    },
    GlobalSecondaryIndexes=[
        {
            'IndexName': 'test_index',
            'KeySchema': [
                {
                    'AttributeName':'second_partiton_key',
                    'KeyType': 'HASH'
                },
                {
                    'AttributeName':'second_sort_key',
                    'KeyType': 'RANGE'
                }
            ],
            'Projection': {
                'ProjectionType': 'ALL'
            },
            'ProvisionedThroughput': {
                'ReadCapacityUnits': 10,
                'WriteCapacityUnits': 10
            }
        }
    ]
)
```

该请求首先指定表名 `test_table`，设置分片键和范围键，并定义属性的类型。然后再添加全局二级索引信息，指定索引名 `test_index` 和其对应的分片键和范围键，并设定索引的读写能力。

创建全局二级索引之后，表 `test_table` 将具有两个新的属性 `global_secondary_indexes`。如下所示：

```json
"TestTable": {
   "AttributeDefinitions": [
      {
         "AttributeName": "id",
         "AttributeType": "S"
      },
     ...
   ],
   "TableName": "TestTable",
  ...
   "GlobalSecondaryIndexes": [
      {
         "IndexArn": "arn:aws:dynamodb:<region>:<account>:table/TestTable/index/GSITableIndex",
         "IndexName": "GSITableIndex",
         "Projection": {
            "NonKeyAttributes": null,
            "ProjectionType": "ALL"
         },
         "KeySchema": [
            {
               "AttributeName": "SecondPartionKey",
               "KeyType": "HASH"
            },
            {
               "AttributeName": "SecondSortKey",
               "KeyType": "RANGE"
            }
         ],
         "ProvisionedThroughput": {
            "NumberOfDecreasesToday": 0,
            "ReadCapacityUnits": 5,
            "WriteCapacityUnits": 5
         }
      }
   ],
  ...
}
```

## 查询操作

对于全局二级索引，DynamoDB 会根据索引的分片键和范围键来匹配数据。如果需要对全局二级索引的范围内的数据进行查询，可以使用两种方法：

1. 使用Query()方法

   ```python
   response = client.query(
       TableName='test_table',
       IndexName='test_index',
       KeyConditionExpression='#partition_key = :partition_value AND #range_key BETWEEN :start_range AND :end_range',
       ExpressionAttributeNames={'#partition_key': 'partition_key', '#range_key':'sort_key'},
       ExpressionAttributeValues={
           ':partition_value': {'S': 'test_value'},
           ':start_range': {'N': '0'},
           ':end_range': {'N': '10'}
       }
   )
   ```

   此处使用 `#partition_key`、`#range_key` 来引用索引的分片键和范围键，并使用 `:partition_value`、`:start_range`、`end_range` 指定查询条件。

2. 使用Scan()方法

   如果查询条件不涉及索引的范围限制，那么也可以使用扫描的方式来查找数据。

   ```python
   response = client.scan(
       TableName='test_table',
       FilterExpression='#partition_key = :partition_value',
       ExpressionAttributeNames={'#partition_key': 'partition_key'},
       ExpressionAttributeValues={
           ':partition_value': {'S': 'test_value'}
       }
   )
   ```

    此处使用FilterExpression来指定查询条件，其中也引用了分片键。

以上两种查询方式都会把匹配到的结果返回给客户端。

## 更新记录

对全局二级索引中的记录进行更新，可以使用UpdateItem()方法。

```python
response = client.update_item(
    TableName='test_table',
    Key={
        'partition_key': {
            'S': 'test_value'
        },
       'sort_key': {
            'N': str(random.randint(1, 10))
        }
    },
    UpdateExpression='SET field=:field',
    ExpressionAttributeValues={
        ':field': {'Value': value}
    },
    ReturnValues="UPDATED_NEW"
)
```

该请求指定表名 `test_table` 及更新条件。这里的Key指定了需要更新的记录的分片键和范围键，同时还需要用 UpdateExpression 设置更新表达式和参数。此外，还可以用ReturnValues选项指定是否返回更新后的记录。

## 删除记录

与全局二级索引类似，对全局二级索引中的记录进行删除可以使用DeleteItem()方法。

```python
response = client.delete_item(
    TableName='test_table',
    Key={
        'partition_key': {
            'S': 'test_value'
        },
       'sort_key': {
            'N': str(random.randint(1, 10))
        }
    }
)
```

该请求指定表名 `test_table` 及待删除记录的分片键和范围键。

# 4. 代码实例与解释说明
## 插入记录
以下实例演示如何在 DynamoDB 中插入记录：

```python
import boto3
from botocore.exceptions import ClientError

client = boto3.client('dynamodb')

try:
    response = client.put_item(
        TableName='test_table',
        Item={
            'partition_key': {
                'S': 'test_value'
            },
           'sort_key': {
                'N': '1'
            },
            'field1': {
                'S': 'test1'
            },
            'field2': {
                'S': 'test2'
            }
        }
    )

    print("Put item succeeded:")
    print(response['ResponseMetadata']['HTTPStatusCode'])
except ClientError as e:
    if e.response['Error']['Code'] == 'ConditionalCheckFailedException':
        print("The conditional request failed")
    else:
        raise e
```

该实例使用 put_item() 函数向 DynamoDB 表 `test_table` 插入一条记录。这里定义了一个字典 `item`，其中包含分片键 `'partition_key'` 和范围键 `'sort_key'`，以及两个额外字段 `'field1'` 和 `'field2'`。由于该表中没有设置全局二级索引，因此不需要显式指定 GSI 的分片键和范围键。

## 查询记录

以下实例演示如何在 DynamoDB 中查询记录：

```python
import random

def query_items():
    try:
        response = client.query(
            TableName='test_table',
            IndexName='test_index',
            Select='ALL_ATTRIBUTES',
            KeyConditionExpression='#partition_key = :partition_value AND #range_key >= :sort_value',
            ExpressionAttributeNames={
                '#partition_key': 'partition_key',
                '#range_key':'sort_key'
            },
            ExpressionAttributeValues={
                ':partition_value': {
                    'S': 'test_value'
                },
                ':sort_value': {
                    'N': str(random.randint(1, 10))
                }
            }
        )

        for i in range(len(response['Items'])):
            print("Record %d:" % (i+1,))
            record = response['Items'][i]
            print("\t partition key:", record['partition_key'])
            print("\t sort key:", record['sort_key'])
            print("\t field1:", record['field1'])
            print("\t field2:", record['field2'])

    except Exception as e:
        print("Could not query items:", e)


if __name__ == '__main__':
    query_items()
```

该实例使用 query() 函数查询 DynamoDB 表 `test_table` 的全局二级索引 `test_index`。由于索引不是分片键和范围键，所以需要分别指定。这里定义了键条件表达式，指定分片键和范围键的值。由于指定了范围键大于等于某个值，所以查询结果应该包含该范围值的记录。

函数定义了一个循环，打印出查询结果中的每条记录的信息，包括分片键、范围键、字段1和字段2。