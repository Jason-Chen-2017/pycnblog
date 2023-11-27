                 

# 1.背景介绍



在Python中，读写文件是一种基础的文件操作方式。由于Python的简洁、易用、广泛应用，使得文件读写变得非常简单。本文将以python语言为例，带领大家快速掌握文件的读写操作。
# 2.核心概念与联系
## 1.1 文件读取与写入模式
- `r`模式（读模式）用于打开只读文件，只能读取其内容，不能修改。如果文件不存在，会报错；
- `w`模式（写模式）用于新建或覆盖已存在的文件，可以对文件进行写入操作。如果文件不存在，则创建新文件；
- `a`模式（追加模式）用于向已存在的文件末尾添加内容，不影响原有内容，如果文件不存在，则创建新文件。

> 以二进制模式读取或写入文件时，需要指定`b`选项，例如：`rb`，`wb`。

## 1.2 文件指针位置
文件读取/写入过程中的位置信息存储在文件指针内部，它记录当前文件读取到的位置。初始化时，文件指针指向文件的开头。可通过调用函数获取或设置文件指针位置。以下是文件指针常用的函数及描述：

 - `tell()` 方法：返回当前文件指针位置。

 - `seek(offset[, whence])` 方法：移动文件指针到指定位置。参数 `offset` 指定偏移量，`whence` 参数决定如何计算出新的位置。默认为 `0`，表示从开头算起，`-1` 表示从当前位置倒退一字节，`-2` 表示从文件末尾倒退一字节。

 - `truncate([size])` 方法：截取文件，删除指针后面的所有内容。参数 `size` 是所保留内容的长度，默认值为当前指针位置。

## 1.3 文件句柄与关闭
每当执行文件读写操作时，都需要创建一个文件对象，即文件句柄。并不是每次访问一个文件都需要创建文件句柄，而是采用延迟加载的方式。只有在真正使用文件句柄的时候才会打开文件，而且使用完毕之后一定要关闭文件句柄释放资源。以下是文件句柄相关的常用函数：

 - `open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None)` 方法：打开指定路径的 file，并且根据不同的 mode 打开对应的文件，创建文件句柄。其他参数含义如下：
   * `buffering`：指定缓冲区大小，单位为字节，`-1` 为系统默认值；
   * `encoding` 和 `errors`：编码相关参数，指定编码格式和处理错误的方式；
   * `newline`：行结束符，不同系统下的换行符可能不同。
 - `close()` 方法：关闭文件句柄，释放系统资源。

## 1.4 文件迭代器
利用 for...in 循环可以方便地遍历文件的所有行。但对于大型文件，一次性读取文件中的所有内容可能会导致内存不足的问题，因此通常采用分批次逐行读取的方法。python 中的文件迭代器是指可以按需读取文件的一种对象。可以使用 `next()` 方法从迭代器中获取下一行内容。文件迭代器一般是由内置函数 `iter()` 来生成的。比如：

```python
with open('filename') as f:
    lines = iter(f) # 获取文件迭代器
    while True:
        try:
            line = next(lines) # 从迭代器中获取下一行
            process_line(line) # 对该行数据进行处理
        except StopIteration: # 当达到文件结尾时抛出 StopIteration 异常
            break
``` 

# 3.核心算法原理和具体操作步骤以及数学模型公式详细讲解
## 1. 创建/打开文件
```python
import os

# 创建空文件
fd = os.open("my_file", os.O_CREAT | os.O_WRONLY)

# 打开文件并写入内容
fd = os.open("my_file", os.O_RDWR | os.O_APPEND)
os.write(fd, b"Hello, world!\n")
os.fsync(fd)

# 关闭文件句柄
os.close(fd)

# 直接使用 with 语句打开文件，自动调用 close() 操作
with open("my_file", "r+") as f:
    pass
``` 
## 2. 读/写文件内容
```python
# 读取整个文件内容
content = ""
with open("my_file", "r") as f:
    content = f.read()
    
print(content) # Hello, world!

# 将字符串写入文件
s = "This is a test."
with open("test.txt", "w") as f:
    f.write(s)
    
# 逐行读取文件内容
for line in open("test.txt"):
    print(line.strip()) # This is a test.
``` 

## 3. 文件指针位置
```python
# 获取文件指针位置
pos = os.lseek(fd, 0, os.SEEK_CUR)

# 设置文件指针位置
os.lseek(fd, offset, os.SEEK_SET)

# 设置文件指针位置，相对于当前位置
os.lseek(fd, offset, os.SEEK_CUR)

# 设置文件指针位置，相对于文件末尾
os.lseek(fd, offset, os.SEEK_END)
``` 

## 4. 文件系统统计信息
```python
# 获取文件大小
statinfo = os.stat("my_file")
filesize = statinfo.st_size

# 获取文件创建时间
ctime = statinfo.st_ctime

# 获取文件最后访问时间
atime = statinfo.st_atime

# 获取文件最后修改时间
mtime = statinfo.st_mtime

# 检查文件是否为目录
is_dir = statinfo.S_ISDIR(statinfo.st_mode)
``` 

## 5. 删除/重命名文件
```python
# 删除文件
os.remove("my_file")

# 重命名文件
os.rename("oldname.txt", "newname.txt")
``` 

# 4.具体代码实例和详细解释说明
## 1. 使用csv模块读写CSV文件
### CSV (Comma Separated Values，逗号分隔值) 文件
CSV 文件是一种纯文本文件，其中内容用逗号分割，并无特殊语法结构。每个单元格均为一列，各列之间用制表符或空格分隔。例如：

```
"name","age","gender"
"Alice",25,"F"
"Bob",30,"M"
"Charlie",35,"M"
``` 

在 CSV 中，第一行为列名（header），列名不可重复；第二行及后续行均为数据行。

### csv 模块
Python 提供了一个叫做 csv 的模块，它提供了一些函数用来读写 CSV 文件。csv 模块可以处理如下格式的数据：

- str 或 bytes 对象
- list of tuples （元组列表）
- dict of lists （字典列表）

使用 csv 模块读写 CSV 文件时，需要先创建一个 csv Writer 对象或 Reader 对象，然后调用对象的 read() / write() 方法。

### 读取 CSV 文件
使用 csv 模块读取 CSV 文件时，可以先创建一个 csv reader 对象，然后调用它的 rows() 方法来获得所有数据行。示例如下：

```python
import csv

with open('data.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader) # 跳过第一行列名
    
    for row in reader:
        name, age, gender = row
        print("{} is {} years old and his gender is {}".format(name, age, gender))
``` 

输出结果：

```
Alice is 25 years old and his gender is F
Bob is 30 years old and his gender is M
Charlie is 35 years old and his gender is M
```

除了上面这种方法外，也可以直接用 csv 文件对象的 dialect 属性来判断 CSV 文件的类型，进而调用相应的解析方法。示例如下：

```python
import csv

with open('data.csv', 'r') as f:
    dialect = csv.Sniffer().sniff(f.readline(), delimiters=[',', '\t'])
    if dialect.delimiter == ',':
        reader = csv.DictReader(f, fieldnames=['name', 'age', 'gender'], restkey='extra columns')
    elif dialect.delimiter == '\t':
        reader = csv.reader(f, delimiter='\t', quotechar='"')
        
    for row in reader:
        if isinstance(row, dict):
            print("{} is {} years old and his gender is {}".format(row['name'], row['age'], row['gender']))
        else:
            extra_columns = row[-len(dialect.lineterminator)-1:-1]
            data = {k: v for k, v in zip(['name', 'age', 'gender'], row[:3])}
            data['extra columns'] = ''.join(extra_columns)
            print(data)
``` 

输出结果：

```
{'name': 'Alice', 'age': '25', 'gender': 'F'}
{'name': 'Bob', 'age': '30', 'gender': 'M'}
{'name': 'Charlie', 'age': '35', 'gender': 'M', 'extra columns': ''}
```

上述方法对不同类型的 CSV 文件提供了不同的解析方案。除此之外，还可以通过设置 skipinitialspace 参数来忽略掉前导空白字符，并设置 doublequote 参数来允许双引号作为字段分隔符。完整的配置参数如下：

```python
import csv

config = {'skipinitialspace': False, 'doublequote': True}
with open('data.csv', **config) as f:
   ...
``` 

更多细节可参考官方文档。

## 2. 使用json模块读写JSON文件
### JSON (JavaScript Object Notation) 文件
JSON 是一种轻量级的数据交换格式，它基于 ECMAScript 4th Edition 中的一个子集。JSON 支持简单的数据类型如字符串、数字、布尔值、数组、对象等。它还有两个主要的特性：

- 可传输：JSON 数据格式是独立于语言的，并且使用 UTF-8 编码，因此非常容易被人类和机器处理。
- 自描述：JSON 数据可以在两种形式之间转换，即从复杂的对象表示法到更紧凑的编码，或者从紧凑的编码到较易读的文本格式。

JSON 文件的内容可以直接映射为 Python 数据类型，比如字符串、数字、布尔值、列表、字典。例如：

```json
{
  "name": "Alice",
  "age": 25,
  "city": "New York",
  "hobbies": ["reading", "travel"]
}
``` 

### json 模块
Python 提供了叫做 json 的模块，它提供了一些函数用来读写 JSON 文件。json 模块支持四种主要的序列化格式：

- str
- bytes
- dict
- iterable of pairs

使用 json 模块读写 JSON 文件时，首先需要将 JSON 数据转换成字典形式，再调用 dumps() 函数序列化成字符串。反之，将字符串反序列化成字典后，再调用 loads() 函数恢复原始数据。示例如下：

```python
import json

# 序列化字典
d = {"name": "Alice", "age": 25, "city": "New York"}
json_str = json.dumps(d)
print(json_str) # {"name": "Alice", "age": 25, "city": "New York"}

# 反序列化字典
json_dict = '{"name": "Alice", "age": 25}'
obj = json.loads(json_dict)
print(type(obj), obj) # <class 'dict'> {'name': 'Alice', 'age': 25}
```

在实际业务场景中，往往需要同时操作多个 JSON 文件，比如合并、过滤、校验、转换等。这时候就需要考虑使用 load() / loads() 函数的多个参数，让它们处理多个 JSON 文件。