                 

# 1.背景介绍


文件（File）是电脑存储信息的重要组成部分之一。对于每个程序员来说，文件操作是一个基本技能，用于读写、创建和删除文件。文件的应用场景非常广泛，例如保存程序配置数据、用户数据等。通过本文学习，你可以掌握Python文件操作的基本知识和技巧，进而更好地理解和运用它。同时，你还将熟悉Python异常处理机制，具备处理各种异常场景的能力。

2.核心概念与联系
## 文件操作常用模块
- os：操作系统相关接口函数。
- sys：系统相关功能函数。
- io：输入输出操作相关接口函数。
- shutil：高级文件操作模块，可以实现文件的复制、移动、删除等操作。
- tempfile：临时文件及目录创建管理工具。
- json：JSON(JavaScript Object Notation)是一种轻量级的数据交换格式，可以方便地进行数据的解析和生成。
- csv：csv(Comma Separated Values，逗号分隔值)文件格式用于保存表格型数据。
- argparse：命令行参数解析模块。
- logging：Python标准日志库，提供程序运行过程中记录日志的功能。

## 文件操作接口概览
- open()：打开一个文件，并返回一个file对象。
- read()：从文件读取数据到字符串。
- write()：向文件写入字符串数据。
- seek()：移动文件读取指针位置。
- tell()：获取当前文件读取指针位置。
- close()：关闭已打开的文件。
- flush()：刷新文件内部缓冲区。
- fileno()：获取文件描述符。
- isatty()：判断是否是一个终端设备。

## 文件操作接口详解
### 文件打开方式
- r: 只读模式，只能读取文件内容，不能修改文件内容。如果文件不存在会抛出 FileNotFoundError 错误。
- w: 覆盖模式，直接替换文件，若文件不存在则自动创建。
- x: 新建模式，只在文件不存在时才允许创建文件，否则报错 FileExistsError 。
- a: 追加模式，从文件末尾追加新的内容。若文件不存在则自动创建。
- +: 更新模式，可读写文件内容。如果文件不存在则自动创建。

```python
f = open('test.txt', 'w') # 以写入模式打开或创建文件
```

```python
try:
    f = open('test.txt', 'r') # 以读取模式打开文件
except IOError as e:
    print("Failed to open file:", e)
else:
    content = f.read()
    print(content)
    f.close()
```

```python
with open('test.txt', 'a') as f: # with语句自动调用close方法
    f.write('Hello World!')
```

```python
import io

buffer = io.StringIO()     # 创建一个BytesIO对象
f = buffer                 # 将BytesIO对象转换为file类型
f.write('Hello ')          
f.write('World!\n')        
print(buffer.getvalue())   # 获取写入的内容
```

```python
import os

os.mkdir('/path/to/dir')    # 在指定路径下创建目录
os.remove('/path/to/file')  # 删除文件
os.listdir('/path/to/dir')  # 返回指定路径下的所有文件和子目录名列表
```

### 文件读取与写入
```python
import codecs

f = open('test.txt', 'rb')      # 以二进制读取模式打开文件
bdata = f.read()                # 读取所有字节数据
sdata = bdata.decode('utf-8')   # 解码为字符串
f.seek(0)                      # 移动读取指针到开头
f.write(b'New data')           # 写入新的字节数据
f.close()                      # 关闭文件

with codecs.open('test.txt', 'w', encoding='gbk') as f:  # 使用codecs模块处理编码
    f.write('中文')                                         # 把中文字符串写入文件
```

### 文件指针
`tell()` 方法获取当前文件指针位置，`seek()` 方法移动文件指针位置到指定的位置。

```python
f = open('test.txt', 'r')
print(f.tell())              # 获取当前文件指针位置
f.seek(0)                    # 移动文件指针到开头
print(f.readline())          # 从开头读取一行文本
f.seek(-2, 2)                # 移动文件指针到倒数第二行开头
print(f.readline())          # 从倒数第二行开始读取
f.seek(len(f.readline()), 1) # 移动文件指针到最后一个字符之后
f.seek(0)                    # 移动文件指针到开头
lines = f.readlines()        # 读取所有文本行，结果为 list 对象
for line in lines:
    print(line, end='')       # 用空串代替换行符打印结果
f.close()                    
```