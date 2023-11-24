                 

# 1.背景介绍


Python作为一门高级语言，具有强大的可扩展性、模块化特性、丰富的标准库、易于学习等优点。作为一种高级语言，它的“脚本语言”的特点也越来越重要。编写脚本程序有很多好处，例如可以快速完成一些重复性任务，节省时间；还能在不同平台上执行，具有更好的适应性。此外，由于Python自身具有较高的运行效率，使得其脚本语言成为IT领域中不可或缺的一部分。因此，掌握Python编程基础知识对于掌握Python语言的应用和技巧至关重要。

在本教程中，主要讲解文件的读写、目录的遍历和异常处理相关知识。

# 2.核心概念与联系
## 文件操作
文件操作是指对文件进行读取、写入、修改、删除等操作。熟练掌握文件操作的基本命令和流程，可以让我们更轻松地处理各种数据源的文件数据，提升工作效率。以下是主要涉及的文件操作的命令：
### 打开文件(open)
使用`open()`函数可以打开一个文件。该函数返回一个文件对象，可以用于后续对文件的操作。语法如下：
```python
file = open("filename", "mode")
```
参数`filename`表示要打开的文件名，`mode`表示文件访问模式（可选）。访问模式有三种类型："r" 表示以只读方式打开文件，即不能对文件做出任何更改；"w" 表示以可写方式打开文件，如果文件不存在则创建新文件；"a" 表示以追加模式打开文件，如果文件不存在则创建一个空文件。
```python
f = open("test.txt", "w+") # 以可读写的方式打开或创建文件test.txt
```
也可以通过`with`语句自动关闭文件，减少异常的发生：
```python
with open("test.txt", "w+") as f:
    f.write("Hello World!")
    print(f.read())
```
### 读取文件(read)
使用`read()`方法可以从文件中读取所有内容并作为字符串返回。若指定了长度参数，则返回指定长度的字节串。
```python
content = f.read()     # 返回文件的所有内容
line_content = f.readline()    # 从当前位置读取一行内容
nbytes = f.readinto(b)   # 将文件的内容读入到缓冲区
```
### 写入文件(write)
使用`write()`方法可以向文件中写入内容，参数可以是字符串或字节串。
```python
result = f.write('hello world')      # 写入字符串
offset = f.seek(0, io.SEEK_END)      # 设置文件指针位置
f.truncate()                        # 清除文件末尾无用内容
```
### 关闭文件(close)
使用`close()`方法可以关闭已打开的文件。关闭文件后将无法再进行读写操作，且可能导致文件被占用而失败。
```python
f.close()
```
## 目录操作
在Python中，可以使用内置的`os`模块对目录进行操作。以下是主要涉及的目录操作命令：
### 查看当前工作目录(getcwd)
使用`getcwd()`方法可以获得当前工作目录的路径。
```python
import os
cwd = os.getcwd()         # 获取当前工作目录
print(cwd)                # 打印当前工作目录路径
```
### 创建目录(mkdir)
使用`mkdir()`方法可以创建新的目录。
```python
os.mkdir('/path/to/newdir')       # 在指定的目录下创建新的目录
```
### 删除目录(rmdir)
使用`rmdir()`方法可以删除一个空目录。
```python
os.rmdir('/path/to/emptydir')      # 删除空目录
```
### 列出目录内容(listdir)
使用`listdir()`方法可以获取目录中的文件列表。
```python
files = os.listdir('.')           # 获取当前目录下的所有文件名称列表
for file in files:
    print(file)                    # 逐个打印文件名称
```
### 修改目录(chdir)
使用`chdir()`方法可以切换当前目录。
```python
os.chdir('/path/to/otherdir')     # 切换到其他目录
```
## 异常处理
当程序运行过程中出现错误时，会抛出异常。为了防止程序崩溃或者因错误影响正常的执行，需要对异常进行处理。Python提供了`try...except...finally`结构来处理异常。
```python
try:
    x = int(input("请输入数字:"))        # 输入数字
    y = 1 / x                           # 求倒数
except ZeroDivisionError:              # 当x为零时，触发异常
    print("分母不能为零！")
else:                                  # 如果没有触发异常，执行这个块
    print("结果为:", y)
finally:                               # 不管是否有异常都会执行这个块
    print("程序结束。")
```

除了捕获异常之外，还有以下几种方法处理异常：
* 使用`assert`关键字进行断言
* `logging`模块记录日志信息
* 使用`try...raise`语句主动引发异常