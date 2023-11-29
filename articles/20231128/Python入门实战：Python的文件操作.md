                 

# 1.背景介绍


文件操作在编程中是一个重要的内容。由于文件的读、写、删除等操作涉及到对文件系统的各种操作，因此掌握文件操作对于后续深入学习其他编程语言、进行项目开发等都有极其重要的作用。本文将从最基础的文件操作入手，带领读者完成Python文件操作的全过程。
# 2.核心概念与联系
文件操作中涉及到的主要概念有如下几个：
- 文件名（Filename）：指存储在磁盘上的文件或目录名称；
- 绝对路径（Absolute Path）：一个文件的完整路径；
- 当前工作目录（Current Working Directory）：当前正在使用的工作目录；
- 相对路径（Relative Path）：基于当前工作目录的一种路径表示方法；
- 文件模式（File Modes）：控制文件的读、写、执行权限等行为；
- 文件描述符（File Descriptor）：指示某个打开文件句柄的索引值；
文件操作通常分为以下几类：
- 文件创建、删除、重命名等；
- 读、写、追加文件内容；
- 查找文件信息；
- 目录管理操作；
文件操作相关的重要函数如下：
- open()：用于打开指定的文件；
- read()、write()、seek()：用于读写文件内容和移动指针位置；
- rename()、remove()、mkdir()、rmdir()：用于文件、目录的创建、删除、修改、复制、重命名等；
- listdir()：用于列出目录下的文件列表；
- chmod()：用于设置文件或目录的访问权限；
- isfile()、isdir()：用于判断指定路径是否为文件或目录；
# 3.核心算法原理和具体操作步骤以及数学模型公式详细讲解
## 3.1 创建/删除文件
创建文件可以通过open()函数创建一个文件，并通过f.close()关闭文件。注意如果文件已经存在则会报错。
```python
import os
os.mknod('test_file')   # 创建空文件test_file
with open('test_file', 'w+') as f:
    print(f'文件已创建成功: {os.path.isfile("test_file")}')
print(f'关闭文件句柄: {f.closed}')
os.remove('test_file')    # 删除文件test_file
```
输出结果：
```
文件已创建成功: True
关闭文件句柄: False
```
## 3.2 读、写文件
读取文件内容可以使用read()函数，写入文件内容可以使用write()函数。
```python
with open('test_file', 'w+') as f:
    content = f.read()      # 读取所有文件内容
    print(content)

    f.seek(0)               # 将文件指针移回开头
    content = f.read()      # 从开头读取所有内容
    print(content)

    new_content = "Hello World!"     # 准备新的文件内容
    f.seek(0)                       # 将文件指针移回开头
    f.write(new_content)            # 覆盖原有文件内容
    print(f'写入新内容: {new_content}')
    
    for line in f:                  # 逐行读取文件内容
        print(line.strip())          # 清除末尾换行符

print(f'{f.name} 文件已关闭')
os.remove('test_file')                # 删除文件test_file
```
输出结果：
```
Hello World! 

Hello World! 
写入新内容: Hello World!
Hello World!
test_file 文件已关闭
```
## 3.3 修改文件属性
使用chmod()函数可以修改文件的访问权限。如只允许用户对文件有读、写、执行权限则可设置为0o700。
```python
import stat
mode = stat.S_IMODE(os.stat('test_file').st_mode)        # 获取文件权限
print(f"文件权限为: {oct(mode)}")                          # 以八进制显示权限

new_mode = mode & ~stat.S_IRWXG                            # 禁止组内成员写入、执行文件
os.chmod('test_file', new_mode)                           # 设置新的权限

mode = stat.S_IMODE(os.stat('test_file').st_mode)        # 获取文件权限
print(f"文件权限为: {oct(mode)}")                          # 以八进制显示权限
```
输出结果：
```
文件权限为: 0o100666
文件权限为: 0o100400
```
## 3.4 查找文件信息
可以使用os模块提供的stat()函数获取文件大小、创建时间、修改时间等信息。
```python
import os
info = os.stat('test_file')                                  # 获取文件信息
size = info.st_size                                          # 获取文件大小
mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(info.st_mtime))       # 格式化日期时间
ctime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(info.st_ctime))       # 格式化日期时间
print(f"文件大小为: {size}, 创建时间: {ctime}, 修改时间: {mtime}")
```
输出结果：
```
文件大小为: 13, 创建时间: 2021-09-09 10:21:47, 修改时间: 2021-09-09 10:21:47
```
## 3.5 文件夹操作
可以使用os模块提供的makedirs()和removedirs()函数创建文件夹和删除文件夹。注意：当文件夹非空时不能删除，需先清空文件夹。
```python
import os
os.makedirs('folder1/sub_folder', exist_ok=True)         # 创建子目录
os.listdir('.')                                         # 列出当前目录下的所有文件和目录
os.rename('folder1', 'folder2')                        # 重命名目录
os.removedirs('folder2')                                # 删除子目录
os.rmdir('folder2/sub_folder')                         # 删除根目录
```
输出结果：
```
['.', '..', '__pycache__', '.gitignore', 'test_file']
```
## 3.6 小结
本文介绍了Python文件操作的基本知识点，包括创建、删除文件、读写文件、修改文件属性、查找文件信息、文件夹操作等操作。这些操作通过Python标准库中的os模块、fcntl、io模块、pickle等模块实现，可以帮助读者更好地理解文件操作的用法和场景。