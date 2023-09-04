
作者：禅与计算机程序设计艺术                    

# 1.简介
  

随着技术的飞速发展、云计算技术的兴起、大数据分析、高性能计算成为主流技术之一，越来越多的人开始使用基于Linux的操作系统。了解Linux系统相关知识并掌握其常用的命令将有助于在日常工作中更加高效地完成工作任务。本文根据常用的Linux命令进行了归纳总结，主要包括用户管理命令、文件目录命令、磁盘管理命令、进程及服务管理命令等，其中各个模块的内容还会继续细分。希望对大家有所帮助。

# 2.概述
Linux是一个开源、免费、UNIX类操作系统，由林纳斯·托瓦兹（<NAME>）和丹尼斯·里奇（Dennis Ritchie）在1991年共同开发，它是一种多用户、多任务、支持多线程和多用户模式的分时操作系统。由于其稳定性、高度可靠性、安全性、开放源码、免费使用和广泛应用，被广泛地用于服务器端、桌面环境、移动终端设备、路由器、网络游戏控制台、高性能计算平台、嵌入式系统等领域。Linux具有“自给自足”的特征，它只需要安装在目标计算机上就可以直接使用，不用依赖于其他什么软件或服务，不需要自己编译、装配，可以说就是非常完美的开源软件。

作为Linux的使用者和开发者，了解各种Linux命令的用法有助于提升技能水平、解决实际问题、提升效率。下面将从多个方面对Linux命令做一个全面的概括。

# 3.用户管理命令

## 用户登录与退出

### login 和 logout 命令

- `login`：该命令用来登录系统，用户可以在登录时指定自己的用户名和密码。

- `logout`：该命令用来注销系统，用户可以使用该命令结束当前登录的会话。

### useradd 命令

- `useradd`：该命令用来创建新的用户账户，该命令可以创建用户账户的同时设置用户的一些属性，如帐号名、有效期限、用户组、加密口令等。

### passwd 命令

- `passwd`：该命令用来修改用户的登录口令，即修改用户的密码。

### su 命令

- `su`：该命令用来切换到root用户，只有root用户才能执行此命令。如果仅仅需要临时切换到某个用户权限运行某些指令，建议使用该命令。

### userdel 命令

- `userdel`：该命令用来删除已经存在的用户账号，并且可以删除对应的所有目录文件。

### groupadd 命令

- `groupadd`：该命令用来创建一个新用户组。

### groupmod 命令

- `groupmod`：该命令用来修改用户组的信息，例如，修改用户组的名称或者 gid 。

### groupdel 命令

- `groupdel`：该命令用来删除一个用户组。

## 用户权限管理命令

### chmod 命令

- `chmod`：该命令用来设定权限，比如读、写、执行权限，用户可以用该命令限制用户对文件的访问权限。

### chown 命令

- `chown`：该命令用来更改文件或者目录的所有者。

### chgrp 命令

- `chgrp`：该命令用来修改文件或目录的用户组。

### su 命令

- `su`：该命令用来切换到root用户，但是不建议频繁使用该命令，因为切换回普通用户之后，无法保存当前工作目录。建议使用su -c command 来切换用户并执行命令。

### sudo 命令

- `sudo`：该命令用来以超级用户(superuser)的身份执行指定的命令。sudo 是 Super User Do 的缩写，也就是超级用户。一般情况下，只有 root 用户才有权限使用 sudo 命令，其他普通用户则无权使用。当普通用户要执行需要权限的命令时，可以通过先用 sudo 执行该命令的方式获得权限。

```bash
# 使用示例：
[myusername@localhost ~]$ ls /root/          # 普通用户尝试查看根目录下的文件，无权查看
ls: cannot open directory '/root/': Permission denied   # 错误提示

[myusername@localhost ~]$ sudo ls /root/    # 用 sudo 获得权限后查看根目录下的文件
bin      etc      lib      lost+found  mnt        sbin     usr
dev      home     media    opt         proc       snap     var
```

以上示例中，普通用户 myusername 试图用 ls 命令查看根目录下的文件，但因权限不足而失败。通过执行 sudo ls /root/ ，使普通用户获得了权限，成功查看根目录下的所有文件。 

### whoami 命令

- `whoami`：该命令用来显示当前登录用户的名字。

### id 命令

- `id`：该命令用来显示当前用户信息，如 uid、gid、groups、用户名等。

### groups 命令

- `groups`：该命令用来显示用户所属的用户组列表。

# 4.文件目录管理命令

## 文件和目录管理命令

### touch 命令

- `touch`：该命令用来创建空白文件，也可以用于更新文件的访问和修改时间戳。

### mkdir 命令

- `mkdir`：该命令用来创建目录，如果目录的上层目录不存在，则会自动创建。

### rmdir 命令

- `rmdir`：该命令用来删除空的子目录。

### rm 命令

- `rm`：该命令用来删除文件或目录，可以一次删除多个文件或目录，并且可以递归删除整个目录树。

### cp 命令

- `cp`：该命令用来复制文件或目录，可以复制整个目录树。

### mv 命令

- `mv`：该命令用来重命名或移动文件或目录，可以移动整个目录树。

### ln 命令

- `ln`：该命令用来创建硬链接或符号链接。

### file 命令

- `file`：该命令用来判断文件类型。

### df 命令

- `df`：该命令用来显示磁盘分区信息。

### du 命令

- `du`：该命令用来显示目录或文件大小。

### tree 命令

- `tree`：该命令用来以树状图列出目录结构。

### find 命令

- `find`：该命令用来搜索文件或目录，并且可以根据条件查找文件或目录。

```bash
# 查找/var目录下所有以".log"结尾的文件
$ find /var -name "*.log"
/var/log/messages.log
/var/log/secure.log
...
```

```bash
# 查找文件大小大于10M的文件
$ find. -size +10M 
./Documents/text.txt
```

```bash
# 查找更改时间在三天前的文件
$ find. -mtime -3
./Documents/Projects/project.doc
./Documents/text.txt
```

```bash
# 查找更改时间在昨天的早上十点之前的文件
$ find. -mmin -720
./Documents/text.txt
```

```bash
# 查找拥有者是root的文件
$ find. -uid 0
/root/.bashrc
```

```bash
# 查找属组是staff的文件
$ find. -gid 50
/home/documents/code/project.py
```

### echo 命令

- `echo`：该命令用来输出字符串，可以实现文字打印、制表符的输出、读取键盘输入等功能。

### head 命令

- `head`：该命令用来显示文件开头部分的内容，默认显示前十行。

### tail 命令

- `tail`：该命令用来显示文件末尾部分的内容，默认显示最后十行。

### cat 命令

- `cat`：该命令用来显示文件内容，可以一次显示多个文件内容。

```bash
# 一次显示多个文件内容
$ cat file1 file2... filen > output_file
```

### less 命令

- `less`：该命令用来分页显示文件内容。

```bash
# 以一页一页的形式浏览文件内容
$ less filename
```

### more 命令

- `more`：该命令也是用来分页显示文件内容，但是比 less 更加灵活。

```bash
# 以一页一页的形式浏览文件内容
$ more filename
```

### vi 命令

- `vi`：该命令用来编辑文件，也可进入程序编辑模式。

```bash
# 进入编辑模式，并编辑第一行
$ vi filename
i           # 进入插入模式
hello world # 在第一行输入文字
ESC         # 退出插入模式
:wq!        # 保存并退出

# 通过命令行编辑文件内容
$ vi filename +number         # 从指定行号处进入编辑模式
$ vi filename +/-offset       # 向上或向下移动一定行数，再进入编辑模式
$ vi filename :s/old/new/g    # 替换文本中的字符串
```

### nano 命令

- `nano`：该命令用来轻量级文本编辑器，功能类似于 vi 命令。

```bash
# 创建并打开一个新文件
$ nano filename

# 删除当前行，并把光标移至下一行
Ctrl+k

# 删除当前行，并把光标移至上一行
Ctrl+u

# 撤销上一步操作
Ctrl+o

# 保存并退出，若文件没有保存过，需按两次Esc键
Ctrl+x
```

### awk 命令

- `awk`：该命令是一种数据分析语言，能够对文本和数据进行处理，是一种强大的文本处理工具。

```bash
# 分隔符为逗号的 CSV 文件统计不同列的平均值
$ cat data.csv | awk -F',' '{ sum += $1; count++ } END { print "Average of first column:", sum/count }'
Average of first column: 5
```

### grep 命令

- `grep`：该命令用来搜索指定字符或模式的文本，并返回匹配的行。

```bash
# 查询 /etc/passwd 中包含 root 的行
$ grep 'root' /etc/passwd
root:x:0:0:root:/root:/bin/bash
daemon:x:1:1:daemon:/usr/sbin:/usr/sbin/nologin
```

### sed 命令

- `sed`：该命令用来流编辑文件，功能类似于 vi 命令。

```bash
# 替换 /etc/passwd 中的所有 root 为 nobody
$ sed -i's/^root/nobody/' /etc/passwd

# 读取 stdin，并替换每行第一个单词为 hello
$ echo "this is a test" | sed's/\b\w\+\b/hello/g'
hello this is a test
```