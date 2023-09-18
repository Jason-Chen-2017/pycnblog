
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Linux命令是一个经久不衰的开源项目，它的命令行界面给用户提供了很多方便的系统管理工具，但是初学者在使用命令的时候常常会遇到一些问题，比如记不住命令，或者遇到了一些难以理解的问题。所以对于刚入门的人来说，了解 Linux 命令的用法是非常重要的。本专栏将从基础命令开始，教你如何高效地掌握 Linux 命令。

22. Linux命令大全——必知必会系列(续)
# 2.背景介绍
由于众所周知的原因，有些同学学习Linux的时间比较长，但是很多新手并没有真正掌握Linux命令的精髓。甚至有的学生只是将Linux作为一个黑盒使用，很少涉及到其内部的运行机制。但是随着Linux的普及以及开源社区的活跃，越来越多的人都关注并应用了Linux，因此了解Linux命令的用法对于无论是学校还是工作都很有必要。如果你对Linux已经有了一定的认识，那么本系列的文章就是为你量身定制。文章的内容主要围绕Linux的基本命令进行讲解，帮助你高效地掌握Linux命令，具备最基本的Linux使用能力。

3.基本概念术语说明
## 3.1. 文件与目录
文件（File）是存储在磁盘上的信息，可以是文本、图像、视频或任何二进制数据等等。而目录（Directory）是用来存储文件的逻辑集合，它类似于文件夹一样，可容纳许多文件和子目录。在 Linux 中，绝对路径和相对路径是两种最常用的路径表示方法。绝对路径指的是从根目录/开始写起的完整路径；相对路径则相对于当前所在位置的路径。

例如：

绝对路径: /home/user/documents/file.txt

相对路径: documents/file.txt 当前所在位置是/home/user，那么相对路径就是 documents/file.txt 。

## 3.2. 命令与参数
命令（Command）是在终端窗口中输入的文字，用于执行特定功能的指令。在 Linux 中，命令的名称和相关的参数（如果有的话）之间用空格隔开。例如：

命令 ls -l /usr/bin

## 3.3. 管道符
管道符（Pipe symbol）是一条连接两个命令的符号。它的作用是使一个命令的输出变成另一个命令的输入。使用管道符时，需要保证前一个命令的所有输出都能够被后一个命令读取到。

例如：

ls -al | grep.sh

上面的命令表示：列出当前目录所有文件信息并把文件名包含“.sh”的文件打印出来。

## 3.4. 重定向符
重定向符（Redirection symbol）用来控制命令的输入和输出。在 Unix/Linux 中，有三种类型的重定向符：

- 标准输入符 (STDIN): 表示从键盘输入。
- 标准输出符 (STDOUT): 表示输出到显示器。
- 标准错误符 (STDERR): 表示输出到错误日志文件。

通过重定向符，你可以将一个命令的输出写入文件或屏幕，也可以从文件读取命令的输入，甚至把某些命令的输出作为另一个命令的输入。

例如：

cat file1.txt > file2.txt

上面命令的意思是：将 file1.txt 的内容复制到 file2.txt 中。

## 3.5. shell脚本
shell脚本（Shell Script）是一个用来自动化执行任务的脚本语言。它可以通过指定各种条件和操作，自动地完成某项复杂的任务。shell脚本通常包含若干命令语句和运算符，它们构成了一个完整的操作流程。

一般来说，Unix/Linux系统上默认使用的shell是Bash，但还有其他的shell如zsh、tcsh等。虽然这些shell有着不同的语法规则和命令集，但都遵循通用的脚本语言。

4.核心算法原理和具体操作步骤以及数学公式讲解
## 4.1. echo命令
echo命令用于打印字符串。

**语法：**

echo [选项]... [字符串]...

**选项：**

- -e 开启反斜杠扩展。
- -n 不输出换行符。

**示例:**

```bash
$ echo "Hello World"         # 输出字符串 "Hello World"
$ echo $PATH                 # 输出环境变量 PATH 中的值
$ echo Hello\nWorld          # 使用 \n 在两行输出 Hello 和 World
$ echo -e "\033[31mHello\033[0m"   # 使用 ANSI 颜色输出红色的 Hello
```

## 4.2. cd命令
cd命令用于切换当前工作目录。

**语法：**

cd [-L|[-P [-e]]] [目录名]

**选项：**

- -L：强制转换为符号链接。
- -P：使用源目录中的物理目录名。
- -e：若目标目录不存在，则创建之。

**示例：**

```bash
$ cd /                   # 进入根目录
$ cd ~                   # 进入用户主目录
$ cd..                  # 返回上级目录
$ cd./folder            # 进入当前目录下的 folder 目录
$ cd ~/Documents/folder  # 进入 Documents 下的 folder 目录
```

## 4.3. pwd命令
pwd命令用于显示当前工作目录的绝对路径。

**语法：**

pwd

**示例：**

```bash
$ pwd                     # 显示当前工作目录的绝对路径
/home/user/Documents/folder
```

## 4.4. mkdir命令
mkdir命令用于创建一个新的目录。

**语法：**

mkdir [-p] [目录]...

**选项：**

- -p：递归创建目录，即创建父目录。

**示例：**

```bash
$ mkdir new_directory     # 创建目录 new_directory
$ mkdir dir1 dir2        # 创建目录 dir1 和 dir2
$ mkdir -p subdir/subsubdir    # 创建父子目录结构
```

## 4.5. rm命令
rm命令用于删除文件或目录。

**语法：**

rm [-rfiI] 文件或目录...

**选项：**

- -r 递归删除目录，删除非空目录时必须加此参数。
- -f 强制删除，忽略不存在的文件，不会提示确认。
- -i 删除前逐一询问确认。
- -I 将删除工作推迟到最后，直到下一次登录时才执行。

**示例：**

```bash
$ rm test.txt           # 删除文件 test.txt
$ rm -r directory       # 递归删除目录 directory
$ rm -if file1 file2    # 强制删除 file1 和 file2，删除前逐一询问确认
```

## 4.6. mv命令
mv命令用于移动或重命名文件或目录。

**语法：**

mv [-fiu] 来源 源... 目标

**选项：**

- -f 强制覆盖已存在的目标文件，不询问。
- -i （交互模式）若目标文件（夹）存在时，询问是否覆盖，回答 y 或 n。
- -u （软链接）若来源为符号链接文件，移动或重命名文件同时保持符号链接特性。

**示例：**

```bash
$ mv test.txt test1.txt    # 重命名文件 test.txt 为 test1.txt
$ mv file1 file2 dir       # 把 file1 移动到 dir 目录下并改名为 file2
$ mv dir1/* dir2          # 将 dir1 目录内的所有内容移动到 dir2 目录下
$ mv -f file1 file2        # 强制覆盖已存在的 file2 文件
$ mv -i file*             # 交互式删除 file* 文件
```

## 4.7. cp命令
cp命令用于复制文件或目录。

**语法：**

cp [-fiuvp] 来源 源... 目标

**选项：**

- -f 强制复制，若目标文件（夹）存在时，不询问。
- -i （交互模式）若目标文件（夹）存在时，询问是否覆盖，回答 y 或 n。
- -u （软链接）若来源为符号链接文件，复制文件同时保持符号链接特性。
- -v （详细模式）显示复制过程。
- -p （保留属性）复制文件时保持文件属性。

**示例：**

```bash
$ cp test.txt test1.txt    # 拷贝文件 test.txt 到文件 test1.txt
$ cp -r dir1 dir2          # 递归拷贝目录 dir1 到 dir2 目录下
$ cp -fv file1 file2       # 复制文件 file1 到 file2 ，显示详细信息，强制覆盖
```

## 4.8. touch命令
touch命令用于更新文件的访问和修改时间。

**语法：**

touch [-acm] 文件

**选项：**

- -a 修改或设定文件或者目录的最后访问时间。
- -c 只更改文件或目录的时间，如果该文件或目录不存在，不报错。
- -m 修改或设定文件或者目录的最后修改时间。

**示例：**

```bash
$ touch file.txt          # 更新文件 file.txt 的最后修改时间
$ touch -am file.txt      # 更新文件 file.txt 的最后访问和修改时间
```

## 4.9. cat命令
cat命令用于查看文件内容。

**语法：**

cat [选项]... [文件]...

**选项：**

- -b 以字节方式查看文件内容。
- -n 显示行号。
- -s 显示文件空白行。

**示例：**

```bash
$ cat file.txt            # 查看文件 file.txt 的内容
$ cat -bn file.txt        # 以字节形式查看文件 file.txt 的内容，显示行号
```

## 4.10. more命令
more命令用于分页查看文件内容。

**语法：**

more [选项] 文件

**选项：**

- +N 从指定行开始显示内容。
- -c 显示单字符，而不是缓冲区满后换页。
- -d 分隔每一行显示指定字符，默认是换行符。
- -n 显示行号。
- -p 提示字符串。
- -s 禁止连续空行为一行。

**示例：**

```bash
$ more file.txt           # 分页显示文件 file.txt 的内容
$ more -n +5 file.txt     # 从第 5 行开始显示文件 file.txt 的内容，显示行号
```

## 4.11. less命令
less命令也是分页查看文件内容，与more命令不同的是，less可以在文件翻页时快速搜索关键词。

**语法：**

less [选项] 文件

**选项：**

- -c 清除 screen 环境下产生的滚动。
- -F 当文件结束时，自动退出。
- -K 启动“静默”模式。
- -M 用小型 “MORE” 模式，开头几行和结尾几行截断显示。
- -N 显示行号。
- -Q 不显示常规的提示字符。
- -S 设置页面大小。
- -V 显示版本信息。
- -X 启动“EXTEND” 模式，标记、标尺和快捷键可用。
- +/pattern 跳转到第一个匹配的行处。
- /pattern 搜索关键字。
-?pattern 搜索关键字，向下搜寻。
- n 显示匹配到的下一行内容。
- N 显示匹配到的上一行内容。

**示例：**

```bash
$ less file.txt           # 分页显示文件 file.txt 的内容
$ less +5 file.txt        # 从第 5 行开始显示文件 file.txt 的内容，显示行号
```

## 4.12. head命令
head命令用于显示文件的开头部分内容。

**语法：**

head [选项]... [文件]...

**选项：**

- -c <字节> 从指定字节位置开始显示内容。
- -n <行数> 指定显示的行数。
- -q 不显示处理信息。
- -v 对非空文件，显示每个发现的匹配行。

**示例：**

```bash
$ head file.txt           # 显示文件 file.txt 的开头内容
$ head -c 10 file.txt     # 从第 10 个字节开始显示文件 file.txt 的内容
$ head -n 3 file.txt      # 显示文件 file.txt 的前 3 行内容
```

## 4.13. tail命令
tail命令用于显示文件末尾部分内容。

**语法：**

tail [选项]... [文件]...

**选项：**

- -c <字节> 从指定字节位置开始显示内容。
- -f 循环读取文件，等待追加更多内容。
- -n <行数> 指定显示的行数。
- -q 不显示处理信息。
- -v 对非空文件，显示每个发现的匹配行。

**示例：**

```bash
$ tail file.txt           # 显示文件 file.txt 的末尾内容
$ tail -f log.txt         # 持续监控文件 log.txt 的末尾，追加内容显示
$ tail -n 3 file.txt      # 显示文件 file.txt 的末尾 3 行内容
```

## 4.14. find命令
find命令用于搜索文件或目录。

**语法：**

find [搜索范围] [匹配条件]

**示例：**

```bash
$ find /etc -name hosts    # 在目录 /etc 下查找文件名为 hosts 的文件
$ find /var -size +1G     # 在目录 /var 下查找大小超过 1GB 的文件
$ find / -perm 0777       # 在整个文件系统查找权限为 777 的文件
```

## 4.15. locate命令
locate命令是 find 命令的增强版，它利用索引数据库快速定位文件。

**语法：**

locate [选项]... {搜索词}...

**选项：**

- -b<区分大小写> 不区分大小写查找。
- -e<范畴文件> 使用指定的范畴文件。
- -i<范畴压缩文件> 处理具有扩展名的压缩包中的文件。
- -l<单词列表文件> 使用指定文件定义的单词列表。
- -o<输出文件> 将结果保存到文件中。
- -S<区分文件类型> 不搜索二进制文件。
- -u<更新数据库> 更新 locate 数据库。

**示例：**

```bash
$ locate update           # 搜索名为 update 的文件
$ locate *.log            # 搜索当前目录下的所有以.log 结尾的文件
$ locate logs -i -o result # 输出搜索结果到文件 result
```

## 4.16. which命令
which命令用于查找并显示可执行文件的位置。

**语法：**

which [命令]...

**示例：**

```bash
$ which ls               # 查找 ls 命令的位置
/bin/ls
$ which shred            # 查找 shred 命令的位置，如果有多个相同命令，只显示第一个找到的命令位置
/usr/games/shred
```

## 4.17. whereis命令
whereis命令是另一种查找可执行文件位置的方式。

**语法：**

whereis [命令|文件]...

**示例：**

```bash
$ whereis ls             # 查找 ls 命令的位置
ls: /bin/ls /usr/bin/ls /sbin/ls /usr/local/bin/ls /usr/games/ls
$ whereis shred          # 查找 shred 命令的位置
shred: /usr/bin/shred /usr/games/shred
```

## 4.18. chmod命令
chmod命令用于修改文件或目录的权限。

**语法：**

chmod [-cfrvR] [--help] 操作目标...

**选项：**

- -c 仅更改文件权限，不显示错误信息。
- -f 如果文件不存在，不会出错。
- -R 递归更改文件权限。
- -v 显示更改权限的详细信息。
- --reference=<参考文件或目录> 使用参考文件或目录的权限设置。

**操作符：**

- u 用户
- g 组
- o 其它用户
- a 所有用户
- r 可读
- w 可写
- x 执行

**示例：**

```bash
$ chmod 755 file.txt      # 将文件 file.txt 的权限设置为 755 ，所有者具有读、写、执行权限，组用户和其它用户只有读、执行权限
$ chmod u=rwx,g=rx,o=r file.txt    # 等价于 chmod 754 file.txt ，即 owner 可以读写执行， group 可以读、执行，others 可以读。
$ chmod og= file.txt       # 将文件 file.txt 的权限设置成 group 和 others 有权利的状态。
$ chmod +x file.sh         # 添加执行权限到文件 file.sh
$ chmod -w-wx file.txt     # 删除除了用户写外的所有权限。
```

## 4.19. chown命令
chown命令用于修改文件或目录的拥有者和组。

**语法：**

chown [-R] 属主名称 文件...

**选项：**

- -R 递归更改文件和目录所有权。

**示例：**

```bash
$ chown user1 file.txt     # 修改文件 file.txt 的所有者为 user1
$ chown user2:group2 file1 file2    # 修改文件 file1 和 file2 的所有者为 user2 ，群组为 group2
$ chown :group3 directory/    # 将 directory/ 的群组修改为 group3
$ chown -R user4:group4 directory/  # 递归修改 directory/ 下的所有文件的拥有者和群组
```

## 4.20. su命令
su命令用于切换用户身份。

**语法：**

su [选项] 新登陆用户

**选项：**

- -c <命令> 切换用户后执行命令。
- -l 锁定当前用户密码，需要输入密码才能切换到 root 用户。
- -m 保留环境变量。
- -p 切换到原始的 tty 上。
- -s <shell> 指定登录后的 shell。

**示例：**

```bash
$ su                    # 切换到 root 用户
# exit                  # 退出 root 用户
$ sudo su               # 使用 root 用户的身份切换到其他用户，需输入密码
Password: 
root@linux:~# whoami
user1
```

## 4.21. gzip命令
gzip命令用于压缩文件。

**语法：**

gzip [选项]... [文件]...

**选项：**

- -c 将压缩的数据输出到标准输出。
- -d 从标准输入中解压。
- -f 压缩或解压缩的文件名，支持通配符。
- -l 压缩级别。
- -n 不检查输入文件。
- -t 检查压缩文件是否正确。
- -v 显示压缩/解压缩过程。

**示例：**

```bash
$ gzip test.txt          # 压缩文件 test.txt
$ gzip -d test.gz         # 解压文件 test.gz
$ gzip -cv test.txt       # 压缩文件 test.txt，并显示压缩过程
```

## 4.22. bzip2命令
bzip2命令用于压缩文件。

**语法：**

bzip2 [选项]... [文件]...

**选项：**

- -c 将压缩的数据输出到标准输出。
- -d 从标准输入中解压。
- -f 压缩或解压缩的文件名，支持通配符。
- -k 保留源文件。
- -v 显示压缩/解压缩过程。
- -z 支持gzip兼容性。

**示例：**

```bash
$ bzip2 test.txt         # 压缩文件 test.txt
$ bzip2 -d test.bz2       # 解压文件 test.bz2
$ bzip2 -zv test.txt      # 测试压缩文件 test.txt 是否正确
```

## 4.23. tar命令
tar命令用于打包和压缩文件。

**语法：**

tar [选项]... [文件]...

**选项：**

- -A 创建新的归档文件，同时增加至归档文件末尾。
- -c 创建新的归档文件。
- -d 从归档文件中释放文件。
- -f <文件> 指定归档文件名。
- -j 解压文件时，将文件解压缩成 bzip2 格式。
- -r 添加文件到归档文件中。
- -t 显示归档文件的内容。
- -z 解压文件时，将文件解压缩成 gzip 格式。
- -v 显示操作过程。

**操作符：**

- c：创建一个新压缩包或解压包。
- d：提取文件或目录至指定目录。
- f：指定压缩包的文件名。
- v：显示详细信息。
- z：调用相应的压缩程序对文件进行压缩或解压缩。
- x：从压缩包中提取文件。

**示例：**

```bash
$ tar cvzf backup.tar.gz /home/user/.bashrc    # 创建压缩文件 backup.tar.gz ，其中包括 /home/user/.bashrc 文件
$ tar xvf backup.tar.gz                         # 解压文件 backup.tar.gz ，并释放到当前目录
$ tar tzvf backup.tar.gz                        # 显示 backup.tar.gz 的内容
$ tar cf home.tar /home                          # 将 /home 目录打包成文件 home.tar
$ tar xf home.tar                              # 将文件 home.tar 解压缩至 /home 目录
```

## 4.24. diff命令
diff命令用于比较两个文件的内容差异。

**语法：**

diff [选项] 文件1 文件2

**选项：**

- -a 输出全部差异的行，包括那些只有在第一个文件出现，第二个文件却没有的行。
- -b 比较的是空白行。
- -c 计算行更改次数。
- -q 安静模式，只显示是否有差异，不显示具体差异的行。
- -u 输出行更改情况，以 unified diff 格式。

**示例：**

```bash
$ diff file1.txt file2.txt         # 比较两个文件 file1.txt 和 file2.txt 的内容差异
$ diff -cbq file1.txt file2.txt    # 以块的方式比较两个文件 file1.txt 和 file2.txt 的内容差异
```

## 4.25. patch命令
patch命令用于合并补丁。

**语法：**

patch [选项]... <打补丁的压缩包> [<打补丁的位置>]

**选项：**

- -d <工作目录> 指定工作目录。
- -i <打补丁的位置> 指定打补丁的位置。
- -p <数字> 指定打补丁时的 strip 层数。
- -s <签名文件> 指定签名文件。
- -v 详细显示处理过程。

**示例：**

```bash
$ patch -p1 < xxx.patch     # 将补丁 xxx.patch 应用到当前目录
$ patch -p0 -d src/kernel -i kernel.patch     # 将补丁 kernel.patch 应用到目录 src/kernel
$ patch -p1 < xxx.tgz      # 将补丁 xxx.tgz 应用到当前目录
```

## 4.26. sort命令
sort命令用于对文件按行排序。

**语法：**

sort [选项]... [文件]...

**选项：**

- -b 使用归并排序，对实数值有效。
- -h 忽略大小写。
- -i 使用内置的字符排序算法。
- -M 将月份表示为 1~12 的整数。
- -k <start>[,<end>] 根据相应字段排序，指定 start 和 end 为要排序的字段，范围由 start 到 end。
- -n 对数值进行排序。
- -r 对记录进行逆序排列。
- -t <分隔符> 指定分隔符。
- -u 排序时，对于重复的行，仅输出一次。
- -z 针对每一行以 NUL 结尾的文件进行排序。

**示例：**

```bash
$ sort file.txt          # 按行对文件 file.txt 排序
$ sort -nr file.txt      # 倒序排列文件 file.txt 中的数值
```

## 4.27. uniq命令
uniq命令用于去掉文件中重复行。

**语法：**

uniq [选项]... [文件]...

**选项：**

- -c 计数，在每行的前面显示该行重复出现的次数。
- -d 只显示重复的行。
- -D 输出重复行的内容。
- -i 在比较时忽略大小写。
- -u 只显示唯一的行。

**示例：**

```bash
$ uniq file.txt          # 去掉文件 file.txt 中重复行
$ uniq -c file.txt       # 显示文件 file.txt 中每行重复出现的次数
```

## 4.28. cut命令
cut命令用于删除文件中的部分行，或者提取文件中的部分行。

**语法：**

cut [选项]... [文件]...

**选项：**

- -b <字节> 指定字节位置切割。
- -d <分隔符> 指定分隔符。
- -f <域> 指定域，域以“,”分割。
- -n 在切割后的行上添加编号。
- -c <字符> 指定字符位置切割。

**示例：**

```bash
$ cut -d''-f2 file.txt   # 显示文件 file.txt 每行的第二个域
$ cut -d''-f2-4 file.txt # 显示文件 file.txt 每行的第二到第四个域
$ cut -b2-5 file.txt       # 显示文件 file.txt 的第三到第六个字节
```

## 4.29. paste命令
paste命令用于合并相邻的行。

**语法：**

paste [选项]... [文件]...

**选项：**

- -d <分隔符> 指定分隔符。
- -s 合并相邻的空行。
- -t 将剪贴板内容粘贴到每一行之后。

**示例：**

```bash
$ paste file1.txt file2.txt         # 合并文件 file1.txt 和 file2.txt 的内容
$ paste -d',' file1.txt file2.txt    # 以逗号分割各字段，合并文件 file1.txt 和 file2.txt 的内容
```

## 4.30. join命令
join命令用于合并两个文件基于某个字段的内容。

**语法：**

join [选项]... FILE1 FILE2

**选项：**

- -a <数字> 将所有行输出，包括未配对的行。
- -e <字符串> 指定填充缺失值时使用的值。
- -i <字符串> 指定使用哪个值来做初始联接。
- -o <字符串> 指定要显示的字段。
- -1 <数字> 指定文件1中每一行的字段数目。
- -2 <数字> 指定文件2中每一行的字段数目。
- -t <字符串> 指定分隔符。

**示例：**

```bash
$ join file1.txt file2.txt     # 基于第一列的值进行关联
$ join -t',' -1 2 -2 1 file1.csv file2.tsv    # 基于第二列的值进行关联
```

## 4.31. awk命令
awk命令用于处理文本文件。

**语法：**

awk [选项] -f progfile [--] file..

**选项：**

- -F <分隔符> 指定输入文件使用的分隔符。
- -v var=value 设置环境变量。

**示例：**

```bash
$ awk '{print NR,$0}' file.txt    # 显示文件 file.txt 中每行的行号和内容
$ awk '/pattern/{cmd}' file.txt   # 在文件 file.txt 中搜索 pattern 并执行 cmd 命令
```

## 4.32. ssh命令
ssh命令用于远程登陆。

**语法：**

ssh [选项]... [-o option]... [-E lnk] hostname [command]

**选项：**

- -a 启用pubkey认证。
- -A 允许从本地机器和remote机器发送连接请求。
- -B 允许连接到指定主机的端口转发。
- -C 压缩传输数据。
- -D <动态端口> 请求动态端口forwarding。
- -e < escape char> 设置转义字符。
- -F configfile 指定配置文件。
- -i identityfile 指定私钥文件。
- -J <jumphost> 通过跳板机连接远程主机。
- -L <port>:localhost:<hostport> 监听本地端口并转发到远程主机。
- -N 取消转发端口。
- -o <option> 参数传递给ssh协议协商。
- -p port 指定远程主机的端口。
- -q 不显示警告消息。
- -R <port>:localhost:<hostport> 请求远程主机的端口转发。
- -S controlpath 指定控制路径。
- -W host:port 选择特定网络接口。
- username@hostname 登录远程主机。

**示例：**

```bash
$ ssh remote_user@remote_host         # 直接登录远程主机
$ ssh -p port remote_user@remote_host  # 指定远程主机的端口号
$ ssh local_user@localhost "ssh remote_user@remote_host"  # 通过跳板机登录远程主机
$ ssh -L 8080:www.example.com:80 -NR 9000:localhost:22 remote_user@remote_host  # 建立本地和远程主机之间的端口转发
```

## 4.33. scp命令
scp命令用于远程复制文件。

**语法：**

scp [选项]... [-i identity_file] source... target

**选项：**

- -1 强制scp命令使用协议ssh1。
- -3 强制scp命令使用协议ssh3。
- -4 强制scp命令使用协议inet。
- -6 强制scp命令使用协议ipv6。
- -B 使用批处理模式。
- -C 压缩文件传输。
- -r 递归复制整个目录。
- -p 保持文件属性。
- -q 静默模式。
- -v 详细输出。
- -P port 服务器的端口号。
- -i identity_file 从指定文件中读取识别信息。

**示例：**

```bash
$ scp local_file remote_user@remote_host:remote_file  # 复制本地文件到远程主机
$ scp -r local_dir remote_user@remote_host:remote_dir  # 递归复制本地目录到远程主机
```