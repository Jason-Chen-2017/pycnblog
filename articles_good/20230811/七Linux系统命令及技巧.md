
作者：禅与计算机程序设计艺术                    

# 1.简介
         

## Linux 是什么？
Linux是一个开源、免费、UNIX类操作系统。它最初由林纳斯·托瓦兹在法国MINIX操作系统基础上开发而成，之后被斯坦福大学李四摩（Linus Torvalds）所加以完善并改名为Linux。
## 为何要学习 Linux 命令？
学习Linux命令能够帮助我们更好地理解计算机底层工作原理，并且掌握一些系统管理、网络配置等关键技能，提升工作效率。
## Linux 的发行版本有哪些？
目前 Linux 有许多发行版本可供选择，包括 Ubuntu、CentOS、Debian、Red Hat、Archlinux、Fedora、OpenSUSE、Manjaro……不同的发行版本之间差异较大，但一般都支持最基本的Linux命令。所以，不管用的是哪个发行版本，掌握基本命令总是有用的。
# 2.基本概念术语说明
## 文件系统
文件系统(File System)是指将数据存储在磁盘上的一种组织方式，每个文件系统都定义了目录结构、文件格式、存取控制列表、用户和群组权限等。不同的文件系统有着自己的特点。常见的文件系统有：
- ext2/ext3: 基于日志的数据结构，兼容于Unix和Linux；
- NTFS：微软开发，用于Windows操作系统；
- FAT32: 只能在少量旧硬件上运行，速度快；
- ReiserFS: 高级文件系统，支持安全加密。

## 用户与权限
### 用户(User): 是指能够登录到计算机并能够使用系统资源的人员。Linux 中通常使用用户名标识用户，用户名只能由大小写英文字母、数字、下划线和减号组成，且长度不能超过32字节。
### 组(Group): 是具有相同特征和权限设置的一组用户。一个用户可以加入多个组，不同组之间可以有重复的成员。在Linux中，组名称由字母、数字、下划线和减号组成，且长度不能超过32字节。
### 身份(Identity): 用户与组的结合体，是 Linux 认证方式的基础。用户身份通过UID(User IDentification)唯一确定，每个用户都有一个独一无二的UID。组身份通过GID(Group IDentification)唯一确定，每个组都有一个独一无二的GID。
### 权限(Permission): 在Linux中，权限分为三类：
- 读(r)：允许用户查看文件内容；
- 写(w)：允许用户修改文件内容或删除文件；
- 执行(x)：允许用户执行文件。
为了方便起见，也可以使用数字表示权限，如7代表rwx。
## 命令与终端
### 命令(Command): 是用来控制计算机完成特定任务的符号序列。命令有两种形式：内置命令和外部命令。内置命令是在操作系统中集成的，不需要调用外部应用程序。外部命令则需要调用外部应用程序才能运行。常见的内置命令有 cd、ls、mkdir、rm等，外部命令有 wget、curl、ping等。
### 终端(Terminal): 是指人机界面，用以与计算机进行交互的界面。通常有图形界面的终端，也有文字模式的终端，甚至还有类似 ssh 的远程终端。在Linux中，可以通过多种方式进入终端，例如：直接按下Ctrl+Alt+T组合键，打开Dash搜索栏后输入terminal，点击图标打开。
# 3.核心算法原理和具体操作步骤以及数学公式讲解
## ls命令
列出指定目录下的所有文件和子目录。语法如下：
```
ls [选项] 文件名...
```
其中，[选项]主要有以下几种：
- -a：显示隐藏文件，即那些名前面有“.”字符的文件；
- -d：仅显示目录本身，而不是列出目录中的文件；
- -F：在列出的文件名称后加上一对引号，表示是否是目录或是链接文件；
- -h：以人性化的方式显示文件大小，比如1K，2M等；
- -l：以详细方式列出文件信息，包含文件大小、拥有者、权限、修改日期等；
- -R：递归列出所有子目录下的文件。

示例：
```
# 列出当前目录的所有文件和目录
ls
# 列出某个目录的所有文件和目录
ls /home/user
# 以详细方式列出某个目录的所有文件和目录
ls -l /home/user
# 列出当前目录及其子目录下所有文件和目录
ls -R
# 查看隐藏文件
ls -a
```
## mkdir命令
创建目录。语法如下：
```
mkdir [-p] 文件夹名称
```
其中，[-p]选项可以创建不存在的父目录。

示例：
```
# 创建文件夹myfolder
mkdir myfolder
# 创建多级文件夹dir1/dir2/dir3
mkdir dir1/dir2/dir3
# 如果父目录不存在，则创建父目录
mkdir -p parent/child
```
## cp命令
复制文件或目录。语法如下：
```
cp [选项] 源文件或目录... 目标文件或目录
```
其中，[选项]主要有以下几种：
- -a：保持文件属性，包括权限、时间戳等；
- -d：当源文件是目录时，才复制整个目录；
- -f：覆盖已存在的文件或目录，且不给出提示；
- -i：与-f选项相反，会询问是否覆盖已存在的文件或目录；
- -p：连同文件的属组一起复制；
- -r,-R：递归处理，把源目录下的所有文件都复制到目标目录下。

示例：
```
# 将文件file1复制为文件file2
cp file1 file2
# 将目录dir1复制为目录dir2
cp -r dir1 dir2
# 将目录dir1下所有文件复制到目录dir2下
cp -r dir1/* dir2
```
## mv命令
移动或重命名文件或目录。语法如下：
```
mv [选项] 源文件或目录... 目标文件或目录
```
其中，[选项]主要有以下几种：
- -b：若需覆盖文件，则覆盖前先备份；
- -f：强制覆盖已存在的文件或目录，不给出提示；
- -i：与-f选项相反，会询问是否覆盖已存在的文件或目录；
- -u：若目标文件已经存在，且mtime较新，才更新目标文件。

示例：
```
# 重命名文件file1为file2
mv file1 file2
# 将文件file1移动到目录dir2
mv file1 dir2
# 移动多个文件到目录dir1
mv file1 file2 file3 dir1
```
## rm命令
删除文件或目录。语法如下：
```
rm [选项] 文件或目录
```
其中，[选项]主要有以下几种：
- -f：强制删除，忽略不存在的文件、无权限的文件或目录；
- -i：互动模式，逐个确认要删除的文件；
- -r,-R：递归删除，将目录下的所有文件和子目录均删除。

示例：
```
# 删除文件file1
rm file1
# 删除目录dir1及其所有内容
rm -r dir1
# 删除多个文件和目录
rm file1 file2 dir1 dir2
```
## cat命令
输出文件的内容。语法如下：
```
cat [选项] 文件
```
其中，[选项]主要有以下几种：
- -n 或 --number：由1开始对所有输出的行数编号；
- -b 或 --number-nonblank：和-n参数相同，只不过对于空白行不编号；
- -s 或 --squeeze-blank：当遇到过多空行时，将两个及两个以上空行压缩成一行；
- -T或--show-tabs：将跳格键^I显示为^I。

示例：
```
# 查看文件file1的内容
cat file1
# 从第二行开始查看文件file1的内容
cat -n file1 | tail +2
# 每三个行打印一个行号
nl file1 | awk '{if (NR%3==1){print NR}}'
```
## touch命令
创建空文件。语法如下：
```
touch 文件
```
该命令与shell中的“>”类似，可以快速创建新文件。

示例：
```
# 创建空文件file1
touch file1
# 创建多个空文件file1、file2、file3
touch file1 file2 file3
```
## chmod命令
更改文件或目录的权限。语法如下：
```
chmod [选项] 模式 文件或目录
```
其中，[选项]主要有以下几种：
- u 或 --owner=USER：更改文件或目录的拥有者；
- g 或 --group=GROUP：更改文件或目录的用户组；
- o 或 --others=OTHERS：更改文件或目录的其他用户；
- a 或 --all=ALL：更改所有相关用户；
- r 或 --read=READ：设置文件或目录可读取；
- w 或 --write=WRITE：设置文件或目录可写入；
- x 或 --execute=EXECUTE：设置文件或目录可执行；
- X 或 --no-exec：执行文件或目录的名字才可查看；
- s 或 --setuid/setgid/sticky：设置用户ID、组ID或粘滞位；
- A 或 --no-preserve-root：不要将文件 chown 到根目录下；
- Z 或 --context：沿用文件的SELINUX上下文。

示例：
```
# 设置文件file1的权限为rwxr-xr-x
chmod 755 file1
# 把所有者、用户组、其他用户的权限分别设置为 rw- --- ---
chmod ug=rw,o=--- file1
# 对目录dir1及其所有内容设置权限为rwxr-xr-x
chmod -R 755 dir1
```
## chown命令
更改文件或目录的拥有者和用户组。语法如下：
```
chown [选项] 用户名 文件或目录
```
其中，[选项]主要有以下几种：
- -c 或 --changes：报告更改的部分信息；
- -f 或 --force：非必要的操作时不给出提示；
- -h 或 --no-dereference：只改变符号连接的指向，而不会影响其对应文件；
- -R 或 --recursive：递归处理，将指定目录下的所有文件及子目录一并处理；
- -v 或 --verbose：详细显示指令的执行过程。

示例：
```
# 把文件file1的拥有者设置为 user1
chown user1 file1
# 把文件file1的拥有者和用户组设置为 user1 和 group1
chown user1:group1 file1
# 修改目录dir1及其所有内容的拥有者和用户组
chown -R user1:group1 dir1
```
## pwd命令
显示当前目录。

示例：
```
pwd
```
## date命令
显示或设置系统的时间。语法如下：
```
date [选项] [+格式]
```
其中，[选项]主要有以下几种：
- -d STRING：根据字符串设定时间；
- -f FILE：从文件中读取日期与时间；
- -j N：返回距离当前时间N天以来的日历时间（以秒计算）；
- -r TIMEZONE：显示与TIMEZONE对应的时间；
- -R：显示UTC时间；
- -s STRING：根据字符串设定系统时间；
- -u：显示GMT时间；
- -V：显示版本信息。

示例：
```
# 显示当前时间
date
# 设置当前时间为2020年1月1日12:00:00
date 20200101120000
# 根据格式显示时间
date "+%Y-%m-%d %H:%M:%S"
# 使用date命令设置系统时间
echo "20200101120000" > /etc/sysconfig/clock
date -s "$(cat /etc/sysconfig/clock)"
```
## find命令
查找文件。语法如下：
```
find [路径] [表达式]
```
其中，[路径]是指查找范围，可以指定某一目录或者多个目录，默认值为当前目录；[表达式]是指匹配条件，可以指定文件名、文件类型、文件权限、用户和用户组、文件大小等，匹配条件可以串联起来，多个匹配条件之间用布尔运算符连接。

示例：
```
# 在/usr目录下查找扩展名为txt的文件
find /usr -name "*.txt"
# 在/home/user目录下查找扩展名为txt的文件，文件大小大于10MB
find /home/user -type f -size +10M -name "*.txt"
# 查找以数字开头的文件名
find. -regex '^\d.*$'
```
## grep命令
搜索文本，并输出匹配行。语法如下：
```
grep [选项] PATTERN 文件
```
其中，[选项]主要有以下几种：
- -c 或 --count：计算匹配行数，但是输出太多内容时很耗费内存；
- -e PATTERN 或 --regexp=PATTERN：指定要搜索的正则表达式；
- -E 或 --extended-regexp：激活扩展的正则表达式功能；
- -f FILE 或 --file=FILE：从文件中获取搜索模式；
- -i 或 --ignore-case：忽略大小写；
- -l 或 --files-with-matches：只列出匹配的文件名，不显示匹配内容；
- -L 或 --files-without-match：只列出没有匹配的文件名；
- -n 或 --line-number：输出匹配行及其行号；
- -q 或 --quiet或--silent：安静模式，只输出最后的结果；
- -r 或 --recursive：递归地搜索当前目录及其子目录；
- -w 或 --word-regexp：只搜索全词；
- -x 或 --only-matching：只输出匹配的部分；
- -y 或 --line-buffered：与-u或--unbuffered参数相反，要求每行一输出。

示例：
```
# 在文件file1中搜索关键字“hello”，并打印出匹配行
grep hello file1
# 在目录dir1及其子目录下搜索关键字“hello”，并打印出匹配行
grep -r hello dir1
# 在文件file1中搜索正则表达式“he.*lo”，并打印出匹配行
grep -E he.*lo file1
# 在文件file1中搜索正则表达式“he\w*lo”，并打印出匹配行
grep -Eo he\w*lo file1
```
## head命令
输出文件的开头内容。语法如下：
```
head [选项] 文件
```
其中，[选项]主要有以下几种：
- -c NUM：从文件前NUM字节开始输出；
- -n NUM：从文件开头输出的行数；
- -q：不输出文件名；
- -v：显示不可打印字符；
- -z：行结束符以\0换行。

示例：
```
# 显示文件file1的前10行
head -n 10 file1
# 不输出文件名，显示文件file1的前10行
head -qn 10 file1
```
## tail命令
输出文件的末尾内容。语法如下：
```
tail [选项] 文件
```
其中，[选项]主要有以下几种：
- -c NUM：从文件末尾向前NUM字节开始输出；
- -f：实时监视文件；
- -n NUM：从文件末尾输出的行数；
- -q：不输出文件名；
- -v：显示不可打印字符；
- -z：行结束符以\0换行。

示例：
```
# 显示文件file1的最后10行
tail -n 10 file1
# 持续实时监视文件file1
tail -fn 10 file1
```
## du命令
显示磁盘使用情况。语法如下：
```
du [选项] 文件或目录
```
其中，[选项]主要有以下几种：
- -a：显示文件大小，除非是符号链接；
- -B SIZE：区块大小；
- -k：以千字节为单位显示文件大小；
- -m：以兆字节为单位显示文件大小；
- -P：与-h参数配合使用，可直接显示更易读的“xxx M”样式大小；
- -s：仅显示总计；
- -X：排除某些文件或目录；
- -x：仅显示非标准文件；
- -h：以友好的可读方式显示文件大小。

示例：
```
# 显示/home/user目录的磁盘使用情况
du /home/user
# 显示当前目录的磁盘使用情况
du -sh *
```
## ln命令
创建链接文件。语法如下：
```
ln [选项] 目标文件 链接文件
```
其中，[选项]主要有以下几种：
- -d：创建链接节点，同时把目标文件视为目录；
- -f：如果目标文件已经存在，就将其删除后再创建链接；
- -i：覆盖既有相同名称的链接文件；
- -n：创建符号链接，不增加链接数；
- -s：创建符号链接，指向原始文件。

示例：
```
# 创建链接文件link1到目标文件file1
ln file1 link1
# 创建硬链接文件hardlink1到目标文件file1
ln file1 hardlink1
# 创建符号链接文件symlink1到目标文件file1
ln -s file1 symlink1
```
## df命令
显示磁盘空间占用信息。语法如下：
```
df [选项] 文件系统
```
其中，[选项]主要有以下几种：
- -a：所有文件系统；
- -B SIZE：区块大小；
- -h：以更易读的方式显示；
- -i：显示inode信息；
- -k：以1KB为单位显示；
- -m：以1MB为单位显示；
- -t TYPE：指定文件系统类型；
- -x devtmpfs tmpfs mqueue hugetlbfs rpc_pipefs fuse.gvfsd-fuse.glusterfs fusectl debugfs squashfs udf zfs ntfs sysfs cifs usbfs devpts shmem nfssub filesystems vboxsf nullfs overlay fsiso9660 gvfs fuse.sshfs ceph fuse.ceph.client ceph.conf ceph.entity mdadm tmpfs2 minix zfs_fuse btrfs lustre tmpfs3
显示指定文件系统的信息。

示例：
```
# 显示当前目录所在文件系统的磁盘使用情况
df -h.
# 显示所有文件系统的磁盘使用情况
df -h
# 显示文件系统类型为ext4的文件系统的磁盘使用情况
df -h -t ext4
```
## free命令
显示内存使用情况。语法如下：
```
free [选项]
```
其中，[选项]主要有以下几种：
- -b：以字节为单位显示；
- -k：以千字节为单位显示；
- -m：以兆字节为单位显示；
- -g：以GIGABYTES为单位显示；
- -h：显示可读性高的容量信息；
- -t：显示总和。

示例：
```
# 显示内存使用情况
free
# 显示内存使用情况，以兆字节为单位显示
free -m
```
## whoami命令
显示当前登陆用户。

示例：
```
whoami
```
## hostname命令
显示主机名。

示例：
```
hostname
```
## export命令
设置环境变量值。

示例：
```
export PATH=$PATH:/opt/bin
```
## alias命令
设置别名。

示例：
```
alias ll='ls -al'
```
## unalias命令
取消别名。

示例：
```
unalias ll
```
## su命令
切换用户。语法如下：
```
su [选项] [-] USERNAME
```
其中，[选项]主要有以下几种：
- -c COMMAND：在新 shell 中执行命令；
- -h：隐藏输入的密码；
- -l：模拟登录 shell；
- -s SHELL：指定登录 shell；
- -v：重用之前的环境变量。

示例：
```
# 以超级用户权限执行命令
sudo su -c "apt update && apt upgrade"
```