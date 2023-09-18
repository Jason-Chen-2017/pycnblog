
作者：禅与计算机程序设计艺术                    

# 1.简介
  

linux命令`whoami`，是查看当前登录用户命令。`whoami`命令是最简单的linux命令，可以用来显示当前登录的用户名。我们可以使用`whoami`命令来查看自己登录系统时使用的用户名。它的基本用法如下所示：
```bash
whoami
```
执行上述命令后，将会输出当前登录的用户名。如果当前没有登录任何账户，则命令不会返回任何结果。如果我们想获取root权限，可以通过sudo命令运行`whoami`命令，获取root权限后即可查看所有用户信息。所以，`whoami`命令在linux系统中很有用。
# 2.基本概念术语说明
## 2.1 whoami命令
whoami命令是显示当前登录用户名的基本命令，其作用类似于windows系统中的“我的电脑”功能，但是在linux系统中没有这个功能，只能通过查看/etc/passwd文件或调用id命令来查看当前登录用户的用户名。

1.`/etc/passwd` 文件：该文件中存储了系统中所有的用户及其相关信息，每条记录都包含了一条用户的信息，包括用户名、真实名字（可选）、加密后的密码、UID、GID、组信息、HOME目录、shell等。

文件中各字段之间的分隔符为冒号(:)。

2.`id` 命令：id命令是linux系统中用来显示用户和组信息的命令。它也可以用来查看当前进程的所有者，但它不是用于查看当前登录的用户名的。

```bash
id
uid=0(root) gid=0(root) groups=0(root)
```

3.`su` 命令：su命令可以用来切换到root身份。su命令的一般形式为：su [-options] [username] 。使用-l选项，可以列出已知用户的帐户信息，如：

```bash
su -l root   # 列出root用户信息
```
# 3.核心算法原理和具体操作步骤以及数学公式讲解
`whoami`命令内部实现非常简单，只需要读取`/etc/passwd`文件的第一行就可以得到当前登录用户的用户名。所以，`whoami`命令的效率非常高。

当我们输入`whoami`命令时，它就会读取`/etc/passwd`文件的第一个域作为自己的用户名并打印出来。因此，如果/etc/passwd文件被修改或者损坏，那么whoami命令也会受到影响。另外，/etc/passwd文件通常只有超级用户才能访问，普通用户无法直接读取。因此，whoami命令对于普通用户来说也是不可用的。

这里就不再赘述了。
# 4.具体代码实例和解释说明
## 4.1 获取当前登录的用户名
```python
import os 

print("Current login user name is:",os.getlogin())
```
## 4.2 修改当前登录的用户名
注意：该功能仅限于root用户。

由于/etc/passwd文件是只读的，因此我们无法直接修改其内容。为此，我们需要借助sudo命令来修改当前登录用户的用户名。

首先，我们需要切换到root用户，然后使用下面的命令修改当前登录用户的用户名：
```bash
sudo usermod -l new_name old_name    # 将old_name更改为new_name
```
其中old_name是旧的用户名，new_name是新的用户名。

例如，假设当前登录用户的用户名为“test”，想要修改成“tuser”，那么我们可以输入如下命令：
```bash
sudo usermod -l tuser test    # 将test更改为tuser
```
修改成功之后，重新登录系统，则当前登录用户的用户名将变为"tuser"。