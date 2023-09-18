
作者：禅与计算机程序设计艺术                    

# 1.简介
  

rm命令（英文全称是“Remove”，移除）用来删除文件或目录，它在Unix/Linux中是一个基础命令，经常被用作系统管理的重要工具之一。熟练掌握rm命令对于系统管理员、开发人员和测试人员都是非常必要的。本教程将详细阐述rm命令的所有功能和用法。
## 1.1 rm命令概览
rm命令是对Linux/Unix下删除文件或目录的命令，它支持许多参数选项和选项组合，能够彻底删除文件及其内容，但同时也具有递归删除目录及其子目录的能力，因此非常危险。在实际工作中，为了避免误删重要文件，通常会使用一些防护措施，比如备份文件，或检查文件是否存在，避免误删系统重要文件等。

rm命令的语法格式如下：

```
rm [options] file_names
```

参数说明：

- options：可选参数，用于控制rm命令行为。常用的参数选项包括：
  - -i：删除时询问确认；
  - -f：强制删除，忽略不存在的文件，不会出现警告信息；
  - -r,-R：递归删除目录及其子目录；
  - -v：显示执行过程；
  - -d：仅删除空目录；
  - -p：删除目录下的配置文件。
- file_names：需要删除的文件名列表。

## 1.2 rm命令功能介绍
rm命令主要用来删除一个或多个文件或目录，可以递归地删除整个目录树。但是，由于rm命令是带有破坏性的命令，因此它一定要谨慎使用。

### （1）删除单个文件
当仅指定了一个文件名时，rm命令默认只删除该文件，而不管它是什么类型的文件。如果该文件是一个目录，则会报错。

示例：

```
$ ls
b.bak  g.doc  h.cfg    i.sh    j.java    k.tar.gz
$ rm a.txt 
$ ls
b.bak  g.doc    h.cfg   i.sh      j.java     k.tar.gz
```

### （2）删除多个文件
如果指定了多个文件名，则rm命令默认会逐一删除每个文件。如果有一个文件不存在或者无法删除，则整个命令会停止运行并返回错误信息。

示例：

```
$ ls
foo  bar  baz  qux
$ rm foo bar baz qux
$ ls
ls: cannot access 'bar': No such file or directory
ls: cannot access 'qux': No such file or directory
```

### （3）删除目录
如果指定的是一个非空目录，则rm命令会报错并要求用户确认是否继续。如果指定的是一个空目录，则直接删除该目录。

示例：

```
$ ls /tmp
[... some files and directories...]
$ rm -rf /tmp/*
```

### （4）强制删除
如果指定了-f或--force选项，则rm命令会尝试强制删除文件或目录，无论它们是否存在。使用此选项应谨慎！

```
$ rm -rf /tmp/*
$ ls /tmp
[nothing]
```

### （5）递归删除目录及其子目录
-r 或 --recursive：递归删除目录及其子目录。如果某个子目录下还有其他目录，则也会一起被删除。

```
$ tree mydir/
mydir/
├── dir1/
│   ├── subfile1
│   └── subsubdir1/
└── dir2
    ├── subfile2
    └── subsubdir2/
        └── subsubfile2

3 directories, 3 files
$ rm -r mydir/
```

### （6）显示执行过程
-v 或 --verbose：显示执行过程，包括文件或目录被成功删除的信息。

```
$ rm *.txt
rm: remove regular empty file '*.txt'? y
$ ls
b.bak  g.doc  h.cfg    i.sh    j.java    k.tar.gz
$ rm -rv./*
removed '.bashrc'
removed '.profile'
removed './testdir/'
removed'some.txt'
total 9
drwx------  2 user group    6 Apr 17 16:46./
drwxrwxrwt 22 root root 4096 Apr 17 16:45../
```

### （7）删除目录下的配置文件
-p 或 --preserve-root：删除目录下的配置文件。对于某些系统程序来说，可能存在配置文件保存在程序目录下的情况，如果不加此选项，这些配置文件就会被删除。

```
$ cd /etc
$ ls
acpi  adduser.conf        alsa         alternatives             apache2           atoprc          bash.bashrc  
[... ]              checkpolicy  cloudinit    console-setup           cron.daily       crontab         cron.weekly    
[... ]               colord       com.apple.Bluetooth  cron.deny             cron.hourly      crypttab        cups           
$ sudo rm -rvf */*.conf
removed '/etc/apt/preferences.d/*.conf'
removed '/etc/kernel/*.conf'
removed '/etc/locale.gen'
removed '/etc/security/access.conf'
[... ]
total 84
drwxr-xr-x  10 0  0  120 Jul  8  2021..
drwxr-xr-x  10 0  0  120 Sep  2  2021.
drwxr-xr-x   2 0  0   60 Feb  4  2021 X11
lrwxrwxrwx   1 0  0   10 Nov 19  2020 altgr.keys -> vconsole.conf
-rw-------   1 0  0   24 Aug 17  2021 apparmor.d
drwxr-xr-x   2 0  0   60 Jan 22  2021 atm
lrwxrwxrwx   1 0  0   11 Jun  1  2021 ascii -> sv/latin1
-rw-r--r--   1 0  0 2317 Oct  7  2021 byobu.ignore
[... ]
$ sudo find. -name "*.conf" -print | xargs sudo chmod go-w
find: ‘./crash/’: Permission denied
chmod: changing permissions of './crash/core.gz': Operation not permitted
find: ‘./installer/cdrom/’: Permission denied
chmod: changing permissions of './installer/cdrom/.disk/by-label/BTRFS_DATA': Operation not permitted
[... ]
```

## 1.3 rm命令注意事项和技巧

### （1）谨慎使用-rf选项
rm命令默认不会递归地删除整个目录树，因此，如果指定了-r或-R选项，则它不会删除父目录。另外，rm命令不能删除根目录，即使指定了选项。因此，在使用rm命令的时候，一定要小心谨慎，不要使用不当的选项。

示例：

```
$ mkdir /tmp/tempdir
$ touch /tmp/tempdir/{file1,file2}
$ rmdir /tmp/tempdir # 删除失败，因为tempdir还不是空目录
rm: failed to remove '/tmp/tempdir': Directory not empty
```

### （2）慎重使用-f选项
rm命令的-f选项是很危险的，它会强制删除文件，即使它们没有权限访问或者是目录里有文件，都会被删除。因此，如果不确定，应该尽量不要使用此选项。

示例：

```
$ echo "hello world">file.txt
$ chmod u=--- x=-- file.txt # 将file.txt设为不可读取和不可写入
$ rm file.txt # 删除失败，因为不可读写的文件会被删除
```

### （3）理解rm命令的特性
rm命令可以删除文件或目录，同时也提供了一些参数选项来控制它的行为。理解rm命令的各个特性和特性之间的关系十分重要。下面给出rm命令的一般特性：

1. 没有回滚机制：rm命令永远不会恢复已删除的文件或目录。
2. 不保证原子性：在rm命令执行过程中，文件的状态可能会突然发生变化，导致无法预料的后果。
3. 只删除目标文件：如果目标文件是符号链接或硬链接，则不会删除源文件。
4. 默认不显示提示信息：默认情况下，rm命令不会显示确认提示信息。
5. 文件名支持通配符：rm命令可以使用通配符进行匹配，例如`rm *.txt`，表示删除当前目录下所有扩展名为`.txt`的文件。

通过对rm命令的特性进行分析，我们可以了解到rm命令的以下特点：

1. 不能删除根目录：由于根目录是最高级目录，所以rm命令不允许删除根目录。因此，如果使用rm命令删除当前目录中的某个文件，那么至少会留下一级目录，不会删除根目录。
2. 不提供撤销机制：删除一个文件或者目录之后，无法恢复，只能从备份中找回。
3. 删除软连接：如果删除一个目录，那么其软链接指向的目录不会被删除。
4. 使用权限位：rm命令依赖于文件权限位来判断删除操作是否合法，如果没有相应的权限，则不会删除文件。