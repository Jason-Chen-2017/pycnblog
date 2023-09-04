
作者：禅与计算机程序设计艺术                    

# 1.简介
         

Git是一个开源的分布式版本控制系统，用于敏捷高效地处理任何或小或大的项目。Git 是 Linus Torvalds 为了帮助管理 Linux 内核开发而创造的一个开放源码的版本控制软件。它的诞生目的就是用廉价易用的方式来共享大量源码。由于 Git 的独特性，使得它被越来越多的人使用。Git 是目前最流行的版本控制工具，其速度快、灵活且能有效地管理各种类型的项目。除了用于个人或小型团队外，Git 也可以服务于中大型项目的版本控制。此外，GitHub 和 GitLab 提供了基于 Git 的版本控制服务。
本文将详细介绍 Git 命令及相关知识点。Git 具有以下一些特性：

1）速度快：每次提交 Git 只需要很少的文件IO，所以速度很快；

2）简单可靠：Git 使用简单的分支和合并机制，并且在任何时候都可以还原；

3）离线工作：只需联网一次就可以完成对仓库的克隆、更新等操作；

4）分支与合并：Git 提供丰富的分支管理功能，使得多人协作开发变得十分容易；

5）丰富的工具支持：Git 提供强大的工具集，包括 git add/rm/mv、git log、git diff、git branch、git checkout、git merge、git rebase、git stash、git tag 等命令。

# 2.安装配置
## 2.1 安装Git
### Windows环境下安装
1.下载Git最新版安装包（https://git-scm.com/downloads），建议选择最新稳定版本。

2.安装时，勾选“Use Git from the Windows Command Prompt”选项，这会在你的电脑上配置好 PATH 环境变量，这样在任意目录打开命令提示符，输入 git 命令就会执行 Git 命令。

3.安装成功后，右键点击开始菜单中的“Git Bash”，在弹出的命令行窗口输入 git --version ，如果显示出版本号表示安装成功。

### MacOS环境下安装
1.打开Terminal并输入如下指令：

```bash
brew install git
```

2.安装成功后，在终端输入 git --version ，如果显示出版本号表示安装成功。

## 2.2 配置Git
### 用户信息设置
Git 需要知道每个 committer 的名字和邮箱才能正常工作。你可以通过 git config 来进行全局配置或者在各个仓库单独配置用户信息。

```bash
# 设置全局用户信息
$ git config --global user.name "your name"
$ git config --global user.email "your email address"

# 在当前仓库中设置用户信息
$ cd /path/to/repository
$ git config user.name "your name"
$ git config user.email "your email address"
```

### 生成SSH密钥
SSH 密钥是一种更安全的远程登录方式，推荐使用 SSH 密钥连接到 GitHub 或 GitLab。

```bash
# 检查是否已存在 SSH 密钥
$ ls -al ~/.ssh

# 如果不存在则生成新的 SSH 密钥
$ ssh-keygen -t rsa -C "your email address"

# 将 SSH 公钥添加至 GitHub 或 GitLab
```

# 3.创建仓库
Git 使用 init 命令来创建一个新的仓库，该仓库会在当前文件夹下生成一个.git 文件夹。

```bash
$ mkdir myproject # 创建项目文件夹
$ cd myproject     # 进入项目文件夹
$ git init          # 初始化仓库
```

# 4.添加文件
## 4.1 添加所有文件
要将所有未跟踪文件（即没有被 Git 管理的文件）添加至暂存区，可以使用 git add * 命令：

```bash
$ git status   # 查看状态
On branch master

No commits yet

Untracked files:
(use "git add <file>..." to include in what will be committed)

file1
file2
...

$ git add *    # 添加所有未跟踪文件
$ git status   # 查看状态
On branch master

No commits yet

Changes to be committed:
(use "git rm --cached <file>..." to unstage)

new file:   file1
new file:   file2
...
```

## 4.2 添加指定文件
要添加指定的单个文件至暂存区，可以使用 git add 文件名 命令：

```bash
$ touch README.md # 新建 README 文件
$ git status      # 查看状态
On branch master

No commits yet

Untracked files:
(use "git add <file>..." to include in what will be committed)

README.md

$ git add README.md  # 添加 README 文件至暂存区
$ git status         # 查看状态
On branch master

No commits yet

Changes to be committed:
(use "git rm --cached <file>..." to unstage)

new file:   README.md
```

## 4.3 撤销暂存区的修改
如果只是临时想要撤销对文件的修改，又不想删除文件，可以使用 git reset HEAD 文件名 命令：

```bash
$ vim README.md           # 修改 README 文件
$ git status              # 查看状态
On branch master

Changes not staged for commit:
(use "git add <file>..." to update what will be committed)
(use "git checkout -- <file>..." to discard changes in working directory)

modified:   README.md

$ git reset HEAD README.md # 撤销对 README 文件的修改
$ git status               # 查看状态
On branch master

No changes added to commit (use "git add" and/or "git commit -a")
```

注意：这条命令不会影响仓库的文件内容，仅仅是撤销暂存区的修改，不会删除文件。如果希望彻底删除文件，需要使用 git rm 文件名 命令。

## 4.4 删除文件
如果要永久删除文件，可以使用 git rm 文件名 命令：

```bash
$ ls                               # 查看当前仓库的文件列表
README.md            file1        file2        ...

$ git rm README.md                 # 从暂存区删除 README 文件
$ git commit -m "delete README"    # 提交删除操作

$ ls                               # 查看当前仓库的文件列表
file1                file2        ...
```

注意：这条命令会从工作目录、暂存区、HEAD 中删除文件，请谨慎使用！

# 5.查看提交历史
## 5.1 查看最近两次提交记录
使用 git log 命令可以查看最近两次提交记录：

```bash
$ git log                            # 查看最近两次提交记录
commit c1e075f54cf70fc92c91d3b7d19ccffea9868e4e (HEAD -> master)
Author: wangyuchen <<EMAIL>>
Date:   Fri Oct 31 15:32:52 2020 +0800

delete README

commit b683ce48fc1a307dd4b19c7f3a240bf4be6e1ba8
Author: wangyuchen <<EMAIL>>
Date:   Wed Oct 29 14:24:53 2020 +0800

create README
```

## 5.2 查看指定提交记录
使用 git show 命令可以查看指定提交记录：

```bash
$ git show b683ce48fc1a307dd4b19c7f3a240bf4be6e1ba8                     # 查看指定提交记录
commit b683ce48fc1a307dd4b19c7f3a240bf4be6e1ba8
Author: wangyuchen <<EMAIL>>
Date:   Wed Oct 29 14:24:53 2020 +0800

create README

diff --git a/README.md b/README.md
new file mode 100644
index 0000000..e69de29
--- /dev/null
+++ b/README.md
@@ -0,0 +1 @@
+# MyProject
```

## 5.3 分页显示提交记录
使用 git log 命令加上 --oneline 可以看到更简洁的提交记录，并且默认分页显示：

```bash
$ git log --oneline  
73fc0dc create README.txt 
783c1ec fix bugs on index page 
a5d0c5c remove unnecessary code 
4ed38ca add navigation menu 
d54f4bc initial commit 
```

使用 --pretty=oneline 参数可以看到更多提交信息：

```bash
$ git log --pretty=oneline   
73fc0dc51b create README.txt 
c628a5d3fd fix bugs on index page 
7a5fafe0db remove unnecessary code 
cd05ef10d1 add navigation menu
84cb17ceaa initial commit
```

使用 --graph 参数可以看到分支图：

```bash
$ git log --oneline --graph  
73fc0dc (HEAD -> master) create README.txt 
* c628a5d fix bugs on index page 
7a5fafe remove unnecessary code 
| * cd05ef1 (feature_branch) add navigation menu 
\ 84cb17c initial commit 
```