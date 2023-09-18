
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 1.1 Git概述
`Git`是一个开源的分布式版本控制系统，最初由林纳斯·托瓦兹（<NAME>）编写，于2005年以BSD授权发布。最初目的是为更好地管理Linux内核开发而设计，后逐渐演化成一个独立的版本控制系统，并且已经成为非常流行的版本控制系统。`Git` 提供了一系列强大的功能，能够有效地管理复杂的项目历史，同时也具有以下特性：

 - 速度快
 - 对数据安全性高
 - 能够处理较大的项目、跨越不同操作系统平台
 - 允许并行开发、分支工作等
 - 有丰富的工具支持

## 1.2 Git的安装
### 1.2.1 Linux/Unix/Mac系统安装
```bash
sudo apt-get install git # Debian/Ubuntu系统安装方式
sudo yum install git # CentOS/RedHat系统安装方式
brew install git # Mac系统安装方式
```
### 1.2.2 Windows系统安装
下载安装包：https://git-scm.com/download/win 

配置环境变量：https://www.jianshu.com/p/9f54a7e3a1d9

## 1.3 Git的基本概念
### 1.3.1 工作区、暂存区、本地仓库
在日常使用`Git`时，通常会建立三个区域：`工作区`、`暂存区`、`本地仓库`。 

`工作区`就是我们的电脑里面的文件目录；`暂存区`中保存了下次将要提交的文件列表信息，在`commit`命令执行后，该区域的数据会被永久保存在`本地仓库`中；而`本地仓库`则是`Git`用来保存项目的元数据和对象数据库的地方，包括所有的版本，作者信息，评论信息等。


### 1.3.2 暂存区
`暂存区`（Stage area）用于暂存将要添加到下一次提交(`commit`)中的文件修改信息，直到实际执行提交操作的时候才移动到`本地仓库`，每个版本库中都有且只有一个暂存区。

### 1.3.3 远程仓库
`远程仓库`又称`远端仓库`，也就是`GitHub`, `Bitbucket`或其他类似的代码托管服务提供商托管的版本库，与本地仓库不同之处在于，远程仓库是其他人的一个仓库，用于分享自己的工作成果，协助他人进行共同开发。所以在实际使用`Git`时，我们需要先克隆远程仓库或者通过`git remote add origin URL`增加新的远程仓库地址。

### 1.3.4 分支
`Git`使用`master`作为默认的分支名，但你依然可以创建其他分支。`master`分支始终代表着最新稳定的代码，一般情况下，`master`分支应该是比较稳定的，开发人员一般只从这个分支上检出代码，然后再在上面进行开发。

另外，当你从远程仓库克隆代码时，你克隆到的本地仓库默认只有`master`分支，如果需要查看其他分支的提交记录，可以通过`git branch -r`查看所有远程分支，通过`git checkout -b dev origin/dev`切换到`dev`分支。

## 2. Git命令简介
`Git` 的命令总体上可以分为两类：

- 客户端命令：与仓库交互的命令，如`clone`，`add`，`commit`，`push`，`pull`等；
- 服务端命令：管理服务器上的仓库的命令，如`fetch`，`merge`，`revert`，`reset`等。

### 2.1 客户端命令
#### 2.1.1 clone 命令
克隆仓库：`git clone [url]`

语法：`git clone <repo> [<dir>]`

参数：
- `<repo>`：Git 服务器的地址，可以是 HTTP(S) 或者 SSH 协议的地址；
- `[dir]`：克隆到本地的仓库名称，默认为当前路径下的目录名称。

示例：
```bash
git clone https://github.com/username/repository.git
```

#### 2.1.2 init 命令
初始化仓库：`git init`

在指定目录创建一个新仓库。如果当前目录已有 `.git` 子目录，该命令将报错退出。

#### 2.1.3 config 命令
显示当前的全局设置：`git config --global --list` 或 `git config -l`

列出指定位置的设置：`git config --list [--local|--global|--system]`

设置配置选项的值：`git config <key> <value>`

删除配置选项：`git config --unset <key>`

#### 2.1.4 status 命令
检查当前文件状态：`git status [-s] [--long]`

参数：
- `-s`: 仅显示简短的状态信息。
- `--long`: 显示详细的状态信息。

示例：
```bash
$ git status
On branch master
Your branch is up to date with 'origin/master'.

nothing to commit, working tree clean
```

#### 2.1.5 diff 命令
显示两个差异的文件：`git diff <file1> <file2>`

显示所有变动的文件：`git diff`

显示当前缓存区与上次提交之间的差异：`git diff --cached`

显示暂存区与上次提交之间的差异：`git diff --staged`

参数：
- `--staged`：显示暂存区与上次提交之间的差异。
- `--cached`：显示暂存区与上次提交之间的差异。

示例：
```bash
$ git diff
diff --git a/README b/README
index 4cb5fb5..abccd2b 100644
--- a/README
+++ b/README
@@ -1 +1 @@
-Hello World!
+Welcome to my repository!
```

#### 2.1.6 add 命令
添加文件至暂存区：`git add <file>`

添加所有改动的文件至暂存区：`git add.`

参数：
- `--force`：强制添加文件至暂存区，即使这样做可能会覆盖掉手动修改过的文件。

示例：
```bash
$ git add README
```

#### 2.1.7 reset 命令
重置当前HEAD指向到指定的提交：`git reset <commit>`

参数：
- `--soft`：不改变工作区的内容，只是更新HEAD指向。
- `--mixed`：除了重置HEAD之外，其他工作目录的内容也会回退到前面一次`git commit`时的样子。
- `--hard`：除了重置HEAD指向之外，其他工作目录的所有内容都会被重置。
- `--keep`：保留相关文件的改动。

示例：
```bash
$ git reset HEAD~1
Unstaged changes after reset:
M       README.md
```

#### 2.1.8 commit 命令
提交修改记录：`git commit -m "message"`

参数：
- `-m`: 添加备注信息。
- `-a`: 将所有已经跟踪过的文件暂存起来一块提交。
- `--amend`: 修改最后一次提交。

示例：
```bash
$ git commit -m "Add new feature"
[master 1c0bfbb] Add new feature
 1 file changed, 1 insertion(+), 1 deletion(-)
```

#### 2.1.9 log 命令
显示提交日志：`git log [-p] [-n <num>] [--oneline]`

参数：
- `-p`: 显示每次提交的内容差异。
- `-n <num>`：显示最近的N个提交。
- `--oneline`: 以单行的方式显示提交日志。

示例：
```bash
$ git log --oneline -n 2
a21dd0e (HEAD -> master) update README.md
e1a23bc Initial commit
```

#### 2.1.10 show 命令
显示一个提交的信息：`git show <commit>`

参数：
- `<commit>`：提交ID或者分支名称。

示例：
```bash
$ git show e1a23bc
commit e1a23bc7f96f568a370591c5a7d42fd9e1f6d767
Author: username <<EMAIL>>
Date:   Sat May 16 17:18:25 2020 +0800

    Initial commit

diff --git a/.gitignore b/.gitignore
new file mode 100644
index 0000000..bca91b1
--- /dev/null
+++ b/.gitignore
@@ -0,0 +1 @@
+.DS_Store
\ No newline at end of file
```

#### 2.1.11 branch 命令
管理分支：`git branch <name>`

参数：
- `<name>`：新建分支名称。

示例：
```bash
$ git branch develop
Switched to a new branch 'develop'
```

#### 2.1.12 merge 命令
合并分支：`git merge <branch>`

参数：
- `<branch>`：被合并的分支名称。

示例：
```bash
$ git merge feature
Updating e1a23bc..8ccbaff
Fast-forward
 README | 2 ++
 1 file changed, 2 insertions(+)
 create mode 100644 test.txt
 ```

#### 2.1.13 rebase 命令
衍合提交：`git rebase <branch>`

参数：
- `<branch>`：基础分支。

示例：
```bash
$ git rebase master
First, rewinding head to replay your work on top of it...
Applying: initial commit
Using index info to reconstruct a base tree...
M       README.md
Falling back to patching base and 3-way merge...
Auto-merging README.md
CONFLICT (modify/delete): README.md deleted in HEAD and modified in BRANCH. Version BRANCH of README.md left in tree.
Patch failed at 0001 initial commit
The copy of the patch that failed is found in:.git/rebase-apply/patch
When you have resolved this problem, run "git rebase --continue".
If you prefer to skip this patch, run "git rebase --skip" instead.
To check out the original branch and stop rebasing, run "git rebase --abort".
```

#### 2.1.14 tag 命令
标记提交：`git tag <tagname>`

参数：
- `<tagname>`：标签名称。

示例：
```bash
$ git tag v1.0
```

#### 2.1.15 fetch 命令
下载远程分支的更新：`git fetch <remote>`

参数：
- `<remote>`：远程主机名称。

示例：
```bash
$ git fetch origin
From github.com:username/repository
   2024b21..a9cfdc2  master     -> origin/master
```

#### 2.1.16 pull 命令
取回远程仓库的变化，合并入当前分支：`git pull <remote> <branch>`

参数：
- `<remote>`：远程主机名称。
- `<branch>`：分支名称。

示例：
```bash
$ git pull origin master
From github.com:username/repository
   dafdece..666efad  master     -> origin/master
Auto-merging README.md
Merge made by the'recursive' strategy.
 README.md | 1 +
 1 file changed, 1 insertion(+)
```

#### 2.1.17 push 命令
上传本地分支的更新：`git push <remote> <branch>`

参数：
- `<remote>`：远程主机名称。
- `<branch>`：分支名称。

示例：
```bash
$ git push origin master
Counting objects: 12, done.
Delta compression using up to 4 threads.
Compressing objects: 100% (7/7), done.
Writing objects: 100% (12/12), 1.46 KiB | 0 bytes/s, done.
Total 12 (delta 4), reused 0 (delta 0)
remote: Resolving deltas: 100% (4/4), completed with 1 local object.
To github.com:username/repository.git
   2024b21..a9cfdc2  master -> master
```