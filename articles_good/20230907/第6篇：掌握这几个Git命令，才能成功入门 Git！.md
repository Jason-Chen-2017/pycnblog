
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 概述
- Git 是目前最流行的版本控制系统之一；
- 在工作中频繁地使用 Git 是一个非常有益的习惯，因为它可以帮助我们记录每次的修改、提交信息等；
- 如果你想学习 Git 的基本用法，那这篇文章就能帮到你！

## 作者简介
- 曾任职于腾讯科技公司的运维工程师，曾参与过多家企业级项目的开发；
- 技术栈：Python + Django + Nginx + MySQL + Redis；
- 个人爱好：摄影，篮球，听音乐，跑步；

本文将从如下几个方面介绍 Git 的基础知识：
1. Git 的安装配置；
2. Git 的基本用法；
3. 分支管理；
4. Github 和 Gitlab 使用方法；
5. Git hooks 自动化脚本；
6. Git 命令速查表。
# 2.环境准备
首先，你需要确认自己的电脑上是否安装了 Git，如果没有，你可以通过如下链接进行下载安装：https://git-scm.com/downloads 。下载完成后，打开终端（或命令行窗口），输入 git --version ，如果显示出版本号则说明安装成功。
```
$ git --version
git version 2.23.0
```
接着，你需要创建一个仓库文件夹并进入该文件夹，在此文件夹下执行初始化命令 `git init` 初始化一个新的 Git 仓库。
```
$ mkdir my_project && cd my_project
$ git init # 初始化一个 Git 仓库
Initialized empty Git repository in /Users/chenchaoyang/my_project/.git/
```
# 3.基本用法
## 配置 Git 用户名和邮箱地址
执行以下命令设置你的用户名和邮箱地址：
```
$ git config --global user.name "your name"
$ git config --global user.email your@email.address
```
当全局配置完成后，再次执行 `git commit`，就会记录你所使用的用户名和邮箱地址。
```
$ git commit -m 'first commit' # 提交代码示例
[master (root-commit) c739d1c] first commit
 1 file changed, 1 insertion(+)
 create mode 100644 test.txt
```
## 添加文件至暂存区
执行 `git add filename` 将指定的文件添加至暂存区。
```
$ touch index.html
$ git add index.html # 添加文件至暂存区
```
## 提交代码到本地仓库
执行 `git commit -m 'description'` 将文件提交到本地仓库。
```
$ git commit -m 'add index.html' # 提交代码示例
[master dff71e6] add index.html
 1 file changed, 0 insertions(+), 0 deletions(-)
 create mode 100644 index.html
```
注意：每次提交时，都会生成一个唯一的哈希值作为标识，可以用来查看每一次提交的内容变化。
## 查看文件状态
执行 `git status` 可以查看当前仓库文件的状态，包括已经跟踪的文件、被改动的文件、暂存区中的文件。
```
$ git status
On branch master
Your branch is up to date with 'origin/master'.
Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        modified:   index.html

no changes added to commit (use "git add" and/or "git commit -a")
```
## 查看历史记录
执行 `git log` 或 `git reflog` 可以查看提交历史记录，其中 `reflog` 显示所有的历史记录，包括已经被删除的提交记录。
```
$ git log
commit dff71e6dd5ed6b5cf8db46b46386c8f7be3561ac (HEAD -> master)
Author: chenchaoyang <<EMAIL>>
Date:   Fri Jul 2 10:18:46 2021 +0800

    add index.html

commit a0990e63dc2f294ba94cb5a509d0e45f2f521e3c
Author: chenchaoyang <<EMAIL>>
Date:   Thu Jun 30 16:31:27 2021 +0800

    initial commit
```
## 撤销操作
### 撤销暂存区文件
执行 `git reset HEAD filename` 可以将暂存区中的文件撤销。
```
$ git checkout -- index.html # 从暂存区撤销文件更改
```
### 撤销修改文件
执行 `git checkout -- filename` 可以将工作区中的文件恢复至最近一次 `git add` 后的状态。
```
$ echo "hello world" > hello.py
$ git add hello.py
$ rm index.html
$ git status
On branch master
Your branch is ahead of 'origin/master' by 1 commit.
  (use "git push" to publish your local commits)

Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        new file:   hello.py

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        deleted:    index.html

$ git checkout -- hello.py # 撤销对 hello.py 的修改
$ ls
hello.py index.html my_project.iml.gitattributes.gitignore README.md
$ git status
On branch master
Your branch is ahead of 'origin/master' by 1 commit.
  (use "git push" to publish your local commits)

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        my_project.iml

```
## 删除文件
执行 `git rm filename` 可以将文件从仓库中删除，同时也会删除工作区和暂存区中的该文件。
```
$ rm hello.py
$ git rm hello.py
rm 'hello.py'
$ git status
On branch master
Your branch is ahead of 'origin/master' by 1 commit.
  (use "git push" to publish your local commits)

Changes to be committed:
  (use "git restore --staged <file>..." to unstage)
        deleted:    hello.py

Changes not staged for commit:
  (use "git add <file>..." to update what will be committed)
  (use "git restore <file>..." to discard changes in working directory)
        deleted:    index.html
```
# 4.分支管理
## 创建分支
执行 `git branch <branch>` 可以创建新分支。
```
$ git branch dev # 创建新分支
$ git branch
  dev
* master
```
执行 `git checkout <branch>` 可以切换到指定分支。
```
$ git checkout dev # 切换到 dev 分支
Switched to branch 'dev'
```
## 拉取分支
执行 `git pull origin <branch>` 可以拉取指定分支上的最新代码。
```
$ git pull origin dev
remote: Enumerating objects: 3, done.
remote: Counting objects: 100% (3/3), done.
remote: Total 3 (delta 0), reused 0 (delta 0), pack-reused 0
Unpacking objects: 100% (3/3), done.
From https://github.com/username/repository
   b08b4fe..dff71e6  dev     -> origin/dev
Updating b08b4fe..dff71e6
Fast-forward
 index.html | 0
 1 file changed, 0 insertions(+), 0 deletions(-)
 create mode 100644 index.html
```
## 提交分支
执行 `git merge <branch>` 可以合并指定分支到当前分支。
```
$ git merge dev # 合并 dev 分支到当前分支
Merge made by the'recursive' strategy.
 index.html | 0
 1 file changed, 0 insertions(+), 0 deletions(-)
 create mode 100644 index.html
```
## 删除分支
执行 `git branch -D <branch>` 可以强制删除指定分支。
```
$ git branch -D dev # 删除 dev 分支
Deleted branch dev (was dff71e6).
```
# 5.Github 和 Gitlab 使用方法
## 在 Github 上创建一个仓库
如果你还没有 Github 账号，可以访问 https://github.com/join?source=header-home 注册一个免费的账号。然后登录 Github 主页，点击右上角头像旁边的加号新建一个仓库：
## 在 Github 中克隆仓库
克隆一个仓库可以使用以下命令：
```
$ git clone https://github.com/username/repository.git
```
这样就可以把远程仓库 Clone 到本地。
## 在 Gitlab 上创建一个仓库
如果你还没有 Gitlab 账号，可以访问 http://gitlab.com 注册一个免费的账号。然后登录 Gitlab 主页，点击左侧导航栏的 Projects --> Create project 创建一个新的项目：
## 在 Gitlab 中克隆仓库
克隆一个仓库可以使用以下命令：
```
$ git clone ssh://git@gitlab.com/username/repository.git
```
这样就可以把远程仓库 Clone 到本地。
# 6.Git hooks 自动化脚本
Git 可以自定义脚本，在特定的事件发生时触发相应的脚本命令。这些脚本通常用于自动化编译、部署等流程。比如，当某个分支的代码被 Push 时，可以自动运行测试脚本。
## 设置 Git hook
Git 通过 `.git/hooks/` 目录下的特定脚本文件实现钩子机制。下面是一个例子：
```bash
#!/bin/sh
echo "Running pre-push hook..."
./test.sh
if [ $? -eq 0 ]
then
    echo "Push succeeded!"
    exit 0
else
    echo "Push failed." >&2
    exit 1
fi
```
这个脚本在 Push 操作之前会检查 test.sh 文件是否存在，并且运行该脚本。如果脚本返回 0，说明 Push 可以继续执行，否则失败退出。

为了让 Git 执行这个脚本，我们需要在 `.git/hooks/` 下面创建一个名为 `pre-push` 的文件，并将上面脚本写入其中。
```bash
touch.git/hooks/pre-push
chmod +x.git/hooks/pre-push
vi.git/hooks/pre-push # 写入脚本内容
```
执行 `git push` 命令时，会调用该脚本，根据脚本输出结果决定是否执行 Push 操作。
```bash
$ git push
Counting objects: 3, done.
Delta compression using up to 8 threads.
Compressing objects: 100% (2/2), done.
Writing objects: 100% (3/3), 292 bytes | 292.00 KiB/s, done.
Total 3 (delta 0), reused 0 (delta 0)
remote: Running pre-push hook...
To https://github.com/username/repository.git
 * [new branch]      feature1 -> feature1
```