
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


一般来说，一个开源项目都需要有README文件作为其文档。该文件主要用于向其他开发者介绍这个项目，让他们可以更好的了解到这个项目的功能、目的、使用方法等。而在实际操作中，往往会遇到添加多个文件到本地仓库的问题，也就是说，对于多文件修改，如何将其添加到仓库，尤其是在版本控制系统Git下。那么接下来就正式开始介绍一下关于Git命令行下如何将所有需要的文件(比如README文件)添加到本地仓库并提交版本的相关知识。
# 2.核心概念与联系
首先要明确一下Git的概念及其关联关系：

 - Git：分布式版本管理工具，是一个开源的版本控制系统。
 - Github：Github是一个面向开源及私有软件项目的托管平台，因为只支持Git作为唯一的版本库格式，故名为GitHub。
 - 本地仓库（Local Repository）：在用户自己的电脑上创建的一个Git工作目录，用来保存对文件进行版本控制。
 - 远程仓库（Remote Repository）：托管在远程服务器上的Git仓库，可以让多人协作开发某个项目。
 - 分支（Branch）：分支是Git的一个重要特性，允许多个同时存在的历史记录。每一个分支都有自己的一套独立的提交历史和提交点。

好了，知道这些概念后，下面进入正题。
# 3.核心算法原理和具体操作步骤以及数学模型公式详细讲解
## 3.1 概念阐述
想要将多个文件或文件夹添加到本地仓库并提交版本的话，就可以用git的`add`命令加文件/文件夹的路径作为参数来实现。下面以一个简单的例子来说明：假设有一个目录，其下包含两个文件：`a.txt`、`b.txt`，那么可以通过以下命令将它们添加到本地仓库并提交版本：

```
$ git status # 查看当前状态
On branch master
Untracked files:
  (use "git add <file>..." to include in what will be committed)

        a.txt
        b.txt

nothing added to commit but untracked files present (use "git add" to track)

$ git add. # 添加文件到暂存区
$ git status # 查看当前状态
On branch master
Changes to be committed:
  (use "git reset HEAD <file>..." to unstage)

        new file:   a.txt
        new file:   b.txt

$ git commit -m "add two files" # 提交文件到版本库
[master d9f1c7d] add two files
 2 files changed, 0 insertions(+), 0 deletions(-)
 create mode 100644 a.txt
 create mode 100644 b.txt
```

这样做有一个缺点就是每次都要手动输入文件路径。如果目录下文件较多，或者需要一次性添加多个文件的话，这种方式就显得非常麻烦了。所以，Git提供了一种更简便的方法来解决这一问题——可以使用`*`通配符来匹配指定的文件。如下例所示：

```
$ ls
README.md	hello_world.py
index.html	test.java

$ git status
On branch master
Untracked files:
  (use "git add <file>..." to include in what will be committed)

	README.md

nothing added to commit but untracked files present (use "git add" to track)

$ git add *
$ git status 
On branch master
Changes to be committed:
  (use "git reset HEAD <file>..." to unstage)

	new file:   index.html
	new file:   test.java
	modified:   README.md

$ git commit -m "add all files"
[master fb3d7ce] add all files
 3 files changed, 0 insertions(+), 0 deletions(-)
 create mode 100644 index.html
 create mode 100644 test.java
1 files changed, 0 insertions(+), 0 deletions(-)
create mode 100644 README.md
```

通过`*`通配符，Git可以自动识别出当前目录下所有的未跟踪的文件，并将它们都加入到暂存区。另外，如果没有新文件被添加到暂存区，那么也不会有提示信息。

除此之外，还可以使用`-A`选项来添加所有文件，包括已跟踪文件和未跟踪文件。

```
$ git add -A
$ git status
On branch master
All changes tracked by repository
```

`add`命令可以在Git的工作流程中随时调用，也可以结合别的命令一起使用，例如`commit`命令。所以，熟练掌握`add`命令，对于日常版本控制工作十分有益。

## 3.2 优化建议

除了`add`命令本身的一些特性外，还有很多情况下，用命令的方式添加文件并不是最佳选择。这里提供一些优化的建议：

- 使用`.gitignore`文件排除不需要添加到本地仓库的文件：Git中的`.gitignore`文件可以帮助我们从本地仓库中忽略掉一些不需要追踪的文件或文件夹。例如，我们在工程根目录下创建一个`.gitignore`文件，内容如下：

  ```
  *.log
  build/
  dist/
  temp/*
  ```

  表示忽略掉当前目录下的所有日志文件（如`*.log`），`build/`和`dist/`目录以及`temp/`目录下的所有文件。这样当我们执行`git add.`命令的时候，这三个文件夹下的内容就会被自动忽略掉。当然，我们也可以用`.gitignore`文件的语法规则来排除特定的文件或文件夹。

- 用提交之前的`status`命令确认所有文件都已经被正确地添加到暂存区：除了确认文件是否都被添加到暂存区外，`status`命令还可以显示每个文件状态的信息，包括文件修改、新增和删除信息。

- 在执行`commit`命令之前检查输出结果：提交前再次运行`status`命令，确认是否还有未提交的修改，防止提交意外覆盖掉了文件。