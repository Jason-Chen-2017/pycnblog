
作者：禅与计算机程序设计艺术                    

# 1.简介
  
 
GitHub是一个开源的版本控制平台，它可以帮助你管理项目中的文件并与他人协作。在实际工作中，项目经常需要多人合作开发代码，而GitHub为这一过程提供了很好的工具支持。

本文将从以下三个方面阐述如何创建新的仓库:

1) 创建一个本地的Git仓库；
2）连接到远程GitHub仓库；
3）推送到远程GitHub仓库。


# 2. Git 相关概念
## 2.1 Git 命令
Git是一个分布式版本控制系统（DVCS），通过git命令，你可以对代码进行版本控制、历史查看等。一些常用的Git命令如下表所示：

| 命令 | 描述 |
|:----:|:----|
| git init    | 初始化一个新库 |
| git add     | 添加文件至暂存区 |
| git commit  | 提交暂存区的文件到本地仓库 |
| git clone   | 从远程克隆一个仓库到本地 |
| git push    | 把本地仓库的内容推送到远程仓库 |
| git pull    | 从远程仓库拉取内容到本地 |
| git log     | 查看提交日志 |
| git status  | 查看仓库当前状态 |
| git diff    | 比较工作区和暂存区之间或HEAD指针指向的commit之间的差异 |


## 2.2 Git工作流
一般情况下，Git的工作流程是：

- 在本地创建一个Git仓库或者克隆一个已有的仓库；
- 通过修改、提交操作将文件添加到暂存区；
- 当暂存区的文件达到一定数量时，执行一次提交操作，将这些文件的快照合并入HEAD指针指向的分支，并生成一个commit对象；
- 如果修改了远程仓库的文件，可以通过pull命令从远程仓库拉取最新版本到本地仓库；
- 要分享自己的代码，可以在本地仓库push到远程仓库即可。

以上工作流图展示了Git基本工作流程：


# 3. 创建本地Git仓库

然后，打开命令提示符或者终端，切换到你想放置Git仓库的目录下，输入如下命令：

```
mkdir myproject && cd myproject
```

新建文件夹`myproject`，并进入该文件夹。

接着，初始化一个Git仓库：

```
git init
```

此时，会在当前目录下生成一个`.git`隐藏目录，里面包含了所有Git管理的信息。

# 4. 连接到远程GitHub仓库
如果你还没有GitHub账号，请先注册一个免费的账号，并设置好你的用户名和邮箱。

登录GitHub后，点击右上角的“+”号，选择“New repository”，按照提示一步步填写仓库信息。


完成之后，点击"Create Repository"按钮完成仓库创建。

为了方便后续操作，记住"Repository name"和"Remote URL"的值，如下图所示：


现在，我们把本地仓库与远程仓库连接起来，先在本地仓库配置一下SSH key。

## 配置SSH Key

在Windows下打开Git Bash，运行命令

```
ssh-keygen -t rsa -C "youremail@gmail.com"
```

一路回车，默认路径保存公钥和私钥文件，文件名分别为id_rsa.pub 和 id_rsa。

如果不想用GitHub账号密码的方式访问GitHub，那么就需要把SSH Key添加到GitHub上。

## 将SSH Key添加到GitHub
进入GitHub主页，点击右上角头像-->Settings-->SSH and GPG keys-->New SSH key。

将之前生成的id_rsa.pub文件的内容复制粘贴进去，Title随便起个名字，如"mypc-key"，然后点击Add SSH key。


这样，GitHub上的SSH Key就成功添加到账户里了，无需输入密码即可正常拉取和推送代码。

## 关联远程仓库

现在，我们已经配置好SSH Key，并且GitHub上的远程仓库也创建好了，可以尝试把本地仓库与远程仓库关联起来。

回到本地仓库，运行如下命令：

```
git remote add origin git@github.com:username/repositoryname.git
```

其中，origin 是本地仓库对远程仓库的唯一标识，可以自定义；username 为你的 GitHub 用户名，repositoryname 为你刚才创建的远程仓库名称。

运行完这个命令后，可以运行 `git remote -v` 命令查看当前仓库的远程仓库情况。输出结果应该类似于：

```
origin	git<EMAIL>:username/repositoryname.git (fetch)
origin	git@github.com:username/repositoryname.git (push)
```

表示本地仓库的 origin 链接到了远程仓库的 url。

# 5. 推送到远程GitHub仓库
既然本地仓库和远程仓库已经关联好了，那就可以直接向远程仓库推送代码了。

运行如下命令：

```
git push -u origin master
```

其中，master 表示你要推送的分支，你可以在 GitHub 页面看到分支列表，确认后替换成相应的分支名称。

第一次推送的时候，你需要加上 `-u` 参数，这样就可以省略 `--set-upstream` 参数了，后面的推送只需要简单的 `git push` 即可。

当你第一次运行这个命令时，你可能需要输入 GitHub 的用户名和密码，确认是否允许授权。

如果一切顺利的话，此时远程仓库的内容应该和本地仓库一致了。

# 6. 小结

本文主要介绍了如何创建本地Git仓库、连接到远程GitHub仓库、推送到远程GitHub仓库，并给出了一个Git基本工作流程。