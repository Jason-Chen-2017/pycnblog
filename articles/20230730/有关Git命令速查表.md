
作者：禅与计算机程序设计艺术                    

# 1.简介
         
1. Git是一个开源的分布式版本控制系统（DVCS），用于快速跟踪文件内容的变化并记录每次更新。通过git命令，可以对本地仓库进行版本管理、协作开发、提交记录等操作。由于其独特的分支、标签、合并、撤销等特性，以及强大的跨平台能力，越来越多的公司开始使用git进行版本管理和代码协作。这也使得git成为最流行的版本管理工具之一。 
         2. 本文整理了常用git命令的相关知识，帮助你快速掌握git的相关功能，提高工作效率。
         # 2. 相关概念
         1. 版本库 (repository)：用来存放项目代码及资源文件的地方，可以理解成一个目录，该目录中会有一个隐藏文件夹.git ，该文件夹里存储着git所有的数据。
         2. 分支 (branch)：每个版本库里都可以有多个分支，相当于一个开发人员或者团队在不同任务上的同时工作的地方。可以将不同的分支合并到一起，实现多人协作开发。
         3. 暂存区 (stage/index): 在你执行git add 命令后，文件就被放在暂存区中等待提交。
         4. HEAD：指向当前所在的分支，在没有任何分支的时候，HEAD指向的是master分支，即主分支。
         5. 提交 (commit)：将暂存区中的文件保存到当前分支的一个特定版本中。
         6. 克隆 (clone)：从远程版本库中拷贝一份到本地。
         7. 拉取 (pull)：获取远程版本库上最新代码并自动合并。
         8. 投递 (push)：将本地代码推送到远程版本库。
         9. 标签 (tag)：给某一特定版本打上标签，以方便后续查看。
         10. 分支合并 (merge)：把不同分支的代码合并到同一分支中去。
         11. 冲突 (conflict)：两个或多个分支因为某些原因不能自动合并，而需要手动解决冲突。
         12. 工作区 (working directory)：在你的电脑上看到的文件目录，即你正在编辑的文件。
         13. 撤销 (revert)：撤销一些操作。
         14. stash: 将当前工作进度暂时存放在堆栈内，方便随时恢复。
         # 3. Git配置
         ```bash
         git config --global user.name "your name"   # 设置用户名
         git config --global user.email "your email address"    # 设置邮箱地址

         # 设置文本编辑器，可选vim 或 nano，默认为vim
         git config --global core.editor vim

         # 查看Git配置信息
         git config --list
         ```
         # 4. 初始化仓库
         ## 创建新仓库
        ```bash
        mkdir myproject     # 创建项目目录
        cd myproject        # 进入项目目录
        git init            # 初始化仓库
        touch README.md     # 创建README文件
        git add README.md   # 添加文件到暂存区
        git commit -m "first commit"  # 提交文件到仓库

        echo "# MyProject" > README.md      # 修改README文件
        git status                          # 查看状态
        git diff                            # 查看修改内容
        git add README.md                   # 添加修改文件到暂存区
        git commit -m "update readme file"  # 提交修改到仓库
        ```
        ## 从已有仓库克隆
        ```bash
        git clone https://github.com/username/myrepo.git      # 从Github克隆仓库
        git clone ssh://user@domain.tld/path/to/repo.git     # 使用SSH协议克隆仓库
        ```
        # 5. 工作流
        ## 分支管理
        ### 创建分支
        ```bash
        git branch dev           # 创建名为dev的分支
        git checkout dev         # 切换到dev分支
        ```
        ### 删除分支
        ```bash
        git branch -d dev       # 删除dev分支
        ```
        ### 合并分支
        ```bash
        git merge master          # 将master分支合并到当前分支
        git branch --merged       # 查看已经合并过的分支
        ```
        ### 列出分支
        ```bash
        git branch             # 查看所有分支
        git branch -a          # 查看包括已删除分支在内的所有分支
        ```
        ### 重命名分支
        ```bash
        git branch -m old_name new_name      # 重命名当前分支
        ```
        ### 同步分支
        如果在远程版本库创建了新的分支，则需要先拉取才能查看到。如下所示：
        ```bash
        git fetch origin              # 获取远程版本库信息
        git branch --remotes           # 查看远程分支
        git checkout -b local_branch origin/remote_branch  # 新建本地分支
        git push origin local_branch               # 推送本地分支到远程分支
        ```
        ## 修订历史
        ### 显示提交历史
        ```bash
        git log                    # 查看提交历史，按时间排序
        git log -p                 # 查看提交历史，按时间排序并显示详细信息
        git reflog                 # 查看所有的提交记录，包括已经删除的提交记录
        ```
        ### 查找更改内容
        ```bash
        git show <commit-id>       # 查看指定提交记录的详情
        git diff <file>            # 比较工作区和暂存区的差异，或比较两次提交之间的差异
        ```
        ### 撤销操作
        #### 撤销工作区更改
        ```bash
        git checkout -- <file>       # 丢弃工作区的改动，回到最近一次git add时的状态
        ```
        #### 撤销暂存区更改
        ```bash
        git reset HEAD <file>        # 恢复暂存区，让文件处于未暂存状态
        git checkout -- <file>       # 撤销所有已缓存的改动，如未add到暂存区，需要先添加再checkout
        ```
        #### 撤销提交记录
        ```bash
        git revert <commit-id>        # 撤销指定提交记录，创建一个新的提交记录，并撤销掉指定的提交记录
        git reset HEAD^              # 取消最后一次提交(谨慎使用！)
        ```
        ### 其他常用命令
        ```bash
        git clean [-f]                # 清理不必要的未跟踪文件和忽略文件
        git remote [-v]               # 查看远程版本库信息
        git remote add origin <url>   # 添加远程版本库
        git pull                      # 获取远程版本库最新代码并自动合并
        git push                      # 将本地代码推送到远程版本库
        git tag                       # 列出所有标签
        git tag v1.0                  # 为当前提交打上标签v1.0
        git tag -a v1.1 -m "version 1.1 release"  # 给已发布的提交打上标签
        git push origin v1.1         # 推送标签到远程版本库
        ```
        # 6. 附录
        ## 常见问题
        ### 如何查看远程分支？
        可以通过以下命令查看：
        ```bash
        git ls-remote <remote_name>  # 查看远程仓库的分支列表，类似于svn list
        ```
        ### 是否可以上传空文件夹？
        不可以。只要文件存在于项目中，即使为空文件夹也是需要提交的。
        ### 当我push时提示“error: src refspec master does not match any”该怎么办？
        这是由于本地分支名字和远程分支名字不一致造成的。通常情况下，本地分支名默认是"master",远程分支名默认是"origin/master"。可以通过以下命令查看本地分支名称：
        ```bash
        git branch
        ```
        通过以下命令设置本地分支名称和远程分支名称一致：
        ```bash
        git branch -u origin/master master
        ```
        然后重新push即可。