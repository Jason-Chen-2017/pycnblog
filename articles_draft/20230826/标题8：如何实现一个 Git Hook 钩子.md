
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 1.1 Git 是什么？
Git是一个开源的分布式版本控制系统（Distributed Version Control System），可以有效、高速地处理从很小到非常大的项目版本管理。
## 1.2 为什么要用 Git Hook 钩子？
我们经常在 Git 中提交代码时需要遵守一些规范，比如代码风格检查、代码编译等。如果每次都手动去跑这些工具的话效率太低了，所以就诞生了 Git Hook。通过 Git Hook 可以在特定的动作发生时自动执行脚本或者程序，这样就可以完成相应的工作流程自动化。
## 1.3 什么是 Git Hook？
Git Hook 是一种脚本文件，它是在特定事件发生后触发运行的一段代码，用于自动化处理工作流。目前常用的 Git Hook 有 pre-commit 和 post-commit 。pre-commit 在执行 git commit 命令之前执行，目的是检查即将提交的代码是否符合预期。post-commit 在执行完 git commit 命令之后执行，通常用来通知团队其他成员或更新远程仓库。

除了上述两个标准 Git Hook 外，还可以自定义其它类型的 Git hook 来满足特殊需求。例如 pre-push 可以在 push 操作之前进行检查，确保没有对远程分支做过不可逆操作。
## 1.4 为什么要使用 Git Hook ？
为了实现 Git Hook 的自动化处理，主要有以下几个原因：
* 提升开发者体验：通过 Git Hook 可以自动提示开发者提交代码时的错误信息，并帮助改善编码规范；
* 统一工作流程：通过 Git Hook 可以使不同的开发者在提交代码时达成一致的工作流程；
* 节约重复劳动：通过 Git Hook 可以减少重复性工作，避免出现无意义的人工操作，提高工作效率。

# 2.基本概念术语说明
## 2.1 Hook类型
目前支持三种类型的 Git Hook：客户端 Hooks、服务端 Hooks 和供应商 Hooks。
### 2.1.1 客户端 Hooks
客户端 Hooks 是安装在客户端上的脚本，它们不能访问私密数据。如 pre-commit 和 pre-push 就是客户端 Hooks 。

客户端 Hooks 只能在本地运行，不依赖于服务器环境。也就是说只要克隆到本地，就可以直接应用到提交的各个阶段。
### 2.1.2 服务端 Hooks
服务端 Hooks 也称为 githooks ，是在 Git 服务端上运行的脚本。他们可以访问 Git 服务端的所有配置和数据。服务端 Hooks 可以针对每个用户或组织提供独特的功能，而且会受到全局配置影响。

### 2.1.3 供应商 Hooks
供应商 Hooks 是由第三方开发者创建的扩展 Git 服务端的功能。它们可以扩展 Git 的功能，或者提供额外的存储或跟踪功能。像 Gitlab 的 Webhooks 就是供应商 Hooks 。

## 2.2 配置文件
Git 使用配置文件.git/hooks 下的 pre-commit、commit-msg 和 post-commit 文件夹来存放 Git Hooks 。


### 2.2.1 客户端 Hooks
客户端 Hooks 只能在本地运行，不会访问 Git 服务端。所以需要把这些 Hooks 拷贝到工作目录下，然后通过以下命令执行：
```bash
$ chmod +x.git/hooks/*   # 执行权限
$.git/hooks/*          # 执行 Hook 脚本
```

### 2.2.2 服务端 Hooks
服务端 Hooks 必须安装在 Git 服务器上，才能在提交代码时得到执行。因此，服务端 Hooks 的安装方式有两种：
#### 2.2.2.1 全局安装
全局安装的服务端 Hooks 会被所有用户共享。在 Linux 上可以通过以下命令安装：
```bash
sudo cp hooks /etc/git/    # 把 Hooks 复制到 Git 目录
sudo chown root:root /etc/git/ -R     # 修改所有权
chmod og-rwx /etc/git/ -R            # 设置文件夹的访问权限
```
然后，修改 Git 服务端的配置文件：
```bash
vi /etc/gitconfig                # 添加以下两行
[receive]
    denyCurrentBranch = ignore
[sendemail]
    smtpUser = your@smtpserver.com
    smtpServer = smtp.yourdomain.com
    from = your@email.address
```

>denyCurrentBranch = ignore 表示忽略分支冲突，因为 Git 不允许推送当前分支。

#### 2.2.2.2 用户级安装
用户级安装的服务端 Hooks 只会对当前用户有效。在 Linux 上可以通过以下命令安装：
```bash
cp hooks ~/.git/           # 把 Hooks 复制到用户目录
echo "export PATH=$PATH:$HOME/.local/bin" >> ~/.bashrc      # 添加环境变量
source ~/.bashrc        # 刷新环境变量
```
然后，测试一下是否安装成功：
```bash
$ echo 'Hello, World!' > test.txt         # 创建测试文件
$ git add test.txt                   # 将测试文件添加到暂存区
$ git commit -m "Add a new file."     # 提交代码
```
此时会看到你的 Git 邮箱收到了提交成功的消息。

## 2.3 Hooks 示例
上面已经介绍了客户端 Hooks、服务端 Hooks 和供应商 Hooks 的基本概念。接下来我们看一下几个 Hooks 的例子。
### 2.3.1 pre-commit
pre-commit 钩子在执行 `git commit` 命令时，默认情况下会运行。如果该命令返回非零值，则提交过程会停止，所有的更改都不会被记录到暂存区和分支中。

我们可以在 `.git/hooks/pre-commit` 文件中编写一些脚本，当执行 `git commit` 时，这些脚本就会运行。

举例如下：
```bash
#!/bin/sh
if [ $(git diff --name-only | wc -l) -gt 0 ]; then
  echo "Only allow adding files to the index." >&2
  exit 1
fi
exit 0
```

这个脚本会检查暂存区中的文件列表，只允许添加文件到索引，而不允许删除、移动、重命名等操作。

### 2.3.2 prepare-commit-msg
prepare-commit-msg 钩子在执行 `git commit` 命令时调用一次，并且可以传递一个参数给脚本。

我们可以在 `.git/hooks/prepare-commit-msg` 文件中写入脚本，向提交信息添加一些内容。

例如，我们可以在提交信息中添加 ticket id 号，这样就不需要记忆 branch name 和 commit message 了。

```bash
#!/bin/sh
TICKET=$(git log -n 1 --pretty=format:"%ct %H" "$1")
sed -i "/^$/d; s/^/$TICKET\n/" "$2"       # Add ticket information to commit message
```

这个脚本会检查前一次提交的 commit message，并且追加 ticket id 号到其开头。注意这里的 `$1` 参数表示最新的提交对象，`$2` 表示刚才编辑的提交信息文件的路径。