
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Ruby（瑞士）又称灰姬（Greek）、红宝石（Colored ruby）、红玉（Red sapphire），是一个高级动态语言，它的解释器也称为MRI (Matz's Ruby Interpreter)，由日本的菅田将之命名，于1995年诞生。它最初被设计用来作为一种脚本语言来进行快速开发，其语法简洁而简单，可读性强，支持多种编程范式，在许多方面都有着独特的特性。如其中的动态语言特性使其具有强大的函数式编程能力，适用于各种场景下的应用。作为一种多用途语言，Ruby提供了丰富的类库和工具，可以支撑多种应用程序的开发，并且广泛应用于web应用、系统管理、网络开发等领域。
本教程旨在帮助读者完成Ruby编程环境的安装配置过程。对于刚接触Ruby编程语言的新手来说，掌握该语言的基本语法和开发技巧非常有必要。通过阅读本教程，读者将了解到如何安装和配置Ruby及其相关的开发工具，从而可以方便地进行Ruby编程。


# 2. 安装Ruby

## Windows平台

在安装DevKit后，打开CMD命令窗口并输入`ridk install`，等待DevKit安装完成。

## Linux平台
Linux用户可以直接使用系统自带的包管理工具进行安装，比如apt或yum。如果你还没有安装过，可以使用以下命令安装Ruby：
```bash
sudo apt-get update
sudo apt-get install -y ruby-full build-essential zlib1g-dev
```
其中，`-y`参数表示自动回答yes。安装完成后，可以通过`ruby -v`命令检查是否安装成功。

## MacOS平台
MacOS用户可以使用brew安装Ruby：
```bash
brew install ruby
```
之后，就可以在终端执行`ruby -v`命令查看版本号了。

# 3. 配置Ruby环境变量
安装完毕后，需要配置环境变量，让命令行可以在任何地方找到Ruby。

## Windows平台
编辑注册表，在`HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\Session Manager\Environment`下创建新的字符串值。变量名为`PATH`，值为`<ruby安装目录>\bin`。例如：
```
C:\Program Files\Ruby27-x64\bin
```
注意，双斜杠代表当前盘符根目录。保存后，重启电脑或者重新登录，命令行就能够正常调用Ruby了。

## Linux平台
编辑`.bashrc`文件，添加如下两行：
```bash
export PATH=$PATH:<ruby安装目录>/bin
source ~/.bashrc
```
保存后，运行`source.bashrc`使设置立即生效。

## MacOS平台
同样是编辑`.zshrc`文件，添加如下两行：
```bash
export PATH=$PATH:/usr/local/opt/ruby/bin:$HOME/.rbenv/shims
eval "$(rbenv init -)"
```
保存后，运行`source.zshrc`使设置立即生效。

# 4. 安装RubyGems
RubyGems是一个Ruby包管理工具，类似于Python的pip或Node的npm。可以帮助我们快速安装和管理Ruby第三方库。

RubyGems提供了一个简单的命令行接口来安装和管理 gems。如果你已经安装了Ruby，那么安装RubyGems也只需一条命令：
```bash
gem install rails
```
上面的命令会安装Rails框架。如果无法连接外网，可以使用国内镜像源加速安装速度。比如：
```bash
gem sources --add https://gems.ruby-china.com/ --remove https://rubygems.org/
gem sources -l # 查看 gem 源列表
```
# 5. 安装集成开发环境(IDE)
为了方便地进行Ruby编程，通常都需要一个集成开发环境(Integrated Development Environment，IDE)。目前主流的Ruby IDE有两种：RubyMine和Ruby BluePrint。这两种IDE都是免费的，可以从各网站下载安装。

## RubyMine
RubyMine是JetBrains公司推出的一款Ruby IDE，功能包括代码编辑、语法高亮、代码自动完成、调试、版本控制、单元测试、数据库管理等，同时也内置了很多有用的插件，比如Rails、RSpec、Rubocop等。它支持Windows、Mac OS X、Ubuntu、CentOS等多个平台。

## Ruby BluePrint
Ruby BluePrint是基于Eclipse平台的一个Ruby IDE，功能包括代码编辑、语法高亮、代码自动完成、调试、版本控制、单元测试、数据库管理等，同时也内置了很多有用的插件，比如Rails、RSpec、Rubocop等。它支持Windows、Mac OS X等平台。

# 6. Hello World!
新建一个文本文档，输入以下代码：
```ruby
puts "Hello world!"
```
保存为`hello.rb`，然后在命令行里进入该目录，输入`ruby hello.rb`运行程序。如果一切顺利，你应该看到屏幕输出"Hello world!"。