
作者：禅与计算机程序设计艺术                    

# 1.简介
  

正则表达式（Regular Expression）是一个用来匹配字符串的模式。它由若干普通字符（例如，a或b），特殊字符（例如，*,., ^等），还有各种括号、竖线、角括号组成。这些符号一起构成一个复杂的模式，能帮助我们在文本中快速查找、替换、剪切、统计和处理特定内容。
本文将介绍Linux命令行下经常使用的正则表达式，包括grep、sed、awk及其它相关工具的用法。同时，还会结合实际案例，讲述正则表达式的一些常用技巧。希望读者能够从中受益。
# 2.什么是正则表达式？
正则表达式（Regular Expression）全称“正则化表达式”，是一种文本匹配的规则或者语法，它可以帮助你在大量文字中搜索、查找特定的单词或字符。它定义了一系列符合某个模式的字符串。正则表达式通常用于文本处理的自动化、搜索和替换功能。通过使用不同的正则表达式语言，你可以精确地指定要搜索的目标，从而有效地节省时间和降低错误。

简单来说，正则表达式就是一个文本匹配的规则或者语法，它使得你可以快速找到文本中符合自己要求的字符串。但正则表达式并不是一门独立的编程语言，而是一门用来描述正则表达式语法的语言。很多其他程序都支持使用正则表达式进行文本处理。比如grep命令（全称Global Regular Expression Print）就是利用正则表达式在文件中查找指定的字符串。

举个例子：假如你想在一篇文章里查找出所有的日期（如"2019-06-07"或"June 7th, 2019"）这样的字符串，该怎么做呢？首先，我们需要确定这个日期的格式，即"YYYY-MM-DD"或"MMMM D, YYYY"。然后，我们可以使用正则表达式对日期进行匹配。

再举个例子：假如你正在编写一个脚本，需要把文本中所有连续重复的字符替换成一个字符（如把"hellooo world!"替换成"hwo!")。该怎么做呢？我们可以使用正则表达式来完成这一工作。

以上仅仅是两个简单的例子，正则表达式可以用于很多复杂的任务，包括电子邮件地址的验证，网页正则解析，密码强度校验等等。正则表达式是一个非常强大的工具，掌握它的用法是提高效率、准确性、可靠性的一项重要能力。

# 3.Linux下的正则表达式工具
下面介绍Linux环境下最常用的几款基于正则表达式的命令行工具。

1) grep 命令
grep（global regular expression print，全局正则表达式打印）是一个在Unix/Linux系统上查找文件里面的匹配行的一个命令。它可以配合正则表达式让我们非常方便地定位想要找的东西。

命令格式如下：
```bash
grep [options] pattern file
```

参数说明：
- options：选项，一般不必理会；
- pattern：要匹配的正则表达式；
- file：要搜索的文件路径。

示例：
```bash
# 查找当前目录下所有后缀名为txt的文件中的Hello World字符串
$ ls *.txt | xargs -I{} sh -c "grep Hello {} || true"

# 在当前目录下查找所有含有数字的文档，并输出其行号
$ grep --line-number "[[:digit:]]" *

# 查找/var目录下所有以log结尾的文件，并输出匹配到的行数
$ find /var -name "*log" | xargs grep "pattern" | wc -l
```

2) sed 命令
sed（stream editor，流编辑器）是一个非常强大的文本编辑器，它可以用来删除、复制、替换文本中的特定字符。我们也可以通过正则表达式来搭配sed命令实现更加精细化的文本处理。

命令格式如下：
```bash
sed [-options] '[pattern]'{sed动作} file(s)
```

参数说明：
- options：选项，一般不必理会；
- pattern：要匹配的正则表达式；
- sed动作：由冒号(:)分隔的两个参数，第一个参数表示要执行的动作，第二个参数表示要修改的内容或操作的对象；
- file：要搜索的文件路径。

示例：
```bash
# 把所有空白字符替换为空格
$ echo " hello  there    guys!   " | tr -d '\n' | sed's/\s\+/ /g' 

# 删除所有数字行，保留所有非数字行
$ cat numbers.txt
1
apple
two
three
four
five
six
seven
eight
nine
ten
11
eleven
twelve
thirteen
14
fifteen
sixteen
17
seventeen
eighteen
nineteen
20
twenty

$ sed '/^[[:digit:]]*$/d' numbers.txt
apple
two
three
four
five
six
seven
eight
nine
ten
eleven
twelve
thirteen
fifteen
sixteen
seventeen
eighteen
nineteen
twenty
```

3) awk 命令
awk（short for AWK script，AWK 脚本）是一个编程语言，它提供了丰富的文本分析能力。它可以读取文本文件，对数据进行处理，最后生成新的输出。我们也可以通过正则表达式来搭配awk命令实现更加精细化的文本处理。

命令格式如下：
```bash
awk '{print $0}' file(s) | command
```

参数说明：
- pattern：要匹配的正则表达式；
- file：要搜索的文件路径。

示例：
```bash
# 将日志文件按类型归类，并写入不同文件
$ awk -F'[][]' '$2 == "/var/log/messages"' log_file > messages.log
$ awk -F'[][]' '$2 == "/var/log/secure"' log_file > secure.log
$ awk -F'[][]' '$2 == "/var/log/auth.log"' log_file > auth.log

# 获取文本中所有英文单词出现频率前20的单词
$ cat text.txt | tr -cs '[:alpha:]' '[\n*]' | sort | uniq -c | sort -nr | head -20
   24 the
   22 and
   16 to
    9 in
     ...
```

4) others
其他基于正则表达式的命令行工具还有：perl，python，ruby等。它们各有特色，具体使用方法可以根据自己的需求了解。

# 4.正则表达式语法
下面我们来介绍一下正则表达式的语法。

## 概念
正则表达式（regular expression）的基本概念和术语如下图所示：


1. 元字符：元字符是指在正则表达式中有特殊含义的字符。例如"."匹配任何字符，"\*"匹配0或多个字符，"+"匹配1或多个字符，"|"表示或，"[]"表示字符集合。
2. 边界：^表示行首，$表示行尾。
3. 分支结构：()表示分支结构，它允许我们选择一条或多条路径来匹配。
4. 限定符：正则表达式有两种限定符：数量和位置。数量限定符用来指定满足匹配条件的字符或字符集的个数。例如，"?"表示前面的元素出现一次或0次，"*"表示前面的元素出现0次或无限次，"{m}"表示前面的元素出现m次，"{m,n}"表示前面的元素出现m到n次。位置限定符用来指定匹配字符串的起始和结束位置。例如，"^abc$"表示匹配字符串从开头到结尾完全是abc。
5. 模式修饰符：模式修饰符用来改变匹配模式。例如，"i"表示忽略大小写，"g"表示全局匹配。

## 元字符
下面介绍几个常用的元字符：

1. \d：匹配任意十进制 digit （0~9）。
2. \D：匹配任意非十进制 digit。
3. \w：匹配任意 alphanumeric character （字母数字字符）。
4. \W：匹配任意 non-alphanumeric character 。
5. \s：匹配任意 whitespace character （空格，制表符，换行符）。
6. \S：匹配任意 non-whitespace character 。
7. \t：匹配 tab character。
8. \r：匹配 carriage return character。
9. \n：匹配 newline character。
10. \b：匹配 word boundary （单词边界）。

## 分支结构
正则表达式的分支结构可以实现逻辑或的操作。例如，"(dog|cat)"可以匹配字符串中的"dog"或者"cat"。

## 限定符
正则表达式的限定符用来控制匹配的次数和位置。以下是常用的限定符：

1.?：问号表示匹配前面元素出现一次或0次。例如，"colou?r"可以匹配"color"或者"colour"。
2. *：星号表示匹配前面元素出现0次或无限次。例如，"do*g"可以匹配"dog"，"dodog", "doggg"等。
3. +：加号表示匹配前面元素至少出现1次。例如，"do+g"可以匹配"dog"，"dodog", "doggg"等，但是不能匹配""。
4. {m}：花括号表示前面的元素匹配 m 次。例如，"do{2}g"只能匹配"dog"。
5. {m,n}：花括号表示前面的元素匹配 m-n 次。例如，"do{2,4}g"可以匹配"dog"，"dodog", "doggg"，但是不能匹配"dogg"。

## 模式修饰符
模式修饰符用来改变匹配模式。以下是常用的模式修饰符：

1. i：表示忽略大小写。例如，"[aA]"匹配"a"或者"A"。
2. g：表示全局匹配。例如，"/the/"全局匹配整个文件，而"/the/"不局限于某一行。
3. m：多行匹配模式。此处暂且不讨论。

# 5.正则表达式实战

## grep 命令

### 查找日志文件中的关键词

有一个日志文件"access.log"记录了网站的访问信息，其中包含用户身份验证失败的信息。为了监控日志文件，我们想知道哪些IP出现过这种情况。

我们可以使用如下命令：
```bash
$ grep "authentication failure" access.log
```
结果显示，在日志文件中存在相应的关键字，输出如下：
```
::1 - frank [10/Oct/2000:13:55:38 -0700] "GET /security/passwords.html HTTP/1.0" 200 2179 "http://www.example.com/login.html" "Mozilla/5.0 (Windows; U; Windows NT 5.1; en-US; rv:1.8.1.6) Gecko/20070725 Firefox/2.0.0.6"
```

### 查找文件中的数字

有一个文本文件"numbers.txt"里面包含数字，但是有些数字没有必要显示完整，因此我们只需保留数字所在的行即可。

我们可以使用如下命令：
```bash
$ grep "^[[:digit:]]+" numbers.txt
1
11
14
17
20
```
结果显示，只有数字所在的行被输出。

### 使用通配符搜索

有一个目录下包含很多文件，我们想知道某个目录下面所有以".txt"结尾的文件中的所有数字。

我们可以使用如下命令：
```bash
$ cd directory
$ ls *.txt | xargs -I{} sh -c "grep \"^[[:digit:]]+\"" {}
```
结果显示，目录下面所有以".txt"结尾的文件中出现的数字被输出。

## sed 命令

### 替换字符串

有一个文本文件"sample.txt"包含了一些数字，我们想把这些数字替换为"*"。

我们可以使用如下命令：
```bash
$ sed's/[0-9]/*/g' sample.txt
```
结果显示，原始文件的数字都被替换为"*"。

### 删除空白行

有一个文本文件"sample.txt"包含了许多空白行和空格，我们想把空白行删除掉。

我们可以使用如下命令：
```bash
$ sed '/^\s*$/d' sample.txt
```
结果显示，空白行已经被删除。

### 替换所有连续重复的字符

有一个文本文件"sample.txt"包含了一些单词，其中有些单词之间可能存在连续相同的字符，例如："hellllloo wooooorrrld!!!!!!"，我们想把所有连续重复的字符替换为一个字符，例如："hello woorld!!!”。

我们可以使用如下命令：
```bash
$ sed's/\([a-zA-Z]\)\?\([a-zA-Z]\)\?\([a-zA-Z]\)\?\([a-zA-Z]\)\?\([a-zA-Z]\)\?\([a-zA-Z]\)\?\([a-zA-Z]\)\?\([a-zA-Z]\)\?\([a-zA-Z]\)/\1\2\3\4\5\6\7\8\9/g' sample.txt
```
结果显示，连续重复的字符已经被替换为一个字符。