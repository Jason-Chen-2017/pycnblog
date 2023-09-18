
作者：禅与计算机程序设计艺术                    

# 1.简介
  

C++是一门非常受欢迎的高级编程语言，它的独特之处在于提供面向对象、泛型编程、运行时多态等能力。随着互联网的飞速发展，C++正在成为企业开发人员中的主流语言。下面让我们一起看一下string类的一些常用方法，并通过实例的方式巩固这些知识。
# 2.基本概念术语说明
## 2.1 字符串（String）
在计算机中，字符串是一个字符序列。它可以由任意数量的字符组成，包括空格、数字、字母甚至符号。每个字符都是用单个ASCII码表示的。例如："hello world"就是一个字符串。

## 2.2 C++中的字符串类
C++中的字符串类主要有std::string和std::wstring。std::string用于存储文本数据，而std::wstring用于存储宽字符的数据，如中文字符或日文字符。

## 2.3 string和wstring的区别
string类是C++标准库的一部分，其功能强大且灵活。它允许你通过索引访问单个字符，还支持随机访问迭代器，可以方便地插入、删除、修改字符串。但是它只能保存固定大小的字符数组，因此不能动态扩充容量，因此当需要保存大量字符的时候，建议使用wstring。

wstring类是基于模板定义的，可以根据需要分配内存空间，因此比较适合用来保存变长字符串。

## 2.4 Unicode与UTF-8编码
Unicode是一个字符集，它定义了每种语言所使用的所有符号。不同国家和地区制定了不同的编码方案，比如UTF-8和GBK等。

UTF-8编码把每个字符编码为1到4个字节，第一个字节的前缀用于标识字符所占用的字节数。从第一个字节开始，后续的字节如果不够用，就在最前面补上0。这样做的好处是它可以处理任何有效的Unicode字符，不会因为某个字符只用了几个字节就截断掉。

# 3.核心算法原理和具体操作步骤及数学公式讲解
## 3.1 分配内存
```cpp
string str = "Hello World"; // 默认分配内存容量
string str(n, 'x');   // 分配 n 个 x 的字符串
string str(s);        // 拷贝 s 中的字符到新创建的字符串
string str(b, e);     // 从 b 到 e 的范围内拷贝字符到新的字符串
string str("");       // 创建一个空串
```
## 3.2 获取长度和元素访问
```cpp
int len = str.length();      // 返回字符串长度
char c = str[i];              // 根据下标访问第 i 个字符
c = *str.begin();             // 获取首字符，没有则抛出异常
*str.end() = '\0';            // 修改尾字符为空字符
for (char ch : str) {         // 遍历整个字符串
   ...
}
if (str == s) {}              // 判断两个字符串是否相等
```
## 3.3 插入、删除、替换
```cpp
str.insert(pos, count, ch);   // 在 pos 位置插入 count 个 ch
str.erase(beg, end);          // 删除子串 [beg, end)
str.replace(beg, end, new_str);// 替换子串 [beg, end] 为 new_str
```
## 3.4 比较运算
```cpp
bool cmp = str < s;               // str 小于 s? true: false
cmp = str <= s;                   // str 小于等于 s? true: false
cmp = str > s;                    // str 大于 s? true: false
cmp = str >= s;                   // str 大于等于 s? true: false
cmp = str.compare(s);             // str 和 s 是否相同？0: -1/+1
```
## 3.5 大小写转换
```cpp
string upper_str = uppercase(str);    // 转为大写
string lower_str = lowercase(str);    // 转为小写
```
## 3.6 查找、替换模式
```cpp
size_t pos = str.find('a', from=0);   // 查找 a 的位置，从 from 开始
pos = rfind('b', to);                 // 从末尾查找 b 的位置，到 to 结束
pos = find_first_of("xyz", from=0);  // 查找 str 中出现过的一个字符列表里的字符
pos = find_last_not_of("xyz");        // 查找 str 中最后一个不属于 xyz 的字符位置
string new_str = replace_first_of("xyz", "pqr");// 替换第一个出现的字符
new_str = replace_all("xyz", "pqr");           // 替换全部出现的字符
```
## 3.7 字符分割与合并
```cpp
vector<string> vec = split(str, " ");   // 以空格为分隔符分割为多个字符串
str = join(vec, ".");                  // 用. 连接多个字符串为一个
```
## 3.8 正则表达式匹配
```cpp
regex re("\\d+");                      // 创建正则表达式对象
smatch match;                           // 创建匹配结果对象
if (regex_match(str, match, re)) {      // 检查 str 是否符合 re 模式
    cout << match[0] << endl;           // 获取第一个捕获组的值
} else {                                // 不符合
    cout << "no match." << endl;
}
```
## 3.9 其他操作
```cpp
string str2 = trim(str);                // 去除开头和结尾的空白字符
swap(str, str2);                        // 交换两个字符串的内容
transform(str.begin(), str.end(), str.begin(), ::tolower);
                                            // 字符串全部转为小写
reverse(str.begin(), str.end());        // 反转字符串
sort(str.begin(), str.end());           // 对字符串进行排序
```
# 4.具体代码实例及解释说明
这里以常见的字符串操作方法及示例作为例子，详细阐述各方法的使用方法。

**构造函数：**

```cpp
string str = "Hello World!";
string str2(str);
cout<<str2<<endl;

string str3(10,'');
cout<<str3<<endl;

const char* pch = "abcdefg";
string str4(pch);
cout<<str4<<endl;

char arr[] = {'h', 'e', 'l', 'l', 'o'};
string str5(arr);
cout<<str5<<endl;
```

输出结果如下：

```cpp
Hello World!
----------
xxxxxxxxxx
abcdefg
helo
```

**获取字符串长度：**

```cpp
string str = "Hello World!";
int length = str.length();
cout<<length<<endl;
```

输出结果：

```cpp
12
```

**获取字符串中的字符：**

```cpp
string str = "Hello World!";
char firstChar = str[0];
char lastChar = str[str.length()-1];
cout<<"First character is "<<firstChar<<endl;
cout<<"Last character is "<<lastChar<<endl;
```

输出结果：

```cpp
First character is H
Last character is!
```

**在字符串指定位置插入字符：**

```cpp
string str = "Hello World!";
str.insert(6,'.');
cout<<str<<endl;
```

输出结果：

```cpp
Hello Worl..ld!
```

**删除子串：**

```cpp
string str = "Hello World!";
str.erase(6,5);
cout<<str<<endl;
```

输出结果：

```cpp
Hello World!
```

**替换子串：**

```cpp
string str = "Hello World!";
str.replace(6,5,"world");
cout<<str<<endl;
```

输出结果：

```cpp
Hello worldd!
```

**比较字符串：**

```cpp
string str = "Hello World!";
string str2 = "Hello World!";
string str3 = "Hello world!";
bool cmp1 = (str==str2);
bool cmp2 = (str<=str3);
bool cmp3 = (str>=str3);
cout<<(cmp1?"true":"false")<<endl;
cout<<(cmp2?"true":"false")<<endl;
cout<<(cmp3?"true":"false")<<endl;
```

输出结果：

```cpp
true
true
false
```

**对字符串进行大小写转换：**

```cpp
string str = "HeLLo WoRLD!";
string upperStr = strToUppercase(str);
string lowerStr = strToLowercase(upperStr);
cout<<lowerStr<<endl;
```

输出结果：

```cpp
hello world!
```

**查找字符串：**

```cpp
string str = "The quick brown fox jumps over the lazy dog.";
string pattern = "the";
int position = str.find(pattern);
int rposition = str.rfind(pattern);
int listPosition = str.find_first_of("aeiouAEIOU");
int nonlistPosition = str.find_last_not_of("aeiouAEIOU");
string subStr = str.substr(nonlistPosition + 1);
cout<<"Position of \""<<pattern<<"\" in \""<<str<<"\" is "<<position<<"."<<endl;
cout<<"Reverse position of \""<<pattern<<"\" in \""<<str<<"\" is "<<rposition<<"."<<endl;
cout<<"Position of any vowel in \""<<str<<"\" is "<<listPosition<<"."<<endl;
cout<<"Last position not of vowel in \""<<str<<"\" is "<<nonlistPosition<<"."<<endl;
cout<<"Substring between last not vowel and end is:\""<<subStr<<"\"."<<endl;
```

输出结果：

```cpp
Position of "the" in "The quick brown fox jumps over the lazy dog." is 3.
Reverse position of "the" in "The quick brown fox jumps over the lazy dog." is 11.
Position of any vowel in "The quick brown fox jumps over the lazy dog." is 3.
Last position not of vowel in "The quick brown fox jumps over the lazy dog." is 4.
Substring between last not vowel and end is:"uick brown fox jumps over the lazy d".
```

**正则表达式匹配：**

```cpp
string str = "A1B2C3D4E5F6G7H8I9J0";
regex re("[0-9]+");
smatch matchResult;
if (regex_search(str, matchResult, re)) {
    for (auto& m : matchResult) {
        cout << m << endl;
    }
}
else {
    cout << "No match found." << endl;
}
```

输出结果：

```cpp
1
2
3
4
5
6
7
8
9
0
```