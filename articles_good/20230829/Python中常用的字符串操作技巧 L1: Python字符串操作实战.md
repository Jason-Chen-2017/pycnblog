
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Python在数据处理、科学计算等领域有着举足轻重的地位。其应用范围广泛且易于上手。对于字符串操作来说也是Python非常重要的一部分。本文将给出一些最常用的Python字符串操作技巧，包括但不限于：字符串拼接、分割、替换、查找、比较、转换等。希望通过这些技巧能够帮助大家快速、高效地对字符串进行操作，提升工作效率。
本文基于Python3环境，内容涵盖中文字符、英文字母、数字、特殊符号等字符类型。如果文章对您有所帮助，欢迎点击关注我的微信公众号“码农翻身”。
# 2.前置条件
1. 安装了Python3环境；
2. 了解Python基础语法和数据结构；
3. 掌握Python字符串相关函数及用法；
4. 有一定编程经验者更佳；
5. 一定的计算机基础知识，如进制、ASCII编码等。
# 3.字符串拼接
字符串拼接，即将多个字符串连接成一个新的字符串。通常有两种方式实现字符串拼接：一种是直接用"+"运算符连接两个或多个字符串，另一种是使用join()方法，该方法可以指定一个分隔符来连接各个字符串。两者的区别主要在于性能和可读性上。
## 用"+"运算符连接字符串
```python
string_a = "Hello World!"
string_b = "This is a test."

result = string_a + string_b
print(result) # Output: Hello World! This is a test.
```
## 使用join()方法连接字符串
```python
strings = ["apple", "banana", "cherry"]
delimiter = ","

result = delimiter.join(strings)
print(result) # Output: apple,banana,cherry
```
### join()方法参数说明
- `sep`: 指定一个分隔符来连接各个字符串。默认值为空格。
- `seq`：需要连接的序列（列表、元组）。例如：`seq = ("hello", "world")`。
- 返回值：返回一个连接后的字符串。
# 4.字符串切分
字符串切分，也称字符串分片，即将一个字符串按照某种模式切分成若干子串。通常有三种方式实现字符串切分：从左边切分、从右边切分、按指定长度切分。
## 从左边切分
```python
string = "This is a sample sentence."
split_position = 4 # split at position 4 (space before's')

left_substring = string[:split_position]
right_substring = string[split_position:]

print("Left substring:", left_substring) # Output: Left substring: This i
print("Right substring:", right_substring) # Output: Right substring: s is a sample sentence.
```
## 从右边切分
```python
string = "This is a sample sentence."
split_position = 9 # split at position 9 (first letter of the second word)

left_substring = string[:-split_position]
right_substring = string[-split_position:]

print("Left substring:", left_substring) # Output: Left substring: This is a sam
print("Right substring:", right_substring) # Output: Right substring: entence.
```
## 通过指定长度切分
```python
string = "abcdefghijklmnopqrstuvwxyz"
chunk_size = 5 # chunk size should be an integer multiple of original length

for i in range(0, len(string), chunk_size):
    print(string[i:i+chunk_size]) # Output: abcde
                      #         fghij
                      #         klmnop
                      #         qrstu
                      #         vwxyz
```
### 参数说明
- `start`，`stop`，`step`：三个参数一起指定切分的起始位置、终止位置、步长。其中，`start`默认为零，表示从头开始；`stop`默认为字符串的结尾处，表示截取到字符串末尾；`step`默认为1，表示一次只切分一个字符。例如：`string[::2]` 表示从头开始，每隔2个字符切一次。
- 返回值：返回一个由切分得到的子串构成的列表。
# 5.字符串替换
字符串替换，指的是用指定的字符串代替指定位置上的字符串。通常有两种方式实现字符串替换：一种是replace()方法，另一种是用特定字符构造新的字符串，再将旧字符串赋值给新字符串即可。
## replace()方法替换字符串
```python
string = "The quick brown fox jumps over the lazy dog."
old_word = "lazy"
new_word = "sleepy"

result = string.replace(old_word, new_word)
print(result) # Output: The quick brown fox jumps over the sleepy dog.
```
## 特定字符构造新字符串
```python
string = "abcABCdefDEFghiGHI"
old_char = "BCdEFgHI"
new_char = "*" * len(old_char)

index_list = [i for i, c in enumerate(string) if c in old_char]
new_string = ""

last_index = 0
for index in sorted(index_list):
    new_string += string[last_index:index]
    new_string += new_char
    last_index = index + 1
    
new_string += string[last_index:]

print(new_string) # Output: a*c*A*Cde*Fgh*I
```
### 特别注意
虽然在字符串替换时可以使用`replace()`方法，但是这种方法只能替换**整体**匹配到的字符串，而无法精确地控制搜索的范围，所以建议优先选择第二种方法。
# 6.字符串查找
字符串查找，也称子串匹配，即找出某个子串是否存在于一个完整的字符串中。常见的字符串查找方法有四种：startswith()、endswith()、find()和rfind()。
## startswith()和endswith()
```python
string = "The quick brown fox jumps over the lazy dog."
prefix = "the"
suffix = "."

if string.startswith(prefix):
    print("String starts with '{}'.".format(prefix)) # Output: String starts with 'the'.
if string.endswith(suffix):
    print("String ends with '{}'.".format(suffix)) # Output: String ends with '.'.
```
## find()和rfind()
```python
string = "The quick brown fox jumps over the lazy dog."
sub_str = "quick"

pos = string.find(sub_str)
if pos!= -1:
    print("'{}' found at position {}.".format(sub_str, pos)) # Output: 'quick' found at position 4.
else:
    print("'{}' not found.".format(sub_str)) # Output: 'quck' not found.

pos = string.rfind(sub_str)
if pos!= -1:
    print("'{}' found at position {} from end.".format(sub_str, len(string)-pos)) # Output: 'quick' found at position 27 from end.
else:
    print("'{}' not found from end.".format(sub_str)) # Output: 'quck' not found from end.
```
### 参数说明
- `sub`：被查找的子串。
- `start`，`end`：用来限制搜索的范围。默认情况下，搜索整个字符串。
- `return`：如果找到子串则返回它的索引位置，否则返回`-1`。
# 7.字符串比较
字符串比较，即根据字符串的内容来判断两个字符串是否相等、大小关系等。常见的字符串比较方法有：cmp()、eq()、ne()、lt()、gt()、le()和ge()。
```python
string_a = "Apple"
string_b = "Banana"

# cmp() method compares two strings and returns an integer according to their comparison
comparison = cmp(string_a, string_b)
if comparison == 0:
    print("{} is equal to {}".format(string_a, string_b)) # Output: Apple is equal to Banana
elif comparison < 0:
    print("{} comes before {}".format(string_a, string_b)) # Output: Apple comes before Banana
else:
    print("{} comes after {}".format(string_a, string_b)) # Output: Banana comes after Apple

# eq(), ne(), lt(), gt(), le(), ge() methods compare two strings and return True or False accordingly
if string_a == string_b:
    print("{} is equal to {}".format(string_a, string_b)) # Output: Apple is equal to Banana
if string_a!= string_b:
    print("{} is not equal to {}".format(string_a, string_b)) # Output: Apple is not equal to Banana
if string_a > string_b:
    print("{} comes after {}".format(string_a, string_b)) # Output: Apple comes after Banana
if string_a >= string_b:
    print("{} comes after or is equal to {}".format(string_a, string_b)) # Output: Apple comes after or is equal to Banana
if string_a < string_b:
    print("{} comes before {}".format(string_a, string_b)) # Output: Apple comes before Banana
if string_a <= string_b:
    print("{} comes before or is equal to {}".format(string_a, string_b)) # Output: Apple comes before or is equal to Banana
```
# 8.字符串排序
字符串排序，即将字符串按照字母顺序排列，或者按照整数顺序排列等。通常有两种方式实现字符串排序：sort()方法和sorted()函数。
## sort()方法排序
```python
words = ['apple', 'banana', 'cherry', 'date', 'elderberry']
words.sort()
print(words) # Output: ['apple', 'banana', 'cherry', 'data', 'elderberry']
```
## sorted()函数排序
```python
numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
sorted_numbers = sorted(numbers)
print(sorted_numbers) # Output: [1, 1, 2, 3, 3, 4, 5, 5, 5, 6, 9]
```
# 9.字符串转换
字符串转换，即把字符串转换为其他形式，如整数、浮点数、列表等。常见的字符串转换方法有int()、float()、list()等。
```python
string = "123"
number = int(string)
print(type(number), number) # Output: <class 'int'> 123

string = "12.34"
fnum = float(string)
print(type(fnum), fnum) # Output: <class 'float'> 12.34

string = "[1, 2, 3]"
lst = list(eval(string))
print(type(lst), lst) # Output: <class 'list'> [1, 2, 3]
```
# 10.编码转换
编码转换，即将一个字符串从一种编码转换为另一种编码，比如UTF-8编码转化为GBK编码。可以通过encode()和decode()方法实现。
```python
utf8_string = "Hello World！"
gbk_bytes = utf8_string.encode('gbk')
gbk_string = gbk_bytes.decode('gbk')
print(type(gbk_bytes), gbk_bytes) # Output: <class 'bytes'> b'\xba\xc3\xb4\xbb\xca\xd4\xcd\xe2\xb3\xf6 \xab\xa4\xb7\xbe!'
print(type(gbk_string), gbk_string) # Output: <class'str'> 海底捞意粥！
```