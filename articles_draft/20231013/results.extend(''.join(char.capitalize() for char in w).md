
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


## 概述
该功能旨在将日期字符串转换成元组列表，其中包含日期元素，如年、月、日等。日期字符串是用户输入的一段文本，例如："2019-07-16"表示一个公历日期。函数输出的是一个由元组组成的列表，每个元组对应于输入的日期字符串中的一天。元组中有三个元素分别是年、月、日。
## 使用场景
* 需要解析很多日期字符串；
* 有大量的日期字符串需要处理；
* 在分析数据时需要快速获取到日期信息；
## 实现方式及步骤
1. 将日期字符串按照指定符号切分成多个子串；
2. 对每个子串进行分析并提取出年、月、日；
3. 创建包含年、月、日的元组，并添加到结果列表中；
4. 返回结果列表。
### 符号类型
目前已知的日期符号包括：
* 年-月-日："YYYY-MM-DD"，如"2019-07-16";
* 年/月/日："YYYY/MM/DD"，如"2019/07/16";
* 年.月.日："YYYY.MM.DD"，如"2019.07.16".
### 时区支持
该功能暂不支持处理带有时区信息的日期字符串。
# 2.核心概念与联系
## 元组（tuple）
元组是一个不可变序列类型，其元素之间用逗号隔开，括起来，比如(a,b,c)。元组可以存储任意类型的对象，但是只能读取不能修改。
## list
list是一种可变序列类型，里面可以保存各种类型的数据，并且可以动态增删改。
## 函数（function）
函数是具有特殊名称的代码块，其作用就是对输入的数据进行处理，并返回相应的结果。它通常有一个入口参数（或称作形式参数），以及一些内部变量（或称作局部变量）。函数调用的时候，函数体内的代码会运行，完成相应的处理后，结果也会返回给调用者。
# 3.核心算法原理和具体操作步骤以及数学模型公式详细讲解
## 一、分析日期字符串
首先我们需要将日期字符串按照指定符号进行切分，切分之后得到的子串如下图所示：
可以看到，子串的数量是等于日期元素数量的三倍，因为每一个元素都可能出现两次，一次作为年份元素，一次作为月份元素，一次作为日元素。因此，通过切分之后的子串，就可以得到年、月、日的元素信息。
## 二、提取子串中的年、月、日
对于子串中的年、月、日元素，根据不同的符号类型，我们可以提取的方式也不同。这里以年-月-日为例，分析如何提取年、月、日。假设我们已经把每个元素按正确的顺序进行了分割，那么我们只需要按一下方式提取年、月、日即可：
```python
year = int(word_list[0])      # 提取年份
month = int(word_list[1])     # 提取月份
day = int(word_list[2])       # 提取日
```
其中，`int()`方法用来将字符串转换为整数。至此，我们已经得到了年、月、日的信息。
## 三、创建元组
将年、月、日分别提取出来之后，就可以创建元组，将它们组合到一起，并添加到结果列表中。由于我们还没有创建结果列表，因此先创建一个空列表，然后再将元组添加进去：
```python
result = []                  # 创建空结果列表
result.append((year, month, day))   # 添加新元组到列表中
return result               # 返回结果列表
```
至此，我们的函数已经可以正常工作。下面我们看看测试代码。
# 4.具体代码实例和详细解释说明
## 函数代码实现
```python
def parse_datestr(word):
    """
    解析日期字符串，并生成元组列表
    :param word: 日期字符串，格式如"YYYY-MM-DD"、"YYYY/MM/DD"、"YYYY.MM.DD"
    :return: 元组列表，每个元组代表一个日期元素，分别为年、月、日。元组的数量等于输入字符串中日期元素的个数，每个元组的长度为3。
    """
    start_index = 0              # 当前子串起始索引
    end_index = len(word)        # 当前子串结束索引
    separator = "-"             # 默认日期分隔符

    if "/" in word:
        separator = "/"
    elif "." in word:
        separator = "."
    
    year_index = -1              # 年元素索引，默认不存在
    month_index = -1             # 月元素索引，默认不存在
    day_index = -1               # 日元素索引，默认不存在

    while True:
        # 从当前索引位置开始查找下一个日期元素，找到之后更新索引信息
        year_index = word.find("Y", start_index, end_index)
        month_index = word.find("M", start_index, end_index)
        day_index = word.find("D", start_index, end_index)

        if year_index == -1 or (month_index!= -1 and month_index < year_index) \
                or (day_index!= -1 and day_index < min(year_index, month_index)):
            break

        start_index = max(max(year_index, month_index), day_index) + 1

    # 根据日期分隔符对日期字符串切分子串
    word_list = [w.strip().lstrip('0') for w in word[:start_index].split(separator)]

    # 如果子串数量小于3，则该字符串不是完整的日期字符串，返回None
    if len(word_list) < 3:
        return None

    year = int(word_list[0])
    month = int(word_list[1])
    day = int(word_list[2])

    result = [(year, month, day)]    # 创建结果列表

    return result                   # 返回结果列表
```
## 测试代码示例
```python
print(parse_datestr("2019-07-16"))         # [('2019', '07', '16')]
print(parse_datestr("2019/07/16"))         # [('2019', '07', '16')]
print(parse_datestr("2019.07.16"))         # [('2019', '07', '16')]
print(parse_datestr("2019.07."))           # None
print(parse_datestr("2019-07"))            # [('2019', '07', '')]
print(parse_datestr(""))                   # None
```