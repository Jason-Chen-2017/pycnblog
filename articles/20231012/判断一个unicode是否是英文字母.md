
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


## 一、需求背景
在处理文本数据的时候，我们经常需要根据语言特性进行各种分析，比如我们需要区分出中文和英文，或者判断一段文本中的实体名词或者词性等信息。那么我们如何判断一个unicode字符是否是一个英文字母呢？我们首先应该对英文字母相关知识有一个基本的了解，下面给大家一些基本的认识：
- 在计算机内部，所有的字符都是用Unicode编码表示的，它是一个符号集，其中包括了几乎所有人类使用的字符，例如汉字、英文字母、数字、符号、空格等。
- Unicode编码一共有两字节(16位)或四字节(32位)，每个编码都对应唯一的一个字符。
- ASCII码只有128个字符，而Unicode编码目前已经超过10万种字符。所以，虽然ASCII码被广泛应用，但其编码范围仍然受限于现实世界中的字符数量。
- 大多数操作系统提供支持Unicode的函数接口，使得开发者可以方便地处理各种各样的字符编码。
- 有些字符只存在于某些语言中，比如中文里的“音调”字符，而英语里没有。
因此，要判断一个unicode字符是否是一个英文字母，最直接的方法就是查看该字符所属的Unicode编码范围。如下表所示：
|编号 |范围|名称|描述|
|--:|:--|:--|:--|
|1|U+0041-U+005A|Latin letters (lowercase)|小写拉丁字母|
|2|U+0061-U+007A|Latin letters (uppercase)|大写拉丁字母|
|3|U+00C0-U+00D6|Latin letters with diacritical marks (latin capital letter A with grave = á, latin capital letter A with circumflex = â,... )|带重音符的拉丁字母（如：“á”）|
|4|U+00D8-U+00F6|Latin letters with diacritical marks (latin capital letter O with grave = ó, latin capital letter O with circumflex = ö,... )|带重音符的拉丁字母（如：“ó”）|
|5|U+00F8-U+00FF|Latin letters with diacritical marks (latin small letter o with stroke / germandbls /...)|带下划线的拉丁字母（如：“ø”）|
|6|U+0100-U+017F|Latin Extended-A|扩展拉丁字母A|
|7|U+0180-U+024F|Latin Extended-B|扩展拉丁字母B|
|8|U+0250-U+02AF|IPA Extensions|国际音标扩展|
|9|U+02B0-U+02FF|Modifier letters|修饰字母|
|10|U+0300-U+036F|Combining diacritical marks|组合重音标记|
|11|U+0370-U+03FF|Greek and Coptic|希腊和哥特字母|
|12|U+0400-U+04FF|Cyrillic|俄语字母|
|13|U+0500-U+052F|Cyrillic Supplement|俄语增补字母|
|14|U+2DE0-U+2DFF|Cyrillic Extended-A|俄语扩充A|
|15|U+A640-U+A69F|Cyrillic Extended-B|俄语扩充B|
|16|U+1E00-U+1EFF|Latin Extended Additional|追加拉丁字母|
|17|U+1F00-U+1FFF|Greek Extended|希腊增补字母|
|18|U+2C60-U+2C7F|Latin Extended-C|拉丁扩充C|
|19|U+0600-U+06FF|Arabic|阿拉伯语字母|
|20|U+FE70-U+FEFF|Arabic Presentation Forms-B|阿拉伯语拼音符号|
|21|U+FB50-U+FDFF|Arabic Pres.Forms A|阿拉伯语字母上方装饰形式|
|22|U+0750-U+077F|Arabic Supplement|阿拉伯语增补字母|
|23|U+08A0-U+08FF|Arabic Extended-A|阿拉伯语扩展A|
|24|U+200C-U+200D|Zero Width Joiner/Non Joiner|U+200C和U+200D的组合|
|25|U+1EE00-U+1EEFF|Arabic Mathematical Alphabetic Symbols|阿拉伯语数学字母符号|
|26|U+0E01-U+0E3A|Thai|泰语字母|
|27|U+1100-U+11FF|Hangul Jamo|韩语基辅音字母|
|28|U+3130-U+318F|Hangul Compatibility Jamo|韩语兼容字母|
|29|U+AC00-U+D7AF|Hangul Syllables|韩语组成音节|
|30|U+3041-U+3096|Hiragana|平假名|
|31|U+30A1-U+30FA|Katakana|片假名|
|32|U+31F0-U+31FF|Kanbun|艮 Bandian Plate|
|33|U+4E00-U+9FFF|CJK Unified Ideographs|中文统一漢字|
|34|U+3400-U+4DB5|CJK Unified Ideographs Extension A|中文漫游扩展A|
|35|U+20000-U+2A6D6|CJK Unified Ideographs Extension B|中文漫游扩展B|
|36|U+2A700-U+2B734|CJK Unified Ideographs Extension C|中文漫游扩展C|
|37|U+2B740-U+2B81D|CJK Unified Ideographs Extension D|中文漫游扩展D|
|38|U+2F00-U+2FD5|Chinese Radicals Supplement|汉字部首补充|
|39|U+2E80-U+2EF3|CJK Radicals Supplement|日文象形字扩展|
|40|U+F900-U+FA2D|CJK Compatibility Ideographs|中文兼容字母|
|41|U+2F800-U+2FA1D|CJK Compatibility Ideographs Supplement|中文兼容字母补充|
除此之外，还有一些字符也会被认为是英文字母，比如数字0到9、半角标点符号、空白符等。但是对于这些字符来说，它们的Unicode编码范围并不一定能确定，比如有的编码可能与其他字符相同。而且，即便确定了字符的Unicode编码范围，也不是所有人的分类都能一一覆盖。因此，仅仅依靠Unicode编码判断一个字符是否是一个英文字母就无疑是不准确且不可靠的，还需要结合语言特性、上下文等更多因素才能做更全面和准确的判断。
## 二、概述
本文将介绍一种简单有效的方法来判断一个unicode字符是否是一个英文字母。这种方法只需要通过比较该字符的Unicode编码值来确定，具体如下：
- 检查其所在的Unicode编码范围是否与前述英文字母Unicode编码范围相交；
- 如果在同一个Unicode编码范围内，则该字符是一个英文字母；否则，不是。
至于哪些字符是单独编制的单个声母还是双拼音单字母，并不影响判断过程。
## 三、核心算法原理及流程图
### 1.Unicode编码空间分类
Unicode编码空间可分为：
- Basic Multilingual Plane (BMP): 第零个基本多语言平面(Basic Multilingual Plane)。其中，从U+0000到U+FFFF之间的Unicode码位构成了BMP。
- Supplementary Planes: 次基本多语言平面之后的额外 planes 。其中，第一个额外plane从U+10000开始，每两个代码单元包含一个代码位置。
- Surrogate Code Points: 代理项代码点，是在UTF-16编码方案中使用的。它们的主要目的是为了处理无法显示的字符。
以上三个Unicode编码平面及其作用分别如下：

|Unicode编码空间|名称|作用|
|-------:|------:|:----------|
|0000-FFFF|Basic Multilingual Plane (BMP)|包含基本的、非私有的字符。例如，所有的拉丁文、希腊文、阿拉伯文、中文及日文字符都在这个平面中。|
|10000-1FFFF|Supplementary Plane 1|包含基本的、非私有的字符。例如，用来描述符号和字体属性的控制字符都在这个平面中。|
|20000-2FFFF|Supplementary Plane 2|包含基本的、非私有的字符。例如，用于计算机图形的字符及相关表情符号都在这个平面中。|
|D800-DFFF|Surrogate Code Points|代理项代码点。它们的主要目的是为了处理无法显示的字符。|

由于各个字符的Unicode编码范围不同，所以采用分类法对其进行管理，把其中基本拉丁文、扩展拉丁文、希腊文、阿拉伯文、中文及日文字符按范围划分如下：

|序号|范围|名称|编码范围|备注|
|----:|:--|:--|:--|:--|
|1|U+0000-U+007F|ASCII Control Chars|U+0000-U+001F，U+007F|ASCII控制字符|
|2|U+0080-U+00FF|Latin-1 Supplement|U+0080-U+009F|欧洲西欧字符集|
|3|U+0100-U+017F|Latin Extended-A|U+0100-U+017F|意大利语，德语，芬兰语，瑞典语，英语等的字母及符号|
|4|U+0180-U+024F|Latin Extended-B|U+0180-U+024F|其他欧洲语言的字母及符号|
|5|U+0250-U+02AF|IPA Extensions|U+0250-U+02AF|国际音标扩展|
|6|U+02B0-U+02FF|Spacing Modifier Letters|U+02B0-U+02FF|修饰字母|
|7|U+0300-U+036F|Combining Diacritical Marks|U+0300-U+036F|组合重音标记|
|8|U+0370-U+03FF|Greek and Coptic|U+0370-U+03FF|希腊字母和古埃及文字|
|9|U+0400-U+04FF|Cyrillic|U+0400-U+04FF|俄语字符集|
|10|U+0500-U+052F|Cyrillic Supplement|U+0500-U+052F|俄语增补字符集|
|11|U+2DE0-U+2DFF|Cyrillic Extended-A|U+2DE0-U+2DFF|俄语扩充A|
|12|U+A640-U+A69F|Cyrillic Extended-B|U+A640-U+A69F|俄语扩充B|
|13|U+1E00-U+1EFF|Latin Extended Additional|U+1E00-U+1EFF|追加拉丁文字符集|
|14|U+1F00-U+1FFF|Greek Extended|U+1F00-U+1FFF|希腊增补字符集|
|15|U+2C60-U+2C7F|Latin Extended-C|U+2C60-U+2C7F|拉丁文扩充C|
|16|U+0600-U+06FF|Arabic|U+0600-U+06FF|阿拉伯语字符集|
|17|U+FE70-U+FEFF|Arabic Presentation Forms-B|U+FE70-U+FEFF|阿拉伯语拼音符号|
|18|U+FB50-U+FDFF|Arabic Pres.Forms A|U+FB50-U+FDFF|阿拉伯语字母上方装饰形式|
|19|U+0750-U+077F|Arabic Supplement|U+0750-U+077F|阿拉伯语增补字符集|
|20|U+08A0-U+08FF|Arabic Extended-A|U+08A0-U+08FF|阿拉伯语扩展A|
|21|U+200C-U+200D|Zero Width Joiner/Non Joiner|U+200C，U+200D|U+200C和U+200D的组合|
|22|U+1EE00-U+1EEFF|Arabic Mathematical Alphabetic Symbols|U+1EE00-U+1EEFF|阿拉伯语数学字母符号|
|23|U+0E01-U+0E3A|Thai|U+0E01-U+0E3A|泰语字母|
|24|U+1100-U+11FF|Hangul Jamo|U+1100-U+11FF|韩语基辅音字母|
|25|U+3130-U+318F|Hangul Compatibility Jamo|U+3130-U+318F|韩语兼容字母|
|26|U+AC00-U+D7AF|Hangul Syllables|U+AC00-U+D7AF|韩语组成音节|
|27|U+3041-U+3096|Hiragana|U+3041-U+3096|平假名|
|28|U+30A1-U+30FA|Katakana|U+30A1-U+30FA|片假名|
|29|U+31F0-U+31FF|Kanbun|U+31F0-U+31FF|艮 Bandian Plate|
|30|U+4E00-U+9FFF|CJK Unified Ideographs|U+4E00-U+9FFF|中文统一漢字|
|31|U+3400-U+4DB5|CJK Unified Ideographs Extension A|U+3400-U+4DB5|中文漫游扩展A|
|32|U+20000-U+2A6D6|CJK Unified Ideographs Extension B|U+20000-U+2A6D6|中文漫游扩展B|
|33|U+2A700-U+2B734|CJK Unified Ideographs Extension C|U+2A700-U+2B734|中文漫游扩展C|
|34|U+2B740-U+2B81D|CJK Unified Ideographs Extension D|U+2B740-U+2B81D|中文漫游扩展D|
|35|U+2F00-U+2FD5|Chinese Radicals Supplement|U+2F00-U+2FD5|汉字部首补充|
|36|U+2E80-U+2EF3|CJK Radicals Supplement|U+2E80-U+2EF3|日文象形字扩展|
|37|U+F900-U+FA2D|CJK Compatibility Ideographs|U+F900-U+FA2D|中文兼容字母|
|38|U+2F800-U+2FA1D|CJK Compatibility Ideographs Supplement|U+2F800-U+2FA1D|中文兼容字母补充|
### 2.算法流程
算法流程如下图所示：
### 3.Python代码实现
```python
def is_english_letter(char):
    # 判断字符是否属于BMP(Basic Multilingual Plane)
    if ord(char) >= 0 and ord(char) <= 0xFFFF:
        code = ord(char)
    elif char in ('\uD800', '\uDFFF'):
        return False # 不允许存在代理项
    else:
        # 将四字节的unicode转化为两个三字节的序列
        b = ord(char) >> 10
        y = ((b & 0x3FF)<<16) + (ord(next(iter(reversed(char))))&0x3FF) + 0x10000 
        code = chr((b<<10)+((ord(char)-0xD800)&0x3FF))+chr(((y>>10)&0x3FF)+0xDC00)
    
    # 判断字符是否属于指定范围
    for range_name, start, end in unicode_ranges:
        if start<=code<=end:
            return True
        
    return False

unicode_ranges = [
    ("ASCII control chars", 0, 0x1F), 
    ("BMP supplementary chars", 0x80, 0xFF),
    ("Latin extended-A", 0x100, 0x17F),
    ("Latin extended-B", 0x180, 0x24F),
    ("IPA extensions", 0x250, 0x2AF),
    ("spacing modifier letters", 0x2B0, 0x2FF),
    ("combining diacritical marks", 0x300, 0x36F),
    ("greek and coptic", 0x370, 0x3FF),
    ("cyrillic", 0x400, 0x4FF),
    ("cyrillic supplement", 0x500, 0x52F),
    ("cyrillic extended-A", 0x2DE0, 0x2DFF),
    ("cyrillic extended-B", 0xA640, 0xA69F),
    ("latin extended additional", 0x1E00, 0x1EFF),
    ("greek extended", 0x1F00, 0x1FFF),
    ("latin extended-C", 0x2C60, 0x2C7F),
    ("arabic", 0x600, 0x6FF),
    ("arabic presentation forms-B", 0xFE70, 0xFEFF),
    ("arabic pres.forms A", 0xFB50, 0xFDFF),
    ("arabic suplement", 0x750, 0x77F),
    ("arabic extended-A", 0x8A0, 0x8FF),
    ("zero width joiners/non-joiners", 0x200C, 0x200D),
    ("arabic mathematical alphabetic symbols", 0x1EE00, 0x1EEFF),
    ("thai", 0xE00, 0xE3A),
    ("hangul jamo", 0x1100, 0x11FF),
    ("hangul compatibility jamo", 0x3130, 0x318F),
    ("hangul syllables", 0xAC00, 0xD7AF),
    ("hiragana", 0x3041, 0x3096),
    ("katakana", 0x30A1, 0x30FA),
    ("kanbun", 0x31F0, 0x31FF),
    ("unified ideographs", 0x4E00, 0x9FFF),
    ("CJK unified ideographs extension A", 0x3400, 0x4DB5),
    ("CJK unified ideographs extension B", 0x20000, 0x2A6D6),
    ("CJK unified ideographs extension C", 0x2A700, 0x2B734),
    ("CJK unified ideographs extension D", 0x2B740, 0x2B81D),
    ("chinese radicals supplement", 0x2F00, 0x2FD5),
    ("CJK radicals supplement", 0x2E80, 0x2EF3),
    ("CJK compatibility ideographs", 0xF900, 0xFA2D),
    ("CJK compatibility ideographs supplement", 0x2F800, 0x2FA1D) 
]
```