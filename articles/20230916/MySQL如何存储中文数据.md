
作者：禅与计算机程序设计艺术                    

# 1.简介
  

MySQL是一个开源关系型数据库管理系统，是最流行的数据库之一。但随着互联网的蓬勃发展，越来越多的网站都将用户数据的存储迁移到了MySQL中。那么，如果需要存储和处理中文的数据呢？是否可以按照常规的方式进行存储和查询呢？

本文将从以下几个方面阐述MySQL如何存储中文数据：

1.MySQL支持的编码字符集及对应存储长度（MB）；
2.使用utf-8或gbk编码创建表时，会出现乱码还是正常显示？；
3.InnoDB引擎是否适合存储中文数据？；
4.MyISAM引擎是否适合存储中文数据？；
5.其它编码方式是否可用于存储中文数据？如GBK、BIG5、GB18030等；
6.MySQL中对于中文检索的相关功能介绍；
7.对于中文数据的排序、统计、分组和其他分析功能，是否有特殊的优化建议？ 

# 2.基本概念术语说明
## 2.1 字符集、字符编码、国家码
在计算机内部，所有信息都是二进制编码形式存储的。这种存储方式和字符集息息相关。一般情况下，计算机为了方便使用，统一采用了某种字符编码方案，该方案用一个或多个字节表示每个字符。不同的字符编码方案对同一种语言的字符集支持不同，例如ASCII编码仅支持英文字母、数字、一些符号，而UTF-8编码则支持更多语言。

在MySQL中，有两个重要的概念“字符集”和“排序规则”，它们共同决定了字符的存储和比较方式。字符集定义了字符所使用的集合，包括所有可打印和不可打印字符，比如汉字、数字、英文单词、标点符号等。排序规则定义了字符串比较的规则，它确定字符集中的字符在比较过程中应该按什么顺序排列。例如，ASCII编码下的英文排序规则是字典序，而UTF-8编码下的中文排序规则可能要更复杂些。

国家码（National Code）也称为国家/地区代码，它用来标识国家或地区。目前，中国共产党和政府为每一个国家分配了一个独一无二的国家码，即国家二维码。国家二维码用16进制数字编码，比如，中国的国家码为+86。

## 2.2 Unicode字符集
Unicode字符集（Universal Character Set，UCS）是由ISO组织制定的标准，用于描述所有字符，包括ASCII字符、希腊语字母、日语假名、俄罗斯字母等等。它于1991年发布第一个版本，最新版是第六次修订，加入了大量新字符，涵盖了世界上所有主要的书面语、非书面语和印刷体。由于其庞大的字符集数量和高度平衡性，使得Unicode成为国际上通用的编码方案。

MySQL从5.5.3版本开始支持Unicode字符集，可以通过配置my.cnf文件设置默认字符集为utf8，这样就可以直接插入或者读取各种语言的文本。并且，如果遇到不支持的编码，mysql会自动尝试转换。不过，不同版本之间的兼容性问题也逐渐被解决。

## 2.3 utf8mb4字符集
utf8mb4是MySQL的utf8字符集的一个变种，它扩充了原有的字符集以支持更多的Unicode字符。它的最大优势是可以使用四个字节来存储所有的字符。因此，如果需要存储中文数据，最好选择这个字符集。但是，utf8mb4并不是只有中文才推荐用这个字符集，任何需要保存完整的Unicode字符都可以考虑用utf8mb4来存储。

## 2.4 gbk字符集
GBK字符集是中国国家标准 GB 13000.1-2000 规定的汉字字符集。它覆盖了汉语的方方面面，包括繁体字、康熙部首、部件符号、笔画结构等。GBK编码实际上是GB2312的超集，包含GB2312中的所有字符，也增加了对繁体字的支持。MySQL从5.5.3版本开始支持gbk字符集，通过配置my.cnf文件设置默认字符集为gbk，这样就可以直接插入或者读取汉字。但是，不同版本之间的兼容性问题也逐渐被解决。

## 2.5 大五码
由于现代汉语字符集繁多，而且各类应用、系统和设备都对不同字符集和编码都有支持，所以，为了能够兼顾各种情况，MySQL提供了大五码字符集（Big Five Coded Character Sets）—— Big5， GB2312，Shift_JIS，EUC-KR，UTF-8等，大致可以支持几乎所有中文场景。

由于GBK和Big5使用相同的字库编码，所以它们可以直接互转。但是，在这种情况下，通常还是建议使用utf8mb4来存储中文数据，因为utf8mb4可以支持更多的Unicode字符，而且不会产生乱码问题。

# 3.MySQL支持的编码字符集及对应存储长度（MB）
|名称 |字符集 |排序规则 |国家码 |默认 |存储长度(MB)|
|:---|:---|:---|:---|:---|:---|
|big5   |big5    |big5_chinese_ci       |-      |          |2
|dec8    |dec8     |dec8_swedish_ci        |-      |          |N/A
|cp850    |cp850     |cp850_general_ci         |-      |          |1
|hp8     |hp8      |hp8_english_ci           |-      |          |1
|koi8r   |koi8r    |koi8r_general_ci        |-      |          |1
|latin1    |latin1     |latin1_swedish_ci        |-      |√          |1
|latin2    |latin2     |latin2_general_ci        |-      |          |1
|swe7    |swe7     |swe7_swedish_ci         |-      |          |N/A
|ascii    |ascii     |ascii_general_ci         |-      |√          |0.25
|ujis    |ujis     |ujis_japanese_ci         |-      |          |N/A
|sjis    |sjis     |sjis_japanese_ci         |-      |          |N/A
|hebrew    |hebrew     |hebrew_general_ci        |-      |          |1
|tis620    |tis620     |tis620_thai_ci          |-      |          |1
|euckr    |euckr    |euckr_korean_ci         |-      |          |1
|koi8u    |koi8u    |koi8u_general_ci        |-      |          |1
|gb2312    |gb2312    |gb2312_chinese_ci        |-      |          |1
|greek    |greek    |greek_general_ci        |-      |          |1
|cp1250    |cp1250     |cp1250_general_ci         |-      |          |1
|gbk    |gbk      |gbk_chinese_ci           |-      |          |1
|latin5    |latin5     |latin5_turkish_ci        |-      |          |1
|armscii8    |armscii8     |armscii8_general_ci        |-      |          |1
|utf8    |utf8     |utf8_general_ci        |utf8   |√          |1
|ucs2    |ucs2     |ucs2_general_ci        |-      |          |N/A
|cp866    |cp866     |cp866_general_ci         |-      |          |1
|keybcs2    |keybcs2     |keybcs2_general_ci        |-      |          |N/A
|macce    |macce     |macce_general_ci        |-      |          |1
|macroman    |macroman     |macroman_general_ci        |-      |          |1
|cp852    |cp852     |cp852_general_ci         |-      |          |1
|latin7    |latin7     |latin7_estonian_cs        |-      |          |1
|utf8mb4    |utf8mb4     |utf8mb4_general_ci        |utf8mb4   |√          |4
|cp1251    |cp1251     |cp1251_bulgarian_ci        |-      |          |1
|utf16    |utf16     |utf16_general_ci        |-      |          |N/A
|utf16le    |utf16le     |utf16le_general_ci        |-      |          |N/A
|cp1256    |cp1256     |cp1256_general_ci        |-      |          |1
|cp1257    |cp1257     |cp1257_lithuanian_ci        |-      |          |1
|binary    |binary     |binary               |-      |          |N/A
|geostd8    |geostd8     |geostd8_general_ci        |-      |          |N/A
|cp932    |cp932     |cp932_japanese_ci        |-      |          |1
|eucjpms    |eucjpms     |eucjpms_japanese_ci        |-      |          |N/A
|gb18030    |gb18030     |gb18030_chinese_ci        |-      |          |1