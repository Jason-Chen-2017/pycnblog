
作者：禅与计算机程序设计艺术                    

# 1.简介
  

在计算机编程领域里，有两种主要语言，C语言和C++语言。它们之间存在着很大的不同，因此掌握其中之一对学习其他语言也至关重要。今天我将通过本文，介绍C和C++的基本概念、相关数据类型、运算符、控制语句、函数、指针、内存管理等方面的区别与联系。

C和C++语言都源自贝尔实验室的B语言，两者非常接近。实际上，C++继承了C的所有优点并添加了新的特性，使得其成为一个功能更强大、性能更佳的高级语言。虽然两者存在许多共同之处，但两者还是具有本质性的区别。

在阅读完本文后，读者应该能够：

1. 对C语言和C++语言有了一个整体的认识；
2. 了解C和C++之间的差异及其背后的历史原因；
3. 在自己的编程中选择合适的语言，达到最佳效果。

# 2.基本概念术语说明
## 2.1 C语言概述
C语言是一种通用、结构化的计算机编程语言。它由<NAME>（贝尔实验室）于1972年创建，是一种低级语言。C语言以过程化、命令式的方式编写程序，并提供低级的数据处理能力。

C语言以“编译”的方式运行，编译器把源代码翻译成机器指令。编译器通常会把源码转换为汇编语言或机器码，然后再交给CPU执行。这样就可以确保程序正确地运行。

C语言的程序可以运行在各种类型的计算机上，包括桌面电脑、服务器、嵌入式系统等。

## 2.2 C++概述
C++是基于C语言的“面向对象”编程语言，也是一种现代化的、通用的、类库丰富的编程语言。C++被设计用于开发底层操作系统和系统软件。

与C语言相比，C++提供了更强大的功能集，包括自动内存管理、模板库、异常处理机制、多重继承等。这些功能使得C++成为系统开发的理想语言。

与Java、C#等语言不同的是，C++没有提供像Java或者C#那样的虚拟机环境，所以在执行速度上要快很多。

## 2.3 数据类型
### 整数类型
| C语言 | C++ | 描述 |
| --- | --- | --- |
| char | char | 单字节字符型 |
| signed char | signed char | 有符号字节类型 |
| unsigned char | unsigned char | 无符号字节类型 |
| short int | short int | 短整型 |
| signed short int | signed short int | 有符号短整型 |
| unsigned short int | unsigned short int | 无符号短整型 |
| int | int | 整型 |
| signed int | signed int | 有符号整型 |
| unsigned int | unsigned int | 无符号整型 |
| long int | long int | 长整型 |
| signed long int | signed long int | 有符号长整型 |
| unsigned long int | unsigned long int | 无符号长整型 |
| long long int | long long int | 超长整型 |
| signed long long int | signed long long int | 有符号超长整型 |
| unsigned long long int | unsigned long long int | 无符号超长整型 |

以上数据类型是C/C++中的基本数据类型，它们定义了一个变量所占据的存储空间的大小。

- char型：char数据类型是一个字节，它可以表示范围从-128~127的整数值。该数据类型是根据需要来决定变量的类型。在32位平台上，char类型是按补码形式存放数据的。例如：char a = -100;则a的值等于十六进制的F4（二进制为11110100）。由于char型只占据一个字节的空间，所以当声明较短的字符串时，可能会导致溢出。
- signed char类型：signed char数据类型也是字节，它的取值范围是-128~127。它与char型相似，只是它有一个符号位。也就是说，对于负数来说，它的符号位是1，而对于正数来说，它的符号位是0。如果出现两个不同的符号位，那么它就会被认为是无效的数据。
- unsigned char类型：unsigned char数据类型也是字节，它的取值范围是0~255。它与char型类似，但它的范围更广，可以表示更大的数字。
- short int类型：short int数据类型是一个短整型。它占据两个字节的存储空间。它的取值范围可以从-32768到32767。
- signed short int类型：signed short int数据类型也是一个短整型，它与short int类型类似，但是它还有一个符号位。如果取值为正数，符号位为0；如果取值为负数，符号位为1。
- unsigned short int类型：unsigned short int数据类型与short int类型类似，但它只能表示无符号整数。它的取值范围可以从0到65535。
- int类型：int数据类型是一个整型。它占据四个字节的存储空间。它的取值范围可以从-2147483648到2147483647。
- signed int类型：signed int数据类型与int类型类似，但它还有一个符号位。对于负数来说，它的符号位是1，对于正数来说，它的符号位是0。
- unsigned int类型：unsigned int数据类型与int类型类似，但它只能表示无符号整数。它的取值范围可以从0到4294967295。
- long int类型：long int数据类型是一个长整型。它占据四个字节的存储空间。它的取值范围可以从-2147483648到2147483647。
- signed long int类型：signed long int数据类型与long int类型类似，但它还有一个符号位。对于负数来说，它的符号位是1，对于正数来说，它的符号位是0。
- unsigned long int类型：unsigned long int数据类型与long int类型类似，但它只能表示无符号整数。它的取值范围可以从0到4294967295。
- long long int类型：long long int数据类型是一个超长整型。它占据八个字节的存储空间。它的取值范围可以从-9223372036854775808到9223372036854775807。
- signed long long int类型：signed long long int数据类型与long long int类型类似，但它还有一个符号位。对于负数来说，它的符号位是1，对于正数来说，它的符号位是0。
- unsigned long long int类型：unsigned long long int数据类型与long long int类型类似，但它只能表示无符号整数。它的取值范围可以从0到18446744073709551615。

注意：
- 如果不加特别说明，对于整形、浮点型、字符型变量，都可以使用后缀'L'(大写)或'l'(小写)，代表long类型。例如：int x=100L。
- 浮点型变量默认是double类型。

### 实数类型
| C语言 | C++ | 描述 |
| --- | --- | --- |
| float | float | 单精度浮点型 |
| double | double | 双精度浮点型 |
| long double | long double | 扩展精度浮点型 |

以上数据类型是C/C++中的实数类型，它们用来存储实数值，一般用于存储算术表达式的结果。float类型是四字节的实数类型，可以表示范围为±3.4E38的实数。double类型是八字节的实数类型，可以表示范围为±1.7E308的实数。long double类型是十六字节的实数类型，可以表示范围更大的实数。

注意：
- 使用浮点数时，应尽量避免使用float类型，因为其精度太低。一般情况下，使用double类型即可。
- 由于不同的硬件平台支持浮点数的精度不同，因此浮点数运算的结果可能不同。

### 字符类型
| C语言 | C++ | 描述 |
| --- | --- | --- |
| char | char | 单字节字符类型 |
| wchar_t | wchar_t | 宽字符类型 |

以上数据类型是C/C++中的字符类型，用于存储单个字符或文本串。char数据类型是一个字节，它可以表示范围从-128~127的整数值。wchar_t数据类型是一个整型，用来存储宽字符，可以表示范围从0到65535的整数值。

注意：
- 宽字符类型wchar_t是为了解决传统的ASCII编码方式不能表示一些特殊符号的问题而引进的，所以不能直接使用它来表示中文字符。

### 复数类型
| C语言 | C++ | 描述 |
| --- | --- | --- |
| complex float | std::complex<float> | 复数单精度浮点型 |
| complex double | std::complex<double> | 复数双精度浮点型 |
| complex long double | std::complex<long double> | 复数扩展精度浮点型 |

以上数据类型是C/C++中的复数类型，它们用来表示复数值。complex float类型用来表示复数值的实部和虚部都是float类型，可以表示±3.4E38的实数。complex double类型用来表示复数值的实部和虚部都是double类型，可以表示±1.7E308的实数。complex long double类型用来表示复数值的实部和虚部都是long double类型，可以表示更大的实数。

## 2.4 运算符
| C语言 | C++ | 描述 |
| --- | --- | --- |
| + | + | 加法 |
| - | - | 减法 |
| * | * | 乘法 |
| / | / | 除法 |
| % | % | 模ulo |
| ++ | ++ | 前置增量 |
| -- | -- | 前置减量 |
| += | += | 后置增量 |
| -= | -= | 后置减量 |
| *= | *= | 后置乘法 |
| /= | /= | 后置除法 |
| &= | &= | 按位与赋值 |
| \|= | \|= | 按位或赋值 |
| ^= | ^= | 按位异或赋值 |
| <<= | <<= | 左移赋值 |
| >>= | >>= | 右移赋值 |
|? : |? : | 条件运算符 |
| < | < | 小于 |
| > | > | 大于 |
| <= | <= | 小于等于 |
| >= | >= | 大于等于 |
| == | == | 等于 |
|!= |!= | 不等于 |
| && | && | 逻辑与 |
| \|\| | \|\| | 逻辑或 |
|! |! | 逻辑非 |

以上运算符是C/C++中的基本运算符，它们用来执行基本的算术、逻辑以及关系运算。

注意：
- 暂不考虑指针运算符、sizeof关键字等。

## 2.5 控制语句
### if语句
| C语言 | C++ | 描述 |
| --- | --- | --- |
| if(expr) statement else statement | if(expr) statement else statement | if...else语句 |

以上语句是C/C++中的if语句，用来判断一个条件是否满足，若满足，则执行if语句块中的语句；否则，执行else语句块中的语句。

### switch语句
| C语言 | C++ | 描述 |
| --- | --- | --- |
| switch (expr){case constant: statement; break;} | switch (expr){case constant: statement; break;} | switch语句 |

以上语句是C/C++中的switch语句，用来实现多分支条件判断，通过比较表达式的值和多个case子句的值，来决定执行哪条case语句。

### while语句
| C语言 | C++ | 描述 |
| --- | --- | --- |
| while (expr) statement | while (expr) statement | while语句 |

以上语句是C/C++中的while语句，用来重复执行语句块，直到表达式的值为假。

### do-while语句
| C语言 | C++ | 描述 |
| --- | --- | --- |
| do{statement} while(expr); | do{statement} while(expr); | do-while语句 |

以上语句是C/C++中的do-while语句，首先执行语句块，然后判断表达式的值，若表达式的值为真，则继续执行语句块，直到表达式的值为假。

### for语句
| C语言 | C++ | 描述 |
| --- | --- | --- |
| for(init; cond; iter) statement | for(init; cond; iter) statement | for语句 |

以上语句是C/C++中的for语句，用来依次执行初始化语句、循环条件语句和迭代语句，重复执行语句块，直到循环条件语句为假。

## 2.6 函数
| C语言 | C++ | 描述 |
| --- | --- | --- |
| void function() {} | void function() {} | 函数声明语法 |
| return expr; | return expr; | 返回语句 |

以上语句是C/C++中的函数声明语法和返回语句，用来定义函数、执行函数调用、返回函数值。

注意：
- 暂不考虑函数参数传递、函数地址、函数指针、递归函数等。

## 2.7 指针
| C语言 | C++ | 描述 |
| --- | --- | --- |
| type* ptr; | type* ptr; | 定义指针语法 |
| *ptr = value; | *ptr = value; | 指针赋值 |
| &ref = *ptr; | auto ref = ptr; | 引用赋值 |

以上语句是C/C++中的指针声明、赋值以及引用的声明和赋值语法。

## 2.8 内存管理
C语言和C++都提供了内存管理机制。但是C++引入了对象技术，使得内存分配和释放变得复杂起来。

### malloc和free函数
| C语言 | C++ | 描述 |
| --- | --- | --- |
| void* malloc(size_t size); | std::malloc | 分配指定字节数的内存块 |
| void free(void* ptr); | std::free | 释放内存块 |

malloc和free函数是C/C++中的内存管理函数。malloc函数用来动态申请一段内存，即使内存已满也不会报错；而free函数用来释放已经分配的内存，一般是在内存泄露的时候使用。

### new和delete运算符
| C语言 | C++ | 描述 |
| --- | --- | --- |
| type* ptr = new type(); | type* ptr = new type; | 使用new运算符申请内存 |
| delete ptr; | delete ptr; | 删除动态分配的内存 |

以上语句是C/C++中的new运算符和delete运算符的声明语法。

## 2.9 其它
| C语言 | C++ | 描述 |
| --- | --- | --- |
| #include | #include | 文件包含 |
| typedef | using | 类型别名 |
| struct | class | 类声明 |
| union | - | 联合声明 |
| enumerated type | enum | 枚举类型 |
| extern | static | 外部变量 |
| register | - | 寄存器变量 |
| volatile | volatile | 可修改的变量 |