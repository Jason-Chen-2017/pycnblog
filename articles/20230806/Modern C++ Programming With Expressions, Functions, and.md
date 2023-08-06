
作者：禅与计算机程序设计艺术                    

# 1.简介
         
　　C++是一个很重要的语言，它的强大的表达能力及其丰富的类库让编程变得十分简单和容易。C++提供了函数、表达式、泛型编程等多种方法来支持高效的开发。本专栏将介绍如何利用C++的这些特性进行编程，包括表达式、函数、容器等技术。
          
         # 2.基本概念术语说明
         　　C++编程中涉及到的一些基础概念和术语如下表所示：
          
            |   名称     |    说明        | 
            | ----------| ---------------:| 
            | 表达式    |     一个运算符和一个或多个操作数构成的语句          | 
            | 函数      |     在程序中定义的可执行代码块，用于完成特定任务            | 
            | 参数      |     表示函数输入值的变量名               | 
            | 返回值    |     表示函数输出的值的变量名           | 
            
            其中，表达式由运算符和操作数组成，表示某些值的计算结果。比如，加法表达式a+b表示两个操作数a和b相加得到的值。在C++中，可以使用各种表达式来对变量值进行赋值、比较大小、逻辑运算等操作。
            
         　　函数是一种声明和定义都有的程序实体，它允许我们将一段代码封装到一个名字中，并可以被其他地方调用。C++中的函数有三种类型：普通函数、重载函数、成员函数。普通函数没有任何返回值，只用于实现功能；重载函数是在同一个作用域内，根据不同的参数类型或者参数数量来区分不同的函数版本；成员函数是类的一部分，只能通过对象访问，而不是独立于对象而存在。例如，vector类的push_back()方法就是一个成员函数。
          
         　　容器是C++中最重要的数据结构之一，用来存储和管理数据集合。常用的容器有数组、列表、向量、散列表、堆、优先队列等。每个容器都提供了一些标准化的方法和接口来操作元素，使得我们不需要关心底层的实现细节。比如，vector类提供了一个insert()方法，可以在指定位置插入新的元素，而list类则提供了头插和尾插两种方式。
          
         　　# 3.核心算法原理和具体操作步骤以及数学公式讲解
          　　本专栏将从以下几个方面讲述C++编程的一些高级技术：表达式、函数、容器等。在这里，我会逐个介绍C++中这些技术的概念、原理和应用场景。

          ## 3.1 表达式
         　　表达式（expression）是C++的一个重要语法构造，是一种抽象语法树（Abstract Syntax Tree，AST）的叶节点，代表了单个的值、操作符或者函数调用。比如，表达式“2+3”包含了加法运算符和两个操作数2和3。表达式可以嵌套组合起来，形成复杂的表达式。表达式一般有四种形式：
          
            - 值表达式（value expression）：表达式的值恒定，可以直接作为值使用。例如，数字、字符串、布尔值、指针、枚举等都是值表达式。
            - 操作符表达式（operator expression）：由运算符和操作数组成的表达式。如a + b。
            - 函数表达式（function expression）：调用某个已定义的函数，并将该函数的返回值赋值给某个变量。如int x = max(3, 7);。
            - 构造表达式（constructor expression）：创建一个对象的表达式。如MyClass myObject;。
            
         　　表达式的主要用途是作为语句的一部分，也可以作为另一个表达式的子表达式。表达式也可以隐式转换为其它类型的表达式。例如，当将int型值赋给bool型变量时，编译器会将int型表达式隐式地转换为bool型表达式。

          ### 3.1.1 常量表达式（constant expressions）
         　　常量表达式（constant expression）是指在编译期间能够确定结果的表达式。常量表达式通常出现在constexpr限定的变量定义、模板参数列表、模板实参中。常量表达式在运行时不会发生变化，所以它们的结果可以在编译期间就得到计算。下面列举一些常量表达式的示例：

            ```c++
            constexpr int a = 1 + 2 * 3; // 9
            
            template<typename T>
            constexpr bool is_even(T value){
                return!(value & 1);
            }
            
            struct Point {
                double x, y;
                
                constexpr Point operator+(const Point& other) const {
                    return {x + other.x, y + other.y};
                }
            };
            
            constexpr auto origin{Point{}};
            ```
            
         　　上面的例子中，变量a是一个常量表达式，因为它是一个整数乘法表达式，它的结果在编译期间就可以确定。函数is_even是一个常量表达式，因为它的判断条件不是变量，而且在编译期间可以知道确切的值。Point::operator+也是一个常量表达式，因为它的操作数都是常量表达式，它们的结果在编译期间就可以确定。最后，变量origin是一个初始值定义，它的结果也是一个常量表达式。
            
         　　常量表达式的另外一个用处是替代宏，因为常量表达式在编译期间就已经确定下来，宏的值不一定每次都相同。

          ### 3.1.2 constexpr函数（constexpr function）
         　　constexpr函数是指可以在编译期间执行的代码。它是指满足一定条件的函数，可以在编译期间计算出结果，并且结果的值在运行时不会改变。在函数体中只允许出现常量表达式，并且除了返回类型外，不能有其他的副作用。constexpr函数的声明必须在函数前加上关键字constexpr，然后再使用圆括号对函数的参数列表进行包裹。下面给出示例：

            ```c++
            constexpr int square(int n) {
                return n*n;
            }
            
            void f(){
                static_assert(square(2)+1 == 5,"");
            }
            ```
            
         　　上面这个示例中，函数square是constexpr函数，它的返回值是常量表达式，所以可以在编译期间就计算出来，并且结果的值在运行时不会改变。static_assert是一条控制语句，只有在条件为假的时候才会触发异常。当条件为真的时候，宇宙中的所有物质都会消失。
            
         　　在模板参数中也可以声明constexpr函数，但需要注意的是，constexpr函数不能递归地调用自己。下面给出示例：

            ```c++
            template <size_t N>
            constexpr size_t factorial() {
                if constexpr (N <= 1) {
                    return 1;
                } else {
                    return N * factorial<N-1>();
                }
            }
            ```
            
         　　这个示例展示了一个constexpr函数factorial，它的返回值也是常量表达式，但是却有一个限制条件，即它只能在编译期间计算。

          ### 3.1.3 变量声明
         　　变量的声明语法如下：

            ```c++
            type variable_name = value;
            ```
            
         　　变量的声明包括类型、变量名、初始化值三个部分。对于那些只能在编译期间确定的值，可以通过constexpr的方式进行声明，这样就可以在编译期间就确定这个值，进而减少运行时的开销。

          ### 3.1.4 sizeof关键字
         　　sizeof关键字用来获取变量、类型或表达式所占据的字节数。它的语法如下：

            ```c++
            sizeof operand
            ```
            
         　　operand可以是变量、类型或表达式。sizeof关键字返回值是一个无符号整数类型，因为不同平台上的内存对齐方式可能不同。如果不知道使用多少字节，建议改用其他方法来确定。

          ### 3.1.5 初始化列表（initializer list）
         　　初始化列表（initializer list）是一个花括号{}括起来的由逗号分隔的值列表，用于在创建对象、传递参数时提供初始值。初始化列表一般与类相关联，它用于构造对象，类似于构造函数。下面是一个示例：

            ```c++
            class MyClass {
                public:
                    explicit MyClass(double val=0): mValue{val} {}
                    
                private:
                    double mValue;
            };
            
            MyClass obj{10.5};
            ```
            
         　　这个例子中，MyClass有一个显式的构造函数，它的参数列表为空，因此可以省略，但是mValue有一个默认值，因此我们需要显示地传入一个初始值。实例化对象obj时，我们给出了初始值10.5。

          ### 3.1.6 decltype关键字
         　　decltype关键字用来获取表达式的静态类型，并且decltype是一种元编程技术。它的语法如下：

            ```c++
            decltype(expression)
            ```
            
         　　表达式可以是任意合法的表达式，它可以是变量、函数调用、算术表达式、指针、引用、成员访问、函数指针等。decltype的作用是告诉编译器待解析的表达式的实际类型，因此可以用来声明一个具有相同类型的局部变量。下面给出一个示例：

            ```c++
            int x = 5, y = 10;
            
            std::cout << "Before swap:" << std::endl;
            std::cout << "x=" << x << ", y=" << y << std::endl;
            
            using std::swap;
            swap(x, y);
            
            std::cout << "After swap:" << std::endl;
            std::cout << "x=" << x << ", y=" << y << std::endl;
            ```
            
         　　这个例子中，x和y都是int型变量，但是由于他们的内存地址是不同的，因此无法直接互换。如果我们想交换x和y的值，我们应该先把这两个变量的值保存起来，然后再将y的值赋给x，最后再将x的值赋给y。然而，如果我们直接交换两个变量的值，那么就会导致行为不可预料。为了保证行为的一致性，我们可以使用std::swap来进行交换，而不是自己手写交换代码。但是如果我们要声明一个局部变量并将它类型设置为decltype(x)，那么decltype(x)的静态类型将是int，而不是std::swap的函数签名。

          ### 3.1.7 模板类型推导
         　　C++中的模板机制是一个很强大的工具，它可以帮助我们避免重复的代码，提升编程效率。模板允许我们创建泛型函数和类，使得它们可以适应不同的类型。但是，使用模板时也需要注意两点：首先，模板是由编译器生成代码，因此它的效率依赖于编译器的优化选项；其次，模板可能会导致编译时间过长，因此我们应该做好计划，并控制模板的使用范围。下面，我们给出模板类型推导的示例：

            ```c++
            template<class T>
            void func(T param){
                cout << typeid(param).name() << endl;
            }
            
            int main(){
                int i = 5;
                float f = 3.5f;
                char c = 'a';
                
                func(i);
                func(f);
                func(c);
                
                return 0;
            }
            ```
            
         　　这个例子中，func函数是一个模板，它的模板参数是一个类型T。我们给出了三个不同类型的值，然后分别调用了func函数。因为func是一个模板，所以编译器需要为每一种类型生成一次代码。但是在模板类型推导的情况下，编译器会根据传入的值自动生成不同的代码，因此我们只需要生成一次代码即可。

          ### 3.1.8 可变参数模板
         　　模板中的参数列表还可以包含可变参数，这意味着这个模板可以接受任意数量的实参。这种特性使得我们编写泛型代码更加灵活，可以处理各种类型的参数。下面给出一个可变参数模板的示例：

            ```c++
            template<typename... Args>
            int sum(Args&&... args){
                int result{};
                ((result += args),...);
                return result;
            }
            
            int main(){
                cout << sum(1,2,3) << endl;
                cout << sum('a', 'b', 'c') << endl;
                string s{"hello"};
                vector<string> v{s};
                cout << sum("abc",v,4) << endl;
            }
            ```
            
         　　这个例子中，sum函数是一个模板，它的参数列表包含可变参数。这个模板可以处理任意数量的实参，并且编译器会为其生成不同的代码。在main函数中，我们调用了sum函数，传入不同类型的参数，并打印了其结果。

          ## 3.2 函数
         　　函数（function）是C++的核心组件之一，可以将一段代码封装到一个命名的实体中，供其他地方调用。C++中提供了多种类型的函数，包括普通函数、重载函数、成员函数等。除此之外，还有函数指针、lambda表达式等高阶技巧。下面我们会逐个介绍C++中的函数知识。

          ### 3.2.1 函数定义
         　　函数的定义语法如下：

            ```c++
            returnType functionName(parameterList){
                body of the function;
            }
            ```
            
         　　returnType是函数的返回类型，可以是void、任意用户自定义类型、引用或指针；parameterList是函数的参数列表，它是一个参数类型列表，以逗号分隔；body of the function是函数的主体，是函数执行的具体指令。函数定义包括函数头和函数体。

          ### 3.2.2 默认参数
         　　C++中支持函数的默认参数，即如果调用函数时没有提供相应的参数值，系统会采用默认参数来代替。下面给出一个默认参数的示例：

            ```c++
            #include <iostream>
            
            using namespace std;
            
            void printMessage(char message[] = "Hello World!") {
               cout << message << endl;
            }
            
            int main() {
               printMessage();  // Outputs "Hello World!"
               printMessage("Greetings!");  // Outputs "Greetings!"
               return 0;
            }
            ```
            
         　　这个例子中，printMessage函数有一个默认参数，即message的值为"Hello World!”。调用printMessage()时没有给出参数值，因此它会使用默认参数；调用printMessage("Greetings!")时，它会使用指定的参数值。

          ### 3.2.3 局部作用域
         　　局部作用域（local scope）是指仅在当前函数中有效的区域。在函数体内定义的变量称作局部变量，它的生命周期只在函数调用期间。局部作用域可以访问外部作用域中的变量，但是外部作用域不能访问局部作用域的变量。

          ### 3.2.4 重载
         　　函数重载（overloading）是指在同一个作用域中，可以存在同名不同参数的函数。它可以提高代码的易读性和复用性。下面给出函数重载的示例：

            ```c++
            #include <iostream>
            
            using namespace std;
            
            void display(int num){
               cout<<"Number is : "<<num<<endl;
            }
            
            void display(float num){
               cout<<"Float number is : "<<num<<endl;
            }
            
            int main() {
               int a = 10;
               float b = 5.5f;
               
               display(a);   //Calls first overloaded version of display()
               display(b);   //Calls second overloaded version of display()
               return 0;
            }
            ```
            
         　　这个例子中，display函数是一个重载函数，它可以接受不同类型的参数，因此可以同时处理整数和浮点数。调用display()时，系统会自动选择匹配的函数版本，以处理对应的参数类型。

          ### 3.2.5 匿名函数
         　　匿名函数（anonymous function）是一个没有名字的函数，它是一个函数对象，可以像变量一样进行传递。它可以用于函数式编程中，用于对函数的编程模型进行统一。下面给出匿名函数的示例：

            ```c++
            #include <algorithm>
            #include <functional>
            #include <iostream>
            
            using namespace std;
            
            int main() {
               int arr[5] = { 10, 20, 30, 40, 50 };
               transform(arr, arr + 5, arr, [](int num){return num / 2;});
               
               for(auto num : arr)
                  cout<<num<<" ";
                  
               return 0;
            }
            ```
            
         　　这个例子中，transform函数是一个泛型算法，它可以对容器中的每个元素应用一定的变换函数。在这个例子中，匿名函数会对每个元素求一半，并将结果放入arr容器中。

          ### 3.2.6 右值引用
         　　右值引用（right value reference）是一个新的引用类型，它可以绑定到临时对象的右值，可以避免拷贝临时对象造成的额外开销。它的语法如下：

            ```c++
            returnType (&func(parameterTypes)) (argumentTypes...)
            ```

         　　在括号中，&用于表示引用，返回类型后面加上一个空格，再加上左小括号，形成引用符号。返回类型之前的&和参数列表之间需要用空格隔开。

          ### 3.2.7 move语义
         　　move语义可以将一个左值转移到右值引用，它是一个非常重要的概念，因为它使得移动语义和拷贝语义的行为保持一致，可以简化我们的代码。下面给出move语义的示例：

            ```c++
            #include <utility>
            #include <iostream>
            
            using namespace std;
            
            int main() {
               int a = 10;
               int b = 20;
               
               cout<<a<<"    "<<b<<endl;
               
               a = ::move(b);   //moves content from b to a without copying it
               cout<<a<<"    "<<b<<endl;
               
               b = 30;
               cout<<a<<"    "<<b<<endl;
               
               return 0;
            }
            ```
            
         　　这个例子中，a和b都是整型变量，且都有初始值。我们首先输出a和b的值，然后将b的值赋值给a，但是只是移动b的内容而不拷贝它。输出a和b的值后，我们又把b的值重新赋值为30。a的值已经从20变为了30，但是b的值仍然为20。这说明，a的值是通过b的值来移动的，因此b的值已经失去了它的原始含义。

          ### 3.2.8 函数指针
         　　函数指针（function pointer）是一个指向函数的指针，可以指向任意类型的函数。它的语法如下：

            ```c++
            returnType (*pointerToFunction)(parameterTypes)
            ```

         　　pointerToFunction是一个指针变量，它指向一个函数，它的返回类型必须与返回类型一致。

          ## 3.3 容器
         　　容器（container）是一个数据结构，用来存储和管理数据集合。C++提供了多种类型的容器，包括数组、列表、向量、链表、哈希表、堆栈、队列等。下面我们将逐一介绍C++中常用的几种容器。

          ### 3.3.1 数组
         　　数组（array）是一个固定大小的顺序容器，里面可以存放同种类型的数据。数组的索引从0开始，可以通过下标访问数组中的元素。数组的声明语法如下：

            ```c++
            dataType arrayName[arraySize];
            ```

         　　dataType是数组中数据的类型，arraySize是数组的大小。数组中的元素通过连续的内存空间存放在一起，因此数组越长，访问效率越高。

          ### 3.3.2 列表
         　　列表（list）是一个双向链表结构，可以从头和尾两端添加或删除元素。它的查找速度较慢，但增删元素速度快。列表的声明语法如下：

            ```c++
            list<dataType> listName;
            ```

         　　dataType是列表中数据的类型。

          ### 3.3.3 向量
         　　向量（vector）是一个动态数组，可以增删元素，大小可以随意调整。它的时间复杂度为O(log(n)),但是查找元素的时间复杂度为O(n)。向量的声明语法如下：

            ```c++
            vector<dataType> vecName;
            ```

         　　dataType是向量中数据的类型。

          ### 3.3.4 字符串
         　　C++中没有直接提供字符串这种数据类型，但是可以通过字符数组实现字符串。字符串的声明语法如下：

            ```c++
            char charArrayName[stringLen + 1];
            ```

         　　stringLen是字符串的长度，注意末尾有一个空格。字符串的每个字符占用一个字节的内存空间，因此它比字符数组稍微大一点，不过能容纳更多的字符。

          ### 3.3.5 迭代器
         　　迭代器（iterator）是一个对象，它可以遍历容器中的元素。它可以前往第一个元素、后往最后一个元素、前往下一个元素、前往上一个元素等。迭代器的类型依赖于容器的类型。下面给出迭代器的声明语法：

            ```c++
            containerType iteratorName;
            ```

         　　containerType是容器的类型，iteratorName是迭代器的名称。

          ### 3.3.6 插入和删除元素
         　　插入元素（insertion operation），删除元素（deletion operation），合并操作（merge operation），排序操作（sort operation），搜索操作（search operation），替换操作（replace operation），随机访问操作（random access operation）等操作都是一些常用的容器操作。

          ### 3.3.7 map和multimap
         　　map和multimap是关联容器，它们提供一种快速检索元素的方法。map的元素是键值对，其关键字是唯一的，而multimap的元素是键值对的集合，关键字可以重复。它们的声明语法如下：

            ```c++
            map<keyType,valueType> mapName;
            multimap<keyType,valueType> mmapName;
            ```

         　　keyType是关键字的类型，valueType是值得类型。

          ### 3.3.8 set和multiset
         　　set和multiset是基于红黑树实现的集合容器，它们提供一种高效的插入和查找操作。set和multiset的元素必须是唯一的，因此它们不能有重复的元素。set和multiset的声明语法如下：

            ```c++
            set<valueType> setName;
            multiset<valueType> msetName;
            ```

         　　valueType是元素的类型。

          ### 3.3.9 heap
         　　堆（heap）是一种特殊的完全二叉树，它满足堆序性质，可以高效地进行元素的插入和删除操作。堆的声明语法如下：

            ```c++
            priority_queue<valueType> pqName;
            ```

         　　valueType是堆中元素的类型。

          ### 3.3.10 queue和stack
         　　队列（queue）是一个先进先出的顺序容器，栈（stack）是一个后进先出的顺序容器。它们都可以从容器的前端添加元素或者弹出元素。队列的声明语法如下：

            ```c++
            queue<valueType> qName;
            stack<valueType> stkName;
            ```

         　　valueType是队列和栈中元素的类型。

          ## 3.4 并发编程
         　　并发编程（concurrency programming）是指在一台计算机上同时运行多个进程或线程，目的是提高程序的执行效率。在现代操作系统中，通过引入线程和进程等概念，我们可以轻松地实现并发编程。下面，我们将讨论C++中线程和锁的基本概念。

          ### 3.4.1 线程
         　　线程（thread）是操作系统调度的最小单位，它是由程序计数器、栈、寄存器等组成的独立执行序列。线程的特点是轻量级、独立、可拥有自己的栈。线程的创建和终止都由操作系统完成，因此应用程序不需要显式地完成线程切换。线程的创建语法如下：

            ```c++
            thread thrdObj(functionName, arg1, arg2,...);
            ```

         　　functionName是新线程执行的函数，arg1, arg2, …是传递给函数的参数。

          ### 3.4.2 锁
         　　锁（lock）是保护共享资源的同步机制。它可以防止多个线程同时访问共享资源，以达到独占访问的效果。锁有两种状态——排他锁（exclusive lock）和共享锁（shared lock）。排他锁是独占的，只能有一个线程持有，共享锁是可共享的，可以在多个线程同时持有。锁的声明语法如下：

            ```c++
            mutex mtxObj; //for exclusive lock
            shared_mutex shrdMtxObj; //for shared lock
            ```

         　　在C++11之后，可以使用std::atomic数据结构来代替mutex，它提供了更好的性能。

          ### 3.4.3 同步问题
         　　同步问题（synchronization problem）是指多个线程之间的通信和协调问题。下面列举一些常见的同步问题：

              - 竞争条件（race condition）：多个线程同时修改同一个资源导致的错误结果。
              - 活跃性（liveness）：程序终止的错误结果。
              - 死锁（deadlock）：两个或多个线程因等待对方占用的资源而陷入僵局。
              - 饥饿（starvation）：一种特殊情况的活跃性问题。
              
         　　解决同步问题有两种方法：信号量（semaphore）和互斥量（mutex）。信号量和互斥量都可以实现同步，但是信号量比互斥量更灵活。

          ### 3.4.4 事件通知
         　　事件通知（event notification）是指通知线程之间某个事件是否发生的机制。在Windows操作系统中，应用程序可以注册事件通知，等待某个事件发生。事件通知的声明语法如下：

            ```c++
            event evtObj;
            ```

         　　evtObj是事件通知对象。

          ## 3.5 反射
         　　反射（reflection）是计算机程序设计中一项重要技术，它可以让程序在运行时取得自身的信息，并根据信息执行不同的操作。在C++中，可以使用反射机制来获取类的信息，创建对象、调用函数等。下面我们将介绍反射的一些基本概念。

          ### 3.5.1 元数据
         　　元数据（metadata）是关于数据的数据。元数据描述了数据的数据，比如数据的类型、大小、布局等。在C++中，可以通过RTTI（Run Time Type Identification）获得类的信息。RTTI的声明语法如下：

            ```c++
            dynamic_cast<destinationType>(sourceObjectPointer)
            ```

         　　dynamic_cast用于将sourceObjectPointer指针转化为destinationType类型的指针。

          ### 3.5.2 属性
         　　属性（property）是某个对象的特征，可以通过反射机制来获取或设置。属性的声明语法如下：

            ```c++
            object.propertyName = propertyValue;
            ```

         　　object是某个类的对象，propertyName是属性的名称，propertyValue是属性的值。

          ### 3.5.3 类信息
         　　类信息（class information）是类及其对象的信息，包括名称、基类、成员函数、成员变量、属性等。在C++中，可以使用typeid运算符获取类的信息，也可以使用反射机制来获取类的信息。

          ## 3.6 模板
         　　模板（template）是C++中一种通用的机制，可以定义一个泛型函数或类，这个函数或类可以针对不同的类型进行实例化，从而实现代码的重用。模板的声明语法如下：

            ```c++
            template <typename parameterType>
            return_type functionName(parameters){
                // code here
            }
            
            template class className<parameterType>;
            ```

         　　模板参数列表包括一个或多个类型参数，通过关键字typename进行标识，return_type是函数的返回类型，parameters是函数的参数列表。模板的实例化语法包括函数模板和类模板。

          ## 3.7 异常
         　　异常（exception）是一种用来表示运行时错误的机制。它可以方便地通知调用者发生了错误，并能帮助定位错误的源头。C++中提供了try-catch语句来处理异常。下面我们给出try-catch的基本语法：

            ```c++
            try {
                // code that may throw an exception
            } catch(exceptionType1 e1) {
                // code to handle exceptionType1 errors
            } catch(exceptionType2 e2) {
                // code to handle exceptionType2 errors
            } catch(...) {
                // default code in case of unhandled exceptions
            }
            ```

         　　exceptionType是捕获的异常类型，e是捕获的异常对象。Catch-all（...）是指在没有明确处理的情况下，处理所有未知的异常。

          ## 3.8 文件输入输出
         　　文件输入输出（file input/output）是一种常见的I/O操作，它可以向文件中写入或读取数据。C++中的fstream类用于文件I/O操作。文件的打开模式包括读、写、追加、文本、二进制。下面我们给出文件的打开语法：

            ```c++
            ifstream infile("inputFileName");
            ofstream outfile("outputFileName");
            ```

         　　ifstream用于读文件，ofstream用于写文件。文件读写的完整语法如下：

            ```c++
            fileName.open(filePath, openMode);
            
            // read or write file contents
            
            fileName.close();
            ```

         　　filePath是文件的路径，openMode是文件的打开模式。