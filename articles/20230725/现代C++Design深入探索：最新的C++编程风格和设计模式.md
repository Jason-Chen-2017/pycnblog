
作者：禅与计算机程序设计艺术                    

# 1.简介
         
随着时代的变化，编程语言也在不断地进化，而学习掌握最新、最好的编程语言显得尤为重要。从C++98开始，到最近才逐渐进入C++17时代，C++已经发展成为世界上主流高级编程语言。随着越来越多的工程师加入到编程行列中，C++语言生态系统正在崛起，各种各样的框架、库、工具层出不穷，总结起来就是一个庞大的体系。但由于C++语言本身有限的特性（例如运行效率、安全性、可移植性等），同时也受到很多方面的限制（例如语言规则繁琐难懂、语法兼容性差等），使得它的应用场景有限。为此，C++社区推出了一些“现代C++”的编码规范，如Google推荐的《C++ Core Guidelines》、LLVM官方提供的《LLVM Coding Standards》、Facebook开源的《Effective Modern C++》等。本文将主要讨论这些编码规范中的最佳实践和模式。
2.最佳实践
## 2.1 提倡统一的命名方式
在阅读完大量的代码之后，命名往往会成为一个沉重的负担。如果单独针对某个模块或项目进行命名，很容易出现不同模块或项目之间的名字冲突。因此，命名的一致性和唯一性是非常重要的。命名的一般做法包括驼峰式、下划线分隔符、匈牙利命名法。统一的命名方式可以让其他工程师、工具可以轻松地理解你的代码。以下是一个命名示例：
```cpp
class Animal {
  public:
    void makeSound(); // good!
  private:
    int m_numberOfLegs; // bad!
    std::string sound_;   // better?
};
```
## 2.2 使用const关键字进行常量限定
在面向对象的编程中，常量是指不能修改的值，这样做可以在一定程度上提高程序的性能。而在C++中，常量通常都是通过const修饰的变量实现的。但是，并不是所有时候都需要用const关键字来修饰变量，我们应该只在必要的时候加上const关键字，减少不必要的误用。另外，对于函数返回值也要注意const关键字的作用。
```cpp
int calculate(int x) const{  // 正确！
  return x * x + y;        // 错误！y没有初始化
}

std::string const getName() const{ // 正确！
  return "John";                 // 不可修改的常量
}

void printMessage(){             // 错误！const修饰的方法不应该修改成员变量
  message_ = "hello world";      // 可修改的成员变量
}
```
## 2.3 使用nullptr而不是NULL作为空指针常量
C++11引入了一个新的关键字nullptr来表示空指针常量，它比NULL更具有表达意义，并且能够明确地标识空指针。所以，建议使用nullptr来替换NULL。
```cpp
char* str = nullptr;     // NULL
bool isNull = (str == nullptr);    // true if str is null
```
## 2.4 在构造函数中避免分配资源
在构造函数中分配资源并不安全，因为在发生异常时可能导致内存泄漏。通常情况下，构造函数应该尽可能简单易懂，不要涉及资源管理相关的代码。比如，可以使用堆栈上预先分配的空间来存储局部变量。
```cpp
class Person {
  public:
    Person(const std::string& name):
      age_(0),
      name_(name){
        address_ = new char[100];   // error! Do not allocate resources in constructor
    }

  private:
    int age_;
    std::string name_;
    char* address_;         // heap-allocated memory for storing address string
};
```
## 2.5 使用std::move来移动对象
在C++11中引入了std::move()函数，用来将右值转为左值引用。在传参过程中，如果希望避免对象被拷贝，则可以使用std::move()来移动对象。std::move()的使用应该遵循两个原则：第一，仅在右值时才使用；第二，在使用前必须对原始对象进行拷贝。
```cpp
void processObject(MyClass&& obj){       // ok
  // use the object here...
  MyClass localCopy = obj;                // avoid copy to save time and space
}

void swapObjects(MyClass& a, MyClass& b){ 
  using std::swap;                        // use of std library inside namespace
  swap(a,b);                               // pass objects by reference and allow move semantics to be used
} 

template<typename T>                      // template function that takes rvalue references as arguments can also take lvalues
void reverseRange(T first, T last){
  while(first < last){                    // handle forward or backward iteration based on type of iterator provided
    swap(*first,*last);                     // dereference pointers before swapping them
    ++first;                                // advance the first pointer
    --last;                                 // decrement the last pointer
  }
}
```
## 2.6 用override和final来规范虚函数的声明
在C++11中，编译器提供了三种方法来声明虚函数：virtual，override和final。其中，virtual用于声明虚函数，override用于指示该虚函数是重新定义父类的虚函数，final用于防止虚函数被覆盖。这三个关键字的使用需要遵守一些规律：
1. override只能用于虚函数重载；
2. final不能用于构造函数或者复制构造函数；
3. 如果某个类中含有覆盖虚函数的成员函数，则这个类中所有虚函数都需要加上override关键字。
```cpp
class Shape {                                    // base class with virtual functions
  public:                                        
    virtual double area() const = 0;              // pure virtual function declaration
    virtual ~Shape() {}                           // destructor needs override keyword
    
};                                                  
                                                       
class Circle : public Shape {                    // derived class with implementation
  public:                                         
    double radius_{0};                             // member variable definition

    double area() const override {                 // redefinition of virtual function from Shape class
      return M_PI * radius_ * radius_;            // circle formula for computing area
    };                                             
                                                        
};                                                   
                                                      
Circle c{};                                        // instance creation of Circle class
                                                     
                                                       
double getArea(const Shape& s) {                   // helper function to compute area
    return s.area();                              // passing reference to shape object allows call to correct overloaded version of area method
}                                                    
                                                     
                                                       
getArea(c);                                        // calling getArea function to compute area
                                                         
```

