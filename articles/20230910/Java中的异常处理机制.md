
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Java 是一种静态类型、面向对象、多线程的编程语言，在面对复杂系统开发时，必然需要异常处理机制。本文从如下几个方面介绍Java的异常处理机制：

1）异常处理机制概述
2）try-catch关键字用法详解
3）throws关键字用法详解
4）异常处理的几种方式及区别
5）异常链（Exception Chaining）
6）自定义异常类
7）Java中的受检异常（Checked Exception）和非受检异常（Unchecked Exception）
8）Java 8 中引入的 Lambda 表达式（Anonymous Functions）以及其与异常处理结合的方法
9）其他注意事项
# 2.概念术语说明
## 2.1 try-catch块
在try块中放置可能出现异常的代码，当异常发生时，控制权就会转移到对应的catch块进行处理。如果没有发生异常，则继续执行try后面的语句。当try块或者catch块中的代码抛出了异常，这个过程称之为“异常抛出”。
## 2.2 throws关键字
当方法内部检测到异常并无法处理时，可以将这个异常抛给调用者。throws 关键字用来声明一个方法可能会抛出的异常类型，调用该方法的地方要捕获或者重新抛出这个异常。
## 2.3 异常栈跟踪
异常栈跟踪就是描述一个异常的产生路径。它一般包括异常类的名字，异常消息，异常产生的位置。通过异常栈跟踪信息，可以方便地定位到异常产生的原因。
## 2.4 try-with-resources语句
JDK 7 引入的新的 try-with-resources 语句，允许在 try 块中自动关闭资源。当程序运行到离开 try 块的时候，会自动调用资源对象的 close() 方法来释放资源。
## 2.5 异常链
当多个异常嵌套抛出时，Java 会将这些异常连接成一条链条，最里面的异常位于链条的底端。可以使用 getCause() 方法获取链条上一个异常。
## 2.6 用户自定义异常类
可以通过继承 Throwable 抽象类创建自己的异常类，并覆盖 getMessage() 和 toString() 方法。也可以定义一些额外的方法来提供更丰富的错误信息。
## 2.7 checked exception vs unchecked exception
在 Java 中，两种类型的异常都继承自 Throwable 抽象类，但是两者又有着不同的表现形式和作用。下面是对这两种异常类型的定义。

1. Checked exception (受检异常)：Java 不要求必须处理这种类型的异常，必须显式地声明或捕获它们。如果 unchecked exception 被忽略掉，那么程序将编译失败。例如，IOException 就是一种受检异常，因为它表示输入/输出相关的问题，如果没有捕获 IOException，程序将无法正常运行。

2. Unchecked exception (非受检异常)：Java 不要求必须处理这种类型的异常，只需要关注它的存在就可以了。Unchecked exception 可以选择不处理，程序仍能正常运行。例如 NullPointerException 表示空指针引用问题，即程序试图访问一个 null 对象上的属性或方法，这种情况对于程序来说并不是致命的，所以不需要强制处理。而 IndexOutOfBoundsException 表示数组索引越界，这个异常很容易被发现和修复，所以不需要用程序去捕获或处理。

一般来说，应该优先使用受检异常，而不是非受检异常。因为受检异常使得程序员承担了更多的责任，并且可以更好地处理异常。除此之外，异常也是一种调试技巧，可以帮助程序员找出代码中的 bug 。因此，应该尽量避免无用的 catch 或 throw。
# 3.异常处理流程图
# 4. Java中异常处理的几种方式及区别
## 4.1 try-catch
语法格式如下：
```
try {
  //可能发生异常的代码
} catch(ExceptionType e) {
  //异常处理代码
}
```
举例如下：
```
public class Test {

  public static void main(String[] args) {
    try {
      int a = 1 / 0; //抛出ArithmeticException
    } catch (ArithmeticException e) {
      System.out.println("catch ArithmeticException");
    }
  }
}
//output: catch ArithmeticException
```
## 4.2 throws
语法格式如下：
```
public static void foo() throws ExceptionType {}
```
foo() 函数中可能发生异常时，可以声明抛出异常类型。如果主函数调用 foo() 时没有处理异常，编译器将报错。

举例如下：
```
class MyException extends Exception{}

public class MainClass{
  
  public static void foo() throws MyException{
    throw new MyException();
  }
  
  public static void main(String[] args){
    try{
        bar();
    } catch (MyException e){
        System.out.println("Caught MyException.");
    }
  }
  
  private static void bar(){
    foo();
  }
}
// output: Caught MyException.
```
## 4.3 finally
finally 块总是在 try-catch 语句之后执行。finally 块主要用于释放资源，确保资源不会泄露。

举例如下：
```
public class FinallyDemo {

    public static void main(String[] args) {
        
        try {
            readFile();
        } catch (Exception ex) {
            ex.printStackTrace();
        } finally {
            closeFile(); //释放资源
        }
        
    }
    
    /**
     * 模拟读取文件
     */
    public static void readFile() throws Exception {
        File file = new File("abc.txt");
        FileInputStream fis = new FileInputStream(file);
        byte b[] = new byte[(int)file.length()];
        fis.read(b);
        String content = new String(b);
        System.out.println(content);
        fis.close();
    }
    
    /**
     * 模拟释放资源
     */
    public static void closeFile() {
        System.out.println("资源已释放！");
    }
}
```
## 4.4 异常链
假设 foo() 函数调用了 bar() 函数，而 bar() 函数又调用了 baz() 函数，如果在 baz() 函数中发生异常，那么 Java 会将异常一直往上传递，直到达到程序入口点才会被捕获并处理。

举例如下：
```
public class MyExceptionA extends Exception {}
public class MyExceptionB extends Exception {}

public class MainClass {
    
    public static void foo() throws MyExceptionA {
        try {
            bar();
        } catch (MyExceptionB ex) {
            throw new MyExceptionA(ex);
        }
    }
    
    public static void bar() throws MyExceptionB {
        baz();
    }
    
    public static void baz() throws MyExceptionB {
        throw new MyExceptionB();
    }
    
    public static void main(String[] args) {
        try {
            foo();
        } catch (MyExceptionA ex) {
            System.out.println("Caught A!");
            while (ex!= null) {
                System.out.println("Caused by " + ex.getClass().getSimpleName());
                if (ex instanceof MyExceptionB &&!ex.getCause().equals(null)) {
                    ex = (MyExceptionA) ex.getCause(); //处理异常链
                } else {
                    break;
                }
            }
        }
    }
}
// output: Caught A! Caused by B
```
## 4.5 用户自定义异常类
可以通过继承 Throwable 抽象类创建自己的异常类，并覆盖 getMessage() 和 toString() 方法。例子如下：
```
public class CustomException extends Exception {
 
    private static final long serialVersionUID = -1L;
     
    private String message;
     

    public CustomException() {
         
    }

    public CustomException(String message) {
          this.message = message;
    }

    @Override
    public String getMessage() {
        return super.getMessage()+":"+this.message; 
    }
 
    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(super.toString()).append(":").append(this.message).append("\n");
         
        StackTraceElement[] elements = Thread.currentThread().getStackTrace();
        for (StackTraceElement element : elements) {
            sb.append("\tat ").append(element).append("\n");
        }
         
        return sb.toString();
    }  
}
```