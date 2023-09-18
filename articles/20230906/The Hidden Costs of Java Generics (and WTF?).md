
作者：禅与计算机程序设计艺术                    

# 1.简介
  


## 什么是泛型？
在面向对象编程语言中，泛型（generics）指的是类型参数化。也就是说，可以将类型参数（type parameter）同某个类或者接口绑定，使得该类或接口可以被不同的数据类型所实例化。例如，Java中的集合框架Collections允许用户指定容器的类型，以适应不同的需求。因此，Java中的泛型最初也是为了解决类型擦除带来的限制而引入的。

```java
List<Integer> list = new ArrayList<>(); //声明一个泛型列表
list.add(new Integer(1));            //添加整数元素
Object obj = list.get(0);              //获取第一个元素
System.out.println(obj instanceof Integer);//输出true
```

上面的例子展示了如何使用泛型创建并使用一个带泛型的ArrayList。当用到list.get(0)时，编译器会将其认为是一个Object类型的引用。因为ArrayList只知道Object是一个引用类型，所以不能直接将整数值赋给它。但由于ArrayList已经被声明成了一个带泛型的类型，所以编译器会对它的get()方法进行类型检查，确保传入的参数是正确的类型。这里就发生了泛型擦除。

泛型在Java编程中扮演着至关重要的角色。它提供了一种灵活和易于使用的方式来编写代码。但是，这种灵活性也带来了一些隐藏的代价。本文旨在探讨泛型背后的一些秘密，包括它们的局限性、影响、风险等。

## 为何需要泛型？

Java泛型最初是为了解决类型擦除带来的限制而引入的。

对于任意一个Java源文件，编译器都会去掉所有泛型信息，然后编译出没有泛型的字节码文件。然后，JVM运行时根据字节码文件里的信息来动态加载这个类，把泛型信息擦除后再调用方法。

通过擦除，编译器不会保留任何关于泛型的类型信息。相反地，它只知道这个类是一个泛型类，并且假设所有的类型都是Object。这样做的原因之一是为了保证运行效率。

通常来说，泛型用来提高代码的灵活性和复用性。比如，在多个线程中处理相同的数据类型，就可以用到泛型。

```java
public class ThreadSafeStack<T> {
    private final Stack<T> stack;
    
    public ThreadSafeStack(){
        this.stack = new Stack<>();
    }
    
    public synchronized void push(T item){
        stack.push(item);
    }
    
    public synchronized T pop(){
        return stack.pop();
    }
}
```

ThreadSafeStack是一个线程安全栈，可以存储任意类型的数据。由于栈是同步的，因此可以由多个线程同时访问它。

但是，Java泛型带来了额外的复杂性。在某些情况下，泛型会让代码变得繁琐，比如在集合框架中，必须强制类型转换才能实现泛型。此外，泛型还增加了运行时的开销。尤其是在代码量较大的项目中，编译时间可能会增长很多。因此，在决定是否使用泛型时，开发者应该要慎重考虑。

## 什么是类型擦除？

类型擦除是Java泛型所存在的问题之一。它意味着当我们声明一个泛型类型时，编译器会自动删除掉所有的泛型信息，仅保留原始类型。换句话说，如果我们定义了一个`List<String>`，实际上编译器只是生成了一个`List`，而不管它具体是什么数据类型。这就是类型擦除。

例如，以下两个变量声明：

```java
List<String> strList;
List<?> wildcardList;
```

其实都声明了一个通配符类型变量。在执行`strList.getClass()`时，结果是`Class<? extends List<String>>`，而在执行`wildcardList.getClass()`时，结果是`Class<? extends List<?>>`。

## 泛型的局限性

泛型虽然在一定程度上提升了代码的灵活性和可读性，但是也同时带来了一些局限性。这些局限性包括：

1. 无法创建泛型数组，只能创建Object数组；
2. 无法安全地传递泛型的值，只能传递Object的值；
3. 有些地方不支持协变返回类型（covariant returns），导致泛型类的设计很难满足；
4. 受限于擦除机制，导致泛型类的运行时类型可能与预期不符。

### 无法创建泛型数组

如上所述，Java泛型会被擦除，导致无法创建泛型数组。这意味着，我们无法像下面这样创建一个具有类型参数T的数组：

```java
T[] array = new T[size];//错误
```

这是因为擦除后，数组的元素类型变成了Object。因此，无法确定泛型数组的元素具体是哪个类型。

不过，可以通过反射的方式来创建泛型数组。首先，先创建一个Object数组，然后通过反射设置其元素类型为T，最后转型为T数组：

```java
T[] createArray(int size){
    Object[] arr = new Object[size];
    try{
        java.lang.reflect.Array.setComponentType(arr.getClass(), Class.forName("type parameter T"));
        @SuppressWarnings("unchecked")
        T[] result = (T[]) arr;
        return result;
    } catch(Exception e){
        throw new RuntimeException(e);
    }
}
```

这个方法接收一个整数作为参数，并返回一个新的泛型数组。它使用反射将Object数组的组件类型设置为类型参数T，然后转型为泛型数组。注意，这个方法可能抛出异常，需要捕获并处理。

### 无法安全地传递泛型的值

与数组一样，泛型也会被擦除，导致无法安全地传递泛型的值。例如，如果有一个函数参数类型为`T`，那就意味着这个参数可以接受任何类型的值。然而，这也意味着我们无法保证传入的参数值正确，只能保证类型正确。

此外，如果函数内部修改了泛型变量的值，那么就会引起潜在的运行时错误。

总之，基于泛型的编程是一种高度依赖类型系统的编程范式，但是由于缺乏类型擦除，导致了很多问题。我们需要更加小心地使用泛型，以避免出现意想不到的问题。