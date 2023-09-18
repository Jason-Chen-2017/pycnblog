
作者：禅与计算机程序设计艺术                    

# 1.简介
  

泛型（Generics）是Java 5引入的重要特征之一，它可以让编译器在编译期间检查类型安全，并自动转换对象类型。然而，Java 泛型语法在某些方面存在一些限制，因此在本文中将详细介绍Java泛型中的两个重要概念——擦除(Type Erasure)和通配符(Wildcard)。希望通过阅读此文，读者能够更好地理解Java泛型的工作原理及其限制，并能充分利用Java泛型的强大功能。

## 1.1 背景介绍
泛型一直都是Java开发人员最感兴趣的话题之一。从最初的简单接口和方法签名到后来的参数化类型、泛型类、泛型集合等各种泛型用法，Java一直都处于一个领先地位。虽然Java提供的泛型机制有很多很酷炫的特性，但是也有一些局限性需要用户了解。比如，由于Java的泛型实现方式，导致其在运行时无法获得真正的泛型信息，因此，对于运行时的类型判断就无能为力了。为了解决这个问题，Java 7引入了一个新的注解处理工具Javac API，通过反射API获取到编译时擦除后的真实类型的元数据。随着JDK版本的更新迭代，Java泛型已经得到了大幅度的改进，目前Java泛型已经成为Java语言的主要特性之一。

在介绍Java泛型之前，先来看一下什么是泛型擦除？

## 1.2 泛型擦除
Java泛型在编译期间进行类型擦除的过程叫做泛型擦除(Type Erasure)，它是指当我们定义泛型类型的时候，泛型类型参数会被擦除掉。具体来说，泛型类型参数会被替换成它们的限定类型或Object类型。例如，假设有一个ArrayList<String> list = new ArrayList<>();，在编译期间，ArrayList<>中的参数String会被擦除，所以实际上编译出来的代码类似于List list = new ArrayList();。这样做的一个结果就是，我们编写的代码不需要关心具体使用的是哪个泛型类型，因为编译器会在编译期间将泛型类型参数擦除掉，只留下它的限定类型或者Object类型。

通过泛型擦除，Java泛型所带来的最大好处就是方便。使用泛型可以使得代码变得更加简单，而且不用担心类型安全问题。但是，泛型擦除也造成了一些不可预料的后果，导致Java泛型并不是一个完美的方案。其中一个就是java中不能使用通配符(Wildcards)。那么为什么Java泛型不能支持通配符呢？这就要讲到通配符的作用了。

## 1.3 泛型中的通配符与类型擦除
通配符(wildcard)是Java泛型的一项重要特性。它允许我们在编译期间使用泛型而无需指定确切的类型。通配符主要用来处理泛型集合上的元素遍历和类型转换的问题。对于泛型类的实例变量和静态变量，我们可以使用通配符对它的类型进行约束。通配符主要有两种：

1.? : Unbounded wildcard type
   这种通配符表示类型可能是任意的，包括null值。在编译期间，编译器将泛型类型参数擦除为Object类型，然后再将?替换为Object类型。但是这种通配符只能作为泛型集合里面的类型参数使用，不能用于泛型类或者方法的返回值。例如，public void printList(List<?> lst) { for (Object obj : lst) { System.out.println(obj); } }

2. <? extends T> : Upper bounded wildcard type
   上界通配符(Upper bounded wildcard type)表示该类型是一个泛型类型或其子类型，即下限为T。在编译期间，编译器将泛型类型参数擦除为T类型，然后再将<? extends T>替换为T类型。<? extends T>可以用来作为泛型集合里面的类型参数，但只能作为方法的返回值使用，不能作为方法的参数类型。例如，public <T> List<T> subList(List<T> lst) { return lst; } public void testSubList() { List<Integer> integers = Arrays.asList(1, 2, 3, 4, 5); List<? extends Integer> intSublist = subList(integers); // OK, returns a List<Integer> Object obj = "hello"; subList(obj); // Error: Incompatible types. Required List<? extends Integer>, found List<Object> }

综上所述，通过泛型擦除，Java泛型帮助我们在编译期间确保类型安全，并且无需在运行时刻去判断泛型类型。然而，Java泛型仍然受到通配符的限制。尤其是在泛型集合和泛型类中的通配符，它们仅仅适用于集合元素类型或者类实例/静态变量的类型，无法用于其他地方。

Java泛型的这些缺陷使得我们必须在设计时充分考虑到类型擦除带来的影响。对于许多开发者来说，在使用泛型时都要小心翼翼，防止出现编译或运行时的异常情况，确保代码质量。