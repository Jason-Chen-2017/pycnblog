
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 什么是泛型？为什么要使用它？
Java 是一种面向对象编程语言，它的强大的泛型机制可以使程序员编写的代码更加灵活、更具弹性。通俗地说，泛型可以让开发人员编写灵活、类型安全、可重用代码。不过，如果你不清楚泛型背后的机制，或者对它有疑惑，那么这个特性就很难用好。所以，本文将深入探讨一下 Java 的泛型机制——“类型擦除”（Type Erasure）以及一些容易混淆的概念。

我们从简单易懂的 Hello World 程序开始学习泛型。这里是一个简单的程序：

```java
public class Main {
    public static void main(String[] args) {
        List<Integer> list = new ArrayList<>();
        list.add(1); // add an integer to the list
        Integer num = list.get(0); // get the first element and assign it to a variable of type Integer
        System.out.println(num); // prints "1"
    }
}
```

上述程序中，`List<Integer>` 表示一个列表，其中元素类型为 `Integer`。我们创建了一个新的空列表，并将整数值 `1` 添加到该列表。然后，我们通过 `list.get(0)` 方法获取该列表中的第一个元素，并将其赋值给一个名为 `num` 的变量，此时 `num` 的类型是 `Integer`。最后，我们打印出 `num` 变量的值，输出为 `"1"`。

这段代码看起来非常简单，但它包含两个重要知识点：

1. 泛型 `<T>`: 在声明变量或参数时添加 `<T>` 将创建一个参数化的类型，即一个类型占位符；`<T>` 可以被任何其他类型替换，如 `String`，`Double`，`Person` 等；
2. 类型擦除（Type Erasure）: 编译器在编译期间，会自动移除所有的泛型信息，并将泛型代码转换成非泛型代码。类型擦除意味着泛型只是让编译器帮我们做了一些类型推导工作，而不是真正的生成多个不同的类文件。

除了 `List` 以外，还有很多地方都可以使用泛型。例如，`HashMap` 中的键和值都是泛型类型：

```java
Map<String, Integer> map = new HashMap<>();
map.put("John", 25); // put a key-value pair into the map
int age = map.get("John"); // retrieve the value associated with the key "John"
System.out.println(age); // outputs "25"
```

此处 `HashMap<String, Integer>` 表示一个字符串到整数值的映射表。我们通过调用 `put()` 方法向映射表中添加一个键值对，其中键为字符串 `"John"`，值为整数 `25`。随后，我们通过调用 `get()` 方法检索对应于键 `"John"` 的值，并赋给一个名为 `age` 的变量，此时 `age` 的类型也是 `Integer`。

除了简单的数据类型，泛型还可以应用于方法和类。例如，`ArrayList` 和 `LinkedList` 都是泛型类，它们可以接收不同类型的元素作为参数：

```java
List<String> strList = new LinkedList<>();
strList.addAll(Arrays.asList("apple", "banana", "orange")); // add elements from an array using Arrays.asList() method
for (String fruit : strList) {
    System.out.print(fruit + " "); // print out each element in the list
}
// Output: "apple banana orange "
```

在上面的代码中，`List<String>` 用于创建 `strList`，我们通过调用 `addAll()` 方法向列表中添加三个元素 `"apple"`, `"banana"`, `"orange"`。为了遍历这个列表，我们使用一个循环，并在每次迭代时访问当前的元素。由于 `strList` 中元素的类型是字符串，因此循环体中的变量 `fruit` 的类型也是字符串。

# 2.泛型相关术语及概念
## 泛型的基本概念
Java 泛型有以下几个特点：

1. 参数化类型（Parameterized Type）: 把类型由原来的具体的类类型，变成参数化的类型。参数化的类型指的是定义在类上的模板，这种模板用来创建特定类型的一组对象。比如，某个类的类型是List<E> ，E代表任意类型。实际上，在编译阶段，编译器会把泛型类型转化成普通类型。
2. 边界（Bound）: 类型参数的一个上下限，只有该类型参数扩展或实现了指定接口才能作为该泛型类型来使用。比如，对于类A和B，如果有一个泛型类C<T extends A & B>,则表示T只能是类A和B的子类。
3. 无限制类型参数（Unbounded Type Parameter）: 没有指定类型参数的泛型类型，称之为无限制类型参数。例如，`List<?> list`，表示List的元素类型是未知的，可以是任何类型。
4. 类型擦除（Type Eraserasure）: 在编译器进行类型检查的时候，擦除掉泛型类型信息，只保留原始类型信息，将泛型类型所有信息都替换为Object类型。

## 泛型类和方法
### 泛型类
泛型类是指拥有类型参数的类。它的形式如下：

```java
class ClassName<TypeParameter1,..., TypeParameterN> {...}
```

其中，`ClassName` 是泛型类的名称，`<...>` 表示参数列表，逗号隔开。每个参数名称一般为单个大写字母。

在使用泛型类时，需要传入相应的参数类型。举例来说，假设有如下泛型类：

```java
public class Pair<K, V> {
    private K key;
    private V value;

    public Pair(K key, V value) {
        this.key = key;
        this.value = value;
    }

    public K getKey() {
        return key;
    }

    public V getValue() {
        return value;
    }
}
```

可以像下面这样使用这个类：

```java
Pair<Integer, String> p1 = new Pair<>(1, "hello");
Pair<String, Double> p2 = new Pair<>("world", 3.14d);
```

这里创建了两个 `Pair` 对象，分别指定了键和值的类型。注意，实例化时应该传入正确的参数类型。另外，泛型类可以包含字段、构造器、静态方法、实例方法等，也可以继承自其他泛型类或非泛型类。

### 泛型方法
泛型方法是指具有类型参数的方法。它的形式如下：

```java
return-type method-name(parameter-list) throws Exception{
  //method body
}
```

其中，`return-type` 是返回值类型，`method-name` 是方法名称，`parameter-list` 是方法参数列表。可以看到，泛型方法跟一般方法没有什么区别，只是在方法名称前面多了一个`<>`符号。

在使用泛型方法时，需要传入相应的参数类型。举例来说，假设有如下泛型方法：

```java
public <T> T max(T x, T y){
    if (x instanceof Comparable && y instanceof Comparable){
        return ((Comparable<T>) x).compareTo(y) > 0? x : y;
    } else {
        throw new IllegalArgumentException("Arguments are not comparable.");
    }
}
```

可以像下面这样调用这个方法：

```java
double d1 = max(3.0, 4.5);    // Returns 4.5
String s2 = max("hello", "world");   // Returns "world"
```

这里，我们调用了泛型方法 `max()` 来比较两个 `double` 或 `String` 类型的值，并返回较大值。注意，泛型方法的参数类型也应该传入正确的类型。

## 通配符
Java泛型还支持一种特殊的类型——通配符（Wildcard）。通配符是指未知类型，可以表示任何类型。可以将通配符看作是一种特殊的类型参数，它的类型是不确定的。下面列出几种通配符的语法形式：

1. `?`：等价于 Object
2. `? extends Type`: 表示 extends 关键字右侧的类，例如 List<? extends Number> 等效于 List<Number>。表示类型的下边界，只能取子类对象
3. `? super Type`: 表示 super 关键字右侧的类，例如 List<? super Integer> 等效于 List<Integer>。表示类型的上边界，只能取父类对象。

例如：

```java
List<? extends Number> nums = new ArrayList<>();
nums.add(new BigDecimal("123.45"));
nums.add(new BigInteger("123"));
nums.add(123);   // compile error
```

第 7 行代码编译报错，因为 List 只能存放 Number 的子类对象，不能存放 int 对象。而使用通配符后就可以解决这个问题：

```java
List<? extends Number> nums = new ArrayList<>();
nums.add(new BigDecimal("123.45"));
nums.add(new BigInteger("123"));
nums.add((Number)123);     // ok!
```