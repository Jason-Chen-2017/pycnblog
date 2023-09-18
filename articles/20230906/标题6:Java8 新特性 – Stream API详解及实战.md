
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 一、Java 8 简介
从 Java 7 到 Java 9，Java 在功能上并没有太大的变化，但从开发体验上来说，Java 的更新换代还是频繁地进行着。Java 8 带来的语言更新主要围绕 Lambda 表达式（函数式编程）、Stream API 和接口私有化等方面。
## 二、为什么要学习 Java 8
Java 8 带来的好处无需多言，可以说是一个重量级特性。作为一个高级工程师，掌握 Java 8 的知识将帮助我更好的理解并应用 Java 面向对象的编程模型。另外，在企业界 Java 8 更像是 JavaScript 中的 ES6。
## 三、学习路线图
# 2.Stream API 相关概念术语说明
## 1.Stream 是什么？
Stream 是一个Java 8新增的API，它为集合处理提供了一种简单、统一的操作方式。Stream 使用数据源（如集合、数组），通过对源数据流水线上的操作，得到结果。Stream 的操作可以是中间操作或最终操作，其中中间操作不会执行任何计算，而最终操作会触发实际计算。
## 2.Stream 有哪些操作？
Stream 支持以下几种操作：
### a.创建 Stream
#### i.直接通过 Collection 创建 Stream
```java
List<String> list = Arrays.asList("apple", "banana", "orange");
Stream<String> stream = list.stream(); // 获取元素为 String 类型的数据流
```
#### ii.Arrays.stream() 方法创建 Stream
```java
int[] numbers = {1, 2, 3};
IntStream intStream = Arrays.stream(numbers); // 获取元素为 Integer 类型的数据流
```
#### iii.Stream.of() 方法创建 Stream
```java
Stream<Integer> integerStream = Stream.of(1, 2, 3); // 获取元素为 Integer 类型的数据流
```
#### iv.Stream.empty() 方法创建空的 Stream
```java
Stream emptyStream = Stream.empty(); // 创建一个空的 Stream
```
### b.中间操作
#### i.filter() 操作
filter() 方法用于过滤出满足条件的元素。该方法接收 Lambda 函数作为参数，只有符合 Lambda 函数返回值为 true 的元素才会保留。
```java
List<String> fruits = Arrays.asList("apple", "banana", "orange", "pear", "pineapple");
fruits.stream().filter(fruit -> fruit.length() > 5).forEach(System.out::println); // 输出 apple banana orange pineapple
```
#### ii.distinct() 操作
distinct() 方法用于去掉重复的元素。
```java
List<String> colors = Arrays.asList("red", "green", "blue", "yellow", "red", "purple");
colors.stream().distinct().forEach(System.out::println); // 输出 red green blue yellow purple
```
#### iii.sorted() 操作
sorted() 方法用于对元素进行排序。
```java
List<String> names = Arrays.asList("Tom", "Jerry", "Mike", "Alice", "Lisa");
names.stream().sorted((n1, n2) -> n1.compareToIgnoreCase(n2)).forEach(System.out::println); // 输出 Alice Jerry Lisa Mike Tom
```
#### iv.limit() 操作
limit() 方法用于限制 Stream 中元素的个数。
```java
List<Integer> numbers = new ArrayList<>();
for (int i = 0; i < 100; i++) {
    numbers.add(i + 1);
}
numbers.stream().limit(5).forEach(System.out::println); // 输出 1 2 3 4 5
```
#### v.skip() 操作
skip() 方法用于跳过前面的几个元素。
```java
List<Integer> numbers = new ArrayList<>();
for (int i = 0; i < 100; i++) {
    numbers.add(i + 1);
}
numbers.stream().skip(5).limit(5).forEach(System.out::println); // 输出 6 7 8 9 10
```
#### vi.map() 操作
map() 方法用于映射每个元素。该方法接收 Lambda 函数作为参数，Lambda 函数的参数就是 Stream 中的元素，返回值则是映射后的元素。
```java
List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
numbers.stream().mapToDouble(num -> Math.pow(num, 2)).forEach(System.out::println); // 输出 1.0 4.0 9.0 16.0 25.0
```
#### vii.flatMap() 操作
flatMap() 方法用于把一个 Stream 中每一个元素都转化成另一个 Stream，然后再把多个 Stream 拼接起来成为一个新的 Stream。
```java
List<String> strings = Arrays.asList("hello world", "goodbye world", "welcome to java world");
strings.stream().flatMap(str -> str.split("\\s+")).distinct().forEach(System.out::println); // 输出 hello world goodbye welcome to java
```
### c.终止操作
#### i.forEach() 操作
forEach() 方法用于遍历 Stream 中的每个元素。该方法接收 Consumer 函数作为参数，Consumer 函数的参数就是 Stream 中的元素，用来对元素进行处理。
```java
List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
numbers.stream().forEach(System.out::println); // 输出 1 2 3 4 5
```
#### ii.count() 操作
count() 方法用于计算 Stream 中的元素个数。
```java
List<Double> doubleNumbers = Arrays.asList(1.1, 2.2, 3.3, 4.4, 5.5, 6.6);
long count = doubleNumbers.stream().count();
System.out.println(count); // 输出 6
```
#### iii.reduce() 操作
reduce() 方法用于归约 Stream 中的元素，将其组合成一个值。该方法接收 BinaryOperator 函数作为参数，BinaryOperator 函数的参数分别是 accumulator（累加器）、current element（当前元素）。accumulator 表示当前累计的值，current element 表示下一个元素；BinaryOperator 函数的作用是计算 accumulator 和 current element 的值并返回。
```java
List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
Optional<Integer> result = numbers.stream().reduce((sum, num) -> sum * num);
result.ifPresent(System.out::println); // 输出 Optional[120]
```
#### iv.collect() 操作
collect() 方法用于收集 Stream 中的元素，并转换成指定类型的集合。该方法接收 Collector 函数作为参数，Collector 函数负责对 Stream 中的元素进行汇总、分组、匹配等操作，并生产结果。Collectors 工具类提供了很多静态方法用于快速创建常用的 Collector。
```java
List<String> words = Arrays.asList("Hello", "World", "This", "is", "a", "test");
String joinedWords = words.stream().collect(Collectors.joining(", "));
System.out.println(joinedWords); // Hello, World, This, is, a, test
```
# 3.如何正确使用 Stream API ？
## 1.应该使用的场景
Stream API 在不同的场景下使用方法也是不同的。如下所示：
### a.集合元素处理
Stream API 可以用来对集合中的元素进行筛选、修改或者删除，也可以用来根据一些规则统计集合中元素的数量、求和、平均值、最大值和最小值。
```java
// 查找列表中长度大于 5 的字符串
List<String> fruits = Arrays.asList("apple", "banana", "orange", "pear", "pineapple");
long count = fruits.stream().filter(fruit -> fruit.length() > 5).count();
System.out.println(count); // 2

// 根据学生名字的首字母统计各个学生人数
List<Student> students = Arrays.asList(new Student("John"), new Student("Mike"),
        new Student("Peter"), new Student("Sarah"), new Student("David"));
Map<Character, Long> studentCountByFirstLetter = students.stream()
       .collect(Collectors.groupingBy(student -> Character.toUpperCase(student.getName().charAt(0)), Collectors.counting()));
System.out.println(studentCountByFirstLetter); // {J=1, P=1, S=1, D=1, M=1}

// 把学生按年龄范围分组，统计每组人数
Map<Integer[], Long> studentAgeRangesAndCounts = students.stream()
       .collect(Collectors.groupingBy(student -> {
            if (student.getAge() <= 20) {
                return new Integer[]{15, 25};
            } else if (student.getAge() <= 30) {
                return new Integer[]{25, 35};
            } else {
                return new Integer[]{35, 45};
            }
        }, Collectors.counting()));
System.out.println(studentAgeRangesAndCounts); // {[15, 25]=2, [25, 35]=2, [35, 45]=1}
```
### b.数组元素处理
Stream API 可以用来对数组中的元素进行筛选、修改或者删除，也可以用来根据一些规则统计数组中元素的数量、求和、平均值、最大值和最小值。
```java
// 对数字数组进行过滤，只保留偶数
int[] numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9};
int evenSum = IntStream.rangeClosed(1, 9).filter(number -> number % 2 == 0).sum();
System.out.println(evenSum); // 20

// 求出数字数组的平方和
double[] doubleNumbers = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
double squareSum = DoubleStream.of(doubleNumbers).mapToDouble(number -> Math.pow(number, 2)).sum();
System.out.println(squareSum); // 165.138
```
### c.文件处理
Stream API 可以用来读取文件中的内容、过滤、映射，还可以使用Collectors.toList()、Collectors.toSet()等收集器收集数据。
```java
File file = new File("/path/to/file.txt");
List<String> lines = Files.lines(file.toPath()).filter(line -> line.startsWith("#")).collect(Collectors.toList());
Set<Integer> integers = Files.lines(file.toPath())
                            .skip(1)
                            .map(Integer::parseInt)
                            .collect(Collectors.toSet());
```