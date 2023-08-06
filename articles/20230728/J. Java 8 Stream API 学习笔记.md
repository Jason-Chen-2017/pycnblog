
作者：禅与计算机程序设计艺术                    

# 1.简介
         

         Java 8引入了Stream API，它提供了一种更高效、更灵活的方式处理数据流。
         
         在本文中，我们将介绍Stream API及其关键特性，包括它的基本使用方法、可应用场景、优缺点等。
         # 2.Stream API概览
         
         ## 2.1.什么是Stream？
         
         从数据结构角度看，Stream可以理解成元素序列。如数组、链表、集合等。在Stream API中，Stream不是数据结构本身，而是一个视图（view）或一个管道（pipeline）。它包含一个源头（source），可以通过Stream上的操作过滤和转换得到想要的数据结果。
         
         Stream只能用于单线程操作，而ParallelStream则支持多线程并行操作。
         
         Stream的特点：
         
         - Stream基于惰性计算(Lazy Evaluation)，只有调用Stream的终止操作(比如count()或者collect())的时候才会执行真正的计算操作。
         - Stream本身不会存储元素。他们只是利用生产者-消费者模式，在内部协调生产数据和消费数据，所以内存 consumption 是最小的。
         - Stream 操作是延迟执行的。这意味着仅当需要结果的时候，操作才会执行。
         - Stream 自己不改变其输入，相反，他们会返回一个新的Stream，这样就允许对操作进行多个层次的组合。
         
         Stream的操作类型：
         
         
         每种操作都提供了一些额外的参数用于控制操作行为，比如limit()操作用于指定结果集大小，forEach()操作用于迭代输出结果。
         
         ## 2.2.创建Stream
         
         ### 2.2.1.通过Collection创建Stream
         
         使用Java 8中的Collectors工具类可以把集合转换成Stream，如下所示：
         
         ```java
import java.util.*;

public class Test {
    public static void main(String[] args) {
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
        
        // 把numbers转成stream
        Stream<Integer> numberStream = numbers.stream();
        
        System.out.println("number stream: " + numberStream.toList());
        
        Stream<Integer> sortedNumbers = numberStream.sorted();
        
        System.out.println("sorted number stream: " + sortedNumbers.toList());
    }
}
```

         上述代码首先生成一个List<Integer>，然后用stream()方法转换为Stream<Integer>。再通过sorted()方法对其排序，并输出到控制台。打印出来的结果为：[1, 2, 3, 4, 5]、[1, 2, 3, 4, 5]。
         
         ### 2.2.2.通过Arrays创建Stream
         
         如果要从数组中创建Stream，可以使用IntStream、DoubleStream、LongStream或其它特定类型的Stream。如下面的代码示例所示：
         
         ```java
import java.util.stream.*;

public class Test {
    public static void main(String[] args) {
        int[] numbers = new int[]{1, 2, 3, 4, 5};

        IntStream evenNumbers = Arrays.stream(numbers).filter(num -> num % 2 == 0);

        System.out.println("Even numbers in the array are: ");
        evenNumbers.forEach(System.out::println);
    }
}
```

           其中，Arrays.stream()方法用于创建一个IntStream对象；filter()方法用于过滤偶数，最后用forEach()方法输出结果。
          
         ### 2.2.3.通过Stream创建Stream
         
         通过调用其他Stream的创建方法也可以生成新Stream。如下所示：
         
         ```java
import java.util.*;

public class Test {
    public static void main(String[] args) {
        Random rand = new Random();
        LongStream longs = LongStream.rangeClosed(1, 10L).mapToObj(l -> l * 10 + rand.nextInt(10));
        List<Long> result = longs.limit(5).boxed().collect(Collectors.toList());
        System.out.println("Randomized and boxed longs:" + result);
    }
}
```

             这里，先用随机数生成器生成了一个1~10间的10个整数，然后映射为长度为12的字符串，即每隔两位插入一个随机的数字。再用splitAsStream()方法分割成两个字符组成的Stream，然后用flatMap()方法合并为一个Stream。最后用boxed()方法转成Boxed类型并限制数量，再用toList()方法收集结果。
            
         ## 2.3.Stream的操作分类
         
         Stream API中的操作分为四大类：
         
         ### 2.3.1.中间操作
         中间操作包括那些返回Stream的方法，这些方法会返回一个还没有执行真正操作的Stream对象，只有当调用终结方法时（比如count()或者collect()），才会真正触发计算操作。
         
         可以看到，这种延迟执行方式让Stream很适合用来实现数据的链式操作。例如，假设我们有如下数据列表：
         
         ```java
List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
```
         
         我们想得到一个由每个数值平方组成的新列表。我们可以按照以下方式实现：
         
         ```java
List<Integer> squares = numbers.stream()
                               .map(n -> n * n)
                               .collect(Collectors.toList());
```

         上述代码中，我们首先用stream()方法把列表转换为Stream；然后用map()方法取出每个数值平方作为新的元素加入到新的列表中；最后用toList()方法获取最终的结果。
         
         ### 2.3.2.终结操作
         终结操作包括那些用于从Stream中提取结果的方法，比如filter()、findAny()、findFirst()等。这些方法都会立刻执行实际的计算操作，并产生结果，而不会返回任何新的Stream对象。
         
         常用的终结操作有forEach()、reduce()、count()、anyMatch()、allMatch()、noneMatch()等。
         
         ### 2.3.3.短路操作
         当多个中间操作连续调用时，由于中间操作都是延迟执行的，因此只有真正需要结果的时候才会执行。而如果某个操作短路了，那么后面的操作也不会被执行。
         
         比如，假设有一个Stream里含有null元素，下面的代码会报空指针异常：
         
         ```java
List<Object> listWithNull = Arrays.asList(1, null, 3, null, 5);
listWithNull.stream().filter(Objects::nonNull).collect(Collectors.toList());
```

         因为filter()操作已经被短路掉了，后面的collect()操作根本不会被执行。为了避免这种情况，我们可以在一个流上使用Optional类，该类可以表示可能存在的值，而不是像NullPointerException一样抛出异常。
         
         ```java
import java.util.*;

public class OptionalTest {
    public static void main(String[] args) {
        List<Object> listWithNull = Arrays.asList(1, null, 3, null, 5);
        List<Object> nonNullValues = listWithNull.stream()
               .map(obj -> obj!= null? Optional.of((Integer) obj) : Optional.empty())
               .filter(Optional::isPresent)
               .map(Optional::get)
               .collect(Collectors.toList());
        System.out.println(nonNullValues);
    }
}
```

              这样，就不会出现空指针异常。如果一个流中的元素类型是OptionalInt、OptionalLong或OptionalDouble，那么map()方法就可以直接返回它自身，就不需要再进行拆包。
             
         ### 2.3.4.排他操作
         排他操作会阻断其他操作继续流水线。常用的排他操作有sorted()、distinct()、limit()和skip()等。
         
         # 3.基础语法
         本节将详细介绍Stream API的基本语法。
         ## 3.1.声明Stream变量
         在Java 8中，所有的Stream操作都通过Streams类来完成，而Streams类的所有静态方法都是创建Stream的工厂方法，因此无需显式地创建Streams对象。不过，为了方便书写，建议声明Stream变量，如下所示：
         
         ```java
import java.util.stream.*;

public class Test {
    public static void main(String[] args) {
        Integer[] nums = {1, 2, 3, 4, 5};
        IntStream s = Arrays.stream(nums);
    }
}
```

              这里，我们声明了一个名为s的IntStream变量，并初始化它的值为{1, 2, 3, 4, 5}。
         ## 3.2.筛选与切片
         ### 3.2.1.filter()方法
         
         filter()方法根据指定的 Predicate 来过滤数据。Predicate 函数接收一个参数，返回true表示保留此元素，false表示排除此元素。例如，以下代码只保留偶数：
         
         ```java
import java.util.*;
import java.util.stream.*;

public class FilterExample {
    public static void main(String[] args) {
        List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5, 6, 7, 8, 9);

        // 创建一个IntStream
        IntStream evens = numbers.stream().filter(num -> num % 2 == 0);

        // 获取所有偶数
        List<Integer> evenNumbers = evens.boxed().collect(Collectors.toList());
        System.out.println("The even numbers from the list are: " + evenNumbers);
    }
}
```

         此例中，我们创建了一个IntStream，它会过滤出输入列表中所有奇数，并将其余的奇数转换为Integer。然后，我们用boxed()方法将其转换为原始类型List<Integer>。

          ### 3.2.2.limit()方法

         limit() 方法用于截取流中元素的数量。limit() 的作用是在计算时对数据流进行截取，对其之后的数据流就不起作用了。以下代码展示了如何使用 limit() 方法将数据流中的前三个元素进行打印：

         ```java
import java.util.*;
import java.util.stream.*;

public class LimitExample {
    public static void main(String[] args) {
        String[] words = {"apple", "banana", "cherry", "date", "elderberry"};

        // 创建一个Stream
        Stream<String> wordStream = Arrays.stream(words);

        // 获取前三个元素
        wordStream.limit(3).forEach(System.out::println);
    }
}
```

         此例中，我们创建了一个字符串数组，并用 Arrays.stream() 方法创建了一个字符串流。接着，我们使用 limit() 方法获取其中的前三个元素。使用 forEach() 方法遍历打印每个元素。

          ### 3.2.3.skip()方法

         skip() 方法用来跳过指定个数的元素，然后返回剩下的元素。以下代码展示了如何使用 skip() 方法跳过第一个元素：

         ```java
import java.util.*;
import java.util.stream.*;

public class SkipExample {
    public static void main(String[] args) {
        String[] fruits = {"apple", "banana", "cherry", "date", "elderberry"};

        // 创建一个Stream
        Stream<String> fruitStream = Arrays.stream(fruits);

        // 跳过第一个元素
        fruitStream.skip(1).forEach(System.out::println);
    }
}
```

         此例中，我们创建了一个字符串数组，并用 Arrays.stream() 方法创建了一个字符串流。接着，我们使用 skip() 方法跳过数组中的第一个元素。使用 forEach() 方法遍历打印剩下的元素。

          ### 3.2.4.distinct()方法

         distinct() 方法用来返回一个去重后的流。以下代码展示了如何使用 distinct() 方法去除重复元素：

         ```java
import java.util.*;
import java.util.stream.*;

public class DistinctExample {
    public static void main(String[] args) {
        String[] colors = {"red", "green", "blue", "red", "yellow", "blue"};

        // 创建一个Stream
        Stream<String> colorStream = Arrays.stream(colors);

        // 去重
        Set<String> uniqueColors = colorStream.distinct().collect(Collectors.toSet());

        // 输出去重后的颜色
        System.out.println("Unique Colors: " + uniqueColors);
    }
}
```

         此例中，我们创建了一个字符串数组，并用 Arrays.stream() 方法创建了一个字符串流。接着，我们使用 distinct() 方法删除重复的元素。最后，我们用 toSet() 方法将结果转换为 Set<String>。

          ## 3.3.映射
          ### 3.3.1.mapToInt()方法

         mapToInt() 方法用于映射每个元素到对应的int值。以下代码展示了如何使用 mapToInt() 将元素加倍：

         ```java
import java.util.*;
import java.util.stream.*;

public class MapToIntExample {
    public static void main(String[] args) {
        double[] temperaturesCelsius = {-20.0, -10.0, 0.0, 10.0, 20.0};

        // 创建一个 DoubleStream
        DoubleStream temps = Arrays.stream(temperaturesCelsius);

        // 对每个元素加 10°C
        int[] tempFahrenheit = temps.mapToInt(temp -> (int) ((temp * 9 / 5) + 32)).toArray();

        for (int fahrenheit : tempFahrenheit) {
            System.out.print(fahrenheit + " ");
        }
    }
}
```

         此例中，我们定义了一个double型数组，并用 Arrays.stream() 方法创建了一个DoubleStream。接着，我们使用 mapToInt() 方法将每个元素乘以 9/5 求出摄氏温度，再加上 32，求出华氏温度。最后，我们使用toArray() 方法将结果转化为整型数组。

          ### 3.3.2.mapToDouble()方法

         mapToDouble() 方法用于映射每个元素到对应的double值。以下代码展示了如何使用 mapToDouble() 将元素乘以 2.5：

         ```java
import java.util.*;
import java.util.stream.*;

public class MapToDoubleExample {
    public static void main(String[] args) {
        int[] numbers = {1, 2, 3, 4, 5};

        // 创建一个 IntStream
        IntStream nums = Arrays.stream(numbers);

        // 用 lambda 表达式将每个元素乘以 2.5
        double sum = nums.mapToDouble(num -> num * 2.5).sum();

        System.out.println("Sum of all elements after mapping with multiplying by 2.5 is: " + sum);
    }
}
```

         此例中，我们定义了一个int型数组，并用 Arrays.stream() 方法创建了一个IntStream。接着，我们使用 mapToDouble() 方法将每个元素乘以 2.5，再求和。最后，我们打印得到的结果。

          ### 3.3.3.map()方法

         map() 方法用于映射每个元素到不同的元素。以下代码展示了如何使用 map() 将每个元素转换为对应的大写形式：

         ```java
import java.util.*;
import java.util.stream.*;

public class MapExample {
    public static void main(String[] args) {
        String[] animals = {"dog", "cat", "elephant", "lion"};

        // 创建一个 Stream
        Stream<String> animalStream = Arrays.stream(animals);

        // 转换每个元素为大写
        List<String> upperCaseAnimals = animalStream.map(String::toUpperCase).collect(Collectors.toList());

        System.out.println("Uppercase Animal Names: " + upperCaseAnimals);
    }
}
```

         此例中，我们定义了一个字符串数组，并用 Arrays.stream() 方法创建了一个字符串流。接着，我们使用 map() 方法将每个元素转换为大写形式。最后，我们使用 collect() 方法将结果转换为 List<String>。

          ### 3.3.4.flatMapping()方法

         flatMapping() 方法与 map() 方法类似，但是它接受的是另一个流，并且会将这个流中的元素扁平化地添加到结果流中。以下代码展示了如何使用 flatMapping() 将数组中的元素连接成一个字符串：

         ```java
import java.util.*;
import java.util.stream.*;

public class FlatMapExample {
    public static void main(String[] args) {
        String[][] matrix = {{".#.", "..#"}, {"###", ".##"}};

        // 创建一个 Stream
        Stream<String[]> matrixStream = Arrays.stream(matrix);

        // 将矩阵元素扁平化
        Stream<String> elementStream = matrixStream.flatMap(Arrays::stream);

        String joinedElements = elementStream.collect(Collectors.joining(", "));

        System.out.println("Joined Elements: " + joinedElements);
    }
}
```

         此例中，我们定义了一个二维字符串数组，并用 Arrays.stream() 方法创建了一个二维字符串流。接着，我们使用 flatMap() 方法将矩阵元素扁平化，得到扁平化后的元素流。我们使用 Arrays.stream() 方法重新创建这个流，然后用 joining() 方法连接成一个字符串。最后，我们打印得到的结果。

          ## 3.4.聚合
          ### 3.4.1.count()方法

         count() 方法用于统计流中元素的个数。以下代码展示了如何使用 count() 方法统计元素的个数：

         ```java
import java.util.*;
import java.util.stream.*;

public class CountExample {
    public static void main(String[] args) {
        String[] fruits = {"apple", "banana", "cherry", "date", "elderberry"};

        // 创建一个 Stream
        Stream<String> fruitStream = Arrays.stream(fruits);

        // 统计元素的个数
        long count = fruitStream.count();

        System.out.println("Number of Fruits: " + count);
    }
}
```

         此例中，我们定义了一个字符串数组，并用 Arrays.stream() 方法创建了一个字符串流。接着，我们使用 count() 方法统计流中元素的个数。

          ### 3.4.2.min()方法

         min() 方法用于返回流中最小的元素。以下代码展示了如何使用 min() 方法获取最小元素：

         ```java
import java.util.*;
import java.util.stream.*;

public class MinExample {
    public static void main(String[] args) {
        String[] fruits = {"apple", "banana", "cherry", "date", "elderberry"};

        // 创建一个 Stream
        Stream<String> fruitStream = Arrays.stream(fruits);

        // 获取最小元素
        Optional<String> smallestFruit = fruitStream.min(Comparator.naturalOrder());

        if (smallestFruit.isPresent()) {
            System.out.println("Smallest Fruit: " + smallestFruit.get());
        } else {
            System.out.println("No minimum value found.");
        }
    }
}
```

         此例中，我们定义了一个字符串数组，并用 Arrays.stream() 方法创建了一个字符串流。接着，我们使用 Comparator.naturalOrder() 方法按字典顺序排序，然后使用 min() 方法获取最小元素。

          ### 3.4.3.max()方法

         max() 方法用于返回流中最大的元素。以下代码展示了如何使用 max() 方法获取最大元素：

         ```java
import java.util.*;
import java.util.stream.*;

public class MaxExample {
    public static void main(String[] args) {
        String[] fruits = {"apple", "banana", "cherry", "date", "elderberry"};

        // 创建一个 Stream
        Stream<String> fruitStream = Arrays.stream(fruits);

        // 获取最大元素
        Optional<String> largestFruit = fruitStream.max(Comparator.reverseOrder());

        if (largestFruit.isPresent()) {
            System.out.println("Largest Fruit: " + largestFruit.get());
        } else {
            System.out.println("No maximum value found.");
        }
    }
}
```

         此例中，我们定义了一个字符串数组，并用 Arrays.stream() 方法创建了一个字符串流。接着，我们使用 Comparator.reverseOrder() 方法倒序排序，然后使用 max() 方法获取最大元素。

          ### 3.4.4.average()方法

         average() 方法用于计算流中元素的平均值。以下代码展示了如何使用 average() 方法计算元素的平均值：

         ```java
import java.util.*;
import java.util.stream.*;

public class AverageExample {
    public static void main(String[] args) {
        int[] numbers = {1, 2, 3, 4, 5};

        // 创建一个 IntStream
        IntStream numStream = Arrays.stream(numbers);

        // 计算平均值
        OptionalDouble avg = numStream.asDoubleStream().average();

        if (avg.isPresent()) {
            System.out.printf("Average of all elements is %.2f%n", avg.getAsDouble());
        } else {
            System.out.println("No values present");
        }
    }
}
```

         此例中，我们定义了一个int型数组，并用 Arrays.stream() 方法创建了一个IntStream。接着，我们使用 asDoubleStream() 方法把流转换为DoubleStream，然后使用 average() 方法计算平均值。

          ## 3.5.排序
          ### 3.5.1.sorted()方法

         sorted() 方法用于对流进行排序。以下代码展示了如何使用 sorted() 方法对元素进行排序：

         ```java
import java.util.*;
import java.util.stream.*;

public class SortExample {
    public static void main(String[] args) {
        String[] fruits = {"apple", "banana", "cherry", "date", "elderberry"};

        // 创建一个 Stream
        Stream<String> fruitStream = Arrays.stream(fruits);

        // 排序
        Stream<String> sortedFruits = fruitStream.sorted();

        sortedFruits.forEach(System.out::println);
    }
}
```

         此例中，我们定义了一个字符串数组，并用 Arrays.stream() 方法创建了一个字符串流。接着，我们使用 sorted() 方法对流进行升序排序。最后，我们使用 forEach() 方法遍历输出排序后的结果。

          ### 3.5.2.reversed()方法

         reversed() 方法用于反转流中元素的顺序。以下代码展示了如何使用 reversed() 方法反转元素顺序：

         ```java
import java.util.*;
import java.util.stream.*;

public class ReversedExample {
    public static void main(String[] args) {
        String[] fruits = {"apple", "banana", "cherry", "date", "elderberry"};

        // 创建一个 Stream
        Stream<String> fruitStream = Arrays.stream(fruits);

        // 反转元素顺序
        Stream<String> reversedFruits = fruitStream.sorted(Collections.reverseOrder()).reversed();

        reversedFruits.forEach(System.out::println);
    }
}
```

         此例中，我们定义了一个字符串数组，并用 Arrays.stream() 方法创建了一个字符串流。接着，我们使用 Collections.reverseOrder() 方法创建 Comparator 对象，然后使用 sorted() 方法对流进行排序，并传入 Comparator 对象进行逆向排序。最后，我们使用 forEach() 方法遍历输出排序后的结果。

          ## 3.6.归约
          ### 3.6.1.reduce()方法

         reduce() 方法用于对流中元素进行聚合。以下代码展示了如何使用 reduce() 方法计算流中元素之和：

         ```java
import java.util.*;
import java.util.stream.*;

public class ReduceExample {
    public static void main(String[] args) {
        int[] numbers = {1, 2, 3, 4, 5};

        // 创建一个 IntStream
        IntStream numStream = Arrays.stream(numbers);

        // 计算和
        OptionalInt sum = numStream.reduce((a, b) -> a + b);

        if (sum.isPresent()) {
            System.out.println("Sum of all elements is: " + sum.getAsInt());
        } else {
            System.out.println("No values present!");
        }
    }
}
```

         此例中，我们定义了一个int型数组，并用 Arrays.stream() 方法创建了一个IntStream。接着，我们使用 reduce() 方法对流进行求和运算，并传入 Lambda 表达式作为聚合函数。最后，我们检查是否有值存在，并打印出来。

          ### 3.6.2.collect()方法

         collect() 方法用于收集流中的元素。以下代码展示了如何使用 collect() 方法将流转换为集合：

         ```java
import java.util.*;
import java.util.stream.*;

public class CollectExample {
    public static void main(String[] args) {
        String[] fruits = {"apple", "banana", "cherry", "date", "elderberry"};

        // 创建一个 Stream
        Stream<String> fruitStream = Arrays.stream(fruits);

        // 转换为集合
        List<String> collectedFruits = fruitStream.collect(Collectors.toList());

        System.out.println("Collected Fruits: " + collectedFruits);
    }
}
```

         此例中，我们定义了一个字符串数组，并用 Arrays.stream() 方法创建了一个字符串流。接着，我们使用 collect() 方法将流转换为 ArrayList，并将结果输出。

          ## 3.7.并行流
          ParallelStream 是 JDK 1.8 中的新接口，该接口提供一种简单且易于使用的方式来并行执行操作。与普通 Stream 不同，ParallelStream 采用了 Fork/Join 模型来充分利用多核系统资源，可以极大提升性能。
          ### 3.7.1.parallel()方法

          parallel() 方法用于将流设置为并行模式。以下代码展示了如何使用 parallel() 方法将流设置为并行模式：

          ```java
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.*;

public class ParallelExample {
    public static void main(String[] args) throws ExecutionException, InterruptedException {
        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

        try {
            // 生成一个有 10 个元素的无限流
            IntStream infiniteNumStream = IntStream.iterate(0, i -> i + 1).limit(10);

            // 设置流为并行模式
            IntStream paralellInfiniteNumStream = infiniteNumStream.parallel();

            // 执行流操作，如找出最大值
            Future<Integer> futureMax = executor.submit(() ->
                    paralellInfiniteNumStream.max().orElseThrow(NoSuchElementException::new));

            // 获取最大值
            int max = futureMax.get();
            System.out.println("Maximum Value: " + max);

        } finally {
            executor.shutdown();
        }
    }
}
```

           此例中，我们使用 iterate() 和 limit() 方法生成一个有 10 个元素的无限流，并设置该流为并行模式。然后，我们提交一个 Callable 对象到线程池中，该对象使用 max() 方法找到并返回流中的最大值。最后，我们获取并输出该最大值。

          ### 3.7.2.sequential()方法

          sequential() 方法用于将流设置为串行模式。以下代码展示了如何使用 sequential() 方法将流设置为串行模式：

          ```java
import java.util.*;
import java.util.stream.*;

public class SequentialExample {
    public static void main(String[] args) {
        int[] numbers = {1, 2, 3, 4, 5};

        // 创建一个 IntStream
        IntStream numStream = Arrays.stream(numbers);

        // 设置流为串行模式
        IntStream seqNumStream = numStream.sequential();

        // 执行流操作，如查找元素
        boolean anyGreaterThanThree = seqNumStream.anyMatch(i -> i > 3);

        System.out.println("Are there any elements greater than three? " + anyGreaterThanThree);
    }
}
```

           此例中，我们定义了一个int型数组，并用 Arrays.stream() 方法创建了一个IntStream。然后，我们使用 sequential() 方法将其设置为串行模式。接着，我们执行流操作，如查询是否存在某个元素大于三。最后，我们输出查询的结果。

          ## 3.8.并行度配置
          默认情况下，Java 8 使用当前机器的 CPU 核心数作为并行度，但也可以通过修改虚拟机参数来更改这一默认设置。以下命令可以查看并行度：

          ```java
java -XX:+UnlockExperimentalVMOptions -XX:+UseEpsilonGC -Xmx4G -Xms4G -version | grep Parallel
```

           命令会打印出相关信息，其中就包含当前机器的并行度设置。你可以使用 -D 选项来临时修改并行度，如下所示：

          ```java
java -Djava.util.stream.parallel.enabled=false YourMainClass
```

           这样的话，你的 Java 应用程序就不会启用并行流功能。