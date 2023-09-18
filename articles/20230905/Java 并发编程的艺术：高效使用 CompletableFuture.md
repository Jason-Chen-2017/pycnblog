
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 一、什么是 CompletableFuture?
CompletableFuture 是 Java 8 中引入的一个全新的并发工具类，它提供了一种基于回调的执行方式，能够让异步任务的执行结果和处理流程更加容易控制。
## 二、为什么要用 CompletableFuture？
Java 并发编程一直以来都是一个难点。而 CompletableFuture 的出现则给了我们一个解决方案——通过 CompletableFuture 可以让编写异步、并行、非阻塞的代码变得更加简单、直观。
## 三、如何使用 CompletableFuture?
### （一）创建 CompletableFuture 对象
首先，需要导入 CompletableFuture 的包，如下：
```java
import java.util.concurrent.CompletableFuture;
```
其次，可以通过以下的方式创建 CompletableFuture 对象：
```java
// 创建 CompletableFuture 对象，默认使用当前线程池中的线程执行任务
CompletableFuture<String> future = new CompletableFuture<>(); 

// 通过 CompletableFuture 提供的方法 supplyAsync() 来指定任务运行在另一个线程中
CompletableFuture<Integer> resultFuture = CompletableFuture.supplyAsync(this::calculation);

// 通过 CompletableFuture 提供的 static 方法 runAsync() 来指定任务在默认线程池中执行
CompletableFuture<Void> voidResultFuture = CompletableFuture.runAsync(() -> {
    System.out.println("Task is running in the default thread pool");
});
```
上面的例子展示了两种不同类型的 CompletableFuture 对象：
- 使用 `new` 关键字创建的 CompletableFuture 对象，当调用方法 get() 时，会等待任务完成后再返回计算结果；
- 使用 `supplyAsync()` 方法创建的 CompletableFuture 对象，会在调用 `get()` 时立即返回计算结果，并且任务不会阻塞主线程；
- 使用 `runAsync()` 方法创建的 CompletableFuture 对象，只负责将任务提交到线程池中进行执行，但是不会获取任何结果。
### （二） CompletableFuture 的回调函数
CompletableFuture 支持对任务的执行结果进行回调处理。回调函数可以指定在任务正常结束或发生异常时应该被调用，也可以用来进一步处理任务的计算结果。
#### ① 当任务成功完成时触发的回调
可以使用 `thenAccept()` 方法注册一个任务成功完成后的回调函数，该回调函数接收计算出的结果作为参数，示例如下：
```java
public class FutureExample {

    public int calculation() throws InterruptedException {
        Thread.sleep(1000); // 模拟耗时的计算过程
        return 1 + 2;
    }
    
    public static void main(String[] args) throws Exception{
        final FutureExample example = new FutureExample();

        CompletableFuture<Integer> completableFuture = 
                CompletableFuture.supplyAsync(example::calculation).
                        thenApply(integer -> integer * 2).  
                        thenApply(integer -> integer - 1);
        
        Integer result = completableFuture.join();
        
        System.out.println("result: " + result);
    }
    
}
```
上述例子中，`thenApply()` 方法用于对上一个任务计算得到的结果做一些处理，这里只是乘以 2 和减去 1。
#### ② 当任务失败时触发的回调
可以使用 `exceptionally()` 方法注册一个任务失败时的回调函数，该回调函数接收导致任务失败的异常对象作为参数，并且只能返回一个替换计算结果的值。如果没有设置异常处理器，默认情况下 CompletableFuture 会向上传递 RuntimeException。
```java
final String value = completableFuture.exceptionally(ex -> "Error occurred").join();
System.out.println("value: " + value);
```
上述例子中，如果任务由于异常而失败，`exceptionally()` 方法会捕获这个异常，然后返回指定的字符串作为替代值。
#### ③ 当所有回调函数都执行完毕后触发的回调
可以使用 `whenComplete()` 方法注册一个任务执行结束后的回调函数，该回调函数接收计算结果（可能为空）和异常（也可能为空），并提供一个统一处理接口。
```java
completableFuture.whenComplete((result, ex) -> {
    if (ex == null) {
        System.out.println("Computation finished with result: " + result);
    } else {
        System.out.println("An error occurred: " + ex.getMessage());
    }
}).join();
```
上述例子中，`whenComplete()` 方法的第二个参数表示异常对象，如果有异常，则输出错误信息；否则，输出计算结果。
### （三） CompletableFuture 的组合
CompletableFuture 可以进行链式调用，从而实现多个 CompletableFuture 对象的组合。
```java
// 创建两个 CompletableFuture 对象
CompletableFuture<Double> future1 = CompletableFuture.completedFuture(1.0d);
CompletableFuture<Double> future2 = CompletableFuture.completedFuture(2.0d);

// 对两个 CompletableFuture 对象进行运算并返回新 CompletableFuture 对象
CompletableFuture<Double> sumFuture = 
        future1.thenCombine(future2, Double::sum);
        
double result = sumFuture.join();
System.out.println("Result: " + result);
```
上述例子中，先创建一个 CompletableFuture 对象，它已经计算出了一个 Double 类型的值，然后再创建一个 CompletableFuture 对象，它将前面那个 CompletableFuture 对象和 2.0d 相加，最后打印出最终的结果。
### （四） CompletableFuture 的取消操作
CompletableFuture 还支持取消操作，即可以取消正在执行的任务。取消操作可以由 CompletableFuture 或它的子类自己处理，也可以由用户自行定义取消策略。
```java
@Test
void testCancelOperation() throws Exception {
    ExecutorService executor = Executors.newFixedThreadPool(2);

    try {
        CompletionStage<String> stage = createCompletionStage(executor);

        assertTrue(stage.toCompletableFuture().cancel(true));

        ExecutionException exception = assertThrows(ExecutionException.class, () ->
                stage.toCompletableFuture().get());

        assertEquals(CancellationException.class, exception.getCause().getClass());
    } finally {
        executor.shutdownNow();
    }
}


private static CompletionStage<String> createCompletionStage(ExecutorService executor) {
    return CompletableFuture.supplyAsync(() -> {
        try {
            TimeUnit.SECONDS.sleep(Long.MAX_VALUE);
        } catch (InterruptedException e) {
            throw new IllegalStateException(e);
        }
        return "done";
    }, executor);
}
```
上述测试用例模拟了用户取消 CompletableFuture 执行的情况。通过调用 toCompletableFuture().cancel(true) 方法，可以取消 CompletableFuture 对象。由于 CompletableFuture 默认使用当前线程池执行任务，因此可以在其他任务线程中判断任务是否已被取消。如果任务被取消，就会抛出 CancellationException 异常。