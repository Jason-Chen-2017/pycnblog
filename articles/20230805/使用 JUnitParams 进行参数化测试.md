
作者：禅与计算机程序设计艺术                    

# 1.简介
         
　　JunitParams 是 JUnit 的一个扩展工具库。它可以让测试方法的参数列表化，并且生成多组参数组合，帮助开发者编写更灵活、可读性强且易于维护的代码。本文将介绍 JunitParams 并演示如何使用它来进行参数化测试。
         ## 2.1 JUnitParams 安装配置
         ### 2.1.1 添加依赖
         在 pom.xml 文件中添加以下依赖即可:

         ```
            <dependency>
               <groupId>pl.pragmatists</groupId>
               <artifactId>JUnitParams</artifactId>
               <version>1.1.1</version>
               <!-- or use the latest version -->
               
               <scope>test</scope>
            </dependency>
         ```

         
         可以通过 `mvn clean install` 命令下载 JUnitParams 到本地仓库。
         ### 2.1.2 初始化框架
         在测试类上添加 `@RunWith(JUnitParamsRunner.class)` 来初始化框架，如下所示:
         
         ```java
           @RunWith(JUnitParamsRunner.class)
           public class ParameterizedTest {
              // your test methods here...
           }
         ```

         
         `@RunWith` 注解用于指定运行器，这里用的是 `JUnitParamsRunner`。
         ### 2.1.3 配置 @Parameters 方法
         `@Parameters` 方法是一个返回 Object[] 或 Collection<Object[]> 类型的静态方法。在此方法中定义了测试方法的参数列表。为了使用 JUnitParams ，需要将测试方法的参数类型修改为 `@Parameter`，并指明参数所在位置（从1开始）。如：
         
         ```java
             @RunWith(JUnitParamsRunner.class)
             public class ParameterizedTest {
                 @Test
                 @Parameters({
                     "1 + 2 = 3",    //parameter at position 1 (first parameter)
                     "2 - 1 = 1"      //parameter at position 2 (second parameter)
                  })
                 public void basicArithmeticOperations(@Parameter(value = 1) int a,
                                                       @Parameter(value = 2) int b,
                                                       @Parameter(value = "{index}+{1}-{0}") String expression,
                                                       @Parameter(value = "(a==b)") boolean result) throws Exception {
                      assertEquals("Expression should be correct.", expression, "(" + a + "==" + b + ")");
                      assertTrue("Result should be true.", result);
                  }
             }
         ```

         
         `@Parameters` 方法注解中的值为一组参数集合。每个参数是由逗号分隔的字符串，其中第一个字符串表示该参数的描述信息。第二个字符串表示参数的实际值。如果希望 `@Parameter` 注解自动填充测试方法的参数位置，可以在第二个字符串中使用 `${index}` 表示当前参数在 `@Parameters` 参数列表中的位置，例如 `@Parameter(value = "${index}+${2}-${1}")`。`$` 符号后面的数字表示当前参数在 `@Parameters` 参数列表中的位置。这样的话，无需手动填写参数位置，注解会自动识别。`@Parameters` 方法也可以返回 Collection<Object[]>，表示生成多个测试用例。
         ### 2.1.4 参数匹配规则
         如果 `@Parameter` 的 value 属性的值不是以 "${ }" 开头，则 JUnitParams 会把其视为参数名，通过反射获取相应的值。可以通过 `@Named()` 和 `@NotNamed()` 来控制匹配规则。
         ### 2.1.5 默认参数值
         如果某个参数没有设置默认值，则 JUnitParams 将会把它作为 null。你可以通过设置 defaultValue 属性来提供默认值，例如：
         
         ```java
             @RunWith(JUnitParamsRunner.class)
             public class ParameterizedTest {
                 @Test
                 @Parameters({
                     "1 + ${num}",          // first parameter is 1 and second parameter is provided with 'num' name
                     "@NotNull(${str})",     // third parameter is not provided but has default value '@NotNull' so it won't cause any problems when used as method argument
                 })
                 public void myTestMethod(@Parameter(name="num", defaultValue="-1") int num,
                                           @Parameter() String str,
                                           @Parameter(defaultValue="@NotNull") Object object) throws Exception {
                      System.out.println(object == null? "null" : object); // prints "null" because no value was provided for this parameter
                 }
             }
         ```

         
         上述例子展示了当第3个参数没有提供时，它的默认值为 "@NotNull"。
         ## 2.2 测试结果
         JUnitParams 生成的测试结果示例如下：
         
         ```java
            .F.
             Time: 0.011
             
             There were failures detected:
             
             1) parametersShouldHaveDescriptionInAnnotationMethod(com.example.ParameterizedTest)
             java.lang.AssertionError: Description in annotation method is missing for parameter at index 0
             	at org.junit.Assert.fail(Assert.java:88)
             	at com.github.nosan.embedded.cassandra.junit.CassandraRule$Builder.<init>(CassandraRule.java:171)
             	at com.github.nosan.embedded.cassandra.junit.CassandraExtension.beforeAll(CassandraExtension.java:95)
             	at org.junit.rules.ExternalResource$1.evaluate(ExternalResource.java:46)
             	at org.junit.internal.runners.statements.FailOnTimeout$CallableStatement.call(FailOnTimeout.java:298)
             	at org.junit.internal.runners.statements.FailOnTimeout$CallableStatement.call(FailOnTimeout.java:292)
             	at java.util.concurrent.FutureTask.run(FutureTask.java:266)
             	at java.lang.Thread.run(Thread.java:748)
             
             2) parametersShouldHaveValueInAnnotationMethod(com.example.ParameterizedTest)
             java.lang.AssertionError: Value in annotation method is missing for parameter at index 0
             	at org.junit.Assert.fail(Assert.java:88)
             	at com.github.nosan.embedded.cassandra.junit.CassandraRule$Builder.<init>(CassandraRule.java:171)
             	at com.github.nosan.embedded.cassandra.junit.CassandraExtension.beforeAll(CassandraExtension.java:95)
             	at org.junit.rules.ExternalResource$1.evaluate(ExternalResource.java:46)
             	at org.junit.internal.runners.statements.FailOnTimeout$CallableStatement.call(FailOnTimeout.java:298)
             	at org.junit.internal.runners.statements.FailOnTimeout$CallableStatement.call(FailOnTimeout.java:292)
             	at java.util.concurrent.FutureTask.run(FutureTask.java:266)
             	at java.lang.Thread.run(Thread.java:748)
             
             3) shouldGenerateMultipleTestCasesUsingCollectionOfArrays(com.example.ParameterizedTest)
             java.lang.NullPointerException
             	at sun.reflect.NativeMethodAccessorImpl.invoke0(Native Method)
             	at sun.reflect.NativeMethodAccessorImpl.invoke(NativeMethodAccessorImpl.java:62)
             	at sun.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
             	at java.lang.reflect.Method.invoke(Method.java:498)
             	at junitparams.internal.InvokeParametrizedMethodRunner.run(InvokeParametrizedMethodRunner.java:46)
             	at junitparams.internal.ParameterisedTestClassRunner.run(ParameterisedTestClassRunner.java:136)
             	at org.junit.runners.Suite.runChild(Suite.java:128)
             	at org.junit.runners.Suite.runChild(Suite.java:27)
             	at org.junit.runners.ParentRunner$3.run(ParentRunner.java:290)
             	at org.junit.runners.ParentRunner$1.schedule(ParentRunner.java:71)
             	at org.junit.runners.ParentRunner.runChildren(ParentRunner.java:288)
             	at org.junit.runners.ParentRunner.access$000(ParentRunner.java:58)
             	at org.junit.runners.ParentRunner$2.evaluate(ParentRunner.java:268)
             	at org.junit.runners.ParentRunner.run(ParentRunner.java:363)
             	at org.apache.maven.surefire.junit4.JUnit4Provider.execute(JUnit4Provider.java:264)
             	at org.apache.maven.surefire.junit4.JUnit4Provider.executeTestSet(JUnit4Provider.java:153)
             	at org.apache.maven.surefire.junit4.JUnit4Provider.invoke(JUnit4Provider.java:124)
             	at org.apache.maven.surefire.booter.ForkedBooter.invokeProviderInSameClassLoader(ForkedBooter.java:200)
             	at org.apache.maven.surefire.booter.ForkedBooter.runSuitesInProcess(ForkedBooter.java:153)
             	at org.apache.maven.surefire.booter.ForkedBooter.main(ForkedBooter.java:103)
             
             FAILURES!!!
             Tests run: 3,  Failures: 3
             
             [INFO] ------------------------------------------------------------------------
             [INFO] BUILD FAILURE
             [INFO] ------------------------------------------------------------------------
             [INFO] Total time:  01:13 min
             [INFO] Finished at: 2020-07-20T01:31:16Z
             [INFO] ------------------------------------------------------------------------
             [ERROR] Failed to execute goal org.apache.maven.plugins:maven-surefire-plugin:2.22.2:test (default-test) on project cassandra-junit-tests: There are test failures.
            ...
         
         ```

         
         从测试报告中可以看到，运行了三个测试方法，成功运行了两个，失败了两个。因为参数个数不对，导致方法内部逻辑报错。
         ## 2.3 小结
         本文简单介绍了 JunitParams 的安装配置及基本用法。它可以方便地生成多组参数组合，提升单元测试的效率。对于那些需要针对同一个输入组合进行多个测试的方法，可以尝试使用 JunitParams 来节省重复的代码。