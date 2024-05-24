
作者：禅与计算机程序设计艺术                    

# 1.简介
         

         **什么是接口？为什么在Java中需要用到它们？**
         
         在本篇博文中，我们将会对接口(interface)这个概念进行详细的介绍，并且探讨它是如何应用于Java编程语言中的，为什么使用它们以及它的一些特性。阅读完本篇文章，你应该能够掌握如下知识点：
        
         - 理解接口概念以及其角色和作用；
         - 从语法层面上理解接口的定义、属性、方法、继承等；
         - 了解接口的设计模式，包括适配器模式、组合模式、代理模式、桥接模式和观察者模式；
         - 使用接口设计原则，提升代码的可维护性；
         - 为何接口能让你的代码更加灵活、松耦合。 
         
         为了帮助读者对接口有一个更好的理解，下面我们一起看看Java中接口的一些特点和用途吧！
     
         # 2.概念术语介绍
         
         ## 2.1 什么是接口？
         
         在计算机编程中，接口（英语：Interface）就是一个契约。它定义了两个或者多个类的行为特征，要求其它类实现这些特征。接口不能被实例化，只能通过其他类的实例来使用。接口中的所有成员方法都是抽象的，意味着没有方法体。只声明方法签名，而不提供实现细节，接口可以被任何类所实现。类可以直接实现多个接口，也可以间接地实现多重继承。
         
         例如，你希望编写一个计算器应用。这个应用允许用户输入两个数字并进行四则运算。那么该应用需要提供的功能如下：
         
         ```java
             public interface Calculator {
                 int add(int a, int b);
                 int subtract(int a, int b);
                 int multiply(int a, int b);
                 int divide(int a, int b);
             }
             
             public class MainClass {
                 public static void main(String[] args) {
                     Calculator calculator = new MyCalculator(); // create instance of the Calculator interface
                     System.out.println("Result: " + calculator.add(10, 5));
                 }
                 
                 private static class MyCalculator implements Calculator {
                     @Override
                     public int add(int a, int b) {
                         return a + b;
                     }
                     
                     @Override
                     public int subtract(int a, int b) {
                         return a - b;
                     }
                     
                     @Override
                     public int multiply(int a, int b) {
                         return a * b;
                     }
                     
                     @Override
                     public int divide(int a, int b) {
                         if (b == 0) {
                             throw new IllegalArgumentException("Cannot divide by zero");
                         }
                         
                         return a / b;
                     }
                 }
             }
         ```
         
         上述例子中，`Calculator`是一个接口，它定义了四个方法用来进行四则运算。`MyCalculator`类实现了`Calculator`接口，并且提供了自己的四则运算实现。当`MainClass`类调用`calculator`对象的`add()`方法时，实际上是在调用`MyCalculator`类的`add()`方法。这种实现方式使得`MainClass`类不需要知道`MyCalculator`类的具体实现。它只需要通过`Calculator`接口与`MyCalculator`类进行交互。
         
         如果某个类只需要从另一个类中获得几个方法的功能，而且不需要额外的数据结构或状态信息，那么就可以使用接口作为该类的一种替代方案。这样可以减少依赖关系和耦合度。
         
         ## 2.2 为什么在Java中需要用到接口？
         
         ### 2.2.1 实现多态
         
         如同子类可以覆盖父类的相同方法一样，接口也支持多态特性。这是因为接口方法签名唯一，因此编译器可以确保接口只有唯一的实现，所以可以安全地向下转型为接口类型。
         
         比如，假设我们有以下代码：
         
         ```java
             public interface Animal {
                 void eat();
             }
             
             public abstract class Pet {
                 protected String name;
                 protected boolean hungry;
                 protected Animal animalType;
                 
                 public Pet(String name, boolean hungry, Animal animalType) {
                     this.name = name;
                     this.hungry = hungry;
                     this.animalType = animalType;
                 }
                 
                 public abstract void play();
                 
                 public void feed() {
                     if (!this.hungry) {
                         System.out.println(this.getName() + " is not hungry!");
                     } else {
                         System.out.println(this.getName() + " is eating...");
                         this.setHungry(false);
                         this.animalType.eat();
                     }
                 }
                 
                 public void setHungry(boolean value) {
                     this.hungry = value;
                 }
                 
                 public String getName() {
                     return this.name;
                 }
             }
             
             public class Cat extends Pet {
                 public Cat(String name, boolean hungry) {
                     super(name, hungry, new Dog());
                 }
                 
                 @Override
                 public void play() {
                     System.out.println(this.getName() + " is playing with its owner.");
                 }
             
                 @Override
                 public void eat() {
                     System.out.println(this.getName() + " is sleeping...");
                 }
             }
             
             public class Dog implements Animal {
                 public void eat() {
                     System.out.println(this.getClass().getSimpleName() + " is eating meat...");
                 }
             }
         ```
         
         在上面代码中，`Animal`接口代表了一系列动物的共同行为，其中包括`eat()`方法。`Cat`类是由`Pet`类继承而来的，它同时还实现了`Animal`接口。
         
         当创建了一个新的`Cat`对象的时候，它会自动按照如下方式初始化自己：
         
         ```java
             Cat cat = new Cat("Tom", true);
             ((Dog)cat.getAnimalType()).eat(); // cast to dog type first then call method on it
             cat.play();
             cat.feed();
         ```
         
         `((Dog)cat.getAnimalType())`语句显式地转换为`Dog`类型，然后调用它的`eat()`方法。虽然这里是针对`Cat`类的特定实例，但是由于`Cat`实现了`Animal`接口，所以`dogType`变量实际上也是一个`Animal`类型的对象。也就是说，`feed()`方法可以在没有警告的情况下正常工作。这就是多态的一种应用场景。
         
         ### 2.2.2 提高代码的可复用性
         
         通过接口，我们可以方便地在不同的上下文环境中使用相同的代码。比如，如果某个第三方库想要使用某个类的方法，但又不想引入该类本身的依赖关系，就可以创建一个接口，让该类实现该接口即可。这样做可以简化开发难度，提高代码的可复用性。
         
         ### 2.2.3 降低耦合度
         
         由于接口只定义了方法签名，因此类可以自由选择实现哪些方法，而不需要考虑其它方法是否存在或如何实现。这就降低了耦合度，促进了代码的模块化和可维护性。
         
         ### 2.2.4 封装变化
         
         通过接口，我们可以避免过多的关注内部实现细节，只需关注外部行为。这就提供了封装变化的能力，防止随着需求的变更导致代码的失效。
         
         # 3.接口的设计模式
         
         有关接口的设计模式主要包括以下几种：
         
         ## 3.1 适配器模式
         
         适配器模式用于把一个接口转换成另一个接口，使得原本因接口不匹配而无法使用的两个对象能在一起工作。
         
         比如，假设有一个名叫`MediaPlayer`的类，它提供了播放音乐的方法，而现在需要连接USB驱动，该驱动提供的文件系统却与播放器不兼容。我们可以使用适配器模式，先创建一个叫`UsbDriverAdapter`的类，它继承`MediaPlayer`并实现了与`UsbDriver`接口一致的`playMusic()`方法，这样就可以使用`UsbDriverAdapter`来播放Usb驱动的音乐文件。
         
         ```java
             public interface MediaPlayer {
                 void playMusic();
             }
             
             public interface UsbDriver {
                 void playMusic();
             }
             
             public class MusicPlayer implements MediaPlayer {
                 @Override
                 public void playMusic() {
                     System.out.println("Playing music from internal hard drive.");
                 }
             }
             
             public class UsbDriverAdapter implements MediaPlayer {
                 private final UsbDriver usbDriver;
                 
                 public UsbDriverAdapter(UsbDriver driver) {
                     this.usbDriver = driver;
                 }
                 
                 @Override
                 public void playMusic() {
                     usbDriver.playMusic();
                 }
             }
         ```
         
         在上述代码中，我们定义了`MediaPlayer`和`UsbDriver`两个接口。`MusicPlayer`类实现了`MediaPlayer`接口，它可以播放音乐文件的本地硬盘。`UsbDriverAdapter`类继承自`MediaPlayer`，同时实现了`UsbDriver`接口。在构造函数中，它接受一个`UsbDriver`类型的对象，并保存起来供后续使用。`UsbDriverAdapter`的`playMusic()`方法只是简单的调用传入的`UsbDriver`对象的`playMusic()`方法。
         
         ```java
             public class Main {
                 public static void main(String[] args) {
                     MusicPlayer player = new MusicPlayer();
                     player.playMusic();
                     
                     UsbDriver driver = new InternalUsbDriver();
                     UsbDriverAdapter adapter = new UsbDriverAdapter(driver);
                     adapter.playMusic();
                 }
             
                 private static class InternalUsbDriver implements UsbDriver {
                     @Override
                     public void playMusic() {
                         System.out.println("Playing music from USB driver.");
                     }
                 }
             }
         ```
         
         在主程序中，我们可以创建两种类型的播放器，分别是`MusicPlayer`和`UsbDriverAdapter`。前者是普通的播放器，后者通过适配器类来访问另一个接口，使得它可以通过另一种接口的特性来播放音乐。最后，我们创建了另外一个`UsbDriver`的实现类，并把它传递给`UsbDriverAdapter`，从而完成了播放音乐的任务。
         
         ## 3.2 组合模式
         
         组合模式用于表示对象的分层结构，即一个对象可以包含零个或多个子对象，而这些子对象本身也可能是复杂的对象，拥有自己的子对象。它使得客户端可以统一地处理单个对象和组合对象，因为他们都具有相同的接口。
         
         比如，假设有一个名叫`Employee`的类，它代表了一个雇员。它可能有多个部门组成，每个部门又可能有多个雇员。我们可以定义出一个接口`Organization`，来描述每个组织的组织架构。
         
         ```java
             public interface Organization {
                 List<Department> getDepartments();
             }
             
             public interface Department {
                 List<Employee> getEmployees();
             }
             
             public class Company implements Organization {
                 private List<Department> departments;
                 
                 public Company() {
                     this.departments = new ArrayList<>();
                 }
                 
                 public void addDepartment(Department department) {
                     this.departments.add(department);
                 }
                 
                 @Override
                 public List<Department> getDepartments() {
                     return Collections.unmodifiableList(this.departments);
                 }
             }
             
             public class Division implements Department {
                 private String name;
                 private List<Employee> employees;
                 
                 public Division(String name) {
                     this.name = name;
                     this.employees = new ArrayList<>();
                 }
                 
                 public void addEmployee(Employee employee) {
                     this.employees.add(employee);
                 }
                 
                 @Override
                 public List<Employee> getEmployees() {
                     return Collections.unmodifiableList(this.employees);
                 }
             }
             
             public class Employee {
                 private String name;
                 
                 public Employee(String name) {
                     this.name = name;
                 }
                 
                 public String getName() {
                     return this.name;
                 }
             }
         ```
         
         在上述代码中，`Company`类实现了`Organization`接口，它代表了一个公司，它可能由多个部门组成，每个部门都可以再包含多个雇员。`Division`类实现了`Department`接口，它代表了一个部门，它可以包含多个雇员。`Employee`类是一个简单类，代表了一个雇员。
         
         可以看到，所有的类都实现了`Organization`、`Department`或`Employee`接口。通过这种接口，客户端可以轻松地处理不同种类的组织结构。
         
         ## 3.3 代理模式
         
         代理模式是一种结构型设计模式，它提供一个替代品或占位符以控制对象的访问。代理可以拦截对原始对象的访问，并根据需要进行额外的操作，如记录日志、验证权限、缓存结果等。
         
         比如，假设有一个名叫`BankAccount`的类，它代表了一个银行账户。我们想在这个类上增加一项功能，即允许客户查看他的账户余额。但是客户并不一定信任这个类，他可能恶意篡改数据，甚至可以尝试伪造请求来冒充他人。此时，我们可以创建一个代理类，它持有对真实账户对象的引用，并在每次调用账户相关的方法之前，检查是否有合法的身份。
         
         ```java
             public interface BankAccount {
                 double checkBalance();
             }
             
             public class RealBankAccount implements BankAccount {
                 private double balance;
                 
                 public RealBankAccount(double balance) {
                     this.balance = balance;
                 }
                 
                 @Override
                 public double checkBalance() {
                     return this.balance;
                 }
                 
                 public void deposit(double amount) {
                     this.balance += amount;
                 }
             }
             
             public class BankAccountProxy implements BankAccount {
                 private final BankAccount account;
                 private final String customerName;
                 
                 public BankAccountProxy(String customerName, BankAccount account) {
                     this.customerName = customerName;
                     this.account = account;
                 }
                 
                 @Override
                 public double checkBalance() {
                     authenticateCustomer(customerName);
                     
                     return this.account.checkBalance();
                 }
                 
                 private void authenticateCustomer(String name) {
                     // perform authentication checks here
                 }
             }
         ```
         
         在上述代码中，`RealBankAccount`类实现了`BankAccount`接口，代表了银行账户。`BankAccountProxy`类是代理类，它持有对真实账户对象的引用，并在每次调用账户相关的方法之前，检查是否有合法的身份。`authenticateCustomer()`方法需要由代理类自己实现，我们假定它会校验客户身份。
         
         ```java
             public class Client {
                 public static void main(String[] args) {
                     BankAccount realAccount = new RealBankAccount(1000.00);
                     
                     BankAccount proxy = new BankAccountProxy("Alice Smith", realAccount);
                     
                     // access bank account methods as usual
                     double balance = proxy.checkBalance();
                     
                     // operations performed through proxy will be authenticated before accessing real account
                     proxy.deposit(500.00);
                     
                     double updatedBalance = proxy.checkBalance();
                     
                     System.out.println("Current balance: $" + updatedBalance);
                 }
             }
         ```
         
         在客户端代码中，我们创建了真实账户和代理账户，并调用他们各自的方法。由于代理类只会拦截对真实账户的访问，并在必要时进行身份认证，所以它们不会受到非法访问的影响。代理类还可以记录客户的所有访问，以便稍后追踪其活动。
         
         ## 3.4 桥接模式
         
         桥接模式也称为柏拉图模式，用于将抽象部分与实现部分分离开来，使得两者可以独立变化。它最主要的优点是隔离了抽象层次和实现部分，使得二者可以从根本上独立地扩展。
         
         比如，假设我们要编写一个图像编辑软件。编辑操作可以由不同的组件完成，比如裁剪、旋转、滤镜等。这些组件一般都属于不同的领域，比如裁剪功能可以由UI组件完成，而滤镜功能可以由底层的图像处理算法实现。
         
         此时，我们可以创建一个叫`Editor`的接口，它包含编辑相关的操作，比如打开图片、保存图片、调整大小、裁剪等。然后，我们就可以创建多个不同的实现类，比如`UserInterfaceComponent`、`ImageProcessingAlgorithm`和`PhotoEditingSoftware`，每个实现类负责实现`Editor`接口中的相应操作。
         
         ```java
             public interface Editor {
                 void openFile();
                 void saveFile();
                 void resize(double width, double height);
                 void crop(Rectangle area);
             }
             
             public class UserInterfaceComponent implements Editor {
                 private Picture picture;
                 
                 public UserInterfaceComponent(Picture picture) {
                     this.picture = picture;
                 }
                 
                 @Override
                 public void openFile() {
                     // implement file opening functionality
                 }
                 
                 @Override
                 public void saveFile() {
                     // implement file saving functionality
                 }
                 
                 @Override
                 public void resize(double width, double height) {
                     // implement resizing logic using UI component
                 }
                 
                 @Override
                 public void crop(Rectangle area) {
                     // implement cropping logic using UI component
                 }
             }
             
             public interface ImageProcessingAlgorithm {
                 void applyFilter(String filterName);
                 void flipHorizontal();
                 void rotateClockwise();
             }
             
             public class SobelEdgeDetectionAlgorithm implements ImageProcessingAlgorithm {
                 @Override
                 public void applyFilter(String filterName) {
                     // implement filtering algorithm for sobel edge detection
                 }
                 
                 @Override
                 public void flipHorizontal() {
                     // implement flipping algorithm
                 }
                 
                 @Override
                 public void rotateClockwise() {
                     // implement rotation algorithm
                 }
             }
             
             public class PhotoEditingSoftware {
                 private Editor editor;
                 
                 public PhotoEditingSoftware(Editor editor) {
                     this.editor = editor;
                 }
                 
                 public void editImage() {
                     editor.openFile();
                     editor.resize(720, 480);
                     editor.applyFilter("Sobel Edge Detection");
                     editor.crop(new Rectangle(100, 100, 500, 500));
                     editor.saveFile();
                 }
             }
         ```
         
         在上述代码中，`Editor`接口定义了编辑相关的操作，包括打开、保存、调整大小和裁剪图片。`UserInterfaceComponent`类实现了`Editor`接口，它负责实现编辑操作的UI逻辑。`ImageProcessingAlgorithm`接口定义了图像处理相关的操作，比如滤镜、翻转、旋转。`SobelEdgeDetectionAlgorithm`类实现了`ImageProcessingAlgorithm`接口，它负责实现图像的边缘检测算法。`PhotoEditingSoftware`类聚合了`Editor`和`ImageProcessingAlgorithm`接口，并在运行时动态地进行组合。
         
         这样一来，客户端代码只需要使用`Editor`接口就能访问编辑功能，而无需关注底层的实现细节。换言之，客户端代码不需要知道底层的图像处理算法，也不必担心底层的UI组件的可用性。这就是桥接模式的优点所在。
         
         ## 3.5 观察者模式
         
         观察者模式用于对象之间的一对多通信。一个对象发生改变时，它的所有依赖者都会收到通知并自动更新。
         
         比如，假设有一个名叫`WeatherData`的类，它代表了天气预报数据。其他的对象可以订阅它，当天气数据发生变化时，它就会自动通知订阅者。
         
         ```java
             import java.util.*;
             
             public class WeatherData {
                 private float temperature;
                 private float humidity;
                 private float pressure;
                 private Subject subject;
                 
                 public WeatherData(Subject subject) {
                     this.subject = subject;
                 }
                 
                 public void setMeasurements(float temperature, float humidity, float pressure) {
                     this.temperature = temperature;
                     this.humidity = humidity;
                     this.pressure = pressure;
                     
                     notifyObservers();
                 }
                 
                 private void notifyObservers() {
                     subject.update(temperature, humidity, pressure);
                 }
                 
                 public void measurementsChanged() {
                     System.out.println("Temperature: " + temperature);
                     System.out.println("Humidity: " + humidity);
                     System.out.println("Pressure: " + pressure);
                     System.out.println("");
                 }
             }
             
             public interface Observer {
                 void update(float temperatur, float humidity, float pressure);
             }
             
             public class CurrentConditionsDisplay implements Observer {
                 private float temperature;
                 private float humidity;
                 
                 @Override
                 public void update(float temperature, float humidity, float pressure) {
                     this.temperature = temperature;
                     this.humidity = humidity;
                     
                     display();
                 }
                 
                 private void display() {
                     System.out.print("Current conditions: ");
                     
                     if (temperature > 10 && temperature < 30) {
                         System.out.println("pretty nice weather today!");
                     } else if (temperature >= 30 && temperature <= 40) {
                         System.out.println("oh my god, it's very hot outside!!!");
                     } else if (temperature > 40) {
                         System.out.println("hell, it's really hot out there!");
                     } else {
                         System.out.println("it's too cold outside, wear some sunscreen!");
                     }
                     
                     System.out.println("Humidity: " + humidity + "%");
                 }
             }
             
             public class StatisticsDisplay implements Observer {
                 private Map<Float, Integer> temperatureCountMap;
                 private Map<Float, Integer> humidityCountMap;
                 
                 public StatisticsDisplay() {
                     this.temperatureCountMap = new HashMap<>();
                     this.humidityCountMap = new HashMap<>();
                 }
                 
                 @Override
                 public void update(float temperature, float humidity, float pressure) {
                     incrementCountForTemperature(temperature);
                     incrementCountForHumidity(humidity);
                     
                     displayStatistics();
                 }
                 
                 private void incrementCountForTemperature(float temperature) {
                     temperatureCountMap.putIfAbsent(temperature, 0);
                     temperatureCountMap.compute(temperature, (k, v) -> ++v);
                 }
                 
                 private void incrementCountForHumidity(float humidity) {
                     humidityCountMap.putIfAbsent(humidity, 0);
                     humidityCountMap.compute(humidity, (k, v) -> ++v);
                 }
                 
                 private void displayStatistics() {
                     System.out.print("Temperature statistics: mean: "
                             + calculateMeanTemperature() + ", range: ["
                             + calculateMinimumTemperature() + "-"
                             + calculateMaximumTemperature() + "]");
                     System.out.println(", count map: " + temperatureCountMap);
                     System.out.println();
                     
                     System.out.print("Humidity statistics: mean: "
                             + calculateMeanHumidity() + ", range: ["
                             + calculateMinimumHumidity() + "-"
                             + calculateMaximumHumidity() + "]");
                     System.out.println(", count map: " + humidityCountMap);
                     System.out.println();
                 }
                 
                 private float calculateMeanTemperature() {
                     float sum = 0;
                     int count = 0;
                     
                     for (float key : temperatureCountMap.keySet()) {
                         sum += key * temperatureCountMap.get(key);
                         count += temperatureCountMap.get(key);
                     }
                     
                     if (count!= 0) {
                         return sum / count;
                     } else {
                         return 0;
                     }
                 }
                 
                 private float calculateMinimumTemperature() {
                     if (temperatureCountMap.isEmpty()) {
                         return 0;
                     }
                     
                     Float minValue = null;
                     for (float key : temperatureCountMap.keySet()) {
                         if (minValue == null || key < minValue) {
                             minValue = key;
                         }
                     }
                     
                     return minValue;
                 }
                 
                 private float calculateMaximumTemperature() {
                     if (temperatureCountMap.isEmpty()) {
                         return 0;
                     }
                     
                     Float maxValue = null;
                     for (float key : temperatureCountMap.keySet()) {
                         if (maxValue == null || key > maxValue) {
                             maxValue = key;
                         }
                     }
                     
                     return maxValue;
                 }
                 
                 private float calculateMeanHumidity() {
                     float sum = 0;
                     int count = 0;
                     
                     for (float key : humidityCountMap.keySet()) {
                         sum += key * humidityCountMap.get(key);
                         count += humidityCountMap.get(key);
                     }
                     
                     if (count!= 0) {
                         return sum / count;
                     } else {
                         return 0;
                     }
                 }
                 
                 private float calculateMinimumHumidity() {
                     if (humidityCountMap.isEmpty()) {
                         return 0;
                     }
                     
                     Float minValue = null;
                     for (float key : humidityCountMap.keySet()) {
                         if (minValue == null || key < minValue) {
                             minValue = key;
                         }
                     }
                     
                     return minValue;
                 }
                 
                 private float calculateMaximumHumidity() {
                     if (humidityCountMap.isEmpty()) {
                         return 0;
                     }
                     
                     Float maxValue = null;
                     for (float key : humidityCountMap.keySet()) {
                         if (maxValue == null || key > maxValue) {
                             maxValue = key;
                         }
                     }
                     
                     return maxValue;
                 }
             }
             
             public class ForecastDisplay implements Observer {
                 private StringBuilder forecastBuilder;
                 
                 public ForecastDisplay() {
                     this.forecastBuilder = new StringBuilder();
                 }
                 
                 @Override
                 public void update(float temperature, float humidity, float pressure) {
                     forecastBuilder.append("* " + temperature + "/F with " + humidity
                                     + "% humidity, expect heavy rains tomorrow
");
                     
                     displayForecast();
                 }
                 
                 private void displayForecast() {
                     System.out.println("Three-day forecast:");
                     System.out.println(forecastBuilder.toString());
                 }
             }
             
             public interface Subject {
                 void registerObserver(Observer observer);
                 void removeObserver(Observer observer);
                 void notifyObservers();
             }
             
             public class WeatherDataSubject implements Subject {
                 private final List<Observer> observers = new ArrayList<>();
                 
                 @Override
                 public void registerObserver(Observer observer) {
                     observers.add(observer);
                 }
                 
                 @Override
                 public void removeObserver(Observer observer) {
                     observers.remove(observer);
                 }
                 
                 @Override
                 public void notifyObservers() {
                     for (Observer observer : observers) {
                         observer.update(weatherData.getTemperature(),
                                         weatherData.getHumidity(),
                                         weatherData.getPressure());
                     }
                 }
                 
                 private WeatherData weatherData;
                 
                 public WeatherDataSubject(WeatherData weatherData) {
                     this.weatherData = weatherData;
                     weatherData.registerSubscriber(this);
                 }
                 
                 public void unregister() {
                     weatherData.unregisterSubscriber(this);
                 }
             }
         ```
         
         在上述代码中，`WeatherData`类实现了天气数据的获取及更新。`Subject`接口定义了观察者模式中的主题接口，它提供了注册、注销和通知观察者的方法。`CurrentConditionsDisplay`类实现了`Observer`接口，它显示当前的温度、湿度和气压，并根据温度变化提供不同的天气描述。`StatisticsDisplay`类统计当前的温度和湿度数据，并输出平均值、范围和计数分布。`ForecastDisplay`类生成简单的天气预报，并打印到控制台。
         
         `WeatherDataSubject`类实现了`Subject`接口，它负责管理订阅者列表，并将数据更新通知到所有订阅者。`WeatherDataSubject`构造函数会自动将自身注册到`WeatherData`对象，当`WeatherData`对象更新数据时，它会自动将数据通知到所有订阅者。当不再需要订阅时，`unregister()`方法可以注销自身。