
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 概述
结构型设计模式（英语：Structural design pattern）是面向对象软件设计领域中非常重要的设计模式。它强调通过组合简单的、单一功能的类或对象，来建立一个更大的结构体系，提供灵活的、可扩展的解决方案。结构型设计模式提供了一种方法论，帮助开发人员创建满足特定需要的复杂系统。结构型设计模式按照其特点可以分成三类：适配器模式、桥接模式、装饰器模式、外观模式、享元模式。本文主要介绍结构型设计模式中的适配器模式。
## 适配器模式
适配器模式（Adapter Pattern）用于将一个接口转换成客户希望的另一个接口。适配器允许被访问的对象能够使用另一个不同的接口。这种类型的设计模式属于结构型模式，因为这种模式帮助客户端在不修改自身源代码的情况下合作。
适配器模式定义了一个新的接口，该接口规范要求所需的功能。新接口是通过复制已有的对象来实现的，同时使用一个代理对象来保持原始对象的接口。客户端既可以使用目标接口，也可以使用原始接口。这样就使得原有类的接口可以使用，同时也能添加一些额外的特性。这种方式让客户端代码能够更容易地与其他系统交互。以下是一个适配器模式的一般结构：


1. Target（目标接口）: 这是客户端期望的接口。
2. Adaptee（被适配者）: 这是现存系统中的某个接口。
3. Adapter（适配器）: 这是用于使Adaptee符合Target接口的类。Adapter继承自Target并实现了Target的所有方法。在Adapter的方法里，调用Adaptee的方法来实现客户端的请求。
4. Client（客户端）: 使用了Target接口的对象。Client与Adaptee和Adapter之间的关系由Client自己的逻辑来确定。

例如，假设有一个客户端希望使用XML数据格式进行通信。但是实际系统可能使用了其他格式的数据。此时可以通过创建一个适配器来实现与其他格式的数据通信。

假设有如下两个接口。其中有一个接口规范要求输出到屏幕上的文字。另外一个接口规范要求输出到文件中的文本。

```java
public interface TextOutput {
    void display(String text);
}

public interface FileOutput {
    void saveToFile(String text);
}
```

如果要输出到屏幕上，那么直接用System.out.println()即可；如果要保存到文件中，则可以创建一个Writer对象，然后把文本写入到该Writer对象中。但假如系统中已经存在了一个TextOutput对象和一个FileOutput对象，为了使这些对象同时能够实现这两个接口，所以需要创建一个适配器。如下图所示：


现在只需要为每个对象创建一个对应的适配器即可，如下图所示：

```java
public class ScreenAdapter implements TextOutput {

    private final Adaptee adaptee;

    public ScreenAdapter(Adaptee adaptee) {
        this.adaptee = adaptee;
    }

    @Override
    public void display(String text) {
        System.out.println("[" + adaptee.getName() + "] " + text);
    }

    // 其他方法
}

public class FileAdapter implements FileOutput {

    private final Adaptee adaptee;

    public FileAdapter(Adaptee adaptee) {
        this.adaptee = adaptee;
    }

    @Override
    public void saveToFile(String text) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter("/path/to/file"))) {
            writer.write("[" + adaptee.getName() + "] " + text);
        }
    }

    // 其他方法
}
```

这里的ScreenAdapter和FileAdapter都是适配器，它们都继承了TextOutput和FileOutput接口。在构造函数里传入了一个Adaptee对象，即真正的实现了TextOutput和FileOutput接口的对象。在display()方法中，屏幕显示信息的格式是"[*AdapteeName*] *text*"。在saveToFile()方法中，写入的文件的内容是"[*AdapteeName*] *text*"。

最后，创建Adaptee的实体对象，再创建ScreenAdapter和FileAdapter对象，就可以让对象通过适配器完成不同接口间的通信。如下例所示：

```java
public static void main(String[] args) {
    Adaptee adaptee = new RealObject();
    TextOutput screenAdapter = new ScreenAdapter(adaptee);
    FileOutput fileAdapter = new FileAdapter(adaptee);
    
    String message = "Hello world!";
    screenAdapter.display(message);
    fileAdapter.saveToFile(message);
}
```

这样就可以达到目的。