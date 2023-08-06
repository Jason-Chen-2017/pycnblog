
作者：禅与计算机程序设计艺术                    

# 1.简介
         

         在软件开发中，架构模式是一个至关重要的因素，它提供了一些指导、规范、模板，能够帮助软件工程师更好地设计、组织和管理软件系统的结构、功能和关系。它的作用是帮助项目团队构建一致性的软件系统，提升可维护性和扩展性，同时降低成本，改善产品质量。软件架构设计模式集合了众多经验丰富的软件设计人员和架构师通过几十年来不断实践总结出的最佳实践。本文是作者在JavaZone 2019上为期两天的演讲《10.Software Architecture Patterns - A pragmatic guide》的文字稿，包含软件架构设计模式的一般介绍和二十种典型模式的阐述。本文档主要目标读者群体为具有相关软件开发经验，对软件架构设计有浓厚兴趣并希望进一步了解这些设计模式的软件架构师、开发人员或项目管理者。
         
         作者简介：黄洪宇，目前就职于优步科技，之前从事嵌入式领域的软件开发工作，精通面向对象编程、设计模式、敏捷开发方法论等。热爱技术，喜欢分享自己的学习心得和经验，因此写下此文档作为分享资料。本文分为前言、正文及后记三个章节，以简要阐述软件架构设计模式的概念，并详细介绍二十种典型模式的演变过程及特点。希望能够帮助读者理解软件架构设计模式背后的设计原则、架构风格、实践原则和工程挑战，为其提供更好的架构决策指引，并激发对软件架构设计有所追求的创新意识。
         
         # 2.术语说明
         **软件架构**：软件系统的静态设计蓝图，包括业务需求、功能特性、技术实现方案以及部署运行环境。它定义了系统的结构、功能、流程和约束条件。
         
         **软件组件**：一个可以独立运行的最小化软件单元，其内部封装数据和处理逻辑，提供一个接口向外界提供服务。软件组件之间通过接口进行通信。
         
         **依赖关系**：在设计、编码、测试或运行时，不同软件模块之间存在的联系。如组件间的调用、数据传递、耦合度、关联度等。
         
         **耦合度**：模块之间的相互依存程度和强弱。越低耦合度的软件模块，其修改范围和影响范围越小；耦合度越高，软件模块之间的耦合越紧密，软件的修改可能导致无法正常运行。耦合度可以通过划分层次结构和设计模式等方式控制。
         
         **抽象层级**：软件系统中各个子系统或模块组成的一种分层结构，用于表示软件系统的结构特征。它通常由应用层、业务层、表示层、持久层、数据访问层等构成。
         
         **面向对象设计**：通过类、对象、继承、组合和多态等概念进行计算机程序的设计。其目的是创建可重用的代码，避免代码重复，提高代码的可维护性、可扩展性和可复用性。
         
         **设计模式**：一套被反复使用、多数人知晓的、经过分类编制的、代码设计经验的总结。其具备普遍适用的性和指导意义，可提高代码的可读性、可扩展性和可靠性。
         
         # 3.软件架构设计模式
         
         ## 3.1.概述
         
         ### 什么是软件架构？
         
         “软件架构”这一词汇的定义日渐流行，甚至已经超越了传统的“系统架构”。软件架构不是简单地划分出一个系统的功能模块，而是在设计阶段就已考虑到各种运行时、部署时的约束条件，以及未来的扩展性、可伸缩性、性能优化等需求，着眼于如何将系统的不同功能模块或者子系统，按照某种顺序，整合到一起，组装成为一个完整且有效的软件系统。
         
         当然，软件架构设计模式也是一系列的模式、原则、方法、工具，它提供了一些指导、规范、模板，能够帮助软件工程师更好地设计、组织和管理软件系统的结构、功能和关系。通过系统化地运用架构设计模式，能够在一定程度上增强软件系统的健壮性、可维护性、可扩展性、可复用性、可移植性、安全性、可测试性以及可用性等方面的能力。
         
         在“软件架构设计模式”一词的产生过程中，软件架构确立的价值观也逐渐明朗，认为软件架构的重要性已经超出了仅仅关注功能和模块的角度，它既需要考虑到系统的静态设计蓝图，也需要关注到系统的动态运行状态，以及系统未来的扩展性、可伸缩性、性能优化、可靠性和可用性等挑战。随着时间的推移，“软件架构设计模式”也在不断更新迭代，并被广泛应用在各个领域。
         
         通过阅读《软件架构设计模式》，可以发现软件架构设计模式对于软件设计师、开发人员和项目管理者来说，都是非常重要的知识基础。它涵盖了软件设计中的不同阶段，如需求分析、系统设计、结构设计、行为设计、实现设计、测试设计、维护设计等多个阶段，以及一些原则、模式、方法、工具等。通过阅读和理解这些设计模式，可以提升自己对软件架构的认识水平，并且更好的规划和管理软件系统的架构设计和开发。
         
         ### 为什么要谈论软件架构设计模式？
         
         在软件架构设计模式的引入之后，软件开发人员开始更多地关注软件架构的整体设计。架构师、项目经理需要根据需求和实际情况，制定一整套架构设计方法，包括架构设计原则、架构风格、实践原则和工程挑战等。然后，通过实践，将这些原则、方法、工具应用到软件的设计之中。
         
         这些设计模式对于软件架构设计者来说，无疑是非常宝贵的资源。因为软件架构设计模式是经过一系列工程实践总结出的，它提供了一些最佳实践，能够帮助软件架构设计者设计出具有良好设计特征的软件系统。另外，软件架构设计模式还提供了一个全新的视角，能够清楚地看到软件架构设计的全貌。而对于那些缺乏架构设计经验的初创公司来说，如果不能站在巨人的肩膀上看世界，很难成功设计出具有竞争力的软件系统。
         
         本书的内容就是为了帮助读者了解软件架构设计模式的概念、原则、模式、方法、工具、实践原则和工程挑战，帮助软件架构设计者更好地掌握软件架构设计的方法论和技术，并顺利完成软件架构设计和开发。
         
        ## 3.2.模式介绍
        （1）责任链模式（Chain of Responsibility pattern）—— 使多个对象都有机会处理请求，从而避免请求的发送者和接受者之间的耦合关系。一个请求可能会沿着一条链一直传递，直到有一个对象处理它为止。
        
        （2）命令模式（Command pattern）—— 将一个请求封装为一个对象，使你可对请求排队或记录请求日志，以及支持可撤销的操作。
        
        （3）适配器模式（Adapter pattern）—— 将一个类的接口转换成客户希望的另一个接口。它使得原本由于接口不兼容而不能一起工作的那些类可以一起工作。
        
        （4）发布-订阅模式（Publish/subscribe pattern）—— 一对多的依赖关系，当发布者把消息发布出去之后，所有订阅了该频道的客户端都会接收到通知。
        
        （5）代理模式（Proxy pattern）—— 为其他对象提供一种代理以控制对这个对象的访问。
        
        （6）组合模式（Composite pattern）—— 将对象组合成树形结构以表示“部分-整体”的层次结构。它用于表示对象的结构层次。
        
        （7）桥接模式（Bridge pattern）—— 分离抽象与实现部分，使它们可以独立变化。
        
        （8）装饰器模式（Decorator pattern）—— 动态地给对象添加额外的职责。
        
        （9）享元模式（Flyweight pattern）—— 用共享的对象来减少内存占用，相同对象只创建一个对象。
        
        （10）模板方法模式（Template method pattern）—— 定义一个操作中算法的骨架，而将一些步骤延迟到子类中。
        
        （11）状态模式（State pattern）—— 允许对象在不同的状态下表现出不同的行为，根据状态改变对象的行为。
        
        （12）策略模式（Strategy pattern）—— 定义了一系列算法，并将每个算法封装起来，让他们可以相互替换，即插即用。
        
        （13）观察者模式（Observer pattern）—— 一对多的依赖关系，当对象改变状态时，所有的依赖对象都会收到通知。
        
        （14）迭代器模式（Iterator pattern）—— 提供一种方法顺序访问一个聚合对象中各个元素，而又无须暴露该对象的内部细节。
        
        （15）解析器模式（Parser pattern）—— 将一个复杂表达式解析成一个易于理解的表示形式，并返回该表示形式的值。
        
        （16）外观模式（Facade pattern）—— 为系统中的多个复杂模块或子系统提供一个统一的接口。
        
        （17）代理模式（Protection Proxy pattern）—— 保护主题的关键代码，并防止它被未授权的访问。
        
        （18）单例模式（Singleton pattern）—— 只生成一个类的唯一实例，该类负责保存全局唯一的数据。
        
        （19）备忘录模式（Memento pattern）—— 在不破坏封装性的前提下，保存和恢复对象之前的状态。
        
        （20）职责链模式（Responsibility chain pattern）—— 请求发送者与一系列的处理对象形成一条链，并沿着这条链传递请求，直到有一个对象处理它为止。

        # 4.实例介绍

       ### 1.责任链模式

        责任链模式的定义为：使多个对象都有机会处理请求，从而避免请求的发送者和接受者之间的耦合关系。一个请求可能会沿着一条链一直传递，直到有一个对象处理它为止。这种模式属于行为型模式。

        责任链模式的结构包括如下几个角色：

        抽象处理者(Handler)：定义了请求处理的方法，并且持有对下一个处理者的引用。

        具体处理者(Concrete Handler)：实现了抽象处理者，负责处理请求，并可以选择性的将请求转交给下一个处理者。

        Client：客户端向处理者发送请求。Client不直接与处理者发生交互，而是通过向第一个具体处理者对象提交请求。

        使用责任链模式时需要注意以下几点：

        1. 请求可以沿着一条链一直传递，直到某个处理者对象能够处理请求为止。
        2. 每个处理者都可选择是否处理请求或将请求传给下一个处理者。
        3. 如果最终没有任何处理者对象能够处理请求，那么该请求将会被丢弃。

        下面给出一个责任链模式的实例。

        情况：假设有一个项目，由多个部门参与。每个部门都有责任人来负责各自的工作。部门之间存在一个领导和成员关系，领导负责管理成员，成员只能自己做事情，不能交由别人处理。而且，项目还存在一个最高领导来统一调度，不能让他的人手单独去做事情。

        根据上述情况，可以使用责任链模式来解决这个问题。首先需要定义一个接口来表示成员处理请求的行为，其中包含处理请求的方法，还包含对下一个处理者的引用。

        ```java
        public interface Member {
            void handleRequest(String request);
            void setNextMember(Member next); // 设置下一个处理者
        }
        ```

        在具体处理者中实现以上接口。

        ```java
        public class Leader implements Member{

            private String name;
            private Member nextLeader;
            private List<Member> members = new ArrayList<>();

            public Leader(String name){
                this.name = name;
            }

            @Override
            public void handleRequest(String request) {

                if (request.equalsIgnoreCase("项目审批")) {
                    System.out.println(this.getName() + "正在审核：" + request);
                    for (Member member : members) {
                        member.handleRequest(request);
                    }
                } else {
                    if (nextLeader!= null) {
                        nextLeader.handleRequest(request);
                    } else {
                        System.out.println("领导权已落空！");
                    }
                }
            }

            @Override
            public void setNextMember(Member next) {
                this.nextLeader = next;
            }

            public String getName() {
                return name;
            }

            public void addMember(Member member){
                members.add(member);
            }

        }

        public class Manager implements Member{

            private String name;
            private Member nextManager;

            public Manager(String name){
                this.name = name;
            }

            @Override
            public void handleRequest(String request) {

                if (request.equalsIgnoreCase("项目启动")) {
                    System.out.println(this.getName() + "正在审核：" + request);
                    nextManager.handleRequest(request);
                } else {
                    if (nextManager!= null) {
                        nextManager.handleRequest(request);
                    } else {
                        System.out.println("下一个主管不存在！");
                    }
                }
            }

            @Override
            public void setNextMember(Member next) {
                this.nextManager = next;
            }

            public String getName(){
                return name;
            }

        }

        public class NormalMember implements Member{

            private String name;

            public NormalMember(String name){
                this.name = name;
            }

            @Override
            public void handleRequest(String request) {
                System.out.println(this.getName() + "正在审核：" + request);
            }

            @Override
            public void setNextMember(Member next) {}

            public String getName(){
                return name;
            }

        }
        ```

        创建两个成员对象：一名主管和三名普通成员。

        ```java
        public static void main(String[] args) {

            Leader leader = new Leader("张三");

            Manager manager1 = new Manager("李四");
            Manager manager2 = new Manager("王五");

            NormalMember normalMember1 = new NormalMember("赵六");
            NormalMember normalMember2 = new NormalMember("孙七");
            NormalMember normalMember3 = new NormalMember("周八");

            leader.setNextMember(manager1);
            manager1.setNextMember(manager2);
            manager2.setNextMember(normalMember1);
            normalMember1.setNextMember(normalMember2);
            normalMember2.setNextMember(normalMember3);

            leader.addMember(manager2);
            leader.addMember(normalMember1);


            leader.handleRequest("项目启动");

            try {
                Thread.sleep(3000);
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            leader.handleRequest("项目审批");


        }
        ```

        执行结果：

        ```java
        李四正在审核：项目启动
        王五正在审核：项目启动
        赵六正在审核：项目启动
        孙七正在审核：项目启动
        周八正在审核：项目启动
        领导权已落空！
        张三正在审核：项目审批
        李四正在审核：项目审批
        王五正在审核：项目审批
        赵六正在审核：项目审批
        孙七正在审核：项目审批
        周八正在审核：项目审批
        ```

        从结果可以看出，在没有主管的情况下，普通成员会被单独审批。

        ### 2.命令模式

        命令模式的定义为：将一个请求封装为一个对象，使你可对请求排队或记录请求日志，以及支持可撤销的操作。命令模式属于行为型模式。

        命令模式的结构包括如下几个角色：

        命令(Command)：声明一个接口，用来执行一个动作。

        ConcreteCommand(具体命令)：实现命令接口。

        Invoker(调用者)：要求命令对象执行请求。

        Receiver(接收者)：执行命令的对象。

        Client：创建一个具体命令对象并设置它的接收者。Invoker对象调用命令对象相应的execute()方法，从而实现命令的调用。

        使用命令模式时需要注意以下几点：

        1. 将一个请求封装成一个对象。
        2. 可在队列中排队、记录请求日志，并支持可撤销的操作。
        3. 可以参数化其他对象，来自定义一个命令的执行过程和triggeredBy()方法。

        下面给出一个命令模式的实例。

        情况：假设有一个电视机遥控器，用户可以通过键盘上的数字按钮来选择预置的频道，也可以通过按钮选取自己想看的节目。由于有限的内存空间，电视机只能容纳固定数量的频道。如果用户尝试选择一个超出最大容量的频道，就会得到一个错误信息。

        此时，可以使用命令模式来解决这个问题。首先，定义一个接口来描述电视机遥控器接收到的命令：

        ```java
        public interface ICommand {
            void execute();
            boolean triggeredBy(int code);
        }
        ```

        然后，定义一个频道切换命令和对应的接收者ChannelSwitcher。ChannelSwitcher有一个数组来存储频道信息，用户选择的频道编号可以作为索引，从而切换到指定的频道。

        ```java
        public class ChannelSwitchCommand implements ICommand {

            private final int channelCode;
            private final ChannelSwitcher switcher;

            public ChannelSwitchCommand(int code, ChannelSwitcher switcher) {
                this.channelCode = code;
                this.switcher = switcher;
            }

            @Override
            public void execute() {
                switcher.changeTo(channelCode);
            }

            @Override
            public boolean triggeredBy(int code) {
                return code == channelCode;
            }

        }

        public class ChannelSwitcher {
            private int currentChannel;
            private final int maxChannels;
            private Channel[] channels;

            public ChannelSwitcher(int maxChannels) {
                this.maxChannels = maxChannels;
                channels = new Channel[maxChannels];
                for (int i = 0; i < maxChannels; i++) {
                    channels[i] = new Channel("Channel " + i);
                }
            }

            public synchronized void changeTo(int code) {
                if (code >= 0 && code < maxChannels) {
                    currentChannel = code;
                } else {
                    throw new IllegalArgumentException("Invalid channel code: " + code);
                }
            }

            public Channel getCurrentChannel() {
                return channels[currentChannel];
            }

            public static class Channel {
                private final String name;

                public Channel(String name) {
                    this.name = name;
                }

                @Override
                public String toString() {
                    return name;
                }
            }

        }
        ```

        创建一个遥控器并添加三个频道：HDMI、Component 和 VGA。每一个频道对应一个频道切换命令。用户输入频道编号后，会创建对应的频道切换命令并执行。

        ```java
        public class RemoteController {

            private final Map<Integer, ICommand> commands = new HashMap<>();
            private final ChannelSwitcher switcher;

            public RemoteController(ChannelSwitcher switcher) {
                this.switcher = switcher;
                initCommands();
            }

            private void initCommands() {
                commands.put(1, new ChannelSwitchCommand(1, switcher));
                commands.put(2, new ChannelSwitchCommand(2, switcher));
                commands.put(3, new ChannelSwitchCommand(3, switcher));
            }

            public void onButtonPressed(int keyCode) {
                ICommand command = commands.get(keyCode);
                if (command!= null && command.triggeredBy(keyCode)) {
                    command.execute();
                } else {
                    System.err.println("Invalid key pressed: " + keyCode);
                }
            }

        }

        public class Main {
            public static void main(String[] args) {
                ChannelSwitcher switcher = new ChannelSwitcher(3);
                RemoteController remoteControl = new RemoteController(switcher);
                remoteControl.onButtonPressed(1);   // HDMI
                System.out.println(switcher.getCurrentChannel());     // Channel 0

                remoteControl.onButtonPressed(2);   // Component
                System.out.println(switcher.getCurrentChannel());     // Channel 1

                remoteControl.onButtonPressed(3);   // VGA
                System.out.println(switcher.getCurrentChannel());     // Channel 2

                remoteControl.onButtonPressed(4);   // Invalid key pressed: 4

                try {
                    Thread.sleep(3000);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }

            }
        }
        ```

        执行结果：

        ```java
        Channel 0
        Channel 1
        Channel 2
        Invalid key pressed: 4
        ```

        从结果可以看出，可以通过键盘上的数字按钮来切换频道。如果用户选择了一个超过最大容量的频道，会得到一个错误信息。

        ### 3.适配器模式

        适配器模式的定义为：将一个类的接口转换成客户希望的另一个接口。它使得原本由于接口不兼容而不能一起工作的那些类可以一起工作。适配器模式属于 Structural 模式。

        适配器模式的结构包括如下几个角色：

        Target（目标接口）：定义客户所期待的接口。

        Adaptee（被适配者）：当前系统存在的接口。

        Adapter（适配器）：它是一个类，它含有Adaptee对象的实例，它的接口与Target一致，是Target接口的一个适配器。在客户端代码中，通过适配器可以调用Adaptee里的方法。

        Object（对象）：该对象含有目标接口所需的所有方法。

        使用适配器模式时需要注意以下几点：

        1. 适配器模式可以把一个类的接口转换成客户希望的另一个接口，适应不同类或对象所提供的接口。
        2. 如果已有的类与其他类兼容，但是接口不符的时候就可以使用适配器。
        3. 如果想要建立一个可以在不同线程之间通讯的应用程序，适配器模式非常有用。
        4. 适配器可以增加类的透明度和复用性。

        下面给出一个适配器模式的实例。

        情况：有一个视频播放器需要播放MKV文件。但现在有一个第三方的软件需要播放RMVB文件。现在需要编写一个适配器来完成这项工作。

        ```java
        public interface MediaPlayer {
            void play(String fileName);
        }

        public class MKVMediaPlayer implements MediaPlayer {

            @Override
            public void play(String fileName) {
                System.out.println("Playing file.mkv :" + fileName);
            }

        }

        public class RMVBtoMKVConverter {

            private final MediaPlayer player;

            public RMVBtoMKVConverter(MediaPlayer player) {
                this.player = player;
            }

            public void convert(String rmvbFile) {
                player.play(".mkv" + rmvbFile.substring(4));
            }

        }

        public class Main {
            public static void main(String[] args) {
                MediaPlayer mkvPlayer = new MKVMediaPlayer();
                RMVBtoMKVConverter converter = new RMVBtoMKVConverter(mkvPlayer);
                converter.convert("/path/to/file.rmvb");    // Playing file.mkv:/path/to/file.rmvb
            }
        }
        ```

        这里，定义了MediaPLayer接口，MKVMediaPlayer是它的一个实现。RMVBtoMKVConverter是适配器，它的构造函数接收一个MediaPlayer对象。通过它的方法，可以将一个RMVB文件转换为MKV文件并播放。

        ### 4.发布-订阅模式

        发布-订阅模式的定义为：定义一组触发事件的对象，并且不指定它们的实时调度顺序。事件的发送者和接收者不知道对方的存在。发送者只是向所有感兴趣的接收者发布消息，接收者则按照自己的时间安排订阅。它属于行为型模式。

        发布-订阅模式的结构包括如下几个角色：

        Subject（主题）：定义了一个接口，用于主题和观察者之间的通讯。

        ConcreteSubject（具体主题）：实现了Subject接口，代表主题对象。维护了一个观察者列表，并提供了注册方法、注销方法、通知方法等。

        Observer（观察者）：定义了一个接口，用于观察者和主题之间的通讯。

        ConcreteObserver（具体观察者）：实现了Observer接口，代表观察者对象。

        Client（客户端）：使用主题对象，向观察者对象发布消息。

        使用发布-订阅模式时需要注意以下几点：

        1. 观察者模式可以让主题和观察者之间松耦合，让主题和观察者可以 independently变化而不影响其他类，同时，它还可以简化主题对象和观察者对象之间的通信。
        2. 主题对象向所有观察者对象广播通知，观察者对象负责订阅和取消订阅。
        3. 该模式可以增加主题对象和观察者对象之间的弹性。

        下面给出一个发布-订阅模式的实例。

        情况：有一个网站有许多文章，可以按月查看，每个月都会有很多阅读量。管理员需要统计各月的阅读量，所以需要建立一个后台统计系统。

        因此，可以采用发布-订阅模式来实现。首先，定义一个接口来表示阅读量的事件：

        ```java
        import java.util.Date;

        public interface IEvent {
            Date getTimestamp();
            String getContent();
        }
        ```

        然后，定义一个文章对象：

        ```java
        import java.util.ArrayList;
        import java.util.List;

        public class Article {
            private String title;
            private List<IEvent> events = new ArrayList<>();

            public Article(String title) {
                this.title = title;
            }

            public String getTitle() {
                return title;
            }

            public void subscribe(IEventSubscriber subscriber) {
                events.add(subscriber);
            }

            public void unsubscribe(IEventSubscriber subscriber) {
                events.remove(subscriber);
            }

            protected void publish(String content) {
                Event event = new Event(new Date(), content);
                for (IEventSubscriber subscriber : events) {
                    subscriber.receive(event);
                }
            }

        }

        public interface IEventSubscriber {
            void receive(IEvent event);
        }

        public class MonthlyStatsProcessor implements IEventSubscriber {

            private String monthName;
            private int readCount;

            @Override
            public void receive(IEvent event) {
                if ("article_read".equals(event.getContent())) {
                    readCount++;
                }
            }

            public MonthlyStatsProcessor(String monthName) {
                this.monthName = monthName;
            }

            public String getMonthName() {
                return monthName;
            }

            public int getReadCount() {
                return readCount;
            }

        }

        public class StatsGenerator {

            private final Article article;

            public StatsGenerator(Article article) {
                this.article = article;
            }

            public void generateMonthlyStats(String monthName) {
                MonthlyStatsProcessor processor = new MonthlyStatsProcessor(monthName);
                article.subscribe(processor);
                try {
                    // Simulate reading the articles...
                    Thread.sleep(1000 * random.nextInt(5));
                } catch (InterruptedException e) {
                    e.printStackTrace();
                } finally {
                    article.unsubscribe(processor);
                }
                System.out.printf("%s stats: %d reads
", monthName, processor.getReadCount());
            }

            private Random random = new Random();

        }

        public class Main {
            public static void main(String[] args) {
                Article article = new Article("Java Programming");
                StatsGenerator generator = new StatsGenerator(article);
                generator.generateMonthlyStats("January");
                generator.generateMonthlyStats("February");
                generator.generateMonthlyStats("March");
            }
        }
        ```

        这里，定义了IEvent接口，表示阅读量的事件。Article类有一个events列表，里面保存了所有注册的观察者对象。publish方法向所有观察者对象广播阅读量的事件，MonthlyStatsProcessor实现了IEventSubscriber接口，用于接收事件并统计阅读量。StatsGenerator用于生成MonthlyStatsProcessor对象，并订阅到Article对象上。当Article对象有新事件发布时，MonthlyStatsProcessor对象会接收到通知，并统计阅读量。最后，生成的统计信息会打印到控制台上。

        执行结果：

        ```java
        January stats: 339 reads
        February stats: 256 reads
        March stats: 150 reads
        ```