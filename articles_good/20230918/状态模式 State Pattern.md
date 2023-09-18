
作者：禅与计算机程序设计艺术                    

# 1.简介
  

在面向对象编程中，状态模式是一种行为设计模式，它可以帮助我们管理对象的状态变化。它把对象的状态封装到一个类中，并通过该类的行为来改变对象状态。这种方式使得对象内在的复杂状态变化变得可控、方便追踪，也可实现不同状态之间的转移。状态模式应用很广泛，比如，计算机系统中的进程调度、电梯控制、交通灯的开关等都采用了状态模式。本文将先对状态模式进行简单介绍，然后再结合实际案例详细介绍它的特点及应用场景。
# 2.定义
状态模式（State Pattern）属于行为型模式，它允许一个对象在其内部状态发生改变时改变它的行为，对象看起来好像修改了它的类。其主要优点如下：

1. 封装了转换逻辑，使得客户端可以直接向同一个接口发送请求，而无需知道对象现在所处的具体状态；
2. 可将状态转换逻辑从客户端代码中分离出来，简化了客户端的复杂性；
3. 可以简化状态机的实现，在没有具体状态的情况下也可以处理逻辑；
4. 可以让代码易读易懂，易于维护；
5. 提供了给对象状态进行“快照”功能，能够提供比继承更有用的扩展点；
6. 可以简化并行开发和测试。

总之，状态模式将不同的行为和逻辑封装到不同的状态类里，每一个状态都是一个子类，当对象的状态发生改变时，会自动切换至对应的状态类，实现了较好的解耦。所以，状态模式是一种非常实用的设计模式，在很多地方都有它的身影。
# 3.问题分析
首先，假设有一个手机 app 的产品经理提出了一个新的需求：希望手机 app 可以根据用户的网络连接状态动态调整通知栏显示的内容，即，当用户打开 WiFi 时显示 Wi-Fi 连接成功的信息，当用户关闭 WiFi 时显示正常网络信息。这个需求中涉及到了三个角色：用户、手机 app 和通知栏。接着，产品经理就进入讨论环节，他列出了以下四种方案：

1. 使用 if...else 分支语句判断当前网络状态，然后设置不同消息文本及提示音，并更新通知栏显示内容。这种方式不够灵活，当要增加新的网络状态时，需要改动代码；并且，通知栏显示消息的方式还受限于系统提供的 API 。

2. 在 app 中添加多套不同的通知栏样式，分别对应不同的网络状态，并在用户登陆时选择相应的通知栏样式。这种方式比较复杂，需要对每个通知栏样式的代码进行完整的编写，而且由于系统限制，可能会造成代码冗余或性能影响。

3. 创建一个抽象状态基类 AbstractState ，它包含两个子类：WifiConnectedState 和 NormalNetworkState，并为他们提供相同的接口方法。同时创建两个实现类 WifiConnectedStateImpl 和 NormalNetworkStateImpl 用于实现它们的业务逻辑。创建一个 Context 对象，负责管理手机 app 当前的状态，Context 通过调用状态类的相关接口方法来切换状态，并调用手机硬件完成通知栏更新操作。这样做虽然解决了灵活性问题，但却违反了“单一职责原则”。

4. 使用状态模式设计模式。定义一个抽象状态基类 AbstractState ，其中包括两个子类：WifiConnectedState 和 NormalNetworkState，它们共同实现相同的方法，并各自实现自己的业务逻辑。定义另一个接口接口 StateMachine ，它包含两个方法：getState() 返回当前的状态；changeState(State state) 方法用来切换状态。创建 Context 对象，它持有一个 StateMachine 对象，并通过 getState() 获取当前状态，然后调用 changeState(State state) 方法切换状态，最后调用手机硬件完成通知栏更新操作。这是最合适的方案，符合“开闭原则”，并且实现了代码的解耦。

接下来，我们一步步分析第四种方案，来了解状态模式的具体运作机制。
# 4.状态模式结构
状态模式的结构如下图所示:


1. State（抽象状态类）：定义了所有具体状态类的共性。
2. ConcreteState（具体状态类）：实现了抽象状态类的接口，具体实现各种状态下的逻辑。
3. Context（上下文对象）：定义了客户请求时的动作，维护一个ConcreteState类型的对象作为当前状态。
4. Client（环境对象）：最终触发状态转换的对象。
# 5. 状态模式角色详解
## 5.1 State（抽象状态类）
抽象状态类 State 是所有具体状态类的父类，它定义了一个接口，用以表示对象的某种状态。它只有一个方法，即改变状态的方法 changeState() ，它会将对象转变为下一个状态。

## 5.2 ConcreteState（具体状态类）
具体状态类 ConcreteState 是抽象状态类的子类，它实现了 State 类的 changeState() 方法，用以完成状态的转换。具体状态类中保存了与此状态相关的属性和行为，并提供相应的接口方法来进行状态切换。

## 5.3 Context（上下文对象）
上下文对象 Context 由三部分组成：一个 ConcreteState 对象，一个 State 接口对象，和一个环境对象。它负责存储和管理当前状态，并按照要求调用状态类的 changeState() 方法，切换到下一个状态。

## 5.4 Client（环境对象）
环境对象 Client 是状态模式的消费者，负责触发状态转换。它获取当前状态，然后调用相应状态类的 changeState() 方法，切换到下一个状态。
# 6. 案例分析
在讲解具体实现之前，先来看一个实际案例。

假设一个手机 app 需要根据用户当前网络连接情况动态调整通知栏显示内容，并且通知栏显示内容要具有良好的可用性、易读性、美观性，同时通知栏显示内容也要考虑国际化、本地化等因素。因此，我们可以使用状态模式来实现这一功能。

## 6.1 用户场景
用户安装了 app，登录 app。此时，app 会根据用户网络连接情况，决定是否显示 Wi-Fi 连接成功的提示信息，或者显示正常网络信息。如果是 Wi-Fi 连接成功，通知栏显示“Wi-Fi 已连接”字样，并播放连接成功提示音；如果是正常网络，通知栏显示“正常网络”字样，并播放正常提示音。用户可以随时点击通知栏上的文字，查看详情。

## 6.2 抽象状态类 AbstractState
首先，我们需要定义一个抽象状态类 AbstractState，它只包含一个方法 changeState(), 用以表示对象的某种状态。由于不同网络状态的逻辑不同，因此需要具体的子类来实现各个状态的逻辑。AbstractState 的子类有：

- NormalNetworkState：该状态表示设备处于正常网络状态。
- WifiConnectedState：该状态表示设备处于 Wi-Fi 连接状态。

### 6.2.1 NormalNetworkState
NormalNetworkState 代表设备处于正常网络状态。它包含两个方法：

1. setMessageTextAndSound()：该方法设置通知栏显示的文字内容和提示音。
2. showNotification()：该方法更新通知栏显示的内容。

```java
public abstract class AbstractState implements State {

    @Override
    public void changeState(Context context) {
        // TODO 判断网络是否连接成功
        if (isConnected()) {
            ((WifiConnectedState)context).setMessageTextAndSound();
            ((WifiConnectedState)context).showNotification();
        } else {
            ((NormalNetworkState)context).setMessageTextAndSound();
            ((NormalNetworkState)context).showNotification();
        }
    }
    
    protected boolean isConnected() {
        return true;
    }
    
}

class NormalNetworkState extends AbstractState {

    private static final String MESSAGE_TEXT = "当前网络";
    private static final int SOUND_ID = R.raw.normal_network;

    @Override
    public void showNotification() {
        NotificationManager manager =
                (NotificationManager) getSystemService(NOTIFICATION_SERVICE);

        Intent intent = new Intent(this, MainActivity.class);
        PendingIntent pendingIntent = PendingIntent.getActivity(this, 0, intent, 0);

        Notification notification = new NotificationCompat.Builder(this)
               .setContentTitle("App Name")
               .setContentText(MESSAGE_TEXT)
               .setSmallIcon(R.mipmap.ic_launcher)
               .setContentIntent(pendingIntent)
               .build();

        manager.notify(0, notification);
    }

    @Override
    public void setMessageTextAndSound() {
        TTSUtils.playTTS(NORMAL_NETWORK, this);
    }
}

```

上面的代码定义了 NormalNetworkState，它包含两个方法：

1. `showMessage()` 方法，该方法用来更新通知栏显示的内容。
2. `setMessageTextAndSound()` 方法，该方法用来设置通知栏显示的文字内容和提示音。

#### 6.2.1.1 showNotification() 方法

```java
@Override
public void showNotification() {
    NotificationManager manager =
            (NotificationManager) getSystemService(NOTIFICATION_SERVICE);

    Intent intent = new Intent(this, MainActivity.class);
    PendingIntent pendingIntent = PendingIntent.getActivity(this, 0, intent, 0);

    Notification notification = new NotificationCompat.Builder(this)
           .setContentTitle("App Name")
           .setContentText(MESSAGE_TEXT)
           .setSmallIcon(R.mipmap.ic_launcher)
           .setContentIntent(pendingIntent)
           .build();

    manager.notify(0, notification);
}
```

这里的 showNotification() 方法创建了一个通知栏的通知，并将其展示在通知栏上。需要注意的是，这个通知栏通知的唯一标识符设置为 0。因为每个手机 app 中的通知栏只能显示一个通知，所以当收到新的通知时，旧的通知就会被替换掉。

#### 6.2.1.2 setMessageTextAndSound() 方法

```java
@Override
public void setMessageTextAndSound() {
    TTSUtils.playTTS(NORMAL_NETWORK, this);
}
```

这里的 setMessageTextAndSound() 方法用来设置通知栏显示的文字内容和提示音。具体地，调用了 TTSUtils 的 playTTS() 方法，来播放提示音。我们暂时不做具体实现，只需记住 TTSUtils 的 playTTS() 方法可以播放指定的提示音文件。

### 6.2.2 WifiConnectedState
WifiConnectedState 表示设备处于 Wi-Fi 连接状态。它包含两个方法：

1. setMessageTextAndSound()：该方法设置通知栏显示的文字内容和提示音。
2. showNotification()：该方法更新通知栏显示的内容。

```java
class WifiConnectedState extends AbstractState {

    private static final String MESSAGE_TEXT = "Wi-Fi 已连接";
    private static final int SOUND_ID = R.raw.wifi_connected;

    @Override
    public void showNotification() {
        NotificationManager manager =
                (NotificationManager) getSystemService(NOTIFICATION_SERVICE);

        Intent intent = new Intent(this, MainActivity.class);
        PendingIntent pendingIntent = PendingIntent.getActivity(this, 0, intent, 0);

        Notification notification = new NotificationCompat.Builder(this)
               .setContentTitle("App Name")
               .setContentText(MESSAGE_TEXT)
               .setSmallIcon(R.mipmap.ic_launcher)
               .setContentIntent(pendingIntent)
               .build();

        manager.notify(0, notification);
    }

    @Override
    public void setMessageTextAndSound() {
        TTSUtils.playTTS(WIFI_CONNECTED, this);
    }
}
```

上面的代码定义了 WifiConnectedState，它包含两个方法：

1. `showMessage()` 方法，该方法用来更新通知栏显示的内容。
2. `setMessageTextAndSound()` 方法，该方法用来设置通知栏显示的文字内容和提示音。

#### 6.2.2.1 showNotification() 方法

```java
@Override
public void showNotification() {
    NotificationManager manager =
            (NotificationManager) getSystemService(NOTIFICATION_SERVICE);

    Intent intent = new Intent(this, MainActivity.class);
    PendingIntent pendingIntent = PendingIntent.getActivity(this, 0, intent, 0);

    Notification notification = new NotificationCompat.Builder(this)
           .setContentTitle("App Name")
           .setContentText(MESSAGE_TEXT)
           .setSmallIcon(R.mipmap.ic_launcher)
           .setContentIntent(pendingIntent)
           .build();

    manager.notify(0, notification);
}
```

这里的 showNotification() 方法创建了一个通知栏的通知，并将其展示在通知栏上。需要注意的是，这个通知栏通知的唯一标识符设置为 0。因为每个手机 app 中的通知栏只能显示一个通知，所以当收到新的通知时，旧的通知就会被替换掉。

#### 6.2.2.2 setMessageTextAndSound() 方法

```java
@Override
public void setMessageTextAndSound() {
    TTSUtils.playTTS(WIFI_CONNECTED, this);
}
```

这里的 setMessageTextAndSound() 方法用来设置通知栏显示的文字内容和提示音。具体地，调用了 TTSUtils 的 playTTS() 方法，来播放提示音。我们暂时不做具体实现，只需记住 TTSUtils 的 playTTS() 方法可以播放指定的提示音文件。


## 6.3 上下文对象 Context
上下文对象 Context 由三部分组成：一个 ConcreteState 对象，一个 State 接口对象，和一个环境对象。它负责存储和管理当前状态，并按照要求调用状态类的 changeState() 方法，切换到下一个状态。

```java
public class NetworkConnectionContext extends ContextWrapper {

    private State currentState;

    public NetworkConnectionContext(Context base) {
        super(base);
        setCurrentState(new NormalNetworkState());
    }

    public void connectToWifi() {
        setCurrentState(new WifiConnectedState());
    }

    public void disconnectFromWifi() {
        setCurrentState(new NormalNetworkState());
    }

    public State getCurrentState() {
        return currentState;
    }

    public void setCurrentState(State newState) {
        if (currentState!= null &&!currentState.equals(newState)) {
            currentState.exit();
        }
        currentState = newState;
        if (currentState!= null) {
            currentState.enter();
            currentState.changeState(this);
        }
    }

}
```

上面代码定义了 NetworkConnectionContext 类，它继承自 ContextWrapper，并重写了部分方法。

### 6.3.1 初始化

```java
private State currentState;

public NetworkConnectionContext(Context base) {
    super(base);
    setCurrentState(new NormalNetworkState());
}
```

这里的构造函数初始化了一个初始状态，即 NormalNetworkState。

### 6.3.2 设置当前状态

```java
public void setCurrentState(State newState) {
    if (currentState!= null &&!currentState.equals(newState)) {
        currentState.exit();
    }
    currentState = newState;
    if (currentState!= null) {
        currentState.enter();
        currentState.changeState(this);
    }
}
```

setCurrentState() 方法用来设置当前状态。在设置新状态之前，会先执行当前状态的 exit() 方法，退出当前状态。之后才会设置新状态，并执行 enter() 方法，进入新状态。如果当前状态为空，则不会执行任何操作。

### 6.3.3 切换状态

```java
public void connectToWifi() {
    setCurrentState(new WifiConnectedState());
}

public void disconnectFromWifi() {
    setCurrentState(new NormalNetworkState());
}
```

connectToWifi() 和 disconnectFromWifi() 方法用来切换状态。它们都会调用 setCurrentState() 方法，并传入相应的状态对象，切换到新的状态。

### 6.3.4 查询当前状态

```java
public State getCurrentState() {
    return currentState;
}
```

getCurrentState() 方法用来查询当前状态。

## 6.4 环境对象 Client
环境对象 Client 是状态模式的消费者，负责触发状态转换。它获取当前状态，然后调用相应状态类的 changeState() 方法，切换到下一个状态。

```java
public class MyApplication extends Application {

    private NetworkConnectionContext networkContext;

    @Override
    public void onCreate() {
        super.onCreate();
        networkContext = new NetworkConnectionContext(getApplicationContext());
    }

    @Override
    public void onTerminate() {
        super.onTerminate();
        networkContext.disconnectFromWifi();
    }
}
```

MyApplication 继承自 Application，负责初始化网络连接上下文对象，并在程序终止的时候，执行一次断开 WiFi 连接。

```java
public class MainActivity extends AppCompatActivity implements View.OnClickListener {

    private Button btnConnect, btnDisconnect;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        initView();
        bindEvent();

        // 注册监听器
        ConnectivityManager connectivityManager =
                (ConnectivityManager)getSystemService(CONNECTIVITY_SERVICE);
        networkCallback = new ConnectivityManager.NetworkCallback(){

            @Override
            public void onAvailable(Network network) {
                networkContext.connectToWifi();
            }

            @Override
            public void onLost(Network network) {
                networkContext.disconnectFromWifi();
            }
        };
        connectivityManager.registerDefaultNetworkCallback(networkCallback);
    }

   ...
    
}
```

MainActivity 启动后，会检查网络连接状态，如果网络已经连接，则会触发切换状态事件。绑定按钮点击事件，当用户点击按钮时，会触发 connectToWifi() 或 disconnectFromWifi() 方法。另外，MainActivity 会注册一个监听器，用来接收 Android 系统关于网络连接状态的变更。当系统检测到网络连接状态发生变化时，会回调 onAvailable() 或 onLost() 方法，这时就会调用网络连接上下文对象的 connectToWifi() 或 disconnectFromWifi() 方法，切换到下一个状态。