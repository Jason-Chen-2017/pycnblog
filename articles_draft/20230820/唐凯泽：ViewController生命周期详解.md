
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 1.1 历时多年的发展历史
从iOS诞生之初，`MVC`, `MVP`, `MVVM`等各种架构模式都被广泛应用于开发应用程序。而在此基础上，Apple又不断推出新的架构模式，如SwiftUI, Combine, RxSwift, Core Data 和 Core ML等等。这些架构模式中最重要的就是`ViewController`，它承担着非常重要的角色，它管理着视图中的所有元素并响应用户输入。但是对于刚入门或者了解这些架构模式的人来说，`ViewController`的生命周期可能不是很容易理解，比如一个简单的页面，当用户点击某个按钮时，应该怎么处理呢？这样的问题没有标准答案，只能靠个人的理解。所以，今天我们将详细介绍一下`ViewController`的生命周期，帮助读者更加深刻地理解这个知识点。
## 1.2 本文所涉及的内容
我们将通过一步步地分析，展示`ViewController`的生命周期。首先会给出一些基本概念，包括导航控制器、TabBarController和Storyboard的作用，然后是`ViewController`的创建过程，以及生命周期各阶段的事件和相应的操作。最后，我们将给出一些误区和警示，以及提出一些建议，帮助读者进一步地了解和应用`ViewController`的生命周期。

*注意：本文不会过分深入底层，只做简单的介绍，避免出现术语模糊、难以理解的名词。*
# 2.基本概念
## 2.1 Navigation Controller
`Navigation Controller`是一个带有返回按钮的容器视图控制器，用于管理它内部子视图控制器之间的导航关系。它能够管理不同的导航栈，每个堆栈对应着可供用户进行导航的一组视图控制器。每当用户点击屏幕上的后退按钮或其他类型的返回手势时，navigation controller会回到之前的视图控制器。除了管理不同导航栈外，Navigation Controller还可以提供不同的转场动画效果，让用户的视觉体验变得更好。


`Navigation Controller`一般都是作为App的根视图控制器存在。它负责管理其下面的各个`ViewController`之间的切换，管理它们的堆栈记录，实现界面之间的互动和动画切换效果。

## 2.2 TabBarController
`TabBarController`是iOS系统自带的一个特殊控制器，主要用来管理iOS设备上面的标签栏样式的主界面的视图控制器。它和`Navigation Controller`一样，也是管理着不同`ViewController`之间的切换。不同的是，`TabBarController`的子视图控制器不止一个，而且是直接跟在标签栏后面，这样就构成了一个类似于标签栏的布局。


当用户点击标签栏上的某个选项时，对应的子视图控制器就会呈现出来。如果用户再次点击同样的选项，则还是回到之前显示的那个子视图控制器。这种布局形式使得`TabBarController`能够快速高效地管理不同功能模块的视图控制器。

## 2.3 Storyboard
`Storyboard`是用来构建UIKit用户界面的重要工具。它提供了一种直观的方式，可以把视图控制器的设计文件和运行时的逻辑分离开来。通过拖拽或直接连接各个控件，就可以快速完成控件之间的约束关系和布局。


在Xcode里面，我们通过拖拽View Controller对象到画布，就可以把它添加到项目里面。也可以通过拖拽Connection对象来建立视图控制器之间的连接。但无论如何，Storyboard都会生成一个`XIB`文件，这个文件最终会编译成为`nib`文件，用来描述`UIViewController`。

# 3.`ViewController`生命周期
## 3.1 创建流程
### 3.1.1 UIKit自动调用的生命周期函数
UIKit在创建`ViewController`的时候，会自动调用一些生命周期函数。

- viewDidLoad()
  - 当`view`已经加载完成，且控件已经约束完成，`viewDidLoad()`方法就会被调用。可以在该方法里做一些初始化工作，例如读取数据、设置默认值、创建自定义的控件等。
- viewWillAppear(_ animated: Bool)
  - 当`view`即将出现在屏幕上时，`viewWillAppear()`方法就会被调用。可以在这里做一些准备工作，例如播放音乐、获取焦点、调整布局等。
- viewDidAppear(_ animated: Bool)
  - 当`view`已经出现在屏幕上时，`viewDidAppear()`方法就会被调用。可以在这里做一些动画效果的修改，例如增加或删除一个控件，使`view`显得更加生动活泼。
- viewWillDisappear(_ animated: Bool)
  - 当`view`即将消失时，`viewWillDisappear()`方法就会被调用。可以在这里做一些收尾工作，例如保存状态、停止播放音乐、失去焦点等。
- viewDidDisappear(_ animated: Bool)
  - 当`view`已经消失时，`viewDidDisappear()`方法就会被调用。可以在这里做一些善后工作，例如更新数据、销毁某个控件等。

### 3.1.2 通过storyboard或XIB创建
在创建`ViewController`的时候，可以通过两种方式：

1. 在Storyboard中，选择菜单栏目中的`Editor`->`Embed in`->`Navigation Controller`，即可创建一个`Navigation Controller`；选择菜单栏目中的`Editor`->`Embed in`->`TabBar Controller`，即可创建一个`TabBar Controller`。这两个选项可以帮助我们快速地创建带有导航栏或标签栏的`ViewController`。

2. 使用XIB文件。如果想要使用XIB文件，则需要先把`XIB`文件拖拽到项目的`Main.storyboard`文件中。然后，打开`Main.storyboard`文件，找到左边的`Object Library`，选择`ViewController`，将其拖拽到右边的画布上。如果要在导航栏或标签栏中嵌入一个`ViewController`，则需要先在`XIB`文件中嵌入其余的`ViewController`。

### 3.1.3 Programmatically创建
如果想要在代码中动态创建`ViewController`，则可以使用`init()`方法和`deinit()`方法。通过继承`UIViewController`，并重写父类的`init()`方法，就可以在指定位置创建`ViewController`。在`deinit()`方法中，则可以做一些清理工作，例如销毁某个控件。

```swift
class MyViewController: UIViewController {
    override init(nibName nibNameOrNil: String?, bundle nibBundleOrNil: Bundle?) {
        super.init(nibName: nibNameOrNil, bundle: nibBundleOrNil)
        
        // Do something here...

        self.title = "My View Controller"
    }
    
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    deinit {
        // Clean up code here...
    }
}
```

## 3.2 生命周期触发的事件
我们知道了`ViewController`的创建过程，以及UIKit自动调用的生命周期函数。接下来，我们将介绍`ViewController`各个生命周期阶段发生的事件，以及相应的操作。

### 3.2.1 第一次启动App时
`UIApplication`会创建一个初始的`window`，根据配置创建根视图控制器(`rootViewController`)并添加到`window`。

当`window`载入完毕之后，UIKit会调用它的`delegate`(通常是`UIWindowSceneDelegate`)的`window(_ window: UIWindow, didBecomeVisibleForFrameOfScreen screen: UIScreen)`方法，告诉它当前的`window`处于可见状态。在`viewDidLoad()`之前，如果`window`内的任何`ViewController`需要请求权限，则需要在此时请求权限。

当`UIApplication`接收到通知(如来电，网络变化)，或在后台唤醒应用时，UIKit会调用`willResignActive(_ application: UIApplication)`, `didEnterBackground(_ application: UIApplication)`, `willEnterForeground(_ application: UIApplication)`, `didBecomeActive(_ application: UIApplication)`等方法。在`didBecomeActive()`方法中，我们可以根据情况执行一些任务，例如刷新数据、启动定时器、重新建立后台传输任务等。

### 3.2.2 从其它`ViewController`跳转到目标`ViewController`
如果在App中有多个`ViewController`，那么它们之间可能需要跳转。UIKit提供的跳转方式有两种：

1. push
   - `UINavigationController`使用`pushViewController()`方法进行跳转，该方法会在栈顶插入一个`ViewController`。新加入的`ViewController`就是栈顶的`ViewController`。


2. present
   - 如果我们希望弹出一个全屏的`ViewController`，或者在当前`ViewController`之上覆盖一个小窗口，我们可以使用`present()`方法。该方法会在当前`ViewController`之上创建新的`ViewController`，并遮盖当前`ViewController`的`view`。


一般情况下，我们使用`push()`方法来进行页面的跳转，但是也有例外。如当前页面是一个列表页，通过列表页跳转到详情页的时候，往往需要同时显示列表页的内容，所以不能使用`push()`，而应该使用`present()`。并且`present()`方法会让用户感觉不到前一个页面的存在。

### 3.2.3 App切到后台
UIKit会调用`applicationWillResignActive(_ application: UIApplication)`方法，通知应用程序即将进入后台。如果应用程序处于非活动状态的时间超过了预定时间，可能会被系统终止掉。因此，`UIApplication`会调用`didEnterBackground(_ application: UIApplication)`方法，来进行一些必要的清理工作，例如关闭数据库连接、结束网络传输、暂停播放音乐等。

### 3.2.4 按下Home键回到桌面
当应用程序进入后台之后，用户可以从Dock上滑动进入桌面，UIKit会调用`applicationWillTerminate(_ application: UIApplication)`方法，告诉应用即将退出。此时，我们应当及时释放资源，释放占用的内存，否则，系统可能会终止我们的进程。另外，我们应当保存应用程序的当前状态，防止系统因内存不足而杀死我们的进程。

### 3.2.5 从其它应用进入当前App
如果用户在另一个App中打开了当前App的某个页面，UIKit会调用`application(_ application: UIApplication, open url: URL, options: [UIApplicationOpenURLOptionsKey : Any] = [:]) -> Bool`方法。我们需要处理这个通知，因为这意味着用户正在打开当前App的某个页面。我们应当在这个方法里做一些恢复状态的操作。