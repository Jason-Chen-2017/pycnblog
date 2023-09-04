
作者：禅与计算机程序设计艺术                    

# 1.简介
  
ViewController是iOS开发中经常使用的一个组件，它是一个窗口管理器，负责显示App界面的各个视觉元素。其生命周期包括初始化、创建、布局、绘制、响应用户操作等，而在不同的ViewController之间也存在相互调用。ViewController的生命周期经历了四种状态，它们分别是：
- init:View控制器的内存空间被分配，但并未完全加载，也就是说，View还没有初始化完成。
- loadView方法:View会被创建并且添加到当前视图控制器上。这个方法不仅仅用于创建View，还要做一些初始化工作比如绑定数据、注册监听等。
- viewWillAppear:View将要出现在屏幕上。此时可以执行一些任务如动画展示或者网络请求等，也可以做一些配置View属性。
- viewDidLoad:View已经在屏幕上显示出来了，这时候可以进行一些配置View的操作，比如重新设置约束、修改UI、添加子View等。
- viewWillDisappear:当ViewController将要离开屏幕的时候。可用于停止一些正在播放音乐或者动画效果的任务，释放一些资源。
- deinit:当ViewController从内存中销毁时调用，通常用来做一些清除工作或收尾工作。
另外，当一个VC中嵌套其他VC（Navigation Controller或者Tab Bar Controller）的时候，子VC也有自己的生命周期。当VC被从屏幕上移除的时候，子VC也会同时被销毁。在VC的生命周期内，父VC可以通过delegate的方式向子VC传递消息。以下是生命周期中的方法及作用。
## 2.基本概念术语说明
1. UIKit Framework:UIKit 是 iOS SDK 的组成部分之一。它提供许多基础类库，如 UIView、UILabel 和 UITextField，能够帮助我们快速构建应用界面。UIKit 的框架包含 View 层级结构体系、控件处理机制、布局管理、事件处理等功能，还包括核心类 NSRunLoop、NSOperationQueue 和 NSThread。

2. UINavigationController:UINavigationController 是一个 UIViewController 子类，负责提供多级页面导航功能。它管理多个 UIViewController 对象之间的切换，为这些对象提供栈式的界面跳转，实现了不同页面间的传值和返回。UINavigationController 支持手势返回，在某个页面上面滑动，可以返回到前一个页面。

3. UITabBarController:UITabBarController 是 UIViewController 的子类，它实现了选项卡式的页面导航，每个选项卡对应于一个页面，通过滑动底部栏可以切换不同的页面。

4. UIResponder:UIResponder 是所有 UIKit 对象（UIView、UIButton、UILabel、UITableView、UICollectionView等）的基类。它定义了事件响应相关的方法，如 touchesBegan:withEvent:、touchesMoved:withEvent:、touchesEnded:withEvent: 等，以及 gesture recognizers 的相关方法，如 gestureRecognizerShouldBegin: 和 gestureRecognizer:shouldRecognizeSimultaneouslyWithGestureRecognizer:。

5. UIView:UIView 是所有 UIKit 对象（如按钮、文本框、标签、表格等）的基本单位。它提供各种属性，如 frame、bounds、center、transform、alpha 等，还有布局属性，如 autoresizingMask、hidden、clipsToBounds、backgroundColor、subviews、constraints 等，这些属性共同作用使得 UIView 可以控制自身的位置和大小，以及其子 View 的布局关系。

6. Frame 和 Bounds：Frame 是 UIView 在 superview 中的坐标系下的边界矩形， Bounds 是 UIView 在内部坐标系下的边界矩形，它决定了 UIView 的尺寸和位置。

7. Autolayout：AutoLayout 是一种在运行时自动计算并调整控件尺寸、位置的机制。通过 AutoLayout，你可以精确地控制控件的尺寸和位置，不需要手动调节。Autolayout 使用基于约束的系统，即你设置好视图之间的约束，系统会自动计算出最佳尺寸和位置，这样就可以让你的视图放置的更加合理、美观。

8. View Controller：View Controller 是 iOS 里的一个重要概念，它是管理 View 的对象。它是一个容器，封装了对 UIView 对象的创建、展示、布局、交互的逻辑。View Controller 通过转场动画、自定义过渡动画，来实现界面切换效果；还可以通过 segue 来实现页面间的传值和返回；还可以管理 child View Controller 。在不同的场景下，UIKit 提供了多个 View Controller 子类，如 UINavigationController、UITabBarController 等，来实现不同类型的页面切换效果。