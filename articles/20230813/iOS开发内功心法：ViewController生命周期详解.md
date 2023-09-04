
作者：禅与计算机程序设计艺术                    

# 1.简介
  

ViewController（VC）是iOS应用中重要的组成部分之一，负责管理用户界面的显示、响应用户输入事件、数据处理及其他业务逻辑实现等功能。每个VC都对应着一个UIViewController类，其生命周期一直伴随着用户的各种交互操作，当VC被创建、释放、加载、显示、消失等状态发生变化时，VC都会发出通知或调用回调方法。因此，对于理解并掌握VC的生命周期至关重要。本文将详细介绍VC的生命周期，并对其中的一些概念及机制进行分析，帮助读者更好地理解并使用VC。
# 2.基本概念术语说明
## 2.1 ViewController
ViewController是一个iOS UIKit框架类，它主要负责管理界面视图的显示和响应用户交互操作。VC是由两个主要部分组成：View（视图）和ViewController。View是指UIViewController的界面，包括所有控件、图层、约束、布局等。ViewController是在运行时生成和管理这些View，包括控制它们的显示、更新、动画、生命周期管理等。所以，ViewController可以说是UIKit框架中最复杂的类之一。
## 2.2 ViewController的四种类型
- UIViewController：系统提供的基础控制器类型，在UIKit库中直接使用，用于展示普通页面；
- UITabBarController：系统提供的TabBar导航栏控制器类型，可切换不同的页面，用于展示多页面嵌套的结构；
- UINavigationController：系统提供的页面导航控制器类型，提供页面间的前进后退切换，用于管理页面堆栈，实现复杂页面流转；
- UICollectionViewController：系统提供的集合视图控制器类型，用于展示具有大量条目的数据集；
每种VC都有自己的生命周期，并且也有自己独有的特性。如，UITableViewController只能显示TableView，UINavigationController只能用来做页面跳转。
## 2.3 VC的生命周期
以下是VC的生命周期：

1. 创建与初始化：当VC被创建时，系统会自动调用其initWithNibName:bundle:或者init方法进行必要的初始化工作；
2. 加载view：当VC第一次加载时，会调用loadView方法来生成它的View。loadView通常是从nib文件加载或者用代码动态创建View对象；
3. 生成view hierarchy：loadView方法生成了View之后，系统会通过其viewDidLoad方法将View添加到ViewController的view上，同时会设置好相关的约束、布局、以及其他属性；
4. viewDidLoad：此方法在view生成完成并且添加到VC的view上之后调用，一般用来设置一些变量、模型数据、事件监听器、以及约束布局等；
5. viewWillAppear：系统即将展示ViewController的view的时候调用的方法，可以在这里做一些动画效果或者数据的加载；
6. viewDidAppear：ViewController的view已经显示出来并且动画效果已经播放完毕后调用的方法；
7. viewWillDisappear：系统即将销毁ViewController的view时调用的方法，可以在这里做一些清理工作；
8. viewDidDisappear：ViewController的view已经完全消失并且动画效果已经播放完毕后调用的方法；
9. dealloc：当VC不再需要时系统会调用dealloc方法，在这个方法里可以做一些内存回收、资源释放等操作。
## 2.4 推迟初始化
VC的创建过程是比较耗时的，因为其涉及到许多操作，比如读取XIB文件、从Storyboard中加载视图、计算约束、设置AutoLayout、注册通知、创建子线程等等。因此，如果能够尽早创建ViewController并进行必要的准备，就可以避免在每次进入该VC时花费额外的时间。这就是推迟初始化的基本原则，通过重载viewDidLoad方法和lazy修饰符来达到这一目的。
```swift
// 在viewDidLoad之前初始化变量
class MyVC: UIViewController {
    
    @IBOutlet weak var myLabel: UILabel!
    let dataProvider = DataProvider() // 数据源

    override func viewDidLoad() {
        super.viewDidLoad()
        
        // 初始化UI
        configureUI()

        // 获取数据
        fetchData()
    }

    private func configureUI() {
        myLabel.text = "Hello World"
    }

    lazy var fetchedData: [String] = self.dataProvider.fetch() // 懒加载数据

    private func fetchData() {
        DispatchQueue.global(qos:.userInitiated).async {
            guard let data = try? self.fetchedData else { return }
            
            DispatchQueue.main.async {
                self.handleData(data)
            }
        }
    }
    
    private func handleData(_ data: [String]) {
        print("Received data \(data)")
    }
}
```
通过懒加载变量，我们仅在访问该变量时才进行真正的初始化操作，这样可以加快ViewController的创建速度，缩短应用启动时间。