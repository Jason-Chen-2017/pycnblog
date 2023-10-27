
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


## 概述
ReactiveX(Rx)是一种基于观察者模式的异步编程接口，它主要用于组成事件驱动型、非阻塞式的应用，以及构建健壮且高度可测试的异步和基于事件的程序。Kotlin语言作为静态类型语言，具备高级特性，特别适合于 ReactiveX 和 RxJava 的开发。因此本文将重点关注 Kotlin 对 ReactiveX 库的支持情况，并用具体的例子实现一个 ReactiveX 风格的 Android 应用。
## ReactiveX 与 RxJava 简介
ReactiveX 是 Reactive Extensions（响应扩展）的缩写，它是一个开发小组，致力于提供统一的 API 来建立异步数据流处理管道。ReactiveX 提供了 Observer 接口，让对象可以订阅 Observable 对象，从而监听其数据流。同时，ReactiveX 提供了一系列的操作符（operator），用于对数据流进行变换、过滤、组合等操作。这些操作符满足了函数式编程中流的惰性求值和无副作用（side effect-free）的特性。ReactiveX 使用观察者模式，Observable 对象维护一组观察者列表，当源数据改变时，会通知所有的观察者，从而使得它们自动更新自己的数据。RxJava 是 ReactiveX 在 Java 中的实现，也是最知名的 ReactiveX 实现版本。RxJava 通过观察者模式封装了并发操作，如线程切换、回调等。
## Kotlin 对 ReactiveX 支持情况
Kotlin 支持 ReactiveX 通过 rxjava-kotlin 库，这个库提供了一些帮助类方便开发者使用 RxJava。比如 Single，Completable，Maybe，Flowable 和 Subject 等。rxjava-kotlin 库还提供了相关的扩展方法，使得 Kotlin 更加像 RxJava，而且功能更强大。比如 Flowable 拓展了集合类的操作符，Flowable 使用 Kotlin 协程，可以轻松实现复杂的流水线操作。Subject 可以被多种数据流对象观察，包括 Observables 或 Flowables。另外还有一些其他的支持库，如 Reactive Streams JVM（用于兼容其他 JVM 平台的 ReactiveX）、RxKotlin （用于 Kotlin 中基于 Observable 的扩展函数）。
## 本文示例项目
### 创建工程
首先创建一个新的 Android Studio 项目，选择 Empty Activity 模板，命名为 "RxDemo"。
在 app/build.gradle 文件中添加依赖：
```groovy
dependencies {
    implementation fileTree(dir: 'libs', include: ['*.jar'])
    // ReactiveX & RxJava dependencies
    compile "io.reactivex.rxjava2:rxjava:${latest_version}"
    compile "io.reactivex.rxjava2:rxandroid:${latest_version}"

    testImplementation 'junit:junit:4.12'
    androidTestImplementation 'androidx.test.ext:junit:1.1.1'
    androidTestImplementation 'androidx.test.espresso:espresso-core:3.2.0'
}
```
其中 `${latest_version}` 为最新版 RxJava2 版本号。然后点击 sync now 同步依赖关系。
### 创建 Model
我们先定义一个 User 实体类：
```kotlin
data class User(val id: Int, val name: String, var age: Int)
```
这里 User 有三个属性：id 表示用户 ID，name 表示用户名，age 表示用户年龄。
### 创建 Repository
为了模拟网络请求，我们需要创建 UserService 接口及其默认实现：
```kotlin
interface UserService {
    fun getUser(userId: Int): Single<User>
}

class DefaultUserService : UserService {
    override fun getUser(userId: Int): Single<User> {
        return if (userId == 0) {
            Single.error(Exception("Invalid userId"))
        } else {
            Single.just(
                User(
                    id = userId,
                    name = "$userId Jackson",
                    age = Random().nextInt(70) + 18
                )
            )
        }
    }
}
```
UserService 的 getUser 方法返回的是 Single<User> 对象，代表了一个可观测的用户对象。在实现中，如果传入的 userId 为 0 ，则直接抛出异常；否则随机生成一个 User 对象并返回。
至此，我们已经完成了模型层的编写。
### 创建 ViewModel
为了实现 MVVM 模式，我们需要创建一个 ViewModel 类，负责管理数据和状态。我们这里只创建一个获取用户信息的方法：
```kotlin
class UserViewModel(private val userService: UserService) : ViewModel() {
    private val _user = MutableLiveData<User>()
    val user: LiveData<User>
        get() = _user

    fun loadUser(userId: Int) {
        viewModelScope.launch {
            try {
                val user = userService.getUser(userId).await()
                _user.postValue(user)
            } catch (e: Exception) {
                Log.e(this::class.qualifiedName, e.message?: "")
            }
        }
    }
}
```
这里 ViewModel 依赖于 UserService 对象，通过它加载用户信息。loadUser 方法的参数为待获取的用户 ID，调用 UserService 的 getUser 方法获取单个用户信息，并将结果保存在 `_user` LiveData 中。注意到这里调用了 `viewModelScope.launch`，这是 Kotlin 协程的使用方式，它允许在 ViewModel 中启动协程。这样做可以确保 ViewModel 内部的逻辑在正确的生命周期内执行，并且不会引起内存泄漏。
### 创建 UI
最后，我们在 activity_main.xml 文件中定义一个 RecyclerView，并将其绑定到 UserViewModel 上：
```xml
<?xml version="1.0" encoding="utf-8"?>
<layout xmlns:app="http://schemas.android.com/apk/res-auto">
  <data>
      <variable
          name="activity"
          type="androidx.appcompat.app.AppCompatActivity" />
      <variable
          name="viewModel"
          type="com.example.rxdemo.viewmodel.UserViewModel" />
  </data>

  <LinearLayout
      android:orientation="vertical"
      android:layout_width="match_parent"
      android:layout_height="match_parent">

      <Button
          android:text="Load User Info"
          android:onClick="onLoadUserInfoClick"/>

      <RecyclerView
          android:id="@+id/recyclerView"
          android:layout_width="match_parent"
          android:layout_height="wrap_content"
          app:layoutManager="androidx.recyclerview.widget.LinearLayoutManager"/>
  </LinearLayout>
</layout>
```
在 onCreate 中设置 viewModel 属性，并初始化 RecyclerView：
```kotlin
override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.activity_main)
    
    recyclerAdapter = RecyclerAdapter()
    recyclerView.adapter = recyclerAdapter
    
    viewModel = ViewModelProvider(this).get(UserViewModel::class.java)
}
```
RecyclerAdapter 是 RecyclerView 的 Adapter，用来展示 User 数据。我们简单地定义一下它的 ViewHolder 类：
```kotlin
class RecyclerAdapter : RecyclerView.Adapter<RecyclerAdapter.ViewHolder>() {
    inner class ViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        val textViewId: TextView = itemView.findViewById(R.id.textViewUserId)
        val textViewName: TextView = itemView.findViewById(R.id.textViewUserName)
        val textViewAge: TextView = itemView.findViewById(R.id.textViewUserAge)

        fun bind(user: User) {
            textViewId.text = user.id.toString()
            textViewName.text = user.name
            textViewAge.text = user.age.toString()
        }
    }

    var users: List<User> = emptyList()
        set(value) {
            field = value
            notifyDataSetChanged()
        }

    override fun getItemCount(): Int = users.size

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val layoutInflater = LayoutInflater.from(parent.context)
        val cellForRow = layoutInflater.inflate(R.layout.cell_row, parent, false)
        return ViewHolder(cellForRow)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        holder.bind(users[position])
    }
}
```
这里 ViewHolder 是 RecyclerView 的子项视图，它包含三个 TextView 对象用于显示 User 的 ID、名称和年龄。RecyclerAdapter 的 users 字段表示要展示的用户数据列表。这里我们重载了 RecyclerView.Adapter 的 onCreateViewHolder 方法，用来创建每个子项视图，并重载了 onBindViewHolder 方法，用来绑定数据给 ViewHolder。
接着我们在 onClick 事件里调用 viewModel 的 loadUser 方法，并将当前输入的 ID 参数传进去：
```kotlin
fun onLoadUserInfoClick(v: View) {
    val inputText = editTextUserId.text.trim()
    if (!inputText.isEmpty()) {
        val userId = inputText.toIntOrNull()?: 0
        if (userId > 0) {
            viewModel.loadUser(userId)
        } else {
            Toast.makeText(applicationContext, "Invalid UserId", Toast.LENGTH_SHORT).show()
        }
    }
}
```
点击按钮后，我们检查输入是否为空，如果不为空，尝试把字符串转换为整数，如果转换失败或者数值小于等于零，则显示错误提示；否则调用 viewModel 的 loadUser 方法。
至此，我们就完成了整套 MVVM 架构，运行一下看看效果吧！