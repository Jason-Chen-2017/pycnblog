
作者：禅与计算机程序设计艺术                    

# 1.背景介绍

：
## 1.1 RxJava简介
RxJava (Reactive Extensions) 是 ReactiveX （Reactive Extensions） 的 Java 版本实现。它提供了很多类似于 LINQ 的高阶函数和操作符，可用于处理异步数据流、事件流和基于回调的编程模式。虽然它的名字叫做 RxJava ，但是实际上，它不是一个独立的项目，而是由微软在.Net 框架中实现并开源发布的一组扩展库，包括 C#、F# 和 Visual Basic 。在 2016 年的时候， Google 提供了 Kotlin 语言对 RxJava 的支持，所以 RxJava 在 Android 开发中越来越受欢迎。
## 1.2 为什么要学习 RxJava？
### 异步编程模型
传统的编程模型都是顺序执行，无论是单线程还是多线程，从编写到调试都需要耗费大量的时间。对于异步编程来说，通常采用的是非阻塞 IO 模型（NIO），也就是说应用层线程不会等待 IO 操作完成，直接切换到其他线程继续处理后续的任务。这样可以极大的提升程序的吞吐量和响应能力。

但是，使用 NIO 有个严重的问题就是复杂性太高。例如，对于复杂的网络交互协议或文件传输，用户需要自己处理各种异常和状态变化，还要考虑诸如超时重试等复杂情况。为了解决这个问题，才有了各种封装好的异步框架，比如 Netty 或 Vert.x 。这些框架已经帮我们封装好了异步 IO 操作，让应用层代码简单易懂。

但异步编程模型本身还有很多缺陷，比如无法有效控制资源，并发数过多容易造成线程上下文切换增加延迟，程序员可能会出现各种各样的并发 bug 。因此，很多程序员倾向于采用同步的方式编写程序，但其实效率很低下。

另一种方法是采用事件驱动模型。当某个事件发生时，主线程把该事件放进消息队列里，然后通知其它线程去处理。这种方式最大的优点是降低了线程之间的耦合度，减少了上下文切换的消耗，进一步提升了程序的运行速度。然而，事件驱动模型也有自己的一些缺陷，比如没有规模化的解决方案，难以处理海量数据。

异步编程模型和事件驱动模型之间还有一条鸿沟，就是程序员对两种模式的理解不同。一般认为，同步和异步分别对应着同步和异步执行的操作，而事件驱动模型则对应着事件通知机制。

### RxJava
RxJava 是由微软开源的一个基于观察者模式的异步编程库，其目标是在 JVM 上实现响应式编程（Reactive Programming）。通过对观察者模式的支持，RxJava 可以帮助我们构造声明式的、可组合的异步和基于事件的数据流。

传统的观察者模式要求我们首先定义 subject（主题），然后再注册 observer（观察者），最后 subject 发送事件给 observer 。而 RxJava 把这一过程反过来，先将 observer 注册到 subject 上，然后 subject 将自身作为参数传递给 observer ，observer 通过 subject 获取数据或者事件。这样可以让 subject 和 observer 彻底分离，subject 不知道 observer 的存在，也不关心 observer 是否正在监听或者消费事件。

另外， RxJava 使用 Observables （可被订阅的对象）来表示事件序列，每个 Observable 可以被多个 Observer 订阅。Observer 可以订阅不同的 Observable ，也可以同时订阅同一个 Observable 来接收来自多个源的事件序列。Observable 和 Observer 之间的关系可以看作是一个双向绑定关系，即任意时刻，任意数量的 Observer 可以订阅同一个 Observable ，或者反之亦然。这样就可以用同一个 Observable 生成多个依赖于它的视图，甚至可以创建完全独立于某个 Observable 的新 Observable ，这些都不需要修改现有的代码。

以上种种优势使得 RxJava 在 Android 和服务器端编程领域非常流行，尤其适用于实时数据处理、后台计算、事件驱动型应用、分布式系统及设备通信等场景。

# 2.核心概念与联系
## 2.1 Subject（主题）
Subject 是 RxJava 中最基本的概念，它代表可观测的对象。可观测对象通常用来触发事件以及将事件通知给观察者。Subject 接口主要有以下几个方面：

1. subscribe() 方法：订阅 Observer ，将自身作为参数传递给 observer；
2. onSubscribe() 方法：订阅时的回调；
3. onNext() 方法：通知事件的回调；
4. onComplete() 方法：通知完成的回调；
5. onError() 方法：通知错误的回调；

通过继承 Subject 类，可以自定义可观测对象的行为。例如， PublishSubject 只有订阅之后才会收到onNext事件，并且onComplete、onError事件只会被传播给已订阅的Observer。

## 2.2 Subscriber（观察者）
Subscriber 是 RxJava 中的核心接口，它定义了一个简单的观察者模式中的实体，负责订阅 Observable 对象，并接收来自 Observable 发出的事件。Subscriber 有两个方法：

1. onStart() 方法：观察者开始订阅时调用，一般用来做一些准备工作；
2. onNext(T t) 方法：接受来自 Observable 的事件，并进行处理；

除了 onStart() 和 onNext(T t) 方法外，Subscriber 还可以通过相应的方法指定当前 Subscriber 对某些类型事件感兴趣。例如，我们可以使用 filter() 方法来过滤掉特定类型的事件。

## 2.3 Operator（操作符）
Operator 是 RxJava 中的一个重要概念，它定义了一系列操作，能改变 Observable 所发出的元素，或者转化 Observable 发出来的元素。操作符实现了 ObservableTransformer 接口，该接口包含三个方法：

1. apply(Observable upstream) 方法：对原始 Observable 进行操作；
2. call(Object... args) 方法：传入一系列变换参数，返回一个新的 Observable；
3. compose(ObservableTransformer other) 方法：对当前 Observable 进行更加复杂的操作。

## 2.4 Scheduler（调度器）
Scheduler 是 RxJava 中用来管理任务调度的重要组件。它允许我们安排任务在特定的时间发生，比如“当 UI 线程空闲时”，“延迟1秒执行”等。Scheduler 接口主要有四个方法：

1. scheduleDirect(Runnable runnable, long delay, TimeUnit unit) 方法：立即执行 Runnable 对象，延迟指定的时间后执行；
2. schedulePeriodically(Runnable runnable, long initialDelay, long period, TimeUnit unit) 方法：按照指定周期重复执行 Runnable 对象；
3. start() 方法：启动 Scheduler ;
4. shutdown() 方法：关闭 Scheduler 。

## 2.5 Backpressure（背压）
Backpressure 是 RxJava 中的术语，它描述的是生产者和消费者的速度差距。当消费者的速度远快于生产者时，背压会导致数据丢失。RxJava 允许我们设置 backpressue 参数来防止数据丢失。Backpressure 有三种级别：

1. BUFFER：缓冲区；
2. ERROR：报错；
3. DROP：丢弃。

通过指定背压策略，RxJava 会自动调整发送速率和缓存大小，确保消费者不会因消费不过来而阻塞。

# 3.核心算法原理和具体操作步骤以及数学模型公式详细讲解
## 3.1 创建操作符
- just() 操作符：创建一个发出固定数据集的 Observable ，使用它可以将集合或数组转换为 Observable 对象。

```java
String[] fruits = {"apple", "banana", "orange"};
Observable<String> observable = Observable.just(fruits);
observable.subscribe(new Consumer<String>() {
    @Override
    public void accept(String fruit) throws Exception {
        System.out.println(fruit);
    }
});
```

- fromArray() 操作符：创建一个发出数组内容的 Observable ，使用它可以将数组转换为 Observable 对象。

```java
int[] nums = {1, 2, 3};
Observable<Integer> observable = Observable.fromArray(nums);
observable.subscribe(new Consumer<Integer>() {
    @Override
    public void accept(Integer num) throws Exception {
        System.out.println(num);
    }
});
```

- interval() 操作符：创建一个定时发出数字的 Observable ，使用它可以创建具有固定间隔时间的 Observable 对象，可以实现轮询操作。

```java
Observable.interval(1, TimeUnit.SECONDS).take(5).subscribe(new Consumer<Long>() {
    @Override
    public void accept(Long aLong) throws Exception {
        System.out.println("Current time: " + new Date());
    }
});
```

## 3.2 变换操作符
- map() 操作符：将源 Observable 中的每项元素经过映射函数转换成一个新的元素，转换后的 Observable 将会发出带有新数据的元素。

```java
Observable.range(1, 5)
       .map(new Function<Integer, Integer>() {
            @Override
            public Integer apply(Integer integer) throws Exception {
                return integer * 2;
            }
        })
       .subscribe(new Consumer<Integer>() {
            @Override
            public void accept(Integer integer) throws Exception {
                System.out.println(integer);
            }
        });
```

- flatMap() 操作符：将源 Observable 中的每项元素转换成 Observable 对象，然后将各个 Observable 合并成一个新的 Observable 对象。flatMap() 操作符会将多个 Observable 中产生的元素拼接起来，形成一个新的 Observable 对象。

```java
Observable.fromIterable(Arrays.asList(Observable.just(1), Observable.just(2)))
        .flatMap(new Function<Observable<Integer>, ObservableSource<Integer>>() {
             @Override
             public ObservableSource<Integer> apply(@NonNull Observable<Integer> o) throws Exception {
                 return o;
             }
         }).subscribe(new Consumer<Integer>() {
             @Override
             public void accept(Integer integer) throws Exception {
                 System.out.println(integer);
             }
         });
```

- concat() 操作符：将多个 Observable 拼接起来，形成一个新的 Observable 对象。concat() 操作符会按顺序将多个 Observable 发射的值拼接到一起，依次输出。

```java
Observable.concat(Observable.fromArray(1, 2, 3), Observable.just(4))
         .subscribe(new Consumer<Integer>() {
              @Override
              public void accept(Integer integer) throws Exception {
                  System.out.println(integer);
              }
          });
```

- groupBy() 操作符：将源 Observable 中的数据按照 keyFunction 指定的方式分组，并返回 GroupedObservable 对象。GroupedObservable 对象实现了 Observable 接口，其中包含子 Observable 中的所有元素。

```java
Observable.range(1, 5)
         .groupBy(new Function<Integer, String>() {
              @Override
              public String apply(Integer integer) throws Exception {
                  if (integer % 2 == 0) {
                      return "Even";
                  } else {
                      return "Odd";
                  }
              }
          })
         .subscribe(new Consumer<GroupedObservable<String, Integer>>() {
              @Override
              public void accept(GroupedObservable<String, Integer> go) throws Exception {
                  System.out.println("Key: " + go.getKey());
                  go.subscribe(new Consumer<Integer>() {
                      @Override
                      public void accept(Integer integer) throws Exception {
                          System.out.println(integer);
                      }
                  });
              }
          });
```

## 3.3 过滤操作符
- filter() 操作符：创建一个新的 Observable ，过滤掉满足条件的元素。

```java
Observable.range(1, 5)
         .filter(new Predicate<Integer>() {
              @Override
              public boolean test(Integer integer) throws Exception {
                  return integer > 3;
              }
          })
         .subscribe(new Consumer<Integer>() {
              @Override
              public void accept(Integer integer) throws Exception {
                  System.out.println(integer);
              }
          });
```

- distinct() 操作符：创建一个新的 Observable ，移除重复的元素。

```java
Observable.just(1, 2, 2, 3, 3, 3)
         .distinct()
         .subscribe(new Consumer<Integer>() {
              @Override
              public void accept(Integer integer) throws Exception {
                  System.out.println(integer);
              }
          });
```

- takeWhile() 操作符：创建一个新的 Observable ，只保留源 Observable 中一直满足条件的数据。

```java
Observable.range(1, 5)
         .takeWhile(new Predicate<Integer>() {
              @Override
              public boolean test(Integer integer) throws Exception {
                  return integer <= 3;
              }
          })
         .subscribe(new Consumer<Integer>() {
              @Override
              public void accept(Integer integer) throws Exception {
                  System.out.println(integer);
              }
          });
```

- skipUntil() 操作符：创建一个新的 Observable ，跳过源 Observable 中直到满足条件的数据。

```java
Observable.just(1, 2, 3, 4, 5)
         .skipUntil(Observable.<Integer>timer(2, TimeUnit.SECONDS).ignoreElements())
         .subscribe(new Consumer<Integer>() {
              @Override
              public void accept(Integer integer) throws Exception {
                  System.out.println(integer);
              }
          });
```

## 3.4 连接操作符
- merge() 操作符：创建一个新的 Observable ，将源 Observable 中的所有元素合并，并按照其发生的时间顺序进行排序。

```java
Observable<String> sourceA = Observable.create(new Publisher<String>() {
    @Override
    protected void subscribeActual(final Subscriber<? super String> s) {
        Timer timer = new Timer();

        for (char c : "ABC".toCharArray()) {
            timer.schedule(new TimerTask() {
                @Override
                public void run() {
                    try {
                        s.onNext("" + c);
                    } catch (Exception e) {
                        s.onError(e);
                    }
                }
            }, 0, 1000); // 每隔1秒发送一次字符
        }
    }
});

Observable<String> sourceB = Observable.create(new Publisher<String>() {
    @Override
    protected void subscribeActual(final Subscriber<? super String> s) {
        Timer timer = new Timer();

        for (char c : "DEF".toCharArray()) {
            timer.schedule(new TimerTask() {
                @Override
                public void run() {
                    try {
                        s.onNext("" + c);
                    } catch (Exception e) {
                        s.onError(e);
                    }
                }
            }, 2000, 1000); // 每隔1秒发送一次字符
        }
    }
});

sourceA.mergeWith(sourceB)
      .observeOn(Schedulers.computation())
      .subscribeOn(Schedulers.io())
      .subscribe(new Consumer<String>() {
           @Override
           public void accept(String str) throws Exception {
               System.out.println(Thread.currentThread().getName() + ": " + str);
           }
       });
```

- zip() 操作符：创建一个新的 Observable ，将来自多个源 Observable 的元素按照索引位置一一配对，然后将结果作为 List 发出。

```java
Observable.zip(Observable.range(1, 3), Observable.range(4, 3),
               new BiFunction<Integer, Integer, Pair<Integer, Integer>>() {
                   @Override
                   public Pair<Integer, Integer> apply(Integer i1, Integer i2) throws Exception {
                       return new Pair<>(i1, i2);
                   }
               })
          .subscribe(new Consumer<Pair<Integer, Integer>>() {
               @Override
               public void accept(Pair<Integer, Integer> pair) throws Exception {
                   System.out.println(pair.first + ", " + pair.second);
               }
           });
```

## 3.5 辅助操作符
- ignoreElements() 操作符：创建一个新的 Observable ，忽略掉源 Observable 发出的元素。

```java
Observable.range(1, 5)
         .ignoreElements()
         .subscribe(new Action() {
              @Override
              public void run() throws Exception {
                  System.out.println("Ignore all elements");
              }
          });
```

- doOnSubscribe() 操作符：为源 Observable 添加一个回调，在订阅时调用。

```java
Observable.range(1, 5)
         .doOnSubscribe(new Consumer<Disposable>() {
              @Override
              public void accept(Disposable d) throws Exception {
                  System.out.println("Start subscribing");
              }
          })
         .subscribe(new Consumer<Integer>() {
              @Override
              public void accept(Integer integer) throws Exception {
                  System.out.println(integer);
              }
          });
```

- observeOn() 操作符：指定观察者所在的 Scheduler 。

```java
Observable.range(1, 5)
         .observeOn(Schedulers.io())
         .subscribe(new Consumer<Integer>() {
              @Override
              public void accept(Integer integer) throws Exception {
                  System.out.println(Thread.currentThread().getName() + ":" + integer);
              }
          });
```

## 3.6 异步操作符
- subscribeOn() 操作符：指定 Observable 生成事件的线程。

```java
Observable.range(1, 5)
         .subscribeOn(Schedulers.io())
         .subscribe(new Consumer<Integer>() {
              @Override
              public void accept(Integer integer) throws Exception {
                  System.out.println(Thread.currentThread().getName() + ":" + integer);
              }
          });
```

- toFlowable() 操作符：将 Observable 转换为 Flowable 对象。Flowable 相比于 Observable ，可以指定 BackpressureStrategy 来处理事件速率不匹配的问题。

```java
Observable.range(1, 5)
         .toFlowable(BackpressureStrategy.BUFFER)
         .blockingSubscribe(new Consumer<Integer>() {
              @Override
              public void accept(Integer integer) throws Exception {
                  System.out.println(integer);
              }
          });
```

- debounce() 操作符：延迟发出元素，如果前面的元素在一段时间内没有发出，则收集后一批元素发出。

```java
Observable.create(new ObservableOnSubscribe<Integer>() {
    @Override
    public void subscribe(@NonNull ObservableEmitter<Integer> emitter) throws Exception {
        Random random = new Random();
        int count = 0;
        
        while(!emitter.isDisposed()){
            emitter.onNext(count++);
            
            Thread.sleep(random.nextInt(3)*1000); // 随机生成一个范围为[0,2]的整数，然后将其乘以1000毫秒转化为毫秒
        }
    }
}).debounce(5,TimeUnit.SECONDS)
  .subscribe(new Consumer<Integer>() {
       @Override
       public void accept(Integer integer) throws Exception {
           System.out.println(integer);
       }
   });
```

- retry() 操作符：在某个元素发出错误时，重新订阅源 Observable 。

```java
Observable<Integer> observable = Observable.create(new ObservableOnSubscribe<Integer>() {
    @Override
    public void subscribe(ObservableEmitter<Integer> emitter) throws Exception {
        if (++index < 3) {
            throw new RuntimeException("Error");
        } else {
            emitter.onNext(1);
            emitter.onComplete();
        }
    }
})
       .retry(2);

observable.subscribe(new Consumer<Integer>() {
    @Override
    public void accept(Integer integer) throws Exception {
        System.out.println(integer);
    }

    @Override
    public void onError(Throwable throwable) {
        System.out.println(throwable.getMessage());
    }
    
    @Override
    public void onComplete() {}
    
});
```

# 4.具体代码实例和详细解释说明
## 4.1 下载图片
```java
public class DownloadImageActivity extends AppCompatActivity implements View.OnClickListener{

    private static final String TAG = DownloadImageActivity.class.getSimpleName();
    private ImageView mImageView;
    private Button mButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_download_image);

        initView();
    }

    private void initView(){
        mImageView = findViewById(R.id.iv_photo);
        mButton = findViewById(R.id.btn_download);

        mButton.setOnClickListener(this);
    }

    @Override
    public void onClick(View v) {
        switch (v.getId()){
            case R.id.btn_download:
                downloadPhoto();
                break;

            default:
                break;
        }
    }

    /**
     * 下载照片
     */
    private void downloadPhoto(){
        Disposable disposable = RetrofitManager.getInstance()
               .getApi()
               .getImageData("https://picsum.photos/200")
               .subscribeOn(Schedulers.io())
               .observeOn(AndroidSchedulers.mainThread())
               .subscribe(new SingleObserver<ResponseBody>() {
                    @Override
                    public void onSubscribe(@NonNull Disposable d) {

                    }

                    @Override
                    public void onSuccess(@NonNull ResponseBody responseBody) {
                        InputStream inputStream = null;
                        try {
                            inputStream = responseBody.byteStream();
                            Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
                            mImageView.setImageBitmap(bitmap);

                        } catch (IOException e) {
                            Log.e(TAG,"Download image failed.",e);
                        } finally {
                            if (inputStream!= null){
                                try {
                                    inputStream.close();
                                } catch (IOException e) {
                                    Log.e(TAG,"Close input stream error.",e);
                                }
                            }
                        }
                    }

                    @Override
                    public void onError(@NonNull Throwable e) {
                        Toast.makeText(getApplicationContext(),"下载失败：" + e.getMessage(),Toast.LENGTH_SHORT).show();
                    }
                });

        addDisposable(disposable);
    }


    private CompositeDisposable mCompositeDisposable;

    /**
     * 添加Disposable
     *
     * @param disposable
     */
    public void addDisposable(Disposable disposable) {
        if (mCompositeDisposable == null) {
            mCompositeDisposable = new CompositeDisposable();
        }
        mCompositeDisposable.add(disposable);
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (mCompositeDisposable!= null) {
            mCompositeDisposable.clear();
        }
    }
}
```
## 4.2 从网络请求数据
```java
public class RequestDataActivity extends BaseActivity {

    private TextView mTextView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_request_data);

        initView();
        requestData();
    }

    private void initView(){
        mTextView = findViewById(R.id.tv_result);
    }

    /**
     * 请求数据
     */
    private void requestData(){
        Disposable disposable = RetrofitManager.getInstance()
               .getApi()
               .getData()
               .subscribeOn(Schedulers.io())
               .observeOn(AndroidSchedulers.mainThread())
               .subscribe(new Consumer<Response<List<ResultBean>>>() {
                    @Override
                    public void accept(Response<List<ResultBean>> listResponse) throws Exception {
                        StringBuilder sb = new StringBuilder();

                        for (ResultBean resultBean : listResponse.body()) {
                            sb.append(resultBean.name + "\n");
                        }

                        mTextView.setText(sb.toString());
                    }
                }, new Consumer<Throwable>() {
                    @Override
                    public void accept(Throwable throwable) throws Exception {
                        mTextView.setText("请求失败：" + throwable.getMessage());
                    }
                });

        addDisposable(disposable);
    }
}
```
## 4.3 文件上传
```java
public class FileUploadActivity extends BaseActivity {

    private EditText mEditText;
    private Button mButton;
    private ProgressDialog mProgressDialog;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_file_upload);

        initView();
    }

    private void initView(){
        mEditText = findViewById(R.id.et_file_path);
        mButton = findViewById(R.id.btn_upload);

        mButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                uploadFile();
            }
        });

        mProgressDialog = new ProgressDialog(FileUploadActivity.this);
        mProgressDialog.setCanceledOnTouchOutside(false);
    }

    /**
     * 上传文件
     */
    private void uploadFile(){
        String filePath = mEditText.getText().toString().trim();

        if (!TextUtils.isEmpty(filePath)){
            showLoading("上传中...");

            MultipartBody.Part part = prepareRequestBody(filePath);

            RetrofitManager.getInstance()
                   .getApi()
                   .uploadFile(part)
                   .subscribeOn(Schedulers.io())
                   .observeOn(AndroidSchedulers.mainThread())
                   .subscribe(new CompletableObserver() {
                        @Override
                        public void onSubscribe(@NonNull Disposable d) {

                        }

                        @Override
                        public void onComplete() {
                            hideLoading();
                            SnackbarUtils.showShort("上传成功！");
                        }

                        @Override
                        public void onError(@NonNull Throwable e) {
                            hideLoading();
                            SnackbarUtils.showShort("上传失败：" + e.getMessage());
                        }
                    });
        }else {
            SnackbarUtils.showShort("请选择要上传的文件路径！");
        }
    }

    /**
     * 准备请求体
     *
     * @param filePath 文件路径
     */
    private MultipartBody.Part prepareRequestBody(String filePath){
        File file = new File(filePath);

        RequestBody requestBody = RequestBody.create(MediaType.parse("multipart/form-data"), file);
        MultipartBody.Builder builder = new MultipartBody.Builder().setType(MultipartBody.FORM);
        builder.addFormDataPart("file\"; filename=\"" + file.getName(), file.getName(), requestBody);

        return builder.build().parts().get(0);
    }

    /**
     * 显示加载框
     *
     * @param msg 显示文字
     */
    public void showLoading(String msg){
        mProgressDialog.setMessage(msg);
        mProgressDialog.show();
    }

    /**
     * 隐藏加载框
     */
    public void hideLoading(){
        mProgressDialog.dismiss();
    }
}
```
## 4.4 WebSocket连接
```java
public class WebSocketConnectionActivity extends BaseActivity implements WebSocketListener {

    private WebSocketClient mWebSocketClient;
    private Handler mHandler;
    private String mUrl = "";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_web_socket_connection);

        initView();

        Intent intent = getIntent();
        if (intent!= null){
            mUrl = intent.getStringExtra("url");
        }

        connectWebSocket();
    }

    private void initView(){
        mHandler = new Handler(Looper.myLooper());
    }

    /**
     * 连接WebSocket
     */
    private void connectWebSocket(){
        URI uri = URI.create(mUrl);
        mWebSocketClient = new WebSocketClient(uri) {
            @Override
            public void onOpen(WebSocket webSocket, Response response) {
                Log.d(TAG, "onOpen:" + response.message());

                JSONObject jsonObject = new JSONObject();
                try {
                    jsonObject.put("action","login");
                    jsonObject.put("userName","admin");
                    jsonObject.put("password","<PASSWORD>");
                    send(jsonObject.toJSONString());
                } catch (JSONException e) {
                    e.printStackTrace();
                }
            }

            @Override
            public void onMessage(WebSocket webSocket, String text) {
                Log.d(TAG, "onMessage:" + text);
                Message message = new Message();
                message.what = 1;
                Bundle bundle = new Bundle();
                bundle.putString("text",text);
                message.setData(bundle);
                mHandler.sendMessage(message);
            }

            @Override
            public void onClosing(WebSocket webSocket, int code, String reason) {
                Log.d(TAG, "onClosing:" + reason);
            }

            @Override
            public void onClosed(WebSocket webSocket, int code, String reason) {
                Log.d(TAG, "onClosed:" + reason);
            }

            @Override
            public void sendMessage(String text) throws IOException {
                synchronized (WebSocketClient.class) {
                    webSocket.send(text);
                }
            }

            @Override
            public void close() throws IOException {
                mWebSocketClient.super.close();
            }
        };

        mWebSocketClient.connect();
    }

    @Override
    public void onDestroy() {
        super.onDestroy();

        if (mWebSocketClient!= null && mWebSocketClient.isOpen()){
            mWebSocketClient.disconnect();
        }
    }

    @Override
    public void onFailure(IOException e, Response response) {
        Log.e(TAG,"onFailure:",e);
    }

    @Override
    public void onMessage(ResponseBody message) {
        super.onMessage(message);
    }

    @Override
    public void onPong(Buffer payload) {
        Log.d(TAG,"onPong:");
    }
}
```