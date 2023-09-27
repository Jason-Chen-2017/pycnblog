
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Go语言是一个开源的编程语言，在网络服务、系统工具、云基础设施等方面都有很大的应用前景。作为一门优秀的新型语言，Go拥有出色的并发性能表现，可以轻松地编写高效且健壮的代码，同时也带来了诸如接口的动态特性和丰富的标准库支持。

针对软件开发领域里面的图形用户界面（GUI）开发，Go语言提供了一些强大而易用的模块和框架，使得GUI程序的编写和调试变得更加方便。本文将讨论Go语言中的GUI开发及相关技术，并给出一个简单的例子，展示如何利用Go语言构建一个简单的GUI程序。最后，我们还将结合实际情况，提出相应的技术改进建议。


# 2.背景介绍
随着互联网和移动互联网的普及，人们对电脑视觉上的响应时间越来越敏感，越来越多的人开始重视视觉上的体验。因此，人机交互界面设计显得尤为重要。

为了满足用户的需求，软件工程师经过不断地迭代，逐步形成了一套完整的计算机图形用户界面（GUI）设计规范。其中包括颜色选择、字体大小、控件样式等视觉元素的设计原则，以及逻辑结构、操作流程、动画效果、提示信息等交互动作的设计指南。

但是，对于一个软件工程师来说，实现一套完整的GUI设计规范并不是一件容易的事情。因此，很多公司会根据自己的业务场景，利用其内部人力资源或外包第三方设计团队完成相关的产品GUI设计。但是这种方式往往需要耗费大量的人力物力，并且无法保证最终达到设计效果的一致性。另外，如果将GUI设计交由其他人士来完成，很可能会出现沟通成本的上升和项目进度的延迟。

为了解决这个问题，近年来基于Web技术的前端框架火爆，例如React、AngularJS等，将Web应用的页面呈现层完全交给浏览器完成。这为Web开发者提供了一种简单有效的方式来实现具有良好视觉效果的GUI，但由于浏览器的限制，这些框架只能实现传统的页面布局、文本显示和输入框等功能。

另一方面，Go语言作为一门新兴的语言，吸收了一些现代语言的一些特点，例如垃圾回收机制、语法简洁、强大的类型系统和接口机制，以及函数式编程的思想。因此，在Go语言中实现一个完整的图形用户界面（GUI）是比较容易的一件事情。

综上所述，本文将阐述Go语言中GUI开发的相关知识，并通过示例程序展示如何构建一个简单的GUI程序，帮助读者了解GUI开发的基本方法及实现方式。


# 3.基本概念术语说明

## 3.1 Goroutine

在Go语言中，每个线程都是用 goroutine 来实现的。goroutine 是一种轻量级线程，可以与其他的 goroutine 同时运行。每当我们启动一个新的 goroutine 时，它就会在独立的堆栈中执行。因此，它没有共享内存、上下文切换的开销，使得它非常适合用于实现异步、并行的任务处理模型。

从 Go 1.11 版本开始，标准库里面引入了 goroutine 的并发模式，包括 channel、select 和 context 等，它们可以让我们构造复杂的并发模型。

## 3.2 Webview

WebView 是 Android 和 iOS 操作系统中用来承载网页的组件。一般情况下，WebView 可以理解为浏览器内核的封装，负责渲染网页内容并向网页提供数据接口。

Go 在桌面端也可以实现 WebView ，主要依赖于 webview 库。目前已有的一些 Go 桌面端 GUI 框架，如 Fltk 和 Gio，都是基于 cgo 调用底层的系统 API，无法直接跨平台使用。因此，借助于 webView 库，就可以通过 Go 语言调用系统 UI 框架，并利用 Go 语言的并发模型来实现异步与并行。

## 3.3 DOM（Document Object Model）

DOM （Document Object Model）是 HTML 文档的对象表示，它定义了 HTML、XML 文档的结构以及它们的关系。

Go 在桌面端可以通过 webview 库访问 DOM 。通过调用 webview 库提供的方法，我们可以对 DOM 中的元素进行增删改查，修改样式属性，设置事件回调函数等。

## 3.4 Channel

Channel 是 Go 语言中用于 goroutine 之间通信的主要方式。一个 channel 只能被两个 goroutine 同时使用，而且严格遵守先入先出的规则。

通过 channel，我们可以让多个 goroutine 之间的数据流动更加灵活、可控。比如，我们可以在某个 goroutine 中读取数据，在另一个 goroutine 中写入数据；或者，我们可以使多个 goroutine 执行同步操作，直到某个条件被满足后再继续；甚至还可以设置超时时间来控制协程之间的等待。

## 3.5 Widget

Widget 是图形用户界面（GUI）中的基本构件，是用户界面元素的最小单位。不同的 Widget 有不同的用法和行为，比如按钮、输入框、标签等。

在 Go 语言中，很多 GUI 框架都会封装相关的 Widget 为具体的类或结构体。这样，我们只需要调用这些类的方法，就可以利用 GUI 框架快速地生成一个完整的用户界面。

## 3.6 MVC 模式

MVC (Model-View-Controller) 模式是一种软件设计模式，它是从结构化编程中衍生出来的。它将软件系统分成三个层次，分别是模型（Model）、视图（View）和控制器（Controller）。

模型层负责管理应用程序的数据，处理业务逻辑；视图层负责显示数据，接收用户输入；而控制器层则负责建立连接，并调度模型和视图之间的交互。

在 Go 语言中，我们可以利用 MVC 模式来组织我们的 GUI 程序，将模型层和视图层解耦，并用消息传递的方式进行通信。

## 3.7 Handler

Handler 是 MVC 架构中的一个关键组件。它是 View 和 Controller 之间的联系纽带，负责处理用户的输入事件并更新 View 上显示的内容。

在 Go 语言中，handler 是作为回调函数类型定义的，它可以监听对应 View 的事件并做出相应的响应。

# 4.核心算法原理和具体操作步骤以及数学公式讲解

我们首先创建一个基本的 GUI 程序，它有一个文本输入框、一个按钮，点击按钮时弹出一个对话框，显示用户输入的内容。

1. 创建 TextView 对象

   ```
   package main
   
   import "github.com/zserge/webview"
   
   func main() {
       //创建 WebView 对象
       w := webview.New(true)
       
       //设置标题
       w.SetTitle("Hello World")
       
       //设置宽度高度
       w.SetSize(800, 600, webview.HintNone)
       
       //设置 URL
       w.Navigate("http://www.google.com")
       
       //设置本地加载 HTML 文件
       w.SetContent(`<html><body>
           <h1>Hello</h1>
           <input type="text">
           <button onclick="showDialog()">Submit</button>
           </body></html>`, "")
       
       //等待网页加载完成之后才显示窗口
       w.Run()
   }
   ```

   
2. 设置 onClickListener

   ```
   package main
   
   import "github.com/zserge/webview"
   
   func showDialog() {
       //获取文本输入框的值
       textInputValue = document.getElementById("textInput").value
       
       //显示对话框
       alert("Your input is: " + textInputValue);
   }
   
   func main() {
       //初始化 WebView
       w := webview.New(false)
   
      ...
       
       //设置按钮的点击事件
       w.Bind("showDialog", showDialog)
   
       //等待网页加载完成之后才显示窗口
       w.Run()
   }
   ```

3. 使用 MVC 模式

   ```
   package main
   
   import "github.com/zserge/webview"
   
   const (
       html = `<!DOCTYPE html>
               <html lang="en">
               <head>
                   <meta charset="UTF-8">
                   <title>My App</title>
               </head>
               <body>
                    <!-- 视图 -->
                   {{if.}}
                       <p>{{.}}</p>
                   {{end}}
                   <form id="myForm">
                       <label for="name">Name:</label>
                       <input type="text" id="name" name="name"><br>
                       <label for="age">Age:</label>
                       <input type="number" id="age" name="age"><br>
                       <input type="submit" value="Submit">
                   </form>
               </body>
               </html>`
       
       dialogHTML = `<html><body style="margin: 0px;">
                     <div style="padding: 10px; background-color: white; border-radius: 5px;">
                         <span>Are you sure?</span><br>
                         <button onclick="onYesClick()">Yes</button>&nbsp;&nbsp;<button onclick="onNoClick()">No</button>
                     </div>
                 </body></html>`
   
   )
   
   var (
       appWindow *webview.WebView
   )
   
   type myApp struct {
       message string
       form    map[string]interface{}
   }
   
   func newApp() *myApp {
       return &myApp{message: "", form: make(map[string]interface{})}
   }
   
   func onFormSubmit(event *js.Object) {
       event.PreventDefault()
       go submitForm()
   }
   
   func renderMessage(msg string) {
       jsStr := fmt.Sprintf("document.querySelector('p').innerHTML = '%s'", msg)
       appWindow.Eval(jsStr)
   }
   
   func onSubmitSuccess(response string) {
       app.message = response
       renderMessage(app.message)
   }
   
   func onSubmitError(err error) {
       app.message = err.Error()
       renderMessage(app.message)
   }
   
   func submitForm() {
       formData := make(url.Values)
       for k, v := range app.form {
           if _, ok := v.(int); ok {
               formData.Add(k, strconv.Itoa(v.(int)))
           } else {
               formData.Add(k, v.(string))
           }
       }
       reqData, _ := http.NewRequest("POST", "https://example.com/api", strings.NewReader(formData.Encode()))
       reqData.Header.Set("Content-Type", "application/x-www-form-urlencoded")
       client := http.Client{}
       resp, err := client.Do(reqData)
       if err!= nil {
           onSubmitError(err)
           return
       }
       bodyBytes, _ := ioutil.ReadAll(resp.Body)
       resp.Body.Close()
       var resultResp ResponseStruct
       json.Unmarshal(bodyBytes, &resultResp)
       onSubmitSuccess(resultResp.Message)
   }
   
   func onYesClick() {
       closeDialog()
   }
   
   func onNoClick() {
       cancelFormSubmission()
   }
   
   func init() {
       runtime.LockOSThread()
       appWindow = webview.New(false)
       defer appWindow.Destroy()
       appWindow.SetTitle("My App")
       appWindow.SetSize(800, 600, webview.HintNone)
       appWindow.Center()
       appWindow.Navigate("data:text/html,"+html)
       css := `html, body { margin: 0px; padding: 0px; width: 100%; height: 100%; }`
       appWindow.InjectCSS(css)
       jsFuncMap := map[string]interface{}{
           "cancelFormSubmission": func() {},
           "closeDialog":          func() {},
           "renderMessage":        renderMessage,
           "submitForm":           submitForm,
           "onFormSubmit":         onFormSubmit,
           "onSubmitSuccess":      onSubmitSuccess,
           "onSubmitError":        onSubmitError,
           "newApp":               newApp,
       }
       jsFuncs := js.Global().Get("Object").New()
       for name, f := range jsFuncMap {
           jsFuncs.Set(name, js.FuncOf(f))
       }
       jsStr := fmt.Sprintf("window.external=%s;", jsFuncs.Interface())
       appWindow.Eval(jsStr)
   }
   
   func main() {}
   ```

   
4. 关于 Go 和 WebView 的综合运用

   ```
   package main
   
   import (
       "encoding/json"
       "fmt"
       "io/ioutil"
       "net/http"
       "runtime"
       "strconv"
       "strings"
       "sync"
       "syscall/js"
       "time"
       
       "github.com/mattn/go-shellwords"
       "github.com/pkg/errors"
       "github.com/tidwall/gjson"
       "github.com/zserge/webview"
   )
   
   const (
       html       = `<html>
                      <head>
                          <script src="jquery-3.5.1.min.js"></script>
                          <style>
                              #container {
                                  display: flex;
                                  justify-content: center;
                                  align-items: center;
                                  height: 100vh;
                              }
                              
                              #loading {
                                  color: gray;
                                  font-size: 2em;
                              }
                              
                              #main {
                                  display: none;
                              }
                          </style>
                      </head>
                      <body>
                          <div id="container">
                              <div id="main">
                                  <h1>Welcome to My App!</h1>
                                  <ul class="list">
                                      <li data-id="1">Item 1</li>
                                      <li data-id="2">Item 2</li>
                                      <li data-id="3">Item 3</li>
                                      <li data-id="4">Item 4</li>
                                  </ul>
                                  <hr>
                                  <form id="myForm">
                                      <label for="name">Name:</label>
                                      <input type="text" id="name" name="name" required><br>
                                      <label for="email">Email:</label>
                                      <input type="email" id="email" name="email" pattern="[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$" required><br>
                                      <input type="checkbox" id="terms" name="terms" required> I accept the terms and conditions.<br>
                                      <button type="submit">Register</button>
                                  </form>
                              </div>
                          </div>
                          
                          <script>
                              $(function () {
                                  $("#myForm").submit(function (e) {
                                      e.preventDefault();
                                      $.ajax({
                                          url: "/register",
                                          method: "post",
                                          data: JSON.stringify($("form").serializeArray()),
                                          dataType: "json",
                                          contentType: "application/json",
                                          success: function (res) {
                                              console.log(res);
                                          },
                                          error: function (xhr, status, err) {
                                              console.error(status, err.toString());
                                          }
                                      });
                                  });
                                  $(".list li").click(function () {
                                      let id = $(this).attr("data-id");
                                      window.external.openLink(id);
                                  })
                              });
                              
                              setTimeout(() => {
                                  $("#loading").hide();
                                  $("#main").show();
                              }, 3000);
                              
                              window.addEventListener('message', receiveMessage, false);
                              
                              function receiveMessage(event) {
                                  console.log('Received message from child:', event.data);
                              }
                              
                              function openLink(linkId) {
                                  parent.postMessage(JSON.stringify({type: 'navigate', linkId}), '*');
                              }
                              
                              function callParentMethod(methodName, params) {
                                  parent.postMessage(JSON.stringify({type:'methodCall', methodName, params}), '*');
                              }
                          </script>
                      </body>
                  </html>`
   
       loadingGif = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
                        <circle cx="50" cy="50" r="45"/>
                        <animateTransform attributeName="transform" dur="1.5s" repeatCount="indefinite" keyTimes="0;1" values="rotate(0 50 50); rotate(360 50 50)" calcMode="discrete" />
                    </svg>`
   
   )
   
   type ResponseStruct struct {
       Message string `json:"message"`
   }
   
   type AjaxResult struct {
       Result   interface{} `json:"result"`
       ErrorCode int    `json:"errorCode"`
       ErrorMsg string `json:"errorMsg"`
   }
   
   type AsyncRequest struct {
       ID     uint32            `json:"id"`
       Method string            `json:"method"`
       Params []interface{}     `json:"params"`
       Done   chan<- AsyncResult `json:"-"`
   }
   
   type AsyncResult struct {
       ID     uint32       `json:"id"`
       Result interface{}  `json:"result"`
       Err    error        `json:"err,omitempty"`
       DoneAt time.Time    `json:"doneAt"`
       Acked  bool         `json:"acked"`
       Mutex  sync.RWMutex `json:"-" db:"mutex_async_result`
   }
   
   type AppData struct {
       FormElements []interface{} `json:"formElements"`
       DialogHTML   string        `json:"dialogHtml"`
       LoadingSVG   string        `json:"loadingSvg"`
       IsLoading    bool          `json:"isLoading"`
   }
   
   type RequestHandler struct {
       SyncMux      sync.RWMutex                     `json:"-" db:"mutex_sync_mux`
       NextAsyncID  uint32                           `json:"nextAsyncID"`
       AsyncResults map[uint32]*AsyncResult           `json:"asyncResults"`
       Data         *AppData                         `json:"data"`
       ChildWnd     *webview.WebView                 `json:"childWnd"`
       JSFunctions  map[string]func(...interface{}) `json:"jsFunctions"`
       ParentWnd    *webview.WebView                 `json:"parentWnd"`
       WndClosed    chan struct{}                    `json:"-"`
   }
   
   var (
       appHandler   *RequestHandler
       lastAsyncReq AsyncRequest
       logger       Logger
       appData      AppData
   )
   
   type Logger interface {
       Println(v...interface{})
   }
   
   func NewConsoleLogger() Logger {
       return log.New(os.Stderr, "[console]", log.LstdFlags|log.Lshortfile)
   }
   
   func NewJSONFileLogger(path string) (Logger, error) {
       file, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0600)
       if err!= nil {
           return nil, errors.Wrap(err, "failed to create log file")
       }
       return log.New(file, "", log.LstdFlags), nil
   }
   
   func StartServer(port int) (*webview.WebView, <-chan struct{}, error) {
       doneChan := make(chan struct{})
       server := http.Server{Addr: fmt.Sprintf(":%d", port)}
       mux := http.NewServeMux()
       mux.HandleFunc("/", handleRoot)
       mux.HandleFunc("/register", handleRegister)
       mux.Handle("/assets/", http.StripPrefix("/assets/", http.FileServer(http.Dir("./assets"))))
       mux.Handle("/jquery-3.5.1.min.js", http.StripPrefix("/jquery-3.5.1.min.js", http.FileServer(http.Dir("."))))
       mux.Handle("/loading.gif", http.StripPrefix("/loading.gif", http.HandlerFunc(handleLoadingGif)))
       mux.Handle("/dialog.html", http.StripPrefix("/dialog.html", http.HandlerFunc(handleDialogHTML)))
       mux.HandleFunc("/shutdown", handleShutdown)
       mux.HandleFunc("/healthcheck", handleHealthCheck)
       mux.HandleFunc("/favicon.ico", handleFavicon)
       mux.HandleFunc("/dialog.js", handleDialogJS)
       mux.HandleFunc("/dialog.css", handleDialogCSS)
       mux.HandleFunc("/callMethod", handleMethodCall)
       mux.Handle("/ws", websocket.Handler(handleWSConn))
       server.Handler = mux
       wg := &sync.WaitGroup{}
       wg.Add(1)
       go func() {
           defer wg.Done()
           err := server.ListenAndServe()
           if err!= nil &&!errors.Is(err, http.ErrServerClosed) {
               panic(err)
           }
           doneChan <- struct{}{}
       }()
       _, _, err := RunWebView()
       return appHandler.ChildWnd, doneChan, err
   }
   
   func ShutdownServer(stopSignal <-chan struct{}) error {
       ctx, cancelFn := context.WithCancel(context.Background())
       interruptCtx := signal.NotifyContext(ctx, syscall.SIGINT, syscall.SIGTERM)
       select {
       case sig := <-interruptCtx.Done():
           switch sig {
           case syscall.SIGINT, syscall.SIGTERM:
               logger.Println("received stop signal:", sig.String())
           default:
               logger.Printf("unexpected signal received: %s\n", sig.String())
           }
           cancelFn()
           time.AfterFunc(5*time.Second, func() {
               os.Exit(-1)
           })
       case <-stopSignal:
           logger.Println("received external stop signal")
           cancelFn()
           time.AfterFunc(5*time.Second, func() {
               os.Exit(-1)
           })
       }
       err := waitForChildToFinish(serverStartTimeout)
       if err == nil {
           serverStopCh <- struct{}{}
       }
       return err
   
   }
   
   func RunWebView() (*webview.WebView, <-chan struct{}, error) {
       appWindow = webview.New(false)
       defer appWindow.Destroy()
       appWindow.SetTitle("My App")
       appWindow.SetSize(1000, 800, webview.HintNone)
       appWindow.Center()
       appWindow.Dispatch(func() {
           initMainWnd()
       })
       doneChan := make(chan struct{})
       go func() {
           select {
           case <-doneChan:
               return
           case <-time.After(serverStartTimeout):
               logger.Printf("%ds elapsed while waiting for connection with frontend\n", serverStartTimeout/time.Second)
               os.Exit(-1)
            }
       }()
       return appWindow, doneChan, nil
   }
   
   func initMainWnd() {
       appData.FormElements = []interface{}{
           {"name": "name", "value": ""},
           {"name": "email", "value": ""},
           {"name": "terms", "value": true},
       }
       appData.DialogHTML = ""
       appData.LoadingSVG = ""
       appData.IsLoading = false
       setMainWndContents(appData)
   }
   
   func SetMainWindowContents(data AppData) {
       appData = data
       setMainWndContents(data)
   }
   
   func setMainWndContents(data AppData) {
       htmlStr := replaceTemplates(html, data)
       evalJSCode("$('#main').replaceWith(\""+htmlStr+"\")")
   }
   
   func ReplaceTemplates(src string, data AppData) string {
       return replaceTemplates(src, data)
   }
   
   func replaceTemplates(src string, data AppData) string {
       templateFuncs := map[string]interface{}{
           "escapeQuotes": escapeQuotes,
           "getOrDefaultBool": getOrDefaultBool,
           "getOrDefaultInt": getOrDefaultInt,
           "getParamByName": getParamByName,
           "toJson": toJson,
       }
       tpl, err := template.New("base").Funcs(templateFuncs).Parse(src)
       if err!= nil {
           logger.Printf("Failed to parse base template - %s\n", err.Error())
           return ""
       }
       buf := &bytes.Buffer{}
       err = tpl.Execute(buf, data)
       if err!= nil {
           logger.Printf("Failed to execute template with data - %s\n", err.Error())
           return ""
       }
       return buf.String()
   }
   
   func escapeQuotes(str string) string {
       str = strings.ReplaceAll(str, "'", "\\'")
       str = strings.ReplaceAll(str, "\"", "\\\"")
       return str
   }
   
   func getOrDefaultBool(arr []interface{}, idx int, defVal bool) bool {
       val, exists := arr[idx].(bool)
       if exists {
           return val
       }
       return defVal
   }
   
   func getOrDefaultInt(arr []interface{}, idx int, defVal int) int {
       val, exists := arr[idx].(float64)
       if exists {
           return int(val)
       }
       return defVal
   }
   
   func getParamByName(query string, name string) string {
       args, err := shellwords.Parse(query)
       if err!= nil {
           return ""
       }
       prefix := "--" + name + "="
       for i := len(args) - 1; i >= 0; i-- {
           arg := args[i]
           if strings.HasPrefix(arg, prefix) {
               return arg[len(prefix):]
           }
       }
       return ""
   }
   
   func toJson(obj interface{}) string {
       b, err := json.Marshal(obj)
       if err!= nil {
           return "{}"
       }
       return string(b)
   }
   
   func loadAssets(filename string) ([]byte, error) {
       filePath := filepath.Join(".", "assets", filename)
       content, err := ioutil.ReadFile(filePath)
       if err!= nil {
           return nil, errors.Wrapf(err, "failed to read asset \"%s\"", filename)
       }
       return content, nil
   }
   
   func sendResponse(writer http.ResponseWriter, resp interface{}) error {
       writer.Header().Set("Content-Type", "application/json")
       writer.WriteHeader(http.StatusOK)
       enc := json.NewEncoder(writer)
       return enc.Encode(&AjaxResult{
           Result:   resp,
           ErrorCode: 0,
           ErrorMsg: "",
       })
   }
   
   func handleRoot(writer http.ResponseWriter, request *http.Request) {
       if request.Method == http.MethodGet {
           htmlStr := ReplaceTemplates(html, appData)
           writeTextResponse(writer, htmlStr)
           return
       }
       writeJsonResponse(writer, HttpError{ErrorCode: 405, ErrorMsg: "Method Not Allowed"})
   }
   
   func handleRegister(writer http.ResponseWriter, request *http.Request) {
       if request.Method == http.MethodPost {
           var reqParams []interface{}
           if err := json.NewDecoder(request.Body).Decode(&reqParams); err!= nil {
               writeJsonResponse(writer, HttpError{ErrorCode: 500, ErrorMsg: err.Error()})
               return
           }
           asyncRes := startAsyncRequest(writer, "registerUser", reqParams)
           asyncRes.Acked = true
           return
       }
       writeJsonResponse(writer, HttpError{ErrorCode: 405, ErrorMsg: "Method Not Allowed"})
   }
   
   func startAsyncRequest(writer http.ResponseWriter, method string, params []interface{}) *AsyncResult {
       resChan := make(chan AsyncResult)
       requestId := atomic.AddUint32(&appHandler.NextAsyncID, 1)
       asyncReq := AsyncRequest{
           ID:     requestId,
           Method: method,
           Params: params,
           Done:   resChan,
       }
       appHandler.SyncMux.RLock()
       _, closed := appHandler.WndClosed
       appHandler.SyncMux.RUnlock()
       if closed {
           notifyAsyncFailure(requestId, errors.New("frontend not connected"), asyncRes)
           return nil
       }
       appHandler.AsyncResults[requestId] = asyncRes
       sendAsyncRequest(asyncReq)
       return asyncRes
   }
   
   func SendSyncRequest(method string, params []interface{}) (interface{}, error) {
       appHandler.SyncMux.RLock()
       _, closed := appHandler.WndClosed
       appHandler.SyncMux.RUnlock()
       if closed {
           return nil, errors.New("frontend not connected")
       }
       requestId := atomic.AddUint32(&appHandler.NextAsyncID, 1)
       asyncReq := AsyncRequest{
           ID:     requestId,
           Method: method,
           Params: params,
       }
       asyncRes := waitAsyncResponse(requestId)
       if asyncRes.Err!= nil {
           return nil, asyncRes.Err
       }
       return asyncRes.Result, nil
   }
   
   func WaitSyncRequests(reqs []*SyncRequest) error {
       for _, req := range reqs {
           res, err := SendSyncRequest(req.Method, req.Params)
           if err!= nil {
               return err
           }
           gjson.Valid(toJson(res)) // check if result is valid JSON object or array before saving it in a variable!
           req.Result = res
       }
       return nil
   }
   
   func HandleSyncRequests(reqs []*SyncRequest, timeout time.Duration) error {
       timer := time.NewTimer(timeout)
       defer timer.Stop()
       reqChan := make(chan *SyncRequest)
       for _, req := range reqs {
           reqChan <- req
       }
       go func() {
           defer close(reqChan)
           for req := range reqChan {
               res, err := SendSyncRequest(req.Method, req.Params)
               if err!= nil {
                   req.Err = err
                   continue
               }
               gjson.Valid(toJson(res)) // check if result is valid JSON object or array before saving it in a variable!
               req.Result = res
           }
       }()
       for {
           select {
           case <-timer.C:
               return errors.New("timeout exceeded")
           case req := <-reqChan:
               if req.Err!= nil {
                   return req.Err
               }
           }
       }
   }
   
   func handleLoadingGif(writer http.ResponseWriter, request *http.Request) {
       if request.Method == http.MethodGet {
           gifData, err := loadAssets("loading.gif")
           if err!= nil {
               writeJsonResponse(writer, HttpError{ErrorCode: 500, ErrorMsg: err.Error()})
               return
           }
           writer.Header().Set("Content-Type", "image/gif")
           writer.Write(gifData)
           return
       }
       writeJsonResponse(writer, HttpError{ErrorCode: 405, ErrorMsg: "Method Not Allowed"})
   }
   
   func handleDialogHTML(writer http.ResponseWriter, request *http.Request) {
       if request.Method == http.MethodGet {
           htmlData, err := loadAssets("dialog.html")
           if err!= nil {
               writeJsonResponse(writer, HttpError{ErrorCode: 500, ErrorMsg: err.Error()})
               return
           }
           dlgStr := ReplaceTemplates(string(htmlData), appData)
           writeTextResponse(writer, dlgStr)
           return
       }
       writeJsonResponse(writer, HttpError{ErrorCode: 405, ErrorMsg: "Method Not Allowed"})
   }
   
   func handleWSConn(conn *websocket.Conn) {
       conn.PayloadType = websocket.BinaryFrame
       session := wsSession{Writer: conn, Reader: conn, isConnected: true}
       sessions[session.Id()] = session
       trySendQueuedMessages(session.Id(), false)
       for {
           mt, payload, err := conn.ReadMessage()
           if err!= nil {
               delete(sessions, session.Id())
               break
           }
           processWsMessage(session, mt, payload)
       }
   }
   
   func trySendQueuedMessages(sessionId uint64, skipIdCheck bool) {
       sess := sessions[sessionId]
       sess.Lock()
       msgsLen := len(sess.queue)
       sess.Unlock()
       if msgsLen > 0 {
           sess.Lock()
           idsToRemove := make([]uint64, 0)
           sentIds := make(map[uint64]struct{})
           messagesToSend := make([]*WebsocketMessage, 0, msgsLen)
           for i := 0; i < msgsLen; i++ {
               msg := sess.queue[i]
               if _, exists := sentIds[msg.Id]; exists || (!skipIdCheck && sess.sentFirstMessageId == 0) {
                   continue
               }
               if msg.MsgType == MsgTypeNotification {
                   sess.lastNotificationSentTs = time.Now()
               }
               messagesToSend = append(messagesToSend, msg)
               sentIds[msg.Id] = struct{}{}
               sess.sentFirstMessageId = msg.Id
           }
           copy(sess.queue[:msgsLen], sess.queue[msgsLen:])
           sess.queue = sess.queue[:len(sess.queue)-msgsLen]
           sess.Unlock()
           for _, msg := range messagesToSend {
               sendMessage(sess, msg)
           }
           idsToRemove = removeExpiredQueuedMessages(idsToRemove)
           if len(idsToRemove) > 0 {
               sess.Lock()
               deleteIdsFromQueue(sess, idsToRemove)
               sess.Unlock()
           }
       }
   }
   
   func removeExpiredQueuedMessages(idsToRemove []uint64) []uint64 {
       nowTs := time.Now()
       sessMapCopy := make(map[uint64]*wsSession)
       for sessionId, sess := range sessions {
           sess.Lock()
           if len(sess.queue) == 0 && ((nowTs.Sub(sess.lastActivityTs)).Seconds() > maxInactivitySec || (nowTs.Sub(sess.lastNotificationSentTs)).Seconds() > notificationTimeoutSec) {
               delete(sessions, sessionId)
               idsToRemove = append(idsToRemove, sessionId)
               continue
           }
           sessMapCopy[sessionId] = sess
           sess.Unlock()
       }
       for sessionId, sess := range sessMapCopy {
           trySendQueuedMessages(sessionId, true)
       }
       return idsToRemove
   
   }
   
   func deleteIdsFromQueue(sess *wsSession, idsToRemove []uint64) {
       queueLen := len(sess.queue)
       for i := queueLen - 1; i >= 0; i-- {
           msg := sess.queue[i]
           for _, id := range idsToRemove {
               if id == msg.Id {
                   copy(sess.queue[i:], sess.queue[i+1:])
                   sess.queue = sess.queue[:queueLen-1]
                   queueLen -= 1
                   i--
                   break
               }
           }
       }
   }