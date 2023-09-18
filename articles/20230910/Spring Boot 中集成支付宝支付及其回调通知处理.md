
作者：禅与计算机程序设计艺术                    

# 1.简介
  

## 1.1什么是Spring Boot？
> Spring Boot 是由 Pivotal 团队提供的全新框架，其设计目的是用来简化新 Spring 应用的初始搭建以及开发过程。该框架使用了特定的方式来进行配置，从而使开发人员不再需要定义样板化的 XML 文件。通过 starter POMs 来增加依赖关系并自动设置属性，Spring Boot 可以快速的为多种开发环境提供一致的初始设置。在 Spring Boot 的帮助下，开发人员可以更加关注于实现业务需求，而不需要花费过多的时间去配置各种各样的框架组件。
> Spring Boot 是一个开源框架，基于 Spring Framework 之上，目标是做到简单、轻量级且易于使用，并在保持传统 Spring 框架核心特性的同时提升系统的可移植性和扩展能力。
## 1.2为什么要学习 Spring Boot 和支付宝？
虽然 Spring Boot 已经成为主流的 Java Web 开发框架，但如果想利用支付宝进行支付的话，就不能直接使用 Spring Boot 自带的支付接口，还需要添加一些额外的配置才能实现对接。因此，了解 Spring Boot 中的相关配置和支付宝的 API 对理解 Spring Boot 的整体架构和功能非常重要。另外，Spring Boot 提供的 starter POM 可以极大的简化项目的集成，让开发者只需要配置少量的 Spring Boot 配置参数即可快速实现对接支付宝支付接口。
# 2.Spring Boot中的支付宝支付
## 2.1集成支付宝支付接口
首先，我们需要按照 Spring Boot 的脚手架快速生成一个 Spring Boot 项目工程。然后，我们在pom文件中加入支付宝的依赖：
```xml
        <dependency>
            <groupId>com.alipay</groupId>
            <artifactId>alipay-sdk-java</artifactId>
            <version>4.1.9</version>
        </dependency>
```
这样，Spring Boot 项目中就可以使用支付宝的相关接口了。接着，我们在 Spring Boot 的配置文件 application.properties 中配置好支付宝的相关信息：
```yaml
alipay:
  app_id: your_app_id    # APPID
  notify_url: http://yourdomain/alipay/notify   # 异步通知地址
  return_url: http://yourdomain/alipay/return   # 同步通知地址（仅扫码支付有效）
  sign_type: RSA2     # 签名方式
  charset: UTF-8     # 字符编码
  gatewayUrl: https://openapi.alipay.com/gateway.do  # 支付宝网关地址（根据沙箱环境还是正式环境修改）
```
这里，我们只是简单地配置了支付宝的基本信息，但是并没有完成整个流程。由于 Spring Boot 启动的时候，默认会扫描指定的包路径下的文件，因此我们需要创建好支付的 Controller 类，并把请求路由到此处：
```java
@RestController
public class PaymentController {
    @Autowired
    private AlipayClient alipayClient;

    /**
     * 创建支付订单
     */
    @RequestMapping("/pay")
    public String pay(@RequestParam("orderNo") String orderNo) throws Exception {
        // 创建 AlipayTradeQueryModel 对象
        AlipayTradeQueryModel model = new AlipayTradeQueryModel();
        model.setOutTradeNo(orderNo);

        // 执行查询订单
        AlipayTradeQueryResponse response = alipayClient.execute(AlipayRequestFactory.createTradeQueryRequest(model));

        // 根据结果判断是否支付成功
        if (response.isSuccess()) {
            return "支付成功";
        } else {
            return "支付失败";
        }
    }
}
```
至此，支付接口集成完毕。但是，当用户点击支付按钮时，仍然需要跳转到支付宝页面进行付款。所以，我们需要在前端添加一个支付按钮，并且调用支付接口来获取支付链接，然后将这个链接传递给客户端。前端的代码如下所示：
```html
<form id="alipayForm" action="${paymentUrl}" method="POST">
    <input type="hidden" name="outTradeNo" value="${orderNo}">
</form>

<button onclick="document.getElementById('alipayForm').submit()">支付</button>
```
`${paymentUrl}` 表示后端的支付接口 URL，`${orderNo}` 表示当前订单号，最终提交表单后，用户就会被重定向到支付宝进行付款。
## 2.2支付接口返回结果的处理
当用户付款成功或失败之后，支付宝会发送异步通知或者同步通知给我们。我们需要接收这些通知并处理，处理的方法就是编写处理通知的 controller 方法。
```java
@RestController
public class NotifyController {
    @Autowired
    private AlipayConfig config;

    /**
     * 支付宝异步通知接口
     */
    @PostMapping("/alipay/notify")
    public ResponseEntity handleNotify(@RequestBody String body) throws Exception {
        // 获取支付宝请求参数对象
        Map<String, String[]> params = HttpUtil.parseQueryString(body);

        // 将参数转化为 Map 对象方便验证签名
        SortedMap<Object, Object> sortedParams = new TreeMap<>(params);
        
        // 验证签名是否正确
        boolean success = AlipaySignature.rsaCheckV1(sortedParams,
                config.getAliPayPublicKey(), 
                "RSA2");

        if (success) {
            // TODO：处理支付成功逻辑
            System.out.println("支付成功！");
            return ResponseEntity.ok().build();
        } else {
            // TODO：处理支付失败逻辑
            System.out.println("支付失败！");
            return ResponseEntity.badRequest().build();
        }
    }
}
```
这里，我们假设支付成功逻辑和支付失败逻辑都是一样的，只是打印了一句日志消息不同而已。注意，这里我们也用了一个 `AlipayConfig` 类来管理支付宝相关的配置信息，包括支付宝公钥等。
# 3.Spring Boot 中的支付宝支付回调
Spring Boot 中的支付宝支付回调处理非常类似于之前集成支付宝支付接口时的处理方式。主要区别在于，这种情况下，我们不能跳转到支付宝的付款界面，因为它是用户浏览器中的第三方应用，用户无法感知到这个操作。因此，一般来说，支付回调的方式分为两种：
* 同步通知回调
* 异步通知回调
## 3.1同步通知回调
同步通知即，服务器直接把响应结果通知给客户端，需要客户端主动轮询服务器获取结果。当用户点击支付按钮时，浏览器会发起 POST 请求到服务端，而这次请求是在用户浏览器中发出的，浏览器同样需要收到服务器的响应结果，才知道支付是否成功，也就是说，这个过程需要双方的协商，所以比较耗时。

如何实现同步通知回调呢？跟之前一样，先在 Spring Boot 的配置文件 application.properties 中配置好支付宝的回调地址：
```yaml
alipay:
 ...
  notify_url: ${host}/api/alipay/syncCallback    # 同步通知地址
  return_url: http://yourdomain/alipay/return        # 同步通知地址
```
`${host}` 表示当前 Spring Boot 服务的主机地址。

然后，我们需要创建一个支付控制器类，并且添加一个同步通知回调方法：
```java
@RestController
public class AliPayController {
    
    /**
     * 同步通知回调方法
     */
    @PostMapping("/api/alipay/syncCallback")
    public ResponseEntity callback() {
        // TODO：处理支付通知结果
        
        // 返回响应
        return ResponseEntity.ok().build();
    }
    
    /**
     * 支付宝同步通知方法
     */
    @GetMapping("/api/alipay/syncReturn/{orderId}")
    public String syncReturn(@PathVariable String orderId) {
        // 查询订单状态并更新订单数据库
        // TODO：...
        
        // 返回支付结果页面
        return "<html><head><title>支付结果</title></head>" +
               "<body>支付成功！<br/><a href='http://www.baidu.com'>继续购物</a></body></html>";
    }
}
```
这里，我们简单的在 `/api/alipay/syncCallback` 下面创建了一个同步通知回调方法，并且也定义了一个支付成功后的返回页面。而支付成功后的同步通知回调，则是在支付宝支付成功后跳转到 `${host}/api/alipay/syncReturn/${orderId}` 页面进行查询订单结果的显示。

## 3.2异步通知回调
异步通知即，服务器主动推送通知给客户端，无需客户端主动轮询。服务器把响应结果通过 HTTP 报文的形式发送给客户端，客户端通过注册到支付宝商户平台上的回调地址进行监听，监听到消息之后就会得到相应的通知。异步通知的方式比同步通知的方式效率高很多。

如何实现异步通知回调呢？我们可以在 Spring Boot 的配置文件 application.properties 中配置好支付宝的异步通知地址：
```yaml
alipay:
 ...
  notify_url: ${host}/api/alipay/asyncCallback  # 异步通知地址
  return_url: http://yourdomain/alipay/return      # 同步通知地址
```
`${host}` 表示当前 Spring Boot 服务的主机地址。

然后，我们需要创建一个支付控制器类，并且添加一个异步通知回调方法：
```java
@RestController
public class AliPayController {
    
    /**
     * 异步通知回调方法
     */
    @PostMapping("/api/alipay/asyncCallback")
    public ResponseEntity asyncCallback() {
        // TODO：处理支付通知结果
        
        // 返回响应
        return ResponseEntity.ok().build();
    }
}
```
这里，我们只是简单的定义了一个异步通知回调方法。

至此，Spring Boot 中的支付宝支付回调接口就全部介绍完毕了。