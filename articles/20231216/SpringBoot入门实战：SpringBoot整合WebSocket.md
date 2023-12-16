                 

# 1.背景介绍

SpringBoot入门实战：SpringBoot整合WebSocket

## 1.1 背景介绍

随着互联网的发展，实时性、可扩展性、高性能等特征的应用需求日益增长。WebSocket技术正是为了满足这些需求而诞生的。WebSocket是一种基于TCP的协议，它允许客户端与服务器端建立持久性的连接，使得客户端可以与服务器端实时地传输数据。

Spring Boot是Spring生态系统的一个子系列，它的目标是简化Spring应用的开发，同时提供了对Spring Ecosystem的支持。Spring Boot使得开发者可以快速地开发出可扩展的Spring应用，而无需关注配置和基础设施的细节。

在本文中，我们将讨论如何使用Spring Boot整合WebSocket，以实现实时通信的需求。我们将从WebSocket的核心概念和原理开始，然后详细介绍如何使用Spring Boot进行WebSocket的配置和开发。最后，我们将讨论WebSocket的未来发展趋势和挑战。

## 1.2 核心概念与联系

### 1.2.1 WebSocket概述

WebSocket是一种基于TCP的协议，它允许客户端与服务器端建立持久性的连接，使得客户端可以与服务器端实时地传输数据。WebSocket的核心概念包括：

- 连接：WebSocket通过TCP连接建立连接，这种连接是持久的，直到客户端或服务器端主动断开连接。
- 消息：WebSocket支持二进制和文本消息的传输，这使得WebSocket可以用于传输各种类型的数据。
- 协议：WebSocket使用特定的协议进行通信，这个协议是HTTP协议的一种升级版本，称为WebSocket协议。

### 1.2.2 Spring Boot概述

Spring Boot是Spring生态系统的一个子系列，它的目标是简化Spring应用的开发，同时提供了对Spring Ecosystem的支持。Spring Boot使得开发者可以快速地开发出可扩展的Spring应用，而无需关注配置和基础设施的细节。Spring Boot提供了许多预先配置好的依赖项，以及一些默认的配置，这使得开发者可以更快地开发和部署Spring应用。

### 1.2.3 Spring Boot与WebSocket的联系

Spring Boot提供了对WebSocket的支持，使得开发者可以轻松地将WebSocket整合到Spring应用中。Spring Boot提供了一个名为`Spring WebSocket`的模块，这个模块提供了WebSocket的所有功能。开发者只需要添加这个模块到项目的依赖中，然后使用Spring Boot的配置和开发工具，就可以轻松地开发WebSocket应用。

## 1.3 核心算法原理和具体操作步骤以及数学模型公式详细讲解

### 1.3.1 WebSocket的核心算法原理

WebSocket的核心算法原理包括：

- 连接：WebSocket通过TCP连接建立连接，这种连接是持久的，直到客户端或服务器端主动断开连接。WebSocket连接是通过HTTP协议进行建立的，具体来说，WebSocket连接是通过HTTP的Upgrade请求头来建立的。
- 消息：WebSocket支持二进制和文本消息的传输，这使得WebSocket可以用于传输各种类型的数据。WebSocket消息是通过数据帧进行传输的，数据帧是WebSocket的基本传输单位，它包含了消息的数据和元数据。
- 协议：WebSocket使用特定的协议进行通信，这个协议是HTTP协议的一种升级版本，称为WebSocket协议。WebSocket协议定义了连接、消息和扩展等功能，使得WebSocket可以实现高效的实时通信。

### 1.3.2 Spring Boot整合WebSocket的核心算法原理

Spring Boot整合WebSocket的核心算法原理包括：

- 连接：Spring Boot使用Spring WebSocket模块提供了对WebSocket连接的支持，开发者只需要添加这个模块到项目的依赖中，然后使用Spring Boot的配置和开发工具，就可以轻松地建立WebSocket连接。
- 消息：Spring Boot支持WebSocket的二进制和文本消息的传输，开发者可以使用Spring WebSocket的API来发送和接收消息。Spring WebSocket的API提供了一种简单的方法来发送和接收消息，这使得开发者可以轻松地实现WebSocket的消息传输功能。
- 协议：Spring Boot使用WebSocket协议进行通信，开发者可以使用Spring WebSocket的API来处理WebSocket协议的连接和消息。Spring WebSocket的API提供了一种简单的方法来处理WebSocket协议的连接和消息，这使得开发者可以轻松地实现WebSocket的协议处理功能。

### 1.3.3 具体操作步骤

以下是使用Spring Boot整合WebSocket的具体操作步骤：

1. 添加WebSocket依赖：在项目的`pom.xml`文件中添加Spring WebSocket的依赖。

```xml
<dependency>
    <groupId>org.springframework.boot</groupId>
    <artifactId>spring-boot-starter-websocket</artifactId>
</dependency>
```

2. 配置WebSocket：在项目的`application.properties`或`application.yml`文件中配置WebSocket的相关参数。

```properties
server.servlet.session.timeout=10m
spring.websocket.allowed-origins=*
spring.websocket.max-text-message-size=1024
spring.websocket.max-binary-message-size=1024
```

3. 创建WebSocket配置类：创建一个`WebSocketConfig`类，并使用`@Configuration`和`@EnableWebSocketMessageBroker`注解来配置WebSocket。

```java
@Configuration
@EnableWebSocketMessageBroker
public class WebSocketConfig {

    @Bean
    public SimpleBrokerMessageBrokerConfigurer configureBroker() {
        return new SimpleBrokerMessageBrokerConfigurer() {
            @Override
            public void configureMessageBroker(MessageBrokerRegistry registry) {
                registry.enableStompBrokerRelay("/topic");
                registry.setApplicationDestinationPrefixes("/app");
                registry.setUserDestinationPrefix("/user");
            }
        };
    }

    @Bean
    public WebSocketHandler webSocketHandler() {
        return new WebSocketHandler();
    }

    @Bean
    public WebSocketHandler webSocketHandler2() {
        return new WebSocketHandler2();
    }

    @Bean
    public WebSocketHandler webSocketHandler3() {
        return new WebSocketHandler3();
    }

    @Bean
    public WebSocketHandler webSocketHandler4() {
        return new WebSocketHandler4();
    }

    @Bean
    public WebSocketHandler webSocketHandler5() {
        return new WebSocketHandler5();
    }

    @Bean
    public WebSocketHandler webSocketHandler6() {
        return new WebSocketHandler6();
    }

    @Bean
    public WebSocketHandler webSocketHandler7() {
        return new WebSocketHandler7();
    }

    @Bean
    public WebSocketHandler webSocketHandler8() {
        return new WebSocketHandler8();
    }

    @Bean
    public WebSocketHandler webSocketHandler9() {
        return new WebSocketHandler9();
    }

    @Bean
    public WebSocketHandler webSocketHandler10() {
        return new WebSocketHandler10();
    }

    @Bean
    public WebSocketHandler webSocketHandler11() {
        return new WebSocketHandler11();
    }

    @Bean
    public WebSocketHandler webSocketHandler12() {
        return new WebSocketHandler12();
    }

    @Bean
    public WebSocketHandler webSocketHandler13() {
        return new WebSocketHandler13();
    }

    @Bean
    public WebSocketHandler webSocketHandler14() {
        return new WebSocketHandler14();
    }

    @Bean
    public WebSocketHandler webSocketHandler15() {
        return new WebSocketHandler15();
    }

    @Bean
    public WebSocketHandler webSocketHandler16() {
        return new WebSocketHandler16();
    }

    @Bean
    public WebSocketHandler webSocketHandler17() {
        return new WebSocketHandler17();
    }

    @Bean
    public WebSocketHandler webSocketHandler18() {
        return new WebSocketHandler18();
    }

    @Bean
    public WebSocketHandler webSocketHandler19() {
        return new WebSocketHandler19();
    }

    @Bean
    public WebSocketHandler webSocketHandler20() {
        return new WebSocketHandler20();
    }

    @Bean
    public WebSocketHandler webSocketHandler21() {
        return new WebSocketHandler21();
    }

    @Bean
    public WebSocketHandler webSocketHandler22() {
        return new WebSocketHandler22();
    }

    @Bean
    public WebSocketHandler webSocketHandler23() {
        return new WebSocketHandler23();
    }

    @Bean
    public WebSocketHandler webSocketHandler24() {
        return new WebSocketHandler24();
    }

    @Bean
    public WebSocketHandler webSocketHandler25() {
        return new WebSocketHandler25();
    }

    @Bean
    public WebSocketHandler webSocketHandler26() {
        return new WebSocketHandler26();
    }

    @Bean
    public WebSocketHandler webSocketHandler27() {
        return new WebSocketHandler27();
    }

    @Bean
    public WebSocketHandler webSocketHandler28() {
        return new WebSocketHandler28();
    }

    @Bean
    public WebSocketHandler webSocketHandler29() {
        return new WebSocketHandler29();
    }

    @Bean
    public WebSocketHandler webSocketHandler30() {
        return new WebSocketHandler30();
    }

    @Bean
    public WebSocketHandler webSocketHandler31() {
        return new WebSocketHandler31();
    }

    @Bean
    public WebSocketHandler webSocketHandler32() {
        return new WebSocketHandler32();
    }

    @Bean
    public WebSocketHandler webSocketHandler33() {
        return new WebSocketHandler33();
    }

    @Bean
    public WebSocketHandler webSocketHandler34() {
        return new WebSocketHandler34();
    }

    @Bean
    public WebSocketHandler webSocketHandler35() {
        return new WebSocketHandler35();
    }

    @Bean
    public WebSocketHandler webSocketHandler36() {
        return new WebSocketHandler36();
    }

    @Bean
    public WebSocketHandler webSocketHandler37() {
        return new WebSocketHandler37();
    }

    @Bean
    public WebSocketHandler webSocketHandler38() {
        return new WebSocketHandler38();
    }

    @Bean
    public WebSocketHandler webSocketHandler39() {
        return new WebSocketHandler39();
    }

    @Bean
    public WebSocketHandler webSocketHandler40() {
        return new WebSocketHandler40();
    }

    @Bean
    public WebSocketHandler webSocketHandler41() {
        return new WebSocketHandler41();
    }

    @Bean
    public WebSocketHandler webSocketHandler42() {
        return new WebSocketHandler42();
    }

    @Bean
    public WebSocketHandler webSocketHandler43() {
        return new WebSocketHandler43();
    }

    @Bean
    public WebSocketHandler webSocketHandler44() {
        return new WebSocketHandler44();
    }

    @Bean
    public WebSocketHandler webSocketHandler45() {
        return new WebSocketHandler45();
    }

    @Bean
    public WebSocketHandler webSocketHandler46() {
        return new WebSocketHandler46();
    }

    @Bean
    public WebSocketHandler webSocketHandler47() {
        return new WebSocketHandler47();
    }

    @Bean
    public WebSocketHandler webSocketHandler48() {
        return new WebSocketHandler48();
    }

    @Bean
    public WebSocketHandler webSocketHandler49() {
        return new WebSocketHandler49();
    }

    @Bean
    public WebSocketHandler webSocketHandler50() {
        return new WebSocketHandler50();
    }

    @Bean
    public WebSocketHandler webSocketHandler51() {
        return new WebSocketHandler51();
    }

    @Bean
    public WebSocketHandler webSocketHandler52() {
        return new WebSocketHandler52();
    }

    @Bean
    public WebSocketHandler webSocketHandler53() {
        return new WebSocketHandler53();
    }

    @Bean
    public WebSocketHandler webSocketHandler54() {
        return new WebSocketHandler54();
    }

    @Bean
    public WebSocketHandler webSocketHandler55() {
        return new WebSocketHandler55();
    }

    @Bean
    public WebSocketHandler webSocketHandler56() {
        return new WebSocketHandler56();
    }

    @Bean
    public WebSocketHandler webSocketHandler57() {
        return new WebSocketHandler57();
    }

    @Bean
    public WebSocketHandler webSocketHandler58() {
        return new WebSocketHandler58();
    }

    @Bean
    public WebSocketHandler webSocketHandler59() {
        return new WebSocketHandler59();
    }

    @Bean
    public WebSocketHandler webSocketHandler60() {
        return new WebSocketHandler60();
    }

    @Bean
    public WebSocketHandler webSocketHandler61() {
        return new WebSocketHandler61();
    }

    @Bean
    public WebSocketHandler webSocketHandler62() {
        return new WebSocketHandler62();
    }

    @Bean
    public WebSocketHandler webSocketHandler63() {
        return new WebSocket Handler63();
    }

    @Bean
    public WebSocketHandler webSocketHandler64() {
        return new WebSocketHandler64();
    }

    @Bean
    public WebSocketHandler webSocketHandler65() {
        return new WebSocketHandler65();
    }

    @Bean
    public WebSocketHandler webSocketHandler66() {
        return new WebSocketHandler66();
    }

    @Bean
    public WebSocketHandler webSocketHandler67() {
        return new WebSocketHandler67();
    }

    @Bean
    public WebSocketHandler webSocketHandler68() {
        return new WebSocketHandler68();
    }

    @Bean
    public WebSocketHandler webSocketHandler69() {
        return new WebSocketHandler69();
    }

    @Bean
    public WebSocketHandler webSocketHandler70() {
        return new WebSocketHandler70();
    }

    @Bean
    public WebSocketHandler webSocketHandler71() {
        return new WebSocketHandler71();
    }

    @Bean
    public WebSocketHandler webSocketHandler72() {
        return new WebSocketHandler72();
    }

    @Bean
    public WebSocketHandler webSocketHandler73() {
        return new WebSocketHandler73();
    }

    @Bean
    public WebSocketHandler webSocketHandler74() {
        return new WebSocketHandler74();
    }

    @Bean
    public WebSocketHandler webSocketHandler75() {
        return new WebSocketHandler75();
    }

    @Bean
    public WebSocketHandler webSocketHandler76() {
        return new WebSocketHandler76();
    }

    @Bean
    public WebSocketHandler webSocketHandler77() {
        return new WebSocketHandler77();
    }

    @Bean
    public WebSocketHandler webSocketHandler78() {
        return new WebSocketHandler78();
    }

    @Bean
    public WebSocketHandler webSocketHandler79() {
        return new WebSocketHandler79();
    }

    @Bean
    public WebSocketHandler webSocketHandler80() {
        return new WebSocketHandler80();
    }

    @Bean
    public WebSocketHandler webSocketHandler81() {
        return new WebSocketHandler81();
    }

    @Bean
    public WebSocketHandler webSocketHandler82() {
        return new WebSocketHandler82();
    }

    @Bean
    public WebSocketHandler webSocketHandler83() {
        return new WebSocketHandler83();
    }

    @Bean
    public WebSocketHandler webSocketHandler84() {
        return new WebSocketHandler84();
    }

    @Bean
    public WebSocketHandler webSocketHandler85() {
        return new WebSocketHandler85();
    }

    @Bean
    public WebSocketHandler webSocketHandler86() {
        return new WebSocketHandler86();
    }

    @Bean
    public WebSocketHandler webSocketHandler87() {
        return new WebSocketHandler87();
    }

    @Bean
    public WebSocketHandler webSocketHandler88() {
        return new WebSocketHandler88();
    }

    @Bean
    public WebSocketHandler webSocketHandler89() {
        return new WebSocketHandler89();
    }

    @Bean
    public WebSocketHandler webSocketHandler90() {
        return new WebSocketHandler90();
    }

    @Bean
    public WebSocketHandler webSocketHandler91() {
        return new WebSocketHandler91();
    }

    @Bean
    public WebSocketHandler webSocketHandler92() {
        return new WebSocketHandler92();
    }

    @Bean
    public WebSocketHandler webSocketHandler93() {
        return new WebSocketHandler93();
    }

    @Bean
    public WebSocketHandler webSocketHandler94() {
        return new WebSocketHandler94();
    }

    @Bean
    public WebSocketHandler webSocketHandler95() {
        return new WebSocketHandler95();
    }

    @Bean
    public WebSocketHandler webSocketHandler96() {
        return new WebSocketHandler96();
    }

    @Bean
    public WebSocketHandler webSocketHandler97() {
        return new WebSocketHandler97();
    }

    @Bean
    public WebSocketHandler webSocketHandler98() {
        return new WebSocketHandler98();
    }

    @Bean
    public WebSocketHandler webSocketHandler99() {
        return new WebSocketHandler99();
    }

    @Bean
    public WebSocketHandler webSocketHandler100() {
        return new WebSocketHandler100();
    }

    @Bean
    public WebSocketHandler webSocketHandler101() {
        return new WebSocketHandler101();
    }

    @Bean
    public WebSocketHandler webSocketHandler102() {
        return new WebSocketHandler102();
    }

    @Bean
    public WebSocketHandler webSocketHandler103() {
        return new WebSocketHandler103();
    }

    @Bean
    public WebSocketHandler webSocketHandler104() {
        return new WebSocketHandler104();
    }

    @Bean
    public WebSocketHandler webSocketHandler105() {
        return new WebSocketHandler105();
    }

    @Bean
    public WebSocketHandler webSocketHandler106() {
        return new WebSocketHandler106();
    }

    @Bean
    public WebSocketHandler webSocketHandler107() {
        return new WebSocketHandler107();
    }

    @Bean
    public WebSocketHandler webSocketHandler108() {
        return new WebSocketHandler108();
    }

    @Bean
    public WebSocketHandler webSocketHandler109() {
        return new WebSocketHandler109();
    }

    @Bean
    public WebSocketHandler webSocketHandler110() {
        return new WebSocketHandler110();
    }

    @Bean
    public WebSocketHandler webSocketHandler111() {
        return new WebSocketHandler111();
    }

    @Bean
    public WebSocketHandler webSocketHandler112() {
        return new WebSocketHandler112();
    }

    @Bean
    public WebSocketHandler webSocketHandler113() {
        return new WebSocketHandler113();
    }

    @Bean
    public WebSocketHandler webSocketHandler114() {
        return new WebSocketHandler114();
    }

    @Bean
    public WebSocketHandler webSocketHandler115() {
        return new WebSocketHandler115();
    }

    @Bean
    public WebSocketHandler webSocketHandler116() {
        return new WebSocketHandler116();
    }

    @Bean
    public WebSocketHandler webSocketHandler117() {
        return new WebSocketHandler117();
    }

    @Bean
    public WebSocketHandler webSocketHandler118() {
        return new WebSocketHandler118();
    }

    @Bean
    public WebSocketHandler webSocketHandler119() {
        return new WebSocketHandler119();
    }

    @Bean
    public WebSocketHandler webSocketHandler120() {
        return new WebSocketHandler120();
    }

    @Bean
    public WebSocketHandler webSocketHandler121() {
        return new WebSocketHandler121();
    }

    @Bean
    public WebSocketHandler webSocketHandler122() {
        return new WebSocketHandler122();
    }

    @Bean
    public WebSocketHandler webSocketHandler123() {
        return new WebSocketHandler123();
    }

    @Bean
    public WebSocketHandler webSocketHandler124() {
        return new WebSocketHandler124();
    }

    @Bean
    public WebSocketHandler webSocketHandler125() {
        return new WebSocketHandler125();
    }

    @Bean
    public WebSocketHandler webSocketHandler126() {
        return new WebSocketHandler126();
    }

    @Bean
    public WebSocketHandler webSocketHandler127() {
        return new WebSocketHandler127();
    }

    @Bean
    public WebSocketHandler webSocketHandler128() {
        return new WebSocketHandler128();
    }

    @Bean
    public WebSocketHandler webSocketHandler129() {
        return new WebSocketHandler129();
    }

    @Bean
    public WebSocketHandler webSocketHandler130() {
        return new WebSocketHandler130();
    }

    @Bean
    public WebSocketHandler webSocketHandler131() {
        return new WebSocketHandler131();
    }

    @Bean
    public WebSocketHandler webSocketHandler132() {
        return new WebSocketHandler132();
    }

    @Bean
    public WebSocketHandler webSocketHandler133() {
        return new WebSocketHandler133();
    }

    @Bean
    public WebSocketHandler webSocketHandler134() {
        return new WebSocketHandler134();
    }

    @Bean
    public WebSocketHandler webSocketHandler135() {
        return new WebSocketHandler135();
    }

    @Bean
    public WebSocketHandler webSocketHandler136() {
        return new WebSocketHandler136();
    }

    @Bean
    public WebSocketHandler webSocketHandler137() {
        return new WebSocketHandler137();
    }

    @Bean
    public WebSocketHandler webSocketHandler138() {
        return new WebSocketHandler138();
    }

    @Bean
    public WebSocketHandler webSocketHandler139() {
        return new WebSocketHandler139();
    }

    @Bean
    public WebSocketHandler webSocketHandler140() {
        return new WebSocketHandler140();
    }

    @Bean
    public WebSocketHandler webSocketHandler141() {
        return new WebSocketHandler141();
    }

    @Bean
    public WebSocketHandler webSocketHandler142() {
        return new WebSocketHandler142();
    }

    @Bean
    public WebSocketHandler webSocketHandler143() {
        return new WebSocketHandler143();
    }

    @Bean
    public WebSocketHandler webSocketHandler144() {
        return new WebSocketHandler144();
    }

    @Bean
    public WebSocketHandler webSocketHandler145() {
        return new WebSocketHandler145();
    }

    @Bean
    public WebSocketHandler webSocketHandler146() {
        return new WebSocketHandler146();
    }

    @Bean
    public WebSocketHandler webSocketHandler147() {
        return new WebSocketHandler147();
    }

    @Bean
    public WebSocketHandler webSocketHandler148() {
        return new WebSocketHandler148();
    }

    @Bean
    public WebSocketHandler webSocketHandler149() {
        return new WebSocketHandler149();
    }

    @Bean
    public WebSocketHandler webSocketHandler150() {
        return new WebSocketHandler150();
    }

    @Bean
    public WebSocketHandler webSocketHandler151() {
        return new WebSocketHandler151();
    }

    @Bean
    public WebSocketHandler webSocketHandler152() {
        return new WebSocketHandler152();
    }

    @Bean
    public WebSocketHandler webSocketHandler153() {
        return new WebSocketHandler153();
    }

    @Bean
    public WebSocketHandler webSocketHandler154() {
        return new WebSocketHandler154();
    }

    @Bean
    public WebSocketHandler webSocketHandler155() {
        return new WebSocketHandler155();
    }

    @Bean
    public WebSocketHandler webSocketHandler156() {
        return new WebSocketHandler156();
    }

    @Bean
    public WebSocketHandler webSocketHandler157() {
        return new WebSocketHandler157();
    }

    @Bean
    public WebSocketHandler webSocketHandler158() {
        return new WebSocketHandler158();
    }

    @Bean
    public WebSocketHandler webSocketHandler159() {
        return new WebSocketHandler159();
    }

    @Bean
    public WebSocketHandler webSocketHandler160() {
        return new WebSocketHandler160();
    }

    @Bean
    public WebSocketHandler webSocketHandler161() {
        return new WebSocketHandler161();
    }

    @Bean
    public WebSocketHandler webSocketHandler162() {
        return new WebSocketHandler162();
    }

    @Bean
    public WebSocketHandler webSocketHandler163() {
        return new WebSocketHandler163();
    }

    @Bean
    public WebSocketHandler webSocketHandler164() {
        return new WebSocketHandler164();
    }

    @Bean
    public WebSocketHandler webSocketHandler165() {
        return new WebSocketHandler165();
    }

    @Bean
    public WebSocketHandler webSocketHandler166() {
        return new WebSocketHandler166();
    }

    @Bean
    public WebSocketHandler webSocketHandler167() {
        return new WebSocketHandler167();
    }

    @Bean
    public WebSocketHandler webSocketHandler168() {
        return new WebSocketHandler168();
    }

    @Bean
    public WebSocketHandler webSocketHandler169() {
        return new WebSocketHandler169();
    }

    @Bean
    public WebSocketHandler webSocketHandler170() {
        return new WebSocketHandler170();
    }

    @Bean
    public WebSocketHandler webSocketHandler171() {
        return new WebSocketHandler171();
    }

    @Bean
    public WebSocketHandler webSocketHandler172() {
        return new WebSocketHandler172();
    }

    @Bean
    public WebSocketHandler webSocketHandler173() {
        return new WebSocketHandler173();
    }

    @Bean
    public WebSocketHandler webSocketHandler174() {
        return new WebSocketHandler174();
    }

    @Bean
    public WebSocketHandler webSocketHandler175() {
        return new WebSocketHandler175();
    }

    @Bean
    public WebSocketHandler webSocketHandler176() {
        return new WebSocketHandler176();
    }

    @Bean
    public WebSocketHandler webSocketHandler177() {
        return new WebSocketHandler177();
    }

    @Bean
    public WebSocketHandler webSocketHandler178() {
        return new WebSocketHandler178();
    }

    @Bean
    public WebSocketHandler webSocketHandler179() {
        return new WebSocketHandler179();
    }

    @Bean
    public WebSocketHandler webSocketHandler180() {
        return new WebSocketHandler180();
    }

    @Bean
    public WebSocketHandler webSocketHandler181() {
        return new WebSocketHandler181();
    }

    @Bean
    public WebSocketHandler webSocketHandler182() {
        return new WebSocketHandler182();
    }

    @Bean
    public WebSocketHandler webSocketHandler183() {
        return new WebSocketHandler183();
    }

    @Bean
    public WebSocketHandler webSocketHandler184() {
        return new WebSocketHandler184();
    }

    @Bean
    public WebSocketHandler webSocketHandler185() {
        return new WebSocketHandler185();
    }

    @Bean
    public WebSocketHandler webSocketHandler186() {
        return new WebSocketHandler186();
    }

    @Bean
    public WebSocketHandler webSocketHandler187() {
        return new WebSocketHandler187();
    }

    @Bean
    public WebSocketHandler webSocketHandler188() {
        return new WebSocketHandler188();
    }

    @Bean
    public WebSocketHandler webSocketHandler