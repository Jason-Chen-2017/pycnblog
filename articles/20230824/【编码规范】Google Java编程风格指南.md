
作者：禅与计算机程序设计艺术                    

# 1.简介
  

这是一份适用于Google公司内部开发人员编写Java代码的编码规范，目的是为了增强代码质量、提高代码可读性、降低维护成本并提升代码的可移植性和可扩展性。该文档可以作为软件工程师在进行项目开发时参考指导，并应用于不同编程语言。
# 2.背景介绍
Java是目前世界上最流行的编程语言之一，它具有简单、灵活、功能丰富等特点，适合构建各种多媒体、游戏、后台系统、移动应用程序、企业级应用等多种类型的软件。近几年，随着微服务架构模式的流行以及云计算的普及，越来越多的软件工程师开始采用面向服务的架构设计和开发软件。由于编码规范对软件开发的一致性、可读性、可维护性等方面具有重要意义，因此，越来越多的软件工程师从事Java软件开发工作，并且提倡使用统一的代码规范来编写Java程序。
Google公司是全球最大的技术公司，它的很多产品都已经成为软件开发者不可或缺的一环。比如Android系统就是由Google开发并推广，而其代码规范就是建立在Google Java编程风格指南基础上制订出来的。同时，其开源软件许可证也是Apache License v2.0。
所以，即使是没有参加过Google工作的人也可以了解到Google公司对于软件开发者的态度和追求。Google公司将软件开发工作视为一种职业化的生产活动，通过制定一系列标准和约束来确保生产出的软件质量，并达到“创造和分享”的共赢效果。这种态度也为国内其他软件公司提供了学习的机会。
# 3.基本概念术语说明
## 3.1 缩进（Indentation）
每一行代码应该首行缩进4个空格。

不推荐：

    void func()
    {
    	if(true){
    		System.out.println("hello world");
    	}else{
    		return;
    	}
    }

推荐：

    void func() {
      if (true) {
        System.out.println("hello world");
      } else {
        return;
      }
    }

## 3.2 空白字符
每个空白字符（包括Tab、空格、换行符等）都应当只用一次。不能出现连续多个空白字符。

不推荐：

    int sum =   1 +    2 ; 
    for   (int i=0;i<n;i++){ 
      //code
    } 

推荐：

    int sum = 1 + 2;
    for (int i = 0; i < n; i++) {
      // code
    } 

## 3.3 括号
### 3.3.1 在控制结构关键字之后必须有一个空格。

例如，if/for/while语句中括号后应该有一个空格；switch语句中的case之后也必须有空格。

不推荐：

    if(x > y)  
      count++; 
      
    switch(color){
      case "red": 
        break;
      default: 
        throw new IllegalArgumentException();
    }  

推荐：

    if (x > y) {
      count++;
    }
    
    switch (color) {
      case "red":
        break;
      default:
        throw new IllegalArgumentException();
    } 

### 3.3.2 每行结束后都要加分号。

推荐：

    String name = "Alice";
    System.out.println(name);

不推荐：

    String name = "Alice"
    System.out.println(name)