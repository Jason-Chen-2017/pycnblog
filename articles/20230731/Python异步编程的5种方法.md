
作者：禅与计算机程序设计艺术                    

# 1.简介
         

         ## 为什么要写这篇文章呢？

         在过去的十年里，Python异步编程已经成为一种主流开发模式，基于asyncio、aiohttp等框架，越来越多的公司开始采用异步编程模式进行系统架构设计和工程实现。但对于初学者来说，了解异步编程并应用到实际项目中却是一项困难的任务。如何让初学者快速理解异步编程、提升编程能力、降低学习曲线，是本文的主要目的。

         2020年，随着硬件性能的提升，Python异步编程也在逐渐走向成熟。目前，基于asyncio、trio和curio等三大异步框架，Python有了更丰富的异步开发库。因此，本文尝试总结并梳理异步编程的五种方法，帮助广大的初学者快速入门、掌握异步编程，提高编程能力和降低学习曲线。希望能够帮到大家。

         ## 目录
         
         - [1. Python异步编程的5种方法](#1-python异步编程的5种方法)
             * [1.1 什么是异步编程](#11-什么是异步编程)
             * [1.2 为什么需要异步编程](#12-为什么需要异步编程)
             * [1.3 async/await语法](#13-asyncawait语法)
             * [1.4 asyncio模块](#14-asyncio模块)
                 + [1.4.1 Asyncio Future对象](#141-asyncio-future对象)
                     - [1.4.1.1 Future对象状态及生命周期](#1411-future对象状态及生命周期)
                     - [1.4.1.2 创建Future对象的方法](#1412-创建future对象的方法)
                         * [协程](#协程)
                         * [生成器函数](#生成器函数)
                             + [普通生成器函数](#普通生成器函数)
                             + [带yield from语句的生成器函数](#带yield-from语句的生成器函数)
                     - [1.4.1.3 执行Future对象的方法](#1413-执行future对象的方法)
                         * [回调函数](#回调函数)
                         * [添加回调函数](#添加回调函数)
                         * [添加异常处理函数](#添加异常处理函数)
                     - [1.4.1.4 Future对象之间的关系](#1414-future对象之间的关系)
                         * [链式回调（callback chaining）](#链式回调callback-chaining)
                     - [1.4.1.5 yield from与协程的转换](#1415-yield-from与协程的转换)
                 + [1.4.2 Asynio EventLoop事件循环](#142-asynio-eventloop事件循环)
                 + [1.4.3 Asynio网络库](#143-asynio网络库)
                     - [1.4.3.1 aiohttp模块](#1431-aiohttp模块)
                 + [1.4.4 利用Asyncio进行并发](#144-利用asyncio进行并发)
                     - [1.4.4.1 使用asyncio.wait()等待多个Future对象完成](#1441-使用asynciowait等待多个future对象完成)
                     - [1.4.4.2 使用asyncio.gather()收集多个Future对象](#1442-使用asynciogather收集多个future对象)
                     - [1.4.4.3 使用asyncio.shield()防止Future被取消](#1443-使用asyncioshield防止future被取消)
                 + [1.4.5 Asynio源码剖析](#145-asynio源码剖析)
             * [1.5 Trio模块](#15-trio模块)
                 + [1.5.1 Trio介绍](#151-trio介绍)
                 + [1.5.2 Trio基本用法](#152-trio基本用法)
                 + [1.5.3 Trio与asyncio比较](#153-trio与asyncio比较)
             * [1.6 Curio模块](#16-curio模块)
                 + [1.6.1 curio介绍](#161-curio介绍)
                 + [1.6.2 curio基本用法](#162-curio基本用法)
                 + [1.6.3 curio与其他异步库的区别](#163-curio与其他异步库的区别)

         
    


    



        