
作者：禅与计算机程序设计艺术                    

# 1.简介
         

         JavaScript（简称JS）是一种跨平台、面向对象的动态脚本语言。它的设计目的是为了能够使得网页的功能更加丰富、互动性更强，它由Netscape公司的Brendan Eich开发，并在1995年作为Navigator 2的内嵌脚本语言而被广泛使用。由于其轻量级、简单、高效率等特点，广受网络工程师喜爱，也得到了业界的广泛认可。它也是当前最热门的编程语言之一。本文将从JavaScript的历史及其发展历程开始，然后介绍JavaScript的一些基础语法和内置对象。最后总结一下本文的主要观点、知识点、不足与需要改进之处。
         
         # 2.JavaScript历史
         
         1995年，网景导航浏览器上市，当时它使用的脚本语言就是JavaScript，用来增强用户界面交互效果。但是因为Java技术快速发展，导致Java Applet技术的出现。但是当时的浏览器支持能力较弱，只能运行一些简单的动画或游戏，缺乏扩展能力。因此，网景公司决定改用解释型脚本语言，只要浏览器支持JavaScript，就可以执行任意的代码。然而这种语言学习曲线比较陡峭，且只能用于客户端应用。
         
         1996年10月，Sun Microsystems推出了Java平台，其浏览器Nashorn就支持运行Java Applet。但是当时的Java技术还没有达到生产环境的要求。为了弥补这个技术短板，Sun公司决定另起炉灶，开发新的Java虚拟机HotSpot VM。由于Java性能相对较差，而且Applet的执行模式存在安全性问题，因此Sun公司转而开发Java Scripting API（简称JSA），以此来为Web开发提供一个统一的脚本语言。
          
         1997年11月，微软正式推出Internet Explorer浏览器，加入对JScript的支持。同年12月，Borland发布Delphi，成为第一个支持JScript的集成开发环境IDE。
          
         2000年初，Mozilla基金会创建了Mozilla项目，开始研制基于Gecko引擎的Mozilla浏览器。但是由于它只是一款普通的浏览器而非网页端应用，无法直接运行JavaScript。为了支持网页端应用，Mozilla基金会于2000年底推出Rhino，一个JavaScript解释器，可以直接在Java虚拟机上运行。2001年3月，IE7发布，开始默认开启JScript，同时还默认开启VBScript。
          
          1997年11月，微软推出Internet Explorer浏览器，加入对JScript的支持；2000年初，Mozilla基金会创建了Mozilla项目，开始研制基于Gecko引擎的Mozilla浏览器；2001年3月，IE7发布，开始默认开启JScript和VBScript；2004年9月，Opera宣布与Sun合作，开始支持Java。
          
          综上所述，JavaScript发展史可分为三个阶段：

          1、原始阶段：1995~1997年间，网景导航浏览器、Sun Java系统以及Mozilla浏览器都支持JScript。

           2、应用阶段：从1997年开始逐渐形成应用。

           3、完善阶段：随着Web技术的发展，JavaScript已经逐步完善，得到广泛的应用。如今JavaScript已经成为多种语言中功能最丰富的一门语言。

         # 3.JavaScript基础语法

         1.变量类型
         
         在JavaScript中，变量类型分为两种：基本数据类型和引用数据类型。基本数据类型包括Number、String、Boolean、Undefined、Null。基本数据类型是不可更改的值，赋值后不能再修改。引用数据类型包括Object、Array、Function。引用数据类型是可变的对象，可以更改引用的对象的属性值。

         声明变量：var age = 20; //声明变量age并初始化值为20

         检测变量类型：typeof(x) 或 instanceof()方法。 typeof()方法返回的是表示类型名称的字符串，instanceof()方法则判断某个对象是否属于某个类。

         检测某个变量是否声明过：typeof x === "undefined"

         if (typeof name!== 'undefined') {
            console.log('Hello '+name+'!');
        } else {
            console.log('Please input your name.');
        }

        2.条件语句

        （1）if...else语句

        var num = prompt("请输入一个数字:");   //输入数字

        if(num%2 == 0){    //如果输入数字是偶数
            alert("该数字是偶数！");
        }else{     //否则
            alert("该数字不是偶数！");
        }

        （2）switch...case语句

        switch(dayOfWeek){
            case 0:
                console.log('星期日');
                break;
            case 1:
                console.log('星期一');
                break;
            default:
                console.log('错误！请输入正确的日期！');
                break;
        }

        （3）for循环语句

        for(var i=0;i<arr.length;i++){  
            console.log(arr[i]);  
        }

        （4）while循环语句

        var count = 0;
        while(count < arr.length){
            console.log(arr[count++]);
        }

        （5）do...while循环语句

        do{
            console.log(count);
        }while(count > 0 && --count);

        3.函数定义

        function sum(a,b){
            return a+b;
        }
        
        console.log(sum(10,20));

        4.事件处理

        var btn = document.getElementById('btn');

        btn.addEventListener('click',function(){  
            console.log('按钮被点击了！'); 
        });

        5.JSON对象

        JSON（JavaScript Object Notation）是一种轻量级的数据交换格式。它基于ECMAScript的一个子集。它是一个文本格式，传输速度快，占用空间小，并易于人阅读和编写。

        var obj = {"name": "张三", "age": 20};

        var jsonStr = JSON.stringify(obj);    //把对象转换为json字符串

        var newObj = JSON.parse(jsonStr);      //把json字符串转换为对象

        console.log(newObj.name + ','+ newObj.age);  //输出结果：张三, 20。