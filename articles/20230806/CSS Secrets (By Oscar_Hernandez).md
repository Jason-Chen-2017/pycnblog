
作者：禅与计算机程序设计艺术                    

# 1.简介
         
2019年是CSS盛极而衰的一年。众多浏览器厂商纷纷推出了新的Web API，使得前端开发者可以实现更加丰富的交互效果。CSS Secrets这本书从头到尾就着重讲述了前端开发者不应该忽视的CSS知识点。其中包括布局、动画、样式管理、响应式设计等方面。读完这本书后，你将对CSS有全新的认识，掌握更多高级技巧。
         此外，作者还设计了一套完整的学习路径，全面覆盖CSS核心知识，并且提供了有价值的参考资源和教程。相信经过这么一段时间的阅读后，你也会对CSS有全面的理解并应用到自己的工作中。
         
 
 
 
 
 
         # 2.基本概念术语说明
         1.Box模型
         
         在CSS中，每一个元素都被看作是一个矩形盒子，这个矩形盒子由四个部分构成，分别是content（内容）、padding（内边距）、border（边框）和margin（外边距）。通过设置这些属性就可以控制元素在页面中的位置、大小及样式。如下图所示:
         

         
         通过设置box-sizing属性，可以让不同类型的边框能填充整个元素的内容区域，如设置box-sizing: border-box;则四个边框都能填充内容区域。
         
         2.盒类型
         
         CSS中的盒类型分为两种：块级盒和行内级盒。当display属性设置为block时，该盒子为块级盒；如果设置为inline，则为行内级盒。块级盒会独占一行，可以设置宽度、高度、内边距、外边距等属性；而行内级盒只能设置宽度、高度、水平方向的内边距、外边距等属性，不能包含任何子元素。如下图所示：
         

         
         某些CSS属性的值可以影响盒子的类型，比如width、height、display、position等属性。例如：
          
          display: inline; /* 为行内级盒 */
          
          display: block; /* 为块级盒 */
          
          position: relative; /* 相对于其正常流进行定位 */
          
          position: absolute; /* 相对于最近的一个定位祖先进行绝对定位 */
          
          width: auto; /* 根据内部内容调整宽度 */
          
          height: 100%; /* 充满父级盒 */
           
          overflow: hidden; /* 设置溢出隐藏 */
       
         3.计算宽度和高度

         当设置元素的宽度和高度时，如果没有设置其他属性值（包括百分比或其他单位），浏览器会根据元素的内容自动分配相应的值。也就是说，如果没有给元素设置宽度，它就会尽可能多地适应内容区域的宽度；同样，如果没有设置高度，它就会尽可能地伸缩以适应内容。但同时，我们也可以通过其他一些属性值或者嵌套元素来影响宽度和高度的计算规则。
         
         首先，通过font-size属性设置字体大小。浏览器根据该属性值计算元素的默认宽度和高度。例如：

           div {
             font-size: 16px;
           }

           /* 默认宽度为16 * 文本长度 + 2 * padding = 16 + 32 = 48px */
           /* 默认高度为16 * 行数 + 2 * padding + 2 * border = 16 + 32 + 2 = 48px */

           span {
             font-size: 32px;
           }

           /* 默认宽度为32 * 文本长度 + 2 * padding = 32 + 64 = 96px */
           /* 默认高度为32 * 行数 + 2 * padding + 2 * border = 32 + 64 + 2 = 96px */
         
         然后，通过line-height属性设置行高。由于行高影响行数的计算，因此它与font-size具有很强的相关性。如果两个元素的font-size相同，但line-height不同，则它们的行数可能会不同。例如：

           p {
             line-height: normal; /* 行高由内容决定 */
           }

           h1 {
             line-height: 1.5em; /* 行高为1.5倍的字号 */
           }
         
         最后，可以通过宽度和高度的单位来指定元素的尺寸，也可以通过百分比和其他单位来设置父容器的尺寸。如下图所示：
         

         
         另外，我们还可以用max-width和min-width属性来限制元素的最大宽度和最小宽度。例如：

            img {
              max-width: 100%; /* 图片宽度不超过100% */
              min-width: 100px; /* 图片宽度至少为100像素 */
            }
            
            button {
              max-width: initial; /* 按钮宽度与默认宽度一致 */
              min-width: inherit; /* 继承父元素的最小宽度 */
            }
         
         总结：通过font-size、line-height、max-width、min-width三个属性影响元素的默认宽度和高度的计算规则，以及通过宽度和高度的单位和百分比设置元素的尺寸，可以帮助我们更好地控制元素的布局。
         
         4.选择器
         
         在CSS中，选择器是用来匹配特定HTML标签、类别、ID或其他特征的。如下图所示：
         

         
         可以通过标签名、类名、ID、属性及组合的方式来选择相应的元素。例如：

           a {
             color: red; /* 对所有<a>标签内的文字设置颜色 */
           }

          .error {
             background-color: yellow; /* 对所有class为"error"的元素设置背景色 */
           }

           input[type="submit"] {
             margin: 10px; /* 对所有<input type="submit">标签设置上下左右边距 */
           }

           nav li {
             list-style: none; /* 清除列表项的标志 */
           }

           ul > li {
             margin-left: 1em; /* 只对直接子元素的第一层级元素添加左侧外边距 */
           }
         
         另外，还有一些高级选择器可以使用，比如: :nth-child(), ::first-child 和 ::before 。::after选择器可用于插入内容到元素末尾。
         
         5.继承和层叠
         
         CSS中的继承和层叠都是用来解决CSS规则冲突的问题。
         
         继承是指子元素继承了父元素的某些样式，比如字体大小、颜色、背景色、文本样式等。使用inherit可以继承父元素的这些样式，可以节省代码量。如下图所示：
         

         
         层叠是指多个CSS规则之间存在优先级关系，CSS定义了四种不同的层叠规则。按从高到低依次为：内联样式（in-line style）、网页样式表（external style sheet）、用户代理样式（user agent styles）、浏览者默认样式（browser default styles）。不同规则之间的冲突关系如下图所示：
         

         
         使用最高优先级的规则可以覆盖其他规则的样式。例如：

            <div class="example">Example Text</div>
            
            /* 外部样式表 */
           .example {
                color: blue;
            }
            
            /* 用户代理样式 */
            div{
                text-transform: uppercase;
            }
            
            /* 浏览器默认样式 */
            body{
                margin: 0;
            }
            
            /* 最终显示的颜色为蓝色，文本全部大写 */
            Example TEXT
         
         有些CSS属性是可以继承的，而有些属性是不可继承的，这样就造成了层叠的局限性。不过CSS3中引入了新的伪类和伪元素，可以满足不同的需求。

         6.过渡与动画
         
         过渡是指两个状态间的平滑过渡过程，可以制作出酷炫的动画效果。CSS3新增了transition和animation属性来实现过渡与动画。

             transition: property duration timing-function delay;
               /* 属性：指定需要过渡的CSS属性
                 持续时间：过渡花费的时间，以秒为单位
                 速度曲线：指定动画速度变化的速率曲线，默认值为ease
                 延迟：延迟多少秒之后开始过渡 */
                 
             animation: name duration timing-function iteration-count direction fill-mode play-state;
               /* 名称：定义动画名称
                 持续时间：动画花费的时间，以秒为单位
                 速度曲线：指定动画速度变化的速率曲线，默认值为ease
                 重复次数：动画重复的次数，默认为1，可以设为infinite
                 方向：动画播放方向，默认forwards，可以设为reverse
                 是否填充：动画结束后是否保持最后一帧，默认none，可以设为forwards或backwards
                 当前状态：动画的初始状态，默认running，可以设为paused */
                  
         下面是一个简单的例子：

              /* HTML结构 */
              <div id="box"></div>
              
              /* CSS */
              #box {
                width: 100px;
                height: 100px;
                background-color: green;
                transition: all 0.5s ease;
              }
              
              #box:hover {
                transform: scale(1.2);
                background-color: purple;
              }
              
              @keyframes move {
                0% { top: 0; left: 0; }
                100% { top: 200px; left: 200px; }
              }
              
              /* 添加动画 */
              #box:hover {
                animation: move 2s linear infinite;
              }

         
         上例中，#box的宽度、高度、背景色、边框、透明度等属性可以过渡动画，而当鼠标悬停时，它会放大并变成紫色，并且使用move动画来移动它的位置。可以看到，transition与animation可以实现各种各样的动画效果。

         7.布局
         
         布局是指确定页面上元素的位置、大小及分布方式，CSS3提供了很多的布局方案，比如Flexbox、Grid Layout和Box Alignment Module。

             Flexbox：一种新型的布局方式，允许将内容按不同的方式布置，而不是像表格那样横向排列。

                 display: flex; /* 将元素设置为flex容器 */
                 
                 justify-content: center; /* 主轴居中 */
                 
                 align-items: center; /* 交叉轴居中 */
                 
                 flex-direction: row | column; /* 主轴方向 */
                 
                 flex-wrap: nowrap | wrap; /* 换行 */
                 
                 order: integer; /* 指定元素的顺序 */
                 
                 align-self: auto | stretch; /* 单个元素的位置 */
                 
             Grid Layout：可以将网格化的布局布置起来，类似于表格的结构。

                 display: grid; /* 将元素设置为grid容器 */
                 
                 grid-template-columns: repeat(3, 1fr); /* 设置网格列数 */
                 
                 grid-gap: 10px; /* 设置网格间隙 */
                 
                 grid-row-start / grid-column-start: line number; /* 指定元素的起始位置 */
                 
                 grid-area: area name; /* 用名称指定元素所在的区域 */
                 
             Box Alignment Module：提供一种更好的方法来垂直对齐和水平对齐。

                 display: box; /* 将元素设置为弹性盒 */
                 
                 align-items: center; /* 垂直对齐 */
                 
                 justify-content: space-between; /* 水平对齐 */
                 
                 place-items: center center; /* 同时垂直、水平对齐 */
                 
                 align-content: center; /* 垂直对齐多个行 */

         
         总结：通过CSS的布局功能，可以有效地对网页的元素进行管理，提升用户的视觉体验。