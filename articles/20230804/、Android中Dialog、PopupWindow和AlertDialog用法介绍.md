
作者：禅与计算机程序设计艺术                    

# 1.简介
         
 Android系统中的对话框（Dialog）、弹出窗口（PopupWindow）和提示信息（AlertDialog），是用户与系统交互的方式之一。本文将从它们各自的作用、用法、特性等方面分别介绍。首先会介绍它们之间的一些共同点，然后详细讨论它们各自的特点和使用场景。
         ## Dialog
          对话框（Dialog）是一种模态的用户界面组件，它提供了一个定制化的用户界面来处理某些相关任务，包括简单的确认或取消操作、输入文本或选择项等。一般来说，对话框用于向用户请求一些简单的数据或让用户进行简单操作，比如显示应用设置、文件选择对话框、警告消息等。Android中的Dialog主要由 AlertDialog 和 AlertDialog.Builder 两个类实现。
         ### AlertDialog
          在 Android SDK 中，我们可以通过 AlertDialog 对象来创建自定义对话框并呈现给用户。 AlertDialog 是最常用的对话框类型，其 API 支持多个选项按钮，单选或多选，可编辑文本框，支持自定义 View，可以添加回调接口来监听事件。 AlertDialog 的一个典型例子如下所示：
         ```java
            new AlertDialog.Builder(MainActivity.this)
               .setTitle("Title") // 设置标题
               .setMessage("Message") // 设置内容
               .setPositiveButton("OK", new DialogInterface.OnClickListener() {
                    @Override
                    public void onClick(DialogInterface dialog, int which) {
                        Toast.makeText(MainActivity.this, "Clicked OK button.", Toast.LENGTH_SHORT).show();
                    }
                })
               .setNegativeButton("Cancel", null) // 没有Negative按钮，设为空值
               .create().show(); // 创建并显示对话框
         ```
         通过 Builder 来创建 AlertDialog 对象，通过 setTitle()、setMessage() 方法设置标题和内容，再调用 setPositiveButton() 方法设置确定按钮文字及点击事件，最后调用 create() 方法生成 AlertDialog 对象，再调用 show() 方法显示对话框。通常情况下，AlertDialog 对象在视图层级上处于顶层，即覆盖整个屏幕，因此需要注意对其他 UI 操作的影响。另外，如果 AlertDialog 对象被创建但没有显示出来，会自动消失。
         ### PopupWindow
          PopupWindow 提供了一种简洁而灵活的机制来展示非控件的内容，可以在屏幕上的任何位置显示。它的 API 可用于丰富的功能集，包括对话框、菜单、工具提示以及通知。PopupWindow 有以下优点：
         * 不依赖于 Activity 或 Fragment，可以独立存在于视图层级中；
         * 可以调整大小、透明度、背景，可以使用动画效果；
         * 也可以嵌入到 View 上，提供更灵活的布局；
         * 适合于自定义悬浮控件、通知中心等场景。
         
         使用 PopupWindow 需要用到两个关键类：PopupWindow 和 PopupWindow.OnDismissListener。其中，PopupWindow 对象用来定义弹出的窗口属性，包括宽高、位置、背景色等；OnDismissListener 接口用于监听 PopupWindow 是否被关闭。下面的例子演示了如何创建一个 PopupWindow 对象并显示它。
         ```java
             private PopupWindow mPopupWindow;
             
             //...
             mPopupWindow = new PopupWindow(View contentView, LayoutParams width, LayoutParams height);
             // 设置弹出窗口的内容 view
             mPopupWindow.setContentView(contentView);
             // 设置弹出窗体的宽度和高度
             mPopupWindow.setWidth(LinearLayout.LayoutParams.WRAP_CONTENT);
             mPopupWindow.setHeight(LinearLayout.LayoutParams.WRAP_CONTENT);
             // 设置弹出窗体可点击，这是为了防止当弹出窗体外的区域也响应 touch event
             mPopupWindow.setTouchable(true);
             // 设置弹出窗体动画效果
             mPopupWindow.setAnimationStyle(R.style.popwin_anim_style);
             // 设置弹出窗体显示的位置
             mPopupWindow.showAsDropDown(mBtnShowPopup, 0, -50);
             // 为 popup window 设置 dismiss listener
             mPopupWindow.setOnDismissListener(new PopupWindow.OnDismissListener() {
                 @Override
                 public void onDismiss() {
                     Log.d(TAG, "Popup Window dismissed");
                 }
             });
         ```
         此例创建一个 PopupWindow 对象，传入 View contentView 对象作为弹出窗口的内容，设置宽高、位置、可点击、动画样式等。调用 showAsDropDown() 方法指定弹出窗体显示的位置，这里采用绝对位置，可以设置偏移量。同时，设置 OnDismissListener 以便于监控弹出窗是否被关闭。
         ## Popover
          弹出式菜单（Popover）是指当用户点击某个元素后，出现一个移动的窗口来承载附属的选项，类似于 iOS 中的 popover。Popover 可以包含不同的操作，如删除、分享、收藏等。Android SDK 中提供了两种方式来创建弹出式菜单：PopupWindow 和 MenuInflater。
         ### PopupWindow
          