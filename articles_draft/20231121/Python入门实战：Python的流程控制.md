                 

# 1.背景介绍


“流程控制”在Python编程中扮演着至关重要的角色，它能帮助我们更加精确地组织我们的代码，并提高其运行效率。本文将以一个简单的流程控制任务——实现一个电话簿程序——为例，通过讲解相关知识点，给读者提供实战操作手册。
# 2.核心概念与联系
流程控制就是根据特定的条件或情况，执行不同的指令或语句。流程控制中的一些核心概念如下所示：

1、顺序结构（Sequence Structure）:按照代码执行的先后顺序执行程序的各个子程序。例如：顺序结构可以用来编写阅读理解测试、预约门票等简单程序；

2、选择结构（Selection Structure）：根据判断条件的成立与否，选择不同路径执行程序的不同分支。例如：if-else结构、switch结构、for-while循环等；

3、迭代结构（Iteration Structure）：重复执行同一段代码，直到满足某个条件才结束。例如：for循环、while循环；

4、跳转结构（Goto Statement）：用于在程序执行过程中跳到指定位置。例如：goto语句。

流程控制有两种形式：

1、顺序型流程控制：即按照代码的顺序执行，每一步都必须按顺序完成；

2、选择型流程控制：即依据某种条件进行判断，然后决定下一步要做什么。

本文将结合电话簿程序的实际例子，来阐述Python语言下的流程控制方式及其语法。
# 3.核心算法原理和具体操作步骤以及数学模型公式详细讲解
## 3.1算法描述：
实现一个电话簿程序，要求用户输入姓名和电话号码，并存储在文件中，同时可根据输入的姓名查询该人对应的电话号码。
## 3.2具体实现步骤：
1.创建电话簿文件，用于存放用户的姓名和电话号码信息。
```python
def create_phonebook():
    """创建一个电话簿文件"""
    with open("phonebook.txt", "w") as f:
        pass
```
2.向电话簿文件写入姓名和电话号码信息，并提示用户输入姓名和电话号码：
```python
def write_to_phonebook(name, phone):
    """向电话簿文件中写入姓名和电话号码信息"""
    print("请输入姓名:")
    name = input()

    print("请输入电话号码:")
    phone = input()
    
    # 将姓名和电话号码保存到电话簿文件
    with open("phonebook.txt", "a+") as f:
        f.write("{} {}\n".format(name, phone))
        print("已成功保存到电话簿文件！")
```
3.从电话簿文件中读取电话号码，并根据用户输入的姓名查找对应的电话号码：
```python
def read_from_phonebook(name):
    """从电话簿文件中读取电话号码"""
    # 从文件中读取所有行
    lines = []
    try:
        with open("phonebook.txt", "r") as f:
            for line in f:
                if len(line) > 1:
                    # 以空格作为分隔符，分割出姓名和电话号码
                    data = line.strip().split(" ")
                    names = [d.lower() for d in data[:-1]]
                    phones = [data[-1] for _ in range(len(names))]

                    # 查找姓名对应的电话号码
                    index = -1
                    for i, n in enumerate(names):
                        if n == name.lower():
                            index = i
                            break
                        
                    if index!= -1:
                        return phones[index]

        print("未找到该人的电话号码！")
        
    except FileNotFoundError:
        print("电话簿文件不存在！")
```

## 3.3数学模型公式详细讲解
无
# 4.具体代码实例和详细解释说明
## 4.1完整的代码实例如下：
create_phonebook()

print("-" * 50)
print("\t欢迎使用电话簿程序\t")
print("-" * 50)

while True:
    print("[1] 新增联系人")
    print("[2] 查询联系人")
    print("[3] 退出程序")
    choice = input("请输入你的选择:")
    
    if choice not in ["1", "2", "3"]:
        continue
    
    elif choice == "1":
        write_to_phonebook("", "")
        
    elif choice == "2":
        print("请输入要查询的姓名:")
        name = input()
        
        result = read_from_phonebook(name)
        if result is not None:
            print("电话号码:", result)
            
    else:
        break
        
print("感谢使用电话簿程序！")
```
## 4.2代码解析：

1.`create_phonebook()`函数用于创建名为`phonebook.txt`的文件，文件的内容为空。
2.`write_to_phonebook()`函数用于向`phonebook.txt`文件写入姓名和电话号码信息，并提示用户输入姓名和电话号码。
3.`read_from_phonebook()`函数用于从`phonebook.txt`文件读取电话号码，并根据用户输入的姓名查找对应的电话号码。

以上三个函数都是需要在程序的主体代码中调用，它们分别完成了电话簿文件的创建、信息写入和信息查询功能。

程序的主体代码是一个无限循环，用于用户交互。首先展示程序简介信息，然后列出菜单选项供用户选择。用户输入数字1代表新增联系人，输入数字2代表查询联系人，输入数字3代表退出程序。用户输入其他内容代表重新输入，重复以上过程。当用户选择退出程序时，程序终止。

## 4.3注意事项：

1.程序运行之前需先确认电脑上有名为`phonebook.txt`的文件，否则会报错；
2.程序支持多次查询相同姓名的人，但只返回最后一次查询结果；
3.程序不区分大小写，所以可以忽略用户名和电话号码输入时的大小写。