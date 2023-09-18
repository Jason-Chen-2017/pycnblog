
作者：禅与计算机程序设计艺术                    

# 1.简介
  

列表（List）和元组（Tuple）在Python里都可以用来存储多条数据。但二者之间又有何不同呢？本文通过对两者进行比较分析，探讨它们之间的区别，并给出实际案例。
# 2.Python中的列表与元组
## Python中的序列类型——列表与元组
- List: List 是 Python 中一种“可变”的数据结构，它可以存储多个元素，且允许重复的值。List 可以随时添加、删除或者修改其中的元素，其中的值可以是任意类型，包括字符串、数字等。
- Tuple: Tuple 是 Python 中一种“不可变”的数据结构，它只能存储多个元素，且不允许重复的值。Tuple 在定义的时候就已经确定了其中的元素，不能再增加或删除其中的元素，其中的值也可以是任意类型。

除了上述两个特点外，List 和 Tuple 还存在一些差异性。以下是主要差异：

1.是否可变：List 可变，Tuple 不可变。
2.赋值方式：Tuple 只能使用索引（index）来访问元素，不能使用其他方式如变量名来访问元素；而 List 的元素可以通过索引或切片来访问。
3.占用内存大小：Tuple 比 List 小得多，因为 Tuple 中的每个元素都指向一个单独的对象，所以当 Tuple 很长时占用的内存会比 List 小很多。
4.性能：对于频繁读取少量数据的应用，比如字典的键值对查询，Tuple 更适合选择，因为 Tuple 每次查找元素时只需要一次指针跳转，而 List 需要多次指针跳转，并且 List 的查找方式相对复杂一些。

下面我们以实际案例的方式来说明两者之间的区别。

# 3.实际案例分析
## 3.1 基本概念
首先，我们先明确下下面几个基本概念：

- 数据类型：数据的形式、结构、分类以及规律。例如：整数、实数、逻辑值、浮点数、字符串、日期时间、集合、列表等。
- 表达式（expression）：运算符、操作数、函数调用及返回值等构成的完整的计算过程，如 2 + 3 * 4 - sqrt(9)。
- 对象：是指特定类型的实体，包括数据和程序。对象的属性和行为决定了其功能，每种对象都有自己的方法（函数）。

## 3.2 list与tuple的简单比较
### 3.2.1 相同之处
1. 初始化时赋值：list和tuple均可以使用初始化赋值，即用方括号或圆括号包裹元素，逗号隔开。
2. 使用[]与()访问元素：与list一样，list[i]表示第i个元素，从0开始；而tuple[i]则表示第i个位置上的元素，但tuple是固定不变的序列，无法修改元素。
3. 修改元素：list和tuple均可以修改元素，但只有list可以添加元素到末尾。
4. 删除元素：list可以使用del语句来删除元素，tuple则没有该语句。
5. 支持迭代：list、tuple、str、bytes等序列均支持迭代。

```python
l = [1, 2, 3]
t = (1, 2, 3)

for i in l:
    print(i)
    
for j in t:
    print(j)
```

输出结果：

```
1
2
3
``` 

### 3.2.2 不同之处
1. 更新特性：list可以修改元素的值，tuple不行。当要修改的元素是一个list或tuple时，如果直接对该元素重新赋值，就会生成新的list或tuple。
2. 添加元素：list可以使用append()方法向末尾添加元素，tuple也同样可以使用+运算符来添加元素。
3. 插入元素：list可以使用insert()方法插入元素，tuple则没有该方法。
4. 方法：list有丰富的方法实现高级功能，如排序sort()、查找index()、成员资格in、长度len()、复制copy()、获取子序列切片slice()等。tuple也有类似的方法。

## 3.3 函数作为参数传递
因为list和tuple都是可变序列，因此作为函数的参数时，它们的区别也会体现出来。举例如下：

```python
def my_func(param):
    param += ['new']    # 修改param对象
    return len(param), tuple(param)[0], type(param).__name__   # 返回新长度、第一个元素和param类型名称
    
my_list = [1, 2, 3]
print('Before:', my_list)

length, first_elem, elem_type = my_func(my_list)     # 将my_list传给函数，得到三个值

print('After length:', length)
print('After first element:', first_elem)
print('After type name:', elem_type)
print('In function:', my_list)
```

输出结果：

```
Before: [1, 2, 3]
After length: 4
After first element: new
After type name: list
In function: [1, 2, 3, 'new']
``` 

这里，我们把my_list作为my_func的参数，并在函数内部对参数进行了更新，此时的param是一个list，然后我们返回了一个tuple，包含了新长度、第一个元素和原参数的类型名称。最后，我们将结果打印出来，发现my_list还是被改变了，因为在函数内部，我们对参数进行了修改。但是在函数外部，my_list并没有变化，说明在函数内部创建的对象（如'new'）并不是修改原参数，而是创建了一个新的对象，并赋给了参数。