                 

# 1.背景介绍


## 一、什么是生成器？
生成器是一种特殊类型的函数，它不仅可以返回值，还可以使用yield关键字来产出值。在for循环中使用yield关键字，可以实现迭代对象元素的生成。
```python
def my_generator():
    yield 1
    yield 'hello'
    yield [1, 2, 3]

my_gen = my_generator() # 创建生成器对象

print(next(my_gen))    # output: 1
print(next(my_gen))    # output: hello
print(next(my_gen))    # output: [1, 2, 3]
```
## 二、为什么要使用生成器？
1）提升内存使用效率：由于生成器不会一次性产生所有结果集，而是在每次需要时才生成当前的结果，因此无需占用太多内存空间；

2）节约资源：如果生成结果集较大且无法全部放入内存，则可以使用生成器来减少内存占用，节省系统资源。

通过一个求素数的例子来了解生成器的应用场景：给定一个正整数n，要求输出其所有的质因数（质数与指数）。一般情况下，我们可以通过遍历的方式找到所有的质数并判断是否整除当前数，如果可以整除则记录该质数及其指数；但是当数字非常大或者质数很多的时候，这种遍历方法会导致大量的计算浪费。因此，我们可以使用生成器来解决这个问题。
```python
def primes_factors(num):
    factors = {}
    for i in range(2, int(num**0.5) + 1):
        if num % i == 0:
            count = 0
            while num % i == 0:
                num //= i
                count += 1
            factors[i] = count
            
    if num > 1:
        factors[num] = 1
    
    return factors
    
print(primes_factors(24))   # output: {2: 3}
print(primes_factors(17))   # output: {17: 1}
```
由此可见，使用生成器的确可以提高程序的运行效率，节约系统资源。

## 三、生成器与迭代器之间的区别
1）概念上的区别：生成器是一个函数，而迭代器是一个接口；

2）创建方式上的区别：生成器不能被调用，只能被赋值或传参到另一个生成器；而迭代器可以被next()等方法调用来获取下一个值。

# 2.核心概念与联系
## 一、什么是迭代器？
迭代器（Iterator）是Python内置的一种数据类型。它表示的是一个数据流，即按照顺序访问集合中的每一个元素，直至处理完整个集合。
## 二、如何创建一个迭代器？
创建一个迭代器的方法有两种：一种是直接创建生成器对象，另一种是使用iter()函数将可迭代对象转换成迭代器。例如：
```python
g = (x*x for x in range(1, 11))     # 使用生成器创建迭代器

lst = ['a', 'b', 'c']
it = iter(lst)                      # 将列表转换成迭代器

for elem in g:                       # 打印生成器的值
    print(elem)
    
for char in it:                      # 打印迭代器的值
    print(char)
```
## 三、迭代器协议
迭代器协议是一个简单的协议，定义了两个方法__iter__() 和 __next__()，用来对迭代器进行迭代。__iter__() 方法返回自己本身，用于启动迭代过程，__next__() 方法返回下一个元素，若没有下一个元素，抛出StopIteration异常结束迭代。
```python
class MyIterator:
    def __init__(self, data):
        self.data = data
        
    def __iter__(self):
        return self
    
    def __next__(self):
        try:
            item = next(self.data)
            return item
        except StopIteration:
            raise StopIteration("No more items.")
        
data = [1, 2, 3, 4, 5]
iterator = MyIterator(data)

while True:
    try:
        value = iterator.__next__()
        print(value)
    except StopIteration as e:
        break
```
## 四、什么是生成器表达式？
生成器表达式（Generator Expression）与列表解析一样，也是一种高级语言结构，但它不是一个列表，而是返回一个生成器对象，可以用来迭代集合。其语法形式为`（expression for target in iterable）`，其中iterable是一个可迭代对象，target表示循环变量。生成器表达式并不会立刻执行，而是返回一个生成器对象，只有在需要访问元素的时候，才会计算出元素。以下是生成器表达式的示例：
```python
result = (x*x for x in range(1, 11))

print(type(result))          # <class 'generator'>

for val in result:           # 惰性求值，不需要遍历
    print(val)               # 在此处会输出第一个数的平方后面9个值
```
注意：生成器表达式只能用在生成器函数中，不能单独使用。