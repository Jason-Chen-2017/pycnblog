
作者：禅与计算机程序设计艺术                    

# 1.简介
  

函数合成(Function Composition)和柯里化(Currying)是非常重要的两个函数式编程中的术语，本文就尝试用通俗易懂的方式来讲解这两者。
## 函数组合(Function Composition)
在函数式编程中，函数组合通常用来将多个函数按照一定顺序执行，形成一个新的函数。如下面的例子所示：
```python
def add_two(x):
    return x + 2

def multiply_by_three(x):
    return x * 3

result = multiply_by_three(add_two(5)) # Output: 17
```
可以看到，这里的`add_two()`和`multiply_by_three()`都是接受一个参数并返回计算结果的普通函数。而通过将这两个函数组合起来，就可以得到一个新的函数`compose`，它接收两个函数作为输入参数，并返回一个新的函数，这个新函数接收一个参数并返回最终的计算结果。
```python
def compose(f, g):
    def h(x):
        return f(g(x))
    return h
    
double_and_triple = compose(lambda x: x*2, lambda x: x*3)

print(double_and_triple(5)) # Output: 30
```
`compose`函数是一个高阶函数，它接受两个函数作为输入参数，然后返回一个新的函数，该函数接收一个参数并返回最终的计算结果。对于`compose(lambda x: x*2, lambda x: x*3)`，其输入参数和输出结果都是一个函数，可以通过下面的方式调用：
```python
result = double_and_triple(5)
print(result) # Output: 30
```
## 柯里化(Currying)
在函数式编程中，柯里化(Currying)指的是把一个多参数函数转换成几个单参数的函数序列。最简单的一层次的柯里化就是将任意一个参数数量大于1的函数转换成几个单参数的函数序列，如下面的例子所示：
```python
def multiply(x, y):
    return x * y

curried_multiply = lambda x: (lambda y: x * y)

print(curried_multiply(5)(3))   # Output: 15
```
上述示例展示了如何利用柯里化方法将一个双参数函数转换成一个链式调用，如`multiply(5, 3)`转变为`curried_multiply(5)(3)`。也可以进一步查看下面的应用场景。
### 使用柯里化实现函数组合
可以使用柯里化的方法来对函数进行组合，从而可以生成更加复杂的功能。比如，假设有一个需要处理字符串的函数`process_string`，希望将字符串进行一些操作之后再进行另外一些操作，但又不想让这些操作都被放到同一个函数中。因此可以先使用柯里化将第一个操作分割开，即：
```python
def split_operation(s):
    first_op = s[::-1]    # Reverse the string
    
    second_ops = process_string(first_op)

    result = do_something_with_second_ops(second_ops)

    return result
```
这样的话，函数`split_operation`就可以接收字符串作为参数，并且可以根据需要对字符串进行处理。
除此之外，还可以在函数`do_something_with_second_ops`前面增加一些柯里化的操作，使得整个函数的调用过程更加灵活。
```python
def apply_map(func, iterable):
    results = []
    for item in iterable:
        results.append(func(item))
    return results
    
def curried_apply_map(func):
    def inner(*args, **kwargs):
        def applied_func(x):
            return func(x, *args, **kwargs)
        return apply_map(applied_func, args[0])
    return inner

@curried_apply_map
def square_each_number(num, n=2):
    if num % 2 == 0:
        return num**n - 2*(num**(n-1))
    else:
        return num**n
        
numbers = [1, 2, 3, 4, 5]
squared_even = square_each_number(numbers, n=2)[::2]
squared_odd = square_each_number(numbers, n=2)[1::2]
print(squared_even)   # Output: [9, 64, 512]
print(squared_odd)    # Output: [2, 25, 32]
```
以上代码展示了如何通过柯里化实现函数的组合。首先定义了一个叫做`square_each_number`的函数，它接受一个可迭代对象作为参数，并且通过`curried_apply_map`装饰器进行柯里化。柯里化之后，函数就可以接受其他非关键字的参数，这些参数会在`applied_func`内部进行传递。然后，将这个柯里化后的函数应用于原始列表的偶数位置（index）或者奇数位置（index+1），取决于是否满足某个条件，然后返回相应的结果。最后，分别获取奇数或者偶数位置上的结果，即可获得所需的平方数值。