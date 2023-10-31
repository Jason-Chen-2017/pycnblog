
作者：禅与计算机程序设计艺术                    

# 1.背景介绍


## 1.1 数据结构简介
计算机的数据结构是指用来组织、存储和处理数据的集合。数据结构与算法是数据存储和处理的重要支撑。数据结构决定了数据的存储方式，并通过选择合适的算法对数据进行处理。如数组、链表、栈、队列、树、图等。

## 1.2 Go语言中的数据结构
Go语言支持多种数据结构。其中比较有名的有数组、切片、字典、哈希表、堆栈、双端队列、树、图等。下面我们就来学习一下Go中这些数据结构的用法。

### 1.2.1 数组array
数组是一个固定大小的顺序集合，可以保存相同或不同类型的数据项。数组元素可以通过索引(index)访问，索引从0开始。

#### 声明数组
```go
var arr [5]int //声明一个整数型的长度为5的数组
arr[0], arr[1], arr[2], arr[3], arr[4] = 1, 2, 3, 4, 5 //初始化数组元素
fmt.Println("arr:", arr)
```

输出:
```
arr: [1 2 3 4 5]
```

#### 遍历数组
```go
for i := 0; i < len(arr); i++ {
    fmt.Printf("%d ", arr[i])
}
fmt.Println()
```

输出:
```
1 2 3 4 5 
```

### 1.2.2 切片slice
切片是一种动态大小的序列类型，可以容纳任意数量的元素。切片由三个元素组成：底层数组、起始位置和长度。

#### 创建切片
```go
//声明一个长度为3的切片
sli := make([]int, 3) 

//初始化切片元素
sli[0], sli[1], sli[2] = 1, 2, 3
fmt.Println("sli:", sli)

//创建容量为5的切片
capSli := make([]int, 5)

//追加元素到切片末尾，若容量不足则会重新分配内存
capSli = append(capSli, 1) //添加一个元素
capSli = append(capSli, 2) //添加另一个元素
capSli = append(capSli, 3) //再次添加一个元素
fmt.Println("capSli:", capSli)
```

输出:
```
sli: [1 2 3]
capSli: [1 2 3 0 0]
```

#### 遍历切片
```go
for _, val := range sli {
    fmt.Printf("%d ", val)
}
fmt.Println()

for i := 0; i < len(capSli); i++ {
    fmt.Printf("%d ", capSli[i])
}
fmt.Println()
```

输出:
```
1 2 3 
1 2 3 0
```

### 1.2.3 map
map是一种无序的键值对容器，可以存放不同类型的值。

#### 创建map
```go
//声明一个空的map
m := make(map[string]int)

//设置键值对
m["apple"] = 1
m["banana"] = 2
m["orange"] = 3

//打印map
fmt.Println("m:", m)
```

输出:
```
m: map[apple:1 banana:2 orange:3]
```

#### 修改和删除map元素
```go
//修改元素值
m["banana"] = 4

//删除元素
delete(m, "orange")

//打印map
fmt.Println("m:", m)
```

输出:
```
m: map[apple:1 banana:4]
```

#### 查找元素
```go
//查找元素值
val, ok := m["banana"]

if!ok {
    fmt.Println("banana not found in map.")
} else {
    fmt.Println("value of banana is", val)
}
```

输出:
```
value of banana is 4
```

### 1.2.4 字符串string
字符串是不可变的字节序列，可以包含任意数据。

#### 创建字符串
```go
str := "Hello World"
```

#### 拼接字符串
```go
str += "!"
fmt.Println(str)
```

输出:
```
Hello World!
```

### 1.2.5 指针pointer
指针指向其它变量存储的地址。

#### 通过取址运算符获取指针
```go
a := 10
p := &a
*p = *p + 5
fmt.Println(*p)
```

输出:
```
15
```