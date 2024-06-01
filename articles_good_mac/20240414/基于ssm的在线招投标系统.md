# 基于SSM的在线招投标系统

## 1. 背景介绍

### 1.1 招投标系统概述

招投标系统是一种用于管理采购过程的软件应用程序。它允许买方(招标方)发布项目需求,卖方(投标方)则可以对这些需求进行投标。该系统的主要目的是提供一个公平、透明的平台,使采购过程更加高效和规范化。

### 1.2 传统招投标模式的缺陷

传统的招投标过程通常是线下进行的,存在以下一些主要缺陷:

- 信息不对称和不透明
- 效率低下,周期长
- 纸质文件管理混乱
- 人为操作增加舞弊风险

### 1.3 在线招投标系统的优势

相比之下,在线招投标系统具有以下优势:

- 提高信息透明度
- 提高工作效率 
- 降低运营成本
- 减少人为干预
- 数据化管理

## 2. 核心概念与联系

### 2.1 系统角色

在线招投标系统通常包括以下三个主要角色:

- 招标方(买方)
- 投标方(卖方)
- 系统管理员

### 2.2 业务流程

典型的业务流程包括:

1. 招标方发布项目需求
2. 投标方查看需求并投标 
3. 招标方评审投标
4. 中标结果公示
5. 签订合同

### 2.3 核心功能模块

为支持上述业务,系统通常包含以下核心模块:

- 项目管理模块
- 投标管理模块 
- 评标管理模块
- 用户权限模块
- 消息通知模块

## 3. 核心算法原理和具体操作步骤

### 3.1 密码加密算法

为确保用户密码的安全性,系统需要使用加密算法对密码进行单向哈希。常用的单向哈希算法有MD5、SHA等。

具体操作步骤如下:

1. 用户输入原始密码
2. 系统使用加密算法(如MD5)对密码进行哈希计算,得到固定长度的哈希值
3. 将哈希值而非原始密码存储在数据库中
4. 用户登录时,系统对用户输入的密码进行相同的哈希计算,并与存储的哈希值进行比对

密码加密公式:

$$
H(x) = h(h(...h(x)))
$$

其中 $H(x)$ 为最终哈希值, $h$ 为单向哈希函数,对 $x$ 进行 $n$ 次迭代计算。

### 3.2 投标评分算法

为公平评审各投标方的投标,系统需要实现一种评分算法。一种常见的做法是将评分项目赋予不同权重,并基于此计算加权分数。

假设有 $m$ 个评分项目,权重分别为 $w_1, w_2, ..., w_m$,且 $\sum_{i=1}^m w_i = 1$。某投标方在各项目的得分为 $s_1, s_2, ..., s_m$,则其加权总分为:

$$
S = \sum_{i=1}^m w_i s_i
$$

该算法的具体实现步骤为:

1. 系统管理员设置评分项目及其权重
2. 评委为每个投标方的每个项目打分
3. 系统计算每个投标方的加权总分 $S$
4. 根据总分从高到低排序,作为中标参考

### 3.3 项目进度计算

为跟踪项目的实施进度,系统需要具备计算和展示进度的功能。

假设一个项目包含 $n$ 个里程碑事件,每个事件完成时间分别为 $t_1, t_2, ..., t_n$,则当前时间 $t$ 时的项目进度为:

$$
\text{Progress} = \frac{\sum_{i=1}^n w_i I(t \geq t_i)}{\sum_{i=1}^n w_i}
$$

其中 $w_i$ 为第 $i$ 个事件的权重, $I(t \geq t_i)$ 为指示函数,当 $t \geq t_i$ 时值为1,否则为0。

该算法实现步骤为:

1. 项目经理为项目设置里程碑事件及权重
2. 记录每个事件的实际完成时间
3. 系统根据当前时间 $t$ 计算进度百分比
4. 在界面上以进度条等形式展示

## 4. 数学模型和公式详细讲解举例说明

### 4.1 密码加密算法详解

我们以MD5算法为例,详细解释单向哈希加密的原理。MD5是一种广泛使用的加密哈希函数,它可以将任意长度的输入消息转换为一个128位(16字节)的哈希值。

MD5算法包括以下主要步骤:

1. **填充**:将输入消息的长度补足为64位的倍数
2. **初始化**:准备4个32位链接变量作为种子
3. **循环处理**:将填充后的消息分成512位(16字词)的数据组,并进行64轮迭代运算
4. **输出**:将最终的4个链接变量级联,生成128位的哈希值

以"Hello World"为例,其MD5哈希值计算过程如下:

1. 填充后的二进制消息为: `0x48656C6C6F20576F726C6480000000000000000000000000000000`
2. 初始链接变量为: `0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476`
3. 经过64轮迭代运算后,最终链接变量为: `0x8F14E9B5, 0x2B1B1472, 0x5B1D1AB8, 0x1E2EF7F0`  
4. 级联后的128位哈希值为: `0x8F14E9B52B1B14725B1D1AB81E2EF7F0`

可以看出,即使只修改一个字符,如将"Hello World"改为"Hello World!",其哈希值就会完全不同。这体现了MD5哈希的"雪崩效应",能有效防止密码被反解。

### 4.2 投标评分算法实例

假设某项目有3个评分项目,权重分别为0.4、0.3和0.3。现有两个投标方A和B,各项目得分如下:

- A: 90分、85分、80分
- B: 88分、92分、90分  

我们可以计算出两个投标方的加权总分:

$$
\begin{aligned}
S_A &= 0.4 \times 90 + 0.3 \times 85 + 0.3 \times 80 = 86 \\
S_B &= 0.4 \times 88 + 0.3 \times 92 + 0.3 \times 90 = 89.6
\end{aligned}
$$

因此,投标方B的加权总分更高,可视为更合适的中标候选人。

### 4.3 项目进度计算示例

某软件项目包含以下5个里程碑事件:

1. 需求评审 (2023-03-01, 权重0.1)
2. 设计评审 (2023-04-15, 权重0.2) 
3. 代码评审 (2023-06-30, 权重0.3)
4. 系统测试 (2023-08-31, 权重0.2)
5. 上线部署 (2023-10-15, 权重0.2)

假设当前时间为2023-07-15,那么到该时间为止,前3个事件已完成,后两个尚未完成。根据进度计算公式:

$$
\begin{aligned}
\text{Progress} &= \frac{0.1 \times 1 + 0.2 \times 1 + 0.3 \times 1 + 0.2 \times 0 + 0.2 \times 0}{0.1 + 0.2 + 0.3 + 0.2 + 0.2} \\
&= \frac{0.6}{1} \\
&= 60\%
\end{aligned}
$$

因此,在2023-07-15时,该项目的实施进度为60%。

## 5. 项目实践:代码实例和详细解释说明

### 5.1 系统架构

本系统采用经典的SSM(Spring + SpringMVC + MyBatis)架构,具体技术栈如下:

- 核心框架: Spring 5.3.9
- Web框架: SpringMVC 5.3.9
- ORM框架: MyBatis 3.5.7
- 数据库: MySQL 8.0.27
- 前端框架: Bootstrap 5.1.3

### 5.2 用户模块实现

以用户模块为例,我们来看看具体的代码实现。

**1. 数据库表结构**

```sql
CREATE TABLE `user` (
  `id` int NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `username` varchar(50) NOT NULL COMMENT '用户名',
  `password` varchar(100) NOT NULL COMMENT '密码',
  `role` varchar(20) NOT NULL COMMENT '角色',
  PRIMARY KEY (`id`),
  UNIQUE KEY `username` (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户表';
```

**2. MyBatis映射文件**

```xml
<mapper namespace="com.example.dao.UserDao">
    <resultMap id="userResultMap" type="com.example.entity.User">
        <id column="id" property="id"/>
        <result column="username" property="username"/>
        <result column="password" property="password"/>
        <result column="role" property="role"/>
    </resultMap>

    <select id="getUserByUsername" parameterType="string" resultMap="userResultMap">
        SELECT * FROM user WHERE username = #{username}
    </select>

    <!-- 其他CRUD方法... -->
</mapper>
```

**3. Service层**

```java
@Service
public class UserServiceImpl implements UserService {

    @Autowired
    private UserDao userDao;

    @Override
    public User getUserByUsername(String username) {
        return userDao.getUserByUsername(username);
    }

    // 其他业务方法...
}
```

**4. 密码加密工具类**

```java
import java.security.MessageDigest;

public class PasswordUtil {
    
    public static String md5(String password) {
        try {
            MessageDigest md = MessageDigest.getInstance("MD5");
            byte[] hashBytes = md.digest(password.getBytes());
            StringBuilder sb = new StringBuilder();
            for (byte b : hashBytes) {
                sb.append(String.format("%02x", b));
            }
            return sb.toString();
        } catch (Exception ex) {
            throw new RuntimeException(ex);
        }
    }
}
```

**5. 登录控制器**

```java
@Controller
@RequestMapping("/user")
public class UserController {

    @Autowired
    private UserService userService;

    @PostMapping("/login")
    public String login(String username, String password, Model model) {
        User user = userService.getUserByUsername(username);
        if (user != null && PasswordUtil.md5(password).equals(user.getPassword())) {
            // 登录成功
            return "redirect:/main";
        } else {
            // 登录失败
            model.addAttribute("error", "Invalid username or password");
            return "login";
        }
    }
}
```

在上面的示例中,我们首先定义了`user`表的结构,包括`id`、`username`、`password`和`role`字段。

然后在MyBatis映射文件中,配置了`getUserByUsername`方法,用于根据用户名查询用户信息。

在Service层的`UserServiceImpl`中,我们注入了`UserDao`实例,并提供了`getUserByUsername`方法的实现。

`PasswordUtil`是一个工具类,提供了MD5加密的静态方法。

最后,在`UserController`中,我们实现了`/user/login`端点,用于处理登录请求。在登录时,首先根据用户名查询用户信息,然后将输入的密码进行MD5加密,并与数据库中存储的密码哈希值进行比对,如果匹配则登录成功。

通过这个实例,我们可以看到如何在SSM架构中实现用户认证功能,并使用MD5算法对密码进行单向哈希加密。

### 5.3 投标评分模块

接下来,我们来看看投标评分模块的实现。

**1. 数据库表结构**

```sql
CREATE TABLE `bid_item` (
  `id` int NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `name` varchar(100) NOT NULL COMMENT '评分项目名称', 
  `weight` decimal(4,2) NOT NULL COMMENT '权重',
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='评分项目表';

CREATE TABLE `bid_score` (
  `id` int NOT NULL AUTO_INCREMENT COMMENT '主键ID',
  `bid_id` int NOT NULL COMMENT '投标ID',
  `item_id` int NOT NULL COMMENT '评分项目ID',
  `score` int NOT NULL COMMENT '分数',
  PRIMARY KEY (`id`),
  FOREIGN KEY (`bid_id`) REFERENCES `bid`(`id`),
  FOREIGN KEY (`item_id`) REFERENCES `bid_item`(`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='投标评分表';
```

**2. Service层**

```java
@Service
public class BidServiceImpl implements Bid