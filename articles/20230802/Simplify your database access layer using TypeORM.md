
作者：禅与计算机程序设计艺术                    

# 1.简介
         

        ## 为什么要使用TypeORM？

        在Node.js或者其他JavaScript运行环境中使用数据库访问层（database access layer）是一个很重要的技能，其原因主要有以下几点:

        1.使用对象关系映射（object-relational mapping，ORM）可以使得代码更加清晰易懂；
        
        2.支持TypeScript，在编码阶段就能够发现一些错误；
        
        3.提供了丰富的API，帮助开发者快速实现常用的功能；
        
        4.社区活跃、文档完善，开发者经验丰富。
        
        使用ORM可以大大减少时间成本，提升编程效率，降低维护难度，同时也方便了后期的数据迁移等工作。但使用ORM并非绝对的必要。如果项目比较简单，不涉及复杂的查询和数据处理，那么直接使用SQL语句或原生的驱动程序即可，无需依赖ORM。
        
        本文将阐述如何使用TypeORM框架进行Node.js开发中的数据库访问层的封装，并进一步给出一些示例代码来展示其用法。
        
        ## 安装TypeORM
        
        TypeORM是一个用于TypeScript和Node.js的ORM框架。它通过提供一个面向对象的接口和控制反转（IoC）模式，让开发人员可以轻松地访问数据库。本文将使用v0.2.27版本。
        
        ### 方法1：使用npm安装
        
        ```bash
        npm install typeorm --save
        ```
        
        ### 方法2：手动下载

        前往https://github.com/typeorm/typeorm/releases页面下载最新版本的TypeORM压缩包，然后解压到本地目录。在你的项目根目录下创建名为`node_modules/@types/`的子目录，然后把下载的`index.d.ts`文件拷贝到该目录下。最后修改package.json文件的`dependencies`字段，添加以下内容：
        
        ```json
        {
            "dependencies": {
                "typeorm": "^0.2.27"
            }
        }
        ```
        
        执行`npm install`命令完成安装。
        
        ## 创建连接
    
        通过TypeORM框架创建一个连接需要以下三步：
    
    1. 创建ORM配置文件，例如ormconfig.json。
    
    2. 使用createConnection()方法创建连接。
    
    3. 获取Repository。
    
    下面详细介绍上述过程。
    
    ### 1. 创建ORM配置文件

    ORM配置文件用于存放连接信息，例如数据库类型、地址、端口号、用户名、密码等。这里假设有一个MySQL数据库“test”用来测试。在项目根目录下创建一个名为ormconfig.json的文件，内容如下：

    ```json
    [
      {
        "name": "default",
        "type": "mysql",
        "host": "localhost",
        "port": 3306,
        "username": "root",
        "password": "password",
        "database": "test",
        "entities": ["src/entity/*.ts"],
        "subscribers": ["src/subscriber/**/*.ts"],
        "migrations": ["src/migration/**/*{.ts,.js}"]
      }
    ]
    ```

    此时ormconfig.json文件的内容应该如下所示：

    ```json
    [
      {
        "name": "default",
        "type": "mysql",
        "host": "localhost",
        "port": 3306,
        "username": "root",
        "password": "password",
        "database": "test",
        "entities": ["src/entity/*.ts"],
        "subscribers": ["src/subscriber/**/*.ts"],
        "migrations": ["src/migration/**/*{.ts,.js}"]
      }
    ]
    ```

    配置文件指定了数据库的相关信息，包括数据库名称、连接方式、主机地址、端口号、用户名、密码等。其中，`name`字段表示连接的名称，通常取值为`default`。
    
    `entities`字段指定了所有实体类文件的位置，这里是使用通配符匹配。这里假设我们已经创建了一个名为`entity`的目录，并将所有的实体类都放在里面。

    `subscribers`字段指定了所有的订阅者文件的位置，这里也是使用通配符匹配。

    `migrations`字段指定了所有的迁移脚本文件的位置，这里也是使用通配符匹配。
    

    ### 2. 使用createConnection()方法创建连接

    连接对象由createConnection()方法创建，其参数为一个包含数据库连接信息的字符串或者一个对象。例如：

    ```typescript
    import { createConnection } from 'typeorm';

    const connection = await createConnection({
      name: 'default',
      type:'mysql',
      host: 'localhost',
      port: 3306,
      username: 'root',
      password: 'password',
      database: 'test',
      entities: ['src/entity/*.ts'],
      subscribers: ['src/subscriber/**/*.ts'],
      migrations: ['src/migration/**/*{.ts,.js}']
    });
    ```

    可以看到，此处的参数和之前定义的ormconfig.json文件中的相同。这里使用await关键字等待连接成功建立。
    
    当连接失败的时候，会抛出一个ConnectionError的异常。我们可以通过try...catch块捕获这个异常，并根据需要做相应的处理。

    ### 3. 获取Repository

    Repository是TypeORM的核心概念之一，它代表着数据库表，它包含对数据库表的一系列操作的方法。通过Repository，我们可以非常容易地执行各种增删改查的操作。下面我们来看一下如何获取Repository。

    每个Entity都对应一个Repository。使用@Entity装饰器标识的类被TypeORM识别为一个Entity，之后可以使用@EntityRepository装饰器标注它的Repository。下面是一段示例代码：

    ```typescript
    @Entity('user') // 指定表名
    export class User {
      @PrimaryGeneratedColumn() // 设置主键生成策略
      id: number;

      @Column()
      name: string;
      
      @Column()
      age: number;
      
      @OneToMany(type => Post, post => post.author) // 一对多关系
      posts: Promise<Post[]>;
    }

    @EntityRepository(User)
    export class UserRepository extends Repository<User> {}

    @Entity('post')
    export class Post {
      @PrimaryGeneratedColumn()
      id: number;
      
      @Column()
      title: string;
      
      @ManyToOne(type => User, user => user.posts) // 一对多关系
      author: Promise<User>;
    }

    @EntityRepository(Post)
    export class PostRepository extends Repository<Post> {}
    ```

    这段代码定义了两个Entity：`User`和`Post`，每个Entity分别定义了自己的属性、关系和Repository。这些定义都是基于实体模型的，而不需要编写任何SQL语句。当连接成功建立时，Repository对象会自动创建。我们可以使用`getRepository()`方法获取到某个Entity对应的Repository。例如：

    ```typescript
    const userRepo = getRepository(User);
    const users = await userRepo.find();
    console.log(users);

    const postRepo = getRepository(Post);
    const post = new Post();
    post.title = 'Hello World!';
    post.author =... // 从用户列表中选择作者
    await postRepo.save(post);
    ```

    通过这种方式，我们就可以通过简单的方法调用来操作数据库。我们甚至可以在多个Repository之间共享查询条件。