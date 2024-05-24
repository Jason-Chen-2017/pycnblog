
作者：禅与计算机程序设计艺术                    

# 1.简介
  


Scrapy是一个成熟、快速、可扩展的开源爬虫框架，它可以帮助开发者轻松地从网上下载所需的数据并进行有效地分析处理。在本篇教程中，我们将用到Scrapy框架，通过简单实践的方式，展示如何利用Scrapy自动下载电影排行榜网站的数据。文章最后还会介绍一些Scrapy的优点及其一些应用场景。

Scrapy是一个Python模块，采用了基于组件构建的架构，允许用户定义爬虫（也称为爬取器）的流程，通过编写抽取逻辑来抓取网页信息，然后使用转换管道对抓取到的内容进行清洗、过滤、解析、存储等操作。 

# 2.安装环境配置

2.1 安装Python

Scrapy需要运行在Python 2.7或更高版本的Python环境下。如果没有安装过Python，可以在官网下载安装包安装即可。
https://www.python.org/downloads/

2.2 安装Scrapy

Scrapy可以使用pip命令进行安装，如下命令即可完成安装：

```python
pip install Scrapy
```

2.3 创建Scrapy项目

在命令行下输入以下命令创建新Scrapy项目：

```python
scrapy startproject myspider
```

该命令创建一个名为myspider的文件夹作为项目根目录，其中包含一个名为settings.py的文件用于设置Scrapy运行参数、一个名为spiders文件夹用于放置爬虫文件。

# 3.编写第一个Spider

Scrapy提供了一个命令行工具`scrapy`，可以通过该工具执行常见的任务如生成项目、创建Spider、运行Spider等。也可以直接编写代码实现相应功能。下面我们先来编写第一个Spider。

3.1 使用Scrapy现成的模版创建Spider

在命令行下输入以下命令创建第一个Spider：

```python
cd myspider   # 进入项目根目录
scrapy genspider movie douban https://movie.douban.com/top250?start=0
```

该命令创建了一个名为movie的Spider，目标网站为豆瓣电影排行榜，初始起始页为0。

创建完毕后，可以打开myspider/spiders/movie.py文件看到Scrapy已经自动生成的代码。

3.2 修改Spider代码

我们要做的是根据网页上的电影详情页地址获取电影的详细信息，并将这些信息保存到本地。我们首先修改起始URL为豆瓣电影排行榜首页的地址https://movie.douban.com/top250，并在parse函数中提取每个电影的名称、评分、导演、主演、评价数、豆瓣链接、海报图片等属性并打印出来。

```python
import scrapy


class MovieSpider(scrapy.Spider):
    name ='movie'

    def start_requests(self):
        url = "https://movie.douban.com/top250"
        yield scrapy.Request(url, callback=self.parse)

    def parse(self, response):
        for each in response.css(".item"):
            title = each.xpath("./div[@class='hd']/a/span[1]/text()").extract_first().strip()
            rating = float(each.xpath("./div[@class='star']/span[@class='rating_num']/text()").extract_first())
            directors = [x.strip() for x in each.xpath("./div[@class='bd']/p[1]//text()[not(ancestor::a)][position()<=2]").extract()]
            actors = [x.strip() for x in each.xpath("./div[@class='bd']/p[1]//text()[not(ancestor::a)][position()>2 and position()<=4]").extract()]
            votes = int("".join([c for c in each.xpath("./div[@class='star']/span[@class='pl']/text()").extract_first() if c.isdigit()]))
            link = each.xpath("./div[@class='hd']/a/@href").extract_first()
            image = each.xpath("./div[@class='pic']/a/img/@src").extract_first()

            print("Title:", title)
            print("Rating:", rating)
            print("Directors:", ", ".join(directors))
            print("Actors:", ", ".join(actors))
            print("Votes:", votes)
            print("Link:", link)
            print("Image:", image)
            print("\n")

        next_page = response.css(".next a::attr('href')").extract_first()
        if next_page is not None:
            next_page = response.urljoin(next_page)
            yield scrapy.Request(next_page, callback=self.parse)
```

3.3 运行Spider

当Spider编写好之后，可以通过以下命令运行它：

```python
cd myspider   # 进入项目根目录
scrapy crawl movie -o movies.csv   # 将结果保存为CSV格式
```

-o选项用于指定输出文件的路径和名称，这里指定为movies.csv。运行结束后，可以查看movies.csv文件，里面保存着所有抓取到的电影信息。

3.4 总结

通过以上步骤，我们可以成功运行Scrapy爬虫，并使用XPath语法从网页上提取出所需的信息。此外，Scrapy还有许多其他强大的功能，如管道、批量抓取、数据存储等，这些都需要进一步学习。不过，Scrapy框架并不仅限于此，它也是一种很好的爬虫解决方案。