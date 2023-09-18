
作者：禅与计算机程序设计艺术                    

# 1.简介
  

Scrapy是一个开源的、使用Python开发的快速高效的用于网页抓取(Web Crawling)的框架。本教程将介绍Scrapy框架的一些主要特性及其应用场景，并展示如何使用它完成简单的网络数据爬取任务，如爬取各类新闻网站的新闻内容等。希望能够帮助读者更好地理解和掌握Scrapy框架，提升个人开发能力和编程水平。

# 2.安装配置
## 安装
### Ubuntu/Debian
Scrapy依赖于Twisted异步框架，如果系统中没有安装该依赖项，可以通过以下命令进行安装：

```
sudo apt-get install python-twisted
```

然后通过pip安装Scrapy：

```
pip install Scrapy
```

### Windows
下载安装包后安装即可。

## 配置
在使用Scrapy之前，需要先配置Scrapy的默认设置，这样就可以方便地运行和调试爬虫脚本。

首先，进入到Scrapy目录下的scrapy.cfg文件，修改下面的配置信息：

```
[settings]
default = myproject.settings
```

这里，“myproject”表示项目名称，可以任意指定。

接着，创建settings.py文件，并编辑其中的设置信息，如下所示：

```python
# -*- coding: utf-8 -*-
BOT_NAME ='mybot'

SPIDER_MODULES = ['myspider.spiders']
NEWSPIDER_MODULE ='myspider.spiders'

DOWNLOADER_MIDDLEWARES = {
    # Engine side
   'scrapy.downloadermiddlewares.robotstxt.RobotsTxtMiddleware': 100,
   'scrapy.downloadermiddlewares.httpauth.HttpAuthMiddleware': 300,
   'scrapy.downloadermiddlewares.downloadtimeout.DownloadTimeoutMiddleware': 350,
   'scrapy.downloadermiddlewares.defaultheaders.DefaultHeadersMiddleware': 400,
   'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': 500,
   'scrapy.downloadermiddlewares.retry.RetryMiddleware': 550,
   'scrapy.downloadermiddlewares.ajaxcrawl.AjaxCrawlMiddleware': 560,
   'scrapy.downloadermiddlewares.redirect.MetaRefreshMiddleware': 580,
   'scrapy.downloadermiddlewares.httpcompression.HttpCompressionMiddleware': 590,
   'scrapy.downloadermiddlewares.redirect.RedirectMiddleware': 600,
   'scrapy.downloadermiddlewares.cookies.CookiesMiddleware': 700,
   'scrapy.downloadermiddlewares.httpproxy.HttpProxyMiddleware': 750,
   'scrapy.downloadermiddlewares.stats.DownloaderStats': 850,
   'scrapy.downloadermiddlewares.httpcache.HttpCacheMiddleware': 900,

    # Downloader side
}

ITEM_PIPELINES = {
   'myspider.pipelines.MyspiderPipeline': 300,
}

LOG_LEVEL = 'INFO'
```

这里，BOT_NAME对应的是爬虫的名字，也是你的Scrapy项目名称，你可以自己指定。

SPIDER_MODULES对应的是Spider模块所在的路径，也就是你创建Spider的文件夹路径；NEWSPIDER_MODULE对应的是默认Spider生成器所在的路径。

DOWNLOADER_MIDDLEWARES定义了Scrapy要使用的下载中间件列表，如需自定义中间件，可以参考Scrapy文档进行编写。

ITEM_PIPELINES定义了Scrapy要使用的item pipeline列表，负责处理爬取到的Item对象，如存储、输出、发送电子邮件等。

LOG_LEVEL定义了Scrapy日志记录级别，可设置为DEBUG、INFO、WARNING、ERROR等。

至此，Scrapy的配置工作已经完成，接下来就可以进行编写爬虫脚本了。