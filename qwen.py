import datetime
import os
import time

from gradio_client import Client
import ast

c = [
    "服务端开发技术原理、方法与实用解决方案：服务端开发职责与技术栈的全面介绍",
    "服务端开发技术原理、方法与实用解决方案：服务端开发的核心流程与进阶路径",
    "服务端开发技术原理、方法与实用解决方案：需求分析与服务端开发",
    "服务端开发技术原理、方法与实用解决方案：抽象建模在应用",
    "服务端开发技术原理、方法与实用解决方案：系统设计与服务端开发",
    "服务端开发技术原理、方法与实用解决方案：数据设计与服务端开发",
    "服务端开发技术原理、方法与实用解决方案：非功能性设计与服务端开发",
    "服务端开发技术原理、方法与实用解决方案：高并发问题与解决方案",
    "服务端开发技术原理、方法与实用解决方案：高性能问题与解决方案",
    "服务端开发技术原理、方法与实用解决方案：高可用问题与解决方案",
    "服务端开发技术原理、方法与实用解决方案：缓存问题与解决方案",
    "服务端开发技术原理、方法与实用解决方案：数据一致性问题与解决方案",
    "服务端开发技术原理、方法与实用解决方案：幂等性问题与解决方案",
    "服务端开发技术原理、方法与实用解决方案：秒杀问题与解决方案",
    "服务端开发技术原理、方法与实用解决方案：接口设计的行业案例与规范",
    "服务端开发技术原理、方法与实用解决方案：日志打印的行业案例与规范",
    "服务端开发技术原理、方法与实用解决方案：异常处理的行业案例与规范",
    "服务端开发技术原理、方法与实用解决方案：代码编写的行业案例与规范",
    "服务端开发技术原理、方法与实用解决方案：代码注释的行业案例与规范",
    "服务端开发技术原理、方法与实用解决方案：安全问题与解决方案",
    "服务端开发技术原理、方法与实用解决方案：性能优化与解决方案",
    "服务端开发技术原理、方法与实用解决方案：负载均衡与解决方案",
    "服务端开发技术原理、方法与实用解决方案：数据库选型与性能评估",
    "服务端开发技术原理、方法与实用解决方案：容灾与业务连续性",
    "服务端开发技术原理、方法与实用解决方案：代码管理与代码托管",
    "服务端开发技术原理、方法与实用解决方案：性能监控与故障排查",
    "服务端开发技术原理、方法与实用解决方案：网络安全与数据保护",
    "服务端开发技术原理、方法与实用解决方案：容器部署与容器编排",
    "服务端开发技术原理、方法与实用解决方案：消息队列与异步处理",
    "服务端开发技术原理、方法与实用解决方案：数据缓存与缓存策略",
    "服务端开发技术原理、方法与实用解决方案：分布式事务与一致性",
    "服务端开发技术原理、方法与实用解决方案：多租户架构与实现",
    "服务端开发技术原理、方法与实用解决方案：微服务架构与实践",
    "服务端开发技术原理、方法与实用解决方案：容器化部署与管理",
    "服务端开发技术原理、方法与实用解决方案：自动化测试与持续集成",
    "服务端开发技术原理、方法与实用解决方案：监控系统与运维",
    "服务端开发技术原理、方法与实用解决方案：安全认证与授权",
    "服务端开发技术原理、方法与实用解决方案：日志分析与可视化",
    "服务端开发技术原理、方法与实用解决方案：容量规划与资源管理",
    "服务端开发技术原理、方法与实用解决方案：服务发现与负载均衡",
    "服务端开发技术原理、方法与实用解决方案：性能调优与瓶颈分析",
    "服务端开发技术原理、方法与实用解决方案：故障处理与恢复",
    "服务端开发技术原理、方法与实用解决方案：容器编排工具比较",
    "服务端开发技术原理、方法与实用解决方案：数据存储与持久化",
    "服务端开发技术原理、方法与实用解决方案：网络通信与协议",
    "服务端开发技术原理、方法与实用解决方案：系统监控与告警",
    "服务端开发技术原理、方法与实用解决方案：性能测试与压测",
    "服务端开发技术原理、方法与实用解决方案：数据分析与挖掘",
    "服务端开发技术原理、方法与实用解决方案：容错与故障转移",
    "服务端开发技术原理、方法与实用解决方案：容器编排最佳实践",
    "服务端开发技术原理、方法与实用解决方案：微服务治理与监控",
    "服务端开发技术原理、方法与实用解决方案：安全防护与漏洞修复",
    "服务端开发技术原理、方法与实用解决方案：代码质量与重构策略",
    "服务端开发技术原理、方法与实用解决方案：分布式缓存与数据一致性",
    "服务端开发技术原理、方法与实用解决方案：消息传递与异步通信",
    "服务端开发技术原理、方法与实用解决方案：容器化架构与部署",
    "服务端开发技术原理、方法与实用解决方案：自动化部署与运维",
    "服务端开发技术原理、方法与实用解决方案：安全漏洞与防护策略",
    "服务端开发技术原理、方法与实用解决方案：代码审查与质量保证",
    "服务端开发技术原理、方法与实用解决方案：性能优化与调优技巧",
    "服务端开发技术原理、方法与实用解决方案：负载均衡与高可用性",
    "服务端开发技术原理、方法与实用解决方案：容灾与故障恢复",
    "服务端开发技术原理、方法与实用解决方案：容器编排与管理工具",
    "服务端开发技术原理、方法与实用解决方案：数据库设计与优化",
    "服务端开发技术原理、方法与实用解决方案：网络通信与安全",
    "服务端开发技术原理、方法与实用解决方案：监控与日志分析",
    "服务端开发技术原理、方法与实用解决方案：性能测试与负载压测",
    "服务端开发技术原理、方法与实用解决方案：数据分析与挖掘",
    "服务端开发技术原理、方法与实用解决方案：容错与故障恢复",
    "服务端开发技术原理、方法与实用解决方案：容器编排最佳实践",
    "服务端开发技术原理、方法与实用解决方案：微服务治理与监控",
    "服务端开发技术原理、方法与实用解决方案：安全防护与漏洞修复",
    "服务端开发技术原理、方法与实用解决方案：代码质量与重构策略",
    "服务端开发技术原理、方法与实用解决方案：分布式缓存与数据一致性",
    "服务端开发技术原理、方法与实用解决方案：消息传递与异步通信",
    "服务端开发技术原理、方法与实用解决方案：容器化架构与部署",
    "服务端开发技术原理、方法与实用解决方案：自动化部署与运维",
    "服务端开发技术原理、方法与实用解决方案：安全漏洞与防护策略",
    "服务端开发技术原理、方法与实用解决方案：代码审查与质量保证",
    "服务端开发技术原理、方法与实用解决方案：性能优化与调优技巧",
    "服务端开发技术原理、方法与实用解决方案：异步编程与事件驱动架构",
    "服务端开发技术原理、方法与实用解决方案：分布式系统设计与实现",
    "服务端开发技术原理、方法与实用解决方案：容器化部署与微服务架构",
    "服务端开发技术原理、方法与实用解决方案：数据存储与数据库优化策略",
    "服务端开发技术原理、方法与实用解决方案：网络通信与协议设计",
    "服务端开发技术原理、方法与实用解决方案：安全防护与漏洞修复",
    "服务端开发技术原理、方法与实用解决方案：性能优化与负载均衡",
    "服务端开发技术原理、方法与实用解决方案：日志分析与监控系统",
    "服务端开发技术原理、方法与实用解决方案：容灾与故障恢复策略",
    "服务端开发技术原理、方法与实用解决方案：容器编排与管理工具",
    "服务端开发技术原理、方法与实用解决方案：自动化部署与持续集成",
    "服务端开发技术原理、方法与实用解决方案：代码审查与质量保证",
    "服务端开发技术原理、方法与实用解决方案：大数据处理与分析",
    "服务端开发技术原理、方法与实用解决方案：消息队列与异步通信",
    "服务端开发技术原理、方法与实用解决方案：高可用性与负载均衡",
    "服务端开发技术原理、方法与实用解决方案：容器化架构与部署",
    "服务端开发技术原理、方法与实用解决方案：自动化部署与运维",
    "服务端开发技术原理、方法与实用解决方案：安全漏洞与防护策略",
    "服务端开发技术原理、方法与实用解决方案：代码审查与质量保证",
    "服务端开发技术原理、方法与实用解决方案：性能优化与调优技巧",
    "服务端开发技术原理、方法与实用解决方案：负载均衡与高可用性",
    "服务端开发技术原理、方法与实用解决方案：容灾与故障恢复",
    "服务端开发技术原理、方法与实用解决方案：容器编排与管理工具",
    "服务端开发技术原理、方法与实用解决方案：数据库设计与优化",
    "服务端开发技术原理、方法与实用解决方案：网络通信与安全",
    "服务端开发技术原理、方法与实用解决方案：监控与日志分析",
    "服务端开发技术原理、方法与实用解决方案：性能测试与负载压测",
    "服务端开发技术原理、方法与实用解决方案：数据分析与挖掘",
    "服务端开发技术原理、方法与实用解决方案：容错与故障恢复",
    "服务端开发技术原理、方法与实用解决方案：消息通信与异步处理",
    "服务端开发技术原理、方法与实用解决方案：高性能计算与并行处理",
    "服务端开发技术原理、方法与实用解决方案：容器化部署与自动扩展",
    "服务端开发技术原理、方法与实用解决方案：数据库优化与索引设计",
    "服务端开发技术原理、方法与实用解决方案：网络安全与防护策略",
    "服务端开发技术原理、方法与实用解决方案：日志管理与分析工具",
    "服务端开发技术原理、方法与实用解决方案：容灾与系统可靠性",
    "服务端开发技术原理、方法与实用解决方案：容器编排与集群管理",
    "服务端开发技术原理、方法与实用解决方案：持续集成与持续部署",
    "服务端开发技术原理、方法与实用解决方案：代码质量与测试覆盖",
    "服务端开发技术原理、方法与实用解决方案：大数据处理与分布式计算",
    "服务端开发技术原理、方法与实用解决方案：消息队列与事件驱动",
    "服务端开发技术原理、方法与实用解决方案：高可用架构与故障恢复",
    "服务端开发技术原理、方法与实用解决方案：容器化部署与运维",
    "服务端开发技术原理、方法与实用解决方案：安全漏洞与攻击防护",
    "服务端开发技术原理、方法与实用解决方案：代码审查与持续集成",
    "服务端开发技术原理、方法与实用解决方案：性能优化与调试技巧",
    "服务端开发技术原理、方法与实用解决方案：负载均衡与高可用",
    "服务端开发技术原理、方法与实用解决方案：容灾与故障恢复策略",
    "服务端开发技术原理、方法与实用解决方案：容器编排与管理工具",
    "服务端开发技术原理、方法与实用解决方案：数据库设计与优化",
    "服务端开发技术原理、方法与实用解决方案：网络通信与安全防护",
    "服务端开发技术原理、方法与实用解决方案：监控与日志分析工具",
    "服务端开发技术原理、方法与实用解决方案：性能测试与负载压力",
    "服务端开发技术原理、方法与实用解决方案：数据分析与挖掘技术",
    "服务端开发技术原理、方法与实用解决方案：容错与故障恢复策略",
    "服务端开发技术原理、方法与实用解决方案：消息通信与异步处理",
    "服务端开发技术原理、方法与实用解决方案：高性能计算与并行处理",
    "服务端开发技术原理、方法与实用解决方案：容器化部署与自动扩展",
    "服务端开发技术原理、方法与实用解决方案：数据库优化与索引设计",
    "服务端开发技术原理、方法与实用解决方案：网络安全与防护策略",

]

client = Client("https://modelscope.cn/api/v1/studio/qwen/Qwen-14B-Chat-Demo/gradio/")


def complete(x):
    y = ''

    try:
        result = client.predict(
            x,  # str in 'Input' Textbox component
            "",  # str (filepath to JSON file) in 'Qwen-14B-Chat' Chatbot component
            fn_index=0
        )

        file = open(result[1], 'r')
        file_content = file.read()

        # 使用 ast.literal_eval 解析文本内容
        parsed_content = ast.literal_eval(file_content)

        # 提取 "b"
        if len(parsed_content) > 0 and len(parsed_content[0]) > 1:
            y = parsed_content[0][1]

    except Exception as e:
        # 处理其他异常
        print("发生了一个错误：", str(e))

    return y


# 创建文件名为title，写入内容 output
def writeMD(title, output):
    # 创建目标目录
    # 获取当前日期
    date = datetime.datetime.today()
    # 格式化成yyyyMMdd
    date_str = date.strftime('%Y%m%d')
    target_directory = './qwen/' + date_str
    os.makedirs(target_directory, exist_ok=True)

    with open(target_directory + '/' + title + '.md', 'w', encoding='utf-8') as file:
        file.write(output)
        print(target_directory + '《' + title + "》文件写入成功！")


if __name__ == '__main__':

    for title in c:
        input = '写一篇技术文章《' + title + '》，文章核心内容包括：1.简介 2.基本概念和关系 3.核心原理讲解 4.具体的数学模型算法公式（latex格式，嵌入使用$$）5.代码实例 6.总结和未来展望.文章内容不少于5000字，markdown格式。'
        start_time = time.time()  # 记录开始时间

        output = complete(input)

        for i in range(0, 2):
            input = output
            output = input + '\n' + complete(input)  # 重复执行 complete 函数
            if output.__contains__('## 6.') or output.__contains__('## 7.'):
                break

        end_time = time.time()  # 记录结束时间
        execution_time = end_time - start_time  # 计算执行时间

        print(output, "\nLength:", len(output), "\nExecution time:", execution_time, "seconds")

        writeMD(title, output)
