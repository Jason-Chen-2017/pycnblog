#!/usr/bin/env bash

# 循环执行
for i in {1..30} ; do
     # 打印序号和时间戳
    echo "    WriteAllBlog Request $i at $(date +%Y-%m-%d_%H:%M:%S)    "
    # 执行curl命令
    curl -X GET "http://127.0.0.1:9000/api/ai/WriteBlogRandomlyWithLLM?model=Open_Chat_MODEL" -H  "Request-Origion:SwaggerBootstrapUi" -H  "accept:*/*"
    # 暂停n秒钟
    sleep 3
done