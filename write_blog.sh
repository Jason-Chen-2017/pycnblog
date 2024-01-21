#!/usr/bin/env bash

# 目标URL
URL="http://127.0.0.1:9000/api/ai/WriteAllBlog"
# 循环执行
while true
do
    # 打印序号和时间戳
    echo "    WriteAllBlog Request $i at $(date +%Y-%m-%d_%H:%M:%S)    "
    # 执行curl命令
    curl -X GET "$URL" -H "Request-Origion:SwaggerBootstrapUi" -H "accept:*/*"
    # 暂停n秒钟
    sleep 1800

done
