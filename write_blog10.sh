#!/usr/bin/env bash

# 目标URL
URL="http://127.0.0.1:9000/api/ai/WriteAllBlog"
# 循环执行n次
for i in {1..10}; do
    # 打印序号和时间戳
    echo "    WriteAllBlog Request $i at $(date +%Y-%m-%d_%H:%M:%S)    "
    # 执行curl命令
    curl -X GET "$URL" -H "Request-Origion:SwaggerBootstrapUi" -H "accept:*/*"
    # 暂停x秒钟
    random_number=$((RANDOM % 3 + 1))
    sleep $random_number
done
