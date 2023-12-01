#!/usr/bin/env bash
# 目标URL: curl -X GET "http://127.0.0.1:9000/api/ai/auto_title" -H "Request-Origion:SwaggerBootstrapUi" -H "accept:*/*"
URL="http://127.0.0.1:9000/api/ai/auto_title"
# 循环执行100次
for i in {1..10}; do
    # 打印序号和时间戳
    echo "Auto Title Request $i at $(date +%Y-%m-%d_%H:%M:%S)"
    # 执行curl命令
    curl -X GET "$URL" -H "Request-Origion:SwaggerBootstrapUi" -H "accept:*/*"
    # 暂停1秒钟
    sleep 1
done
