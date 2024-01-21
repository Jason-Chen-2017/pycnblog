#!/usr/bin/env bash

# auto_title
# 目标URL: curl -X GET "http://127.0.0.1:9000/api/ai/auto_title" -H "Request-Origion:SwaggerBootstrapUi" -H "accept:*/*"
auto_title="http://127.0.0.1:9000/api/ai/auto_title"
auto_aigc="http://127.0.0.1:9000/api/ai/auto_aigc"

# 循环执行
while true
do

    # 打印序号和时间戳
    echo "Auto Title Request $i at $(date +%Y-%m-%d_%H:%M:%S)"

    # $auto_title
    curl -X GET "$auto_title" -H "Request-Origion:SwaggerBootstrapUi" -H "accept:*/*"

    echo "Auto AIGC Request $i at $(date +%Y-%m-%d_%H:%M:%S)"
    # auto_aigc
    curl -X GET "$auto_aigc" -H  "Request-Origion:SwaggerBootstrapUi" -H  "accept:*/*"

    # 暂停n秒钟
    sleep 3600

done
