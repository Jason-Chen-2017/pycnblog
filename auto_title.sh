#!/usr/bin/env bash
# 目标URL: curl -X GET "http://127.0.0.1:9000/api/ai/auto_title" -H "Request-Origion:SwaggerBootstrapUi" -H "accept:*/*"
auto_title="http://127.0.0.1:9000/api/ai/auto_title"
 # 打印序号和时间戳
echo "Auto Title Request $i at $(date +%Y-%m-%d_%H:%M:%S)"
curl -X GET "$auto_title" -H "Request-Origion:SwaggerBootstrapUi" -H "accept:*/*"
