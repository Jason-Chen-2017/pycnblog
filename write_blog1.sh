#!/usr/bin/env bash

# 目标URL
URL="http://127.0.0.1:9000/api/ai/WriteAllBlog"

echo "    WriteAllBlog Request : $(date +%Y-%m-%d_%H:%M:%S)    "
# 执行curl命令
curl -X GET "$URL" -H "Request-Origion:SwaggerBootstrapUi" -H "accept:*/*"

