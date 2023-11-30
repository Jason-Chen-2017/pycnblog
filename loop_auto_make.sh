#!/usr/bin/env bash

while true; do
  # 执行你的任务
  echo "执行任务：make o  时间：$(date +%Y-%m-%d_%H:%M:%S)"
  make a
  # 等待1小时
  sleep 3600
done
