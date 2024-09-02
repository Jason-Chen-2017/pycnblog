#!/usr/bin/env bash

while true
do
  echo "执行任务：auto make book 时间：$(date +%Y-%m-%d_%H:%M:%S)"
  make mb
  sleep 30
  make gu
  sleep 3600
done