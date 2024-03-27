#!/usr/bin/env bash

while true
do
  echo "执行任务：auto make mc 时间：$(date +%Y-%m-%d_%H:%M:%S)"

  make mc
  sleep 60

  make fm
  sleep 10

  make g

  sleep 3600

done