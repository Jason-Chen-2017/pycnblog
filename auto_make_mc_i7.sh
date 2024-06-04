#!/usr/bin/env bash

while true
do
  echo "执行任务：auto make mc 时间：$(date +%Y-%m-%d_%H:%M:%S)"

  make mc
  sleep 60

  make fc
  sleep 10

  make gm

  sleep 3600

done