#!/usr/bin/env bash

while true
do
  echo "执行任务：auto make 时间：$(date +%Y-%m-%d_%H:%M:%S)"
  make m
  sleep 30
  make f
  sleep 30
  make gu
  sleep 3600
done