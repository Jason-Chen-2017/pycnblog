#!/usr/bin/env bash
echo "执行任务：human make 时间：$(date +%Y-%m-%d_%H:%M:%S)"
make m
sleep 60
make f
sleep 10
make g