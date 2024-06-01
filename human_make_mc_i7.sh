#!/usr/bin/env bash
echo "执行任务：mac human make 时间：$(date +%Y-%m-%d_%H:%M:%S)"
make mci7
sleep 60
make fci7
sleep 10
make gm