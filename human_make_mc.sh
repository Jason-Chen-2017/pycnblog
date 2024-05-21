#!/usr/bin/env bash
echo "执行任务：mac human make 时间：$(date +%Y-%m-%d_%H:%M:%S)"
make mc
sleep 60
make fc
sleep 10
make gm