#!/usr/bin/env bash

# 获取当前日期和时间
msg="$(date)"

# 添加所有文件到git
git add *

# 提交更改到仓库
git commit -m "$msg"
# https://github.com/settings/tokens
git remote set-url origin https://Jason-Chen-2017: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@github.com/Jason-Chen-2017/pycnblog.git

git push origin master