#!/usr/bin/env bash
ps -ef|grep 'electron'|awk '{print $2}'| xargs kill -9