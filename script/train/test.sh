#!/bin/bash

# 设置代理环境变量
export http_proxy=http://192.168.1.52:7891
export https_proxy=http://192.168.1.52:7891
export HTTP_PROXY=http://192.168.1.52:7891
export HTTPS_PROXY=http://192.168.1.52:7891


# 测试网络连接
echo "测试网络连接..."
curl -I https://api.wandb.ai --max-time 30

wandb sync wandb/offline-run-20250605_235910-rkszza01 --id $(date +%s)
