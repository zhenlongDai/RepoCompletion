#!/bin/bash
# 测试网络连接
echo "测试网络连接..."
curl -I https://api.wandb.ai --max-time 30

wandb sync wandb/offline-run-20250605_235910-rkszza01 --id $(date +%s)
