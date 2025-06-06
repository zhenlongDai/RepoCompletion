#!/bin/bash

# 启动第一个服务（后台），记录PID
# CUDA_VISIBLE_DEVICES=4 trl vllm-serve --model /data/dzl/package/CodeLLM/Qwen2.5-Coder-7B-Instruct \
#     --port 8300 
# SERVE_PID=$!

# # 等待端口8000开放，最多等待60秒
# for i in {1..100}; do
#     if nc -z localhost 8000; then
#         echo "vllm-serve 已启动"
#         break
#     fi
#     echo "等待 vllm-serve 启动...($i)"
#     sleep 1
# done

# 启动第二个服务（前台）
# export http_proxy=http://192.168.1.52:7891
# export https_proxy=http://192.168.1.52:7891
# export HTTP_PROXY=http://192.168.1.52:7891
# export HTTPS_PROXY=http://192.168.1.52:7891
export CUDA_VISIBLE_DEVICES=1,2,3
export WANDB_CONSOLE=off
export RICH_DISABLE=1
export WANDB_MODE=offline
#export NCCL_P2P_DISABLE=1
#export NCCL_IB_DISABLE=1
accelerate launch --main_process_port 29300 --num_processes 3 --config_file configs/accelerate_configs/zero2.yaml train/grpo.py \
    --config configs/Qwen2.5-7B-Instruct/grpo/config_focusContext.yaml
# 如果第二个服务退出，自动kill第一个
#kill $SERVE_PID