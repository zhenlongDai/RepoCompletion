#!/bin/bash

# 启动第一个服务（后台），记录PID
# CUDA_VISIBLE_DEVICES=4 trl vllm-serve --model /data/dzl/package/CodeLLM/Qwen2.5-Coder-7B-Instruct \
#     --port 8300 --tensor_parallel_size 2
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
export CUDA_VISIBLE_DEVICES=1,3,4
export WANDB_CONSOLE=off
export RICH_DISABLE=1
export WANDB_MODE=offline
#export NCCL_P2P_DISABLE=1
#export NCCL_IB_DISABLE=1
accelerate launch --main_process_port 29300 --num_processes 3 --config_file configs/accelerate_configs/zero2.yaml train/grpo.py \
    --config configs/Qwen2.5-7B-Instruct/grpo/config_context2.yaml
# 如果第二个服务退出，自动kill第一个
#kill $SERVE_PID


CUDA_VISIBLE_DEVICES=1,3,4 trl vllm-serve --model /data/dzl/package/CodeLLM/Qwen2.5-Coder-7B-Instruct \
    --port 8400 
# CUDA_VISIBLE_DEVICES=3 trl vllm-serve --model /data/dzl/package/CodeLLM/Qwen2.5-Coder-7B-Instruct \
#     --port 8500 
# CUDA_VISIBLE_DEVICES=4 trl vllm-serve --model /data/dzl/package/CodeLLM/Qwen2.5-Coder-7B-Instruct \
#     --port 8600 