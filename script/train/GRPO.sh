#!/bin/bash

# act the first service (background), record PID
CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model /data/dzl/package/CodeLLM/Qwen2.5-Coder-7B-Instruct \
    --port 8100 --tensor_parallel_size 2
SERVE_PID=$!

# wait for port 8000 to be open, max wait time is 100 seconds
for i in {1..100}; do
    if nc -z localhost 8100; then
        echo "vllm-serve started successfully on port 8100"
        break
    fi
    echo "waiting for vllm-serve...($i)"
    sleep 1
done

export CUDA_VISIBLE_DEVICES=1,3,4
export WANDB_CONSOLE=off
export RICH_DISABLE=1
export WANDB_MODE=offline

accelerate launch --main_process_port 29300 --num_processes 3 --config_file configs/accelerate_configs/zero2.yaml train/grpo.py \
    --config configs/Qwen2.5-7B-Instruct/grpo/config_context2.yaml

