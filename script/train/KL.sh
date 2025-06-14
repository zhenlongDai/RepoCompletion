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
# export http_proxy=http://192.168.1.52:7891
# export https_proxy=http://192.168.1.52:7891
# export HTTP_PROXY=http://192.168.1.52:7891
# export HTTPS_PROXY=http://192.168.1.52:7891
export CUDA_VISIBLE_DEVICES=1,3,4
export WANDB_CONSOLE=off
export RICH_DISABLE=1
export WANDB_MODE=offline
#export NCCL_P2P_DISABLE=1
#export NCCL_IB_DISABLE=1
# accelerate launch --main_process_port 29300 --num_processes 3 --config_file configs/accelerate_configs/zero2.yaml train/choose_trainer.py \
#     --config configs/Qwen2.5-7B-Instruct/grpo/config_GRPO_context_KL.yaml
# # 如果第二个服务退出，自动kill第一个
# #kill $SERVE_PID
# export CUDA_VISIBLE_DEVICES=1,3,4
# bash script/evaluation/eval_dev_model.sh 

accelerate launch --main_process_port 29300 --num_processes 3 --config_file configs/accelerate_configs/zero2.yaml train/choose_trainer.py \
    --config configs/Qwen2.5-7B-Instruct/grpo/config_GRPO_KL.yaml



model_name="Qwen2.5-7B-Instruct-KL-IDES"
python -m evaluation.eval_model_pipeline \
    --config evaluation/config_file/evalConfig.yaml \
    --generation_params.inference_config generation/config_file/GRPOConfig.yaml \
    --models_dir ./weights/java/$model_name \
    --output_dir ./output_dir/dev_evaluation/$model_name \
    --generation_params.dev_data_path_dir "datasets/repobench/java" \
    --generation_params.test_data_path_dir "datasets/repobench/java/test" \
    --generation_params.use_lora true 

CUDA_VISIBLE_DEVICES=1,3,4 trl vllm-serve --model /data/dzl/package/CodeLLM/Qwen2.5-Coder-7B-Instruct \
    --port 8400 
# CUDA_VISIBLE_DEVICES=3 trl vllm-serve --model /data/dzl/package/CodeLLM/Qwen2.5-Coder-7B-Instruct \
#     --port 8500 
# CUDA_VISIBLE_DEVICES=4 trl vllm-serve --model /data/dzl/package/CodeLLM/Qwen2.5-Coder-7B-Instruct \
#     --port 8600 