export CUDA_VISIBLE_DEVICES=5
export NCCL_P2P_DISABLE=1
export WANDB_MODE=offline
accelerate launch --main_process_port 29400 --num_processes 1 --config_file configs/accelerate_configs/zero2.yaml train/sft.py \
    --config configs/Qwen2.5-7B-Instruct/sft/config_demo.yaml