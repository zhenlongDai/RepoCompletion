export CUDA_VISIBLE_DEVICES=0,1,3,4
export NCCL_P2P_DISABLE=1
export WANDB_MODE=offline
accelerate launch --main_process_port 29400 --num_processes 4 --config_file configs/accelerate_configs/zero2.yaml train/sft.py \
    --config configs/deepseek-coder-6.7b-instruct/sft/config_demo.yaml

accelerate launch --main_process_port 29400 --num_processes 4 --config_file configs/accelerate_configs/zero2.yaml train/sft.py \
    --config configs/deepseek-coder-6.7b-instruct/sft/config_demo_java.yaml