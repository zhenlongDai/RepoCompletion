export CUDA_VISIBLE_DEVICES=4,5,6,7
export NCCL_P2P_DISABLE=1
accelerate launch --config_file configs/accelerate_configs/zero2.yaml train/sft.py \
    --config configs/Qwen2.5-7B-Instruct/sft/config_demo.yaml