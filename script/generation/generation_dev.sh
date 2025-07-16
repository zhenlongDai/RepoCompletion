export CUDA_VISIBLE_DEVICES=0,1
python -m generation.InferencePipeline --config generation/config_file/GRPOConfig.yaml \
    --model_path /data/dzl/package/CodeLLM/Qwen2.5-Coder-7B-Instruct \
    --data_path_dir datasets/repobench \
    --save_file_path output_dir/dev_evaluation/Qwen2.5-7B-Instruct-GRPO/generations/weights/java/Qwen2.5-7B-Instruct-GRPO/checkpoint-164_generation.json \
    --language java \
    --dtype float16 \
    --gpu_memory_utilization 0.7 \
    --temperature 0.7 \
    --dp_size 2 \
    --tp_size 1 \
    --use_lora True \
    --lora_path weights/java/Qwen2.5-7B-Instruct-GRPO/checkpoint-164\
    --eval_mode "dev"