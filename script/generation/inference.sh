
export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
model_name="Qwen2.5-Coder-0.5B-Instruct"
python -m generation.InferencePipeline \
    --config "generation/config_file/inferenceConfig.yaml" \
    --model "CodeLLM/$model_name" \
    --save_file_path "output_dir/generation/$model_name.json" \
    --dtype "float16" \
    --dp_size 2 \
    --tp_size 2
    