
export CUDA_VISIBLE_DEVICES=6,7
export NCCL_P2P_DISABLE=1
model_name="Qwen2.5-Coder-7B-Instruct"
language="python"
eval_dataset_name="repobench"
python -m generation.InferencePipeline \
    --config "generation/config_file/inferenceConfig.yaml" \
    --model "../../package/CodeLLM/$model_name" \
    --data_path_dir "datasets/$eval_dataset_name/$language/test" \
    --language "$language" \
    --save_file_path "output_dir/generation/$eval_dataset_name/$language/$model_name.json" \
    --dtype "float16" \
    --dp_size 2 \
    --tp_size 1 \
    --gpu_memory_utilization 0.7
    