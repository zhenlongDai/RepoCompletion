export CUDA_VISIBLE_DEVICES=2,3
model_name="Qwen2.5-Coder-7B-Instruct"
save_mode_name="Qwen2.5-7B-Instruct-CodeI"
save_name="$save_mode_name"
language="java"
eval_dataset_name="repobench"
without_context=true
lora_path="./weights/$language/$save_mode_name/checkpoint-902"
#如果without_context为true，则不使用上下文，prediction_file的路径需要修改
if [ "$without_context" = true ]; then
    save_file_path="./output_dir/generation/$eval_dataset_name/$language/without_context/$save_name.json"
else
    save_file_path="./output_dir/generation/$eval_dataset_name/$language/$save_name.json"
fi

cmd="python -m generation.InferencePipeline \
    --config generation/config_file/inferenceConfig.yaml \
    --dp_master_port 0 \
    --model ../../package/CodeLLM/$model_name \
    --data_path_dir datasets/$eval_dataset_name/$language/test \
    --language $language \
    --save_file_path $save_file_path \
    --dtype float16 \
    --dp_size 2 \
    --tp_size 1 \
    --gpu_memory_utilization 0.7 \
    --temperature 0.7 \
    --use_lora true \
    --lora_path $lora_path
    "

if [ "$without_context" = true ]; then
    cmd="$cmd --without_context $without_context"
fi

eval $cmd

     
    