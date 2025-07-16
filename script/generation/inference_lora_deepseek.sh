export CUDA_VISIBLE_DEVICES=2,5
export NCCL_P2P_DISABLE=1
language="python"
model_name="Qwen2.5-Coder-7B-Instruct"
save_name="$model_name-Lora"
lora_path="./weights/$language/Qwen2.5-7B-Instruct-Lora/checkpoint-475"
eval_dataset_name="cceval" #repobench/cceval
if [ "$eval_dataset_name" = "repobench" ]; then
    data_path_dir="datasets/$eval_dataset_name/$language/test"
else
    data_path_dir="datasets/$eval_dataset_name/$language"
fi

without_context=false
#如果without_context为true，则不使用上下文，prediction_file的路径需要修改
if [ "$without_context" = true ]; then
    save_file_path="./output_dir/generation/$eval_dataset_name/$language/without_context/$save_name.json"
else
    save_file_path="./output_dir/generation/$eval_dataset_name/$language/$save_name.json"
fi

cmd="python -m generation.InferencePipeline \
    --config generation/config_file/inferenceConfig.yaml \
    --eval_dataset_name $eval_dataset_name \
    --model_path ../../package/CodeLLM/$model_name \
    --data_path_dir $data_path_dir \
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



# export CUDA_VISIBLE_DEVICES=2
# model_name="Qwen2.5-Coder-7B-Instruct"
# save_name="$model_name-Lora"
# language="python"
# eval_dataset_name="repobench"
# without_context=false
# lora_path="./weights/$language/Qwen2.5-7B-Instruct-Lora/checkpoint-475"
# #如果without_context为true，则不使用上下文，prediction_file的路径需要修改
# if [ "$without_context" = true ]; then
#     save_file_path="./output_dir/generation/$eval_dataset_name/$language/without_context/$save_name.json"
# else
#     save_file_path="./output_dir/generation/$eval_dataset_name/$language/$save_name.json"
# fi

# cmd="python -m generation.InferencePipeline \
#     --config generation/config_file/inferenceConfig.yaml \
#     --model ../../package/CodeLLM/$model_name \
#     --data_path_dir datasets/$eval_dataset_name/$language/test \
#     --language $language \
#     --save_file_path $save_file_path \
#     --dtype float16 \
#     --dp_size 1 \
#     --tp_size 1 \
#     --gpu_memory_utilization 0.7 \
#     --temperature 0.7 \
#     --use_lora true \
#     --lora_path $lora_path
#     "

# if [ "$without_context" = true ]; then
#     cmd="$cmd --without_context $without_context"
# fi

# eval $cmd
