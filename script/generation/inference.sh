export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
model_name="CodeLlama-34b-Instruct"
save_name="$model_name"
language="java"
eval_dataset_name="repobench" #repobench/cceval
prompt_mode="split" #split/comment
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
    --prompt_mode $prompt_mode \
    --model_path ../../package/CodeLLM/$model_name \
    --data_path_dir $data_path_dir \
    --language $language \
    --save_file_path $save_file_path \
    --dtype float16 \
    --dp_size 1 \
    --tp_size 2 \
    --gpu_memory_utilization 0.9 \
    --temperature 0.7"

if [ "$without_context" = true ]; then
    cmd="$cmd --without_context $without_context"
fi

eval $cmd