export CUDA_VISIBLE_DEVICES=0,3
export NCCL_P2P_DISABLE=1
model_name="starcoder2-7b"
save_name="$model_name"
language="python"
eval_dataset_name="repobench"
without_context=true
#如果without_context为true，则不使用上下文，prediction_file的路径需要修改
if [ "$without_context" = true ]; then
    save_file_path="./output_dir/generation/$eval_dataset_name/$language/without_context/$save_name.json"
else
    save_file_path="./output_dir/generation/$eval_dataset_name/$language/$save_name.json"
fi

cmd="python -m baseline.InferencePipeline \
    --config baseline/config_file/inferenceConfig.yaml \
    --model_name $model_name \
    --eval_dataset_name $eval_dataset_name \
    --model_path /data/LLMs/$model_name \
    --data_path_dir datasets/$eval_dataset_name/$language/test \
    --language $language \
    --save_file_path $save_file_path \
    --dtype float16 \
    --dp_size 2 \
    --tp_size 1 \
    --gpu_memory_utilization 0.9 \
    --temperature 0.7"

if [ "$without_context" = true ]; then
    cmd="$cmd --without_context $without_context"
fi

eval $cmd