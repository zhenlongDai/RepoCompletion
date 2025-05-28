language="java"
eval_dataset_name="repobench"
without_context=false
model_name="Qwen2.5-Coder-7B-Instruct-lora-greedy"
#如果without_context为true，则不使用上下文，prediction_file的路径需要修改
if [ "$without_context" = true ]; then
    prediction_file="./output_dir/generation/$eval_dataset_name/$language/without_context/$model_name.json"
else
    prediction_file="./output_dir/generation/$eval_dataset_name/$language/$model_name.json"
fi

python -m evaluation.eval_metric \
    --eval_dataset_name "$eval_dataset_name" \
    --language "$language" \
    --data_dir_path "datasets/$eval_dataset_name/$language/test" \
    --prediction_file "$prediction_file" \
    --ts_lib "build/$language-lang-parser.so"
    