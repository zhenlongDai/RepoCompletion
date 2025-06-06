language="java"
eval_dataset_name="repobench"
model_name="Qwen2.5-Coder-7B-Instruct-lora"

# 文件1：不使用context
prediction_file1="./output_dir/generation/$eval_dataset_name/$language/without_context/$model_name.json"
# 文件2：使用context
prediction_file2="./output_dir/generation/$eval_dataset_name/$language/$model_name.json"

python -m evaluation.eval_compare \
    --eval_dataset_name "$eval_dataset_name" \
    --language "$language" \
    --data_dir_path "datasets/$eval_dataset_name/$language/test" \
    --pred1 "$prediction_file1" \
    --pred2 "$prediction_file2" \
    --ts_lib "build/$language-lang-parser.so"
    