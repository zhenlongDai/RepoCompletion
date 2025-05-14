language="java"
eval_dataset_name="repobench"

python -m evaluation.eval_metric \
    --eval_dataset_name "$eval_dataset_name" \
    --language "$language" \
    --data_dir_path "datasets/$eval_dataset_name/$language/test" \
    --prediction_file "./output_dir/generation/$eval_dataset_name/$language/Qwen2.5-Coder-0.5B-Instruct.json" \
    --ts_lib "build/$language-lang-parser.so"
    