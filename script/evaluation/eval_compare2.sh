language="python"
eval_dataset_name="repobench"

# 文件1：不使用context
prediction_file1="./output_dir/generation/$eval_dataset_name/$language/without_context/deepseek-coder-6.7b-instruct.json"
# 文件2：使用context
prediction_file2="/data/dzl/RL_project/RepoCompletion/output_dir/dev_evaluation/repobench/python/deepseek-coder-6.7b-instruct-EM/results/checkpoint-500_generation.json"

python -m evaluation.eval_compare \
    --eval_dataset_name "$eval_dataset_name" \
    --language "$language" \
    --data_dir_path "datasets/$eval_dataset_name/$language/test" \
    --pred1 "$prediction_file1" \
    --pred2 "$prediction_file2" \
    --ts_lib "build/$language-lang-parser.so"
    