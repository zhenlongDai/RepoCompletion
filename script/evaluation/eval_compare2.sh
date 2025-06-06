language="java"
eval_dataset_name="repobench"
model_name="Qwen2.5-Coder-7B-Instruct-GRPO"

# 文件1：不使用context
prediction_file1="/data/dzl/RL_project/RepoCompletion/output_dir/generation/repobench/java/without_context/Qwen2.5-Coder-7B-Instruct.json"
# 文件2：使用context
prediction_file2="/data/dzl/RL_project/RepoCompletion/output_dir/dev_evaluation/Qwen2.5-7B-Instruct-CodeI2EM/results/checkpoint-820_generation.json"

python -m evaluation.eval_compare \
    --eval_dataset_name "$eval_dataset_name" \
    --language "$language" \
    --data_dir_path "datasets/$eval_dataset_name/$language/test" \
    --pred1 "$prediction_file1" \
    --pred2 "$prediction_file2" \
    --ts_lib "build/$language-lang-parser.so"
    