language="java"
eval_dataset_name="repobench"
without_context=false
model_name="aixcoder-7b-v2"
eval_mode="test"
#如果without_context为true，则不使用上下文，prediction_file的路径需要修改
if [ "$without_context" = true ]; then
    prediction_file="./output_dir/generation/$eval_dataset_name/$language/without_context/$model_name.json"
else
    prediction_file="./output_dir/generation/$eval_dataset_name/$language/$model_name.json"
fi
#prediction_file="/data/dzl/RL_project/RepoCompletion/output_dir/dev_evaluation/repobench/python/Qwen2.5-7B-Instruct-intent-KL_CL-codeIDF12ES/results/checkpoint-300_generation.json"
python -m evaluation.eval_metric \
    --eval_dataset_name "$eval_dataset_name" \
    --language "$language" \
    --data_dir_path "datasets/$eval_dataset_name/$language/test" \
    --prediction_file "$prediction_file" \
    --ts_lib "build/$language-lang-parser.so" \
    --eval_mode $eval_mode
    