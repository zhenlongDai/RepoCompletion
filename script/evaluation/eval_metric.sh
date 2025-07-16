language="python"
without_context=false
model_name="starcoder2-15b"
eval_mode="test"
#如果without_context为true，则不使用上下文，prediction_file的路径需要修改
eval_dataset_name="repobench" #repobench/cceval
if [ "$eval_dataset_name" = "repobench" ]; then
    data_path_dir="datasets/$eval_dataset_name/$language/test"
else
    data_path_dir="datasets/$eval_dataset_name/$language"
fi

if [ "$without_context" = true ]; then
    prediction_file="./output_dir/generation/$eval_dataset_name/$language/without_context/$model_name.json",
else
    prediction_file="./output_dir/generation/$eval_dataset_name/$language/$model_name.json"
fi
prediction_file="/data/dzl/RL_project/RepoCompletion/output_dir/generation/repobench/python/CodeLlama-34b-Instruct.json"
python -m evaluation.eval_metric \
    --eval_dataset_name "$eval_dataset_name" \
    --language "$language" \
    --data_dir_path $data_path_dir \
    --prediction_file "$prediction_file" \
    --ts_lib "build/$language-lang-parser.so" \
    --eval_mode $eval_mode
    