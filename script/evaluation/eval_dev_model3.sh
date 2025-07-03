model_name="Qwen2.5-7B-Instruct-ablation-intent"
language="java"
dataset_name="repobench"
python -m evaluation.eval_model_pipeline \
    --config evaluation/config_file/evalConfig.yaml \
    --generation_params.inference_config generation/config_file/GRPOConfig.yaml \
    --models_dir ./weights/java/$model_name \
    --output_dir ./output_dir/dev_evaluation/$dataset_name/$language/$model_name \
    --generation_params.dev_data_path_dir "datasets/$dataset_name/$language" \
    --generation_params.test_data_path_dir "datasets/$dataset_name/$language/test" \
    --generation_params.use_lora true 

