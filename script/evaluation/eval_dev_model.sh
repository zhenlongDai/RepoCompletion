model_name="Qwen2.5-7B-Instruct-Context-KL-ES"
python -m evaluation.eval_model_pipeline \
    --config evaluation/config_file/evalConfig.yaml \
    --generation_params.inference_config generation/config_file/focusContextConfig.yaml \
    --models_dir ./weights/java/$model_name \
    --output_dir ./output_dir/dev_evaluation/$model_name \
    --generation_params.dev_data_path_dir "datasets/repobench/java" \
    --generation_params.test_data_path_dir "datasets/repobench/java/test" \
    --generation_params.use_lora true 