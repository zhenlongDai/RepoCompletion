export CUDA_VISIBLE_DEVICES=0,1
cuda_visible_devices="0,1"
model_name="deepseek-coder-6.7b-instruct-EM"
language="python"
ts_lib="build/$language-lang-parser.so"
base_model_path="/data/dzl/package/CodeLLM/deepseek-coder-6.7b-instruct"
dataset_name="repobench"
python -m evaluation.eval_model_pipeline \
    --config evaluation/config_file/evalConfig.yaml \
    --generation_params.inference_config generation/config_file/GRPOConfig.yaml \
    --language $language \
    --ts_lib $ts_lib \
    --models_dir ./weights/$language/$model_name \
    --output_dir ./output_dir/dev_evaluation/$dataset_name/$language/$model_name \
    --generation_params.dev_data_path_dir "datasets/$dataset_name/$language" \
    --generation_params.test_data_path_dir "datasets/$dataset_name/$language/test" \
    --generation_params.model_path $base_model_path \
    --generation_params.cuda_visible_devices $cuda_visible_devices \
    --generation_params.use_lora true 

