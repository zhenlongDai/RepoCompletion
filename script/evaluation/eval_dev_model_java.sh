export CUDA_VISIBLE_DEVICES=2
cuda_visible_devices="2"
model_name="Qwen2.5-7B-Instruct-intent-KL-CL-mix_any_codeIDF1_2ES"
language="java"
ts_lib="build/$language-lang-parser.so"
model_path="/data/dzl/package/CodeLLM/Qwen2.5-Coder-7B-Instruct"
dataset_name="repobench"
python -m evaluation.eval_model_pipeline \
    --config evaluation/config_file/evalConfig.yaml \
    --generation_params.inference_config generation/config_file/IntentConfig.yaml \
    --language $language \
    --ts_lib $ts_lib \
    --models_dir ./weights/$language/$model_name \
    --output_dir ./output_dir/dev_evaluation/$dataset_name/$language/$model_name \
    --generation_params.dev_data_path_dir "datasets/$dataset_name/$language" \
    --generation_params.test_data_path_dir "datasets/$dataset_name/$language/test" \
    --generation_params.model_path $model_path \
    --generation_params.cuda_visible_devices $cuda_visible_devices \
    --generation_params.use_lora true 

