# RepoCompletion

#### Environmental installation



2. eval Environment
```shell
uv venv IntentEnv --python 3.11 && source IntentEnv/bin/activate && uv pip install --upgrade pip
uv pip install tree_sitter==0.20.1 
uv pip install timeout-decorator==0.5.0
uv pip install fuzzywuzzy==0.18.0
uv pip install flashinfer-python
uv pip install cachetools==5.5.2
uv pip install vllm==0.8.3
uv pip install /data/dzl/package/pack/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl --no-build-isolation
```
if meet `AttributeError: 'LoRALRUCache' object has no attribute '_LRUCache__update'` when you use vllm=0.8.3
excute `uv pip install cachetools==5.5.2`
- Build tree sitter via `bash evaluation/build_treesitter.sh`
#### download dataset
1. download test data
you should execute the script `script/process_dataset.sh`

2. download train data


