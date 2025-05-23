export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export VLLM_LOGGING_LEVEL=DEBUG
export NCCL_DEBUG=WARN
python -m generation.generation_test \
    --model="/data/dzl/package/CodeLLM/Qwen2.5-Coder-0.5B-Instruct" \
    --dp-size=2 \
    --tp-size=1 