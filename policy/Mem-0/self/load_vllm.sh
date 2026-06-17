CUDA_VISIBLE_DEVICES=3 vllm serve \
    /mnt/hwdata/cfy/RMBench/policy/Mem-0/checkpoints/Qwen3-VL-8B-Instruct-cover_blocks_uniform \
    --port 8123 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85