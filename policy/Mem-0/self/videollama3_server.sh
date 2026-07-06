export CUDA_VISIBLE_DEVICES=6

/mnt/hwdata/cfy/miniconda3/envs/videollama3/bin/python \
    scripts/serve_videollama3_planner.py \
    --host 0.0.0.0 \
    --port 8009 \
    --repo_path /mnt/hwdata/cfy/VideoLLaMA3 \
    --base_model DAMO-NLP-SG/VideoLLaMA3-7B \
    --lora_path /mnt/hwdata/cfy/VideoLLaMA3/work_dirs/rmbench_cover_blocks_stream_frames_random_demo_clean/checkpoint-520 \
    --device cuda:0 \
    --fps 1 \
    --max_frames 128 \
    --max_new_tokens 128
