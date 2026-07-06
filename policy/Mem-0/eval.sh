#!/bin/bash

policy_name=Mem-0

export CUDA_VISIBLE_DEVICES=7
echo -e "\033[33mGPU to use: 7\033[0m"

cd ../..  # move to project root

# Example command, you can change the arguments as needed

PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/${policy_name}/deploy_policy.yml --overrides \
    --task_name cover_blocks \
    --planner_type videollama3_server \
    --use_classifier_switch False \
    --planner_query_interval 10 \
    --planner_switch_confirm_steps 1 \
    --execution_ckpt ./policy/Mem-0/checkpoints/cover_blocks/final_step30000.pt \
    --state_stats_path ./policy/Mem-0/assets/cover_blocks/norm_stats.json \
    --global_task "On the table, red, green, and blue blocks are arranged randomly along with three lids. From the current viewpoint, cover the blocks from left to right using the lids, and then uncover them again in the sequence red, green, and blue." \
    --videollama3_server.url "http://127.0.0.1:8009" \
    --action_horizon 30 # Changeable
