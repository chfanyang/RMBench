#!/bin/bash

policy_name=Mem-0

export CUDA_VISIBLE_DEVICES=4
echo -e "\033[33mGPU to use: 0\033[0m"

cd ../..  # move to project root

# Example command, you can change the arguments as needed

# M(1) evaluation format
PYTHONWARNINGS=ignore::UserWarning \
python script/eval_policy.py --config policy/${policy_name}/deploy_policy.yml --overrides \
    --task_name put_back_block \
    --execution_ckpt ./policy/Mem-0/checkpoints/final_step30000.pt \
    --state_stats_path ./policy/Mem-0/assets/put_back_block/norm_stats.json \
    --global_task "There are four mats, one block, and a button on the table. One block is on one of the mats. First, put the block to the center, then press the button. Then, put the block back in its original position." \
    --vllm_url "http://localhost:8000" \
    --action_horizon 30 # Changeable

# M(n) evaluation format
# PYTHONWARNINGS=ignore::UserWarning \
# python script/eval_policy.py --config policy/${policy_name}/deploy_policy.yml --overrides \
#     --task_name cover_blocks \
#     --execution_ckpt ./policy/Mem-0/checkpoints/model.pt \
#     --state_stats_path ./policy/Mem-0/assets/model/norm_stats.json \
#     --global_task "On the table, red, green, and blue blocks are arranged randomly along with three lids. From the current viewpoint, cover the blocks from left to right using the lids, and then uncover them again in the sequence red, green, and blue." \
#     --vllm_url "http://localhost:8000" \
#     --action_horizon 8 # Changeable
