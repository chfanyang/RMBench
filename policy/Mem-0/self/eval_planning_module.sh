export CUDA_VISIBLE_DEVICES=1
# python self/eval_planning_module.py \
#     --data_json llamafactory_data/battery_try_eval/battery_try_eval_high_level_finetune_data.json \
#     --image_base_dir llamafactory_data/battery_try_eval \
#     --base_url http://localhost:8123/v1 \
#     --model /mnt/hwdata/cfy/RMBench/policy/Mem-0/checkpoints/Qwen3-VL-8B-Instruct-battery_try \
#     --max_samples 150 \
#     --out planner_eval.jsonl

# python self/eval_planning_module.py \
#     --data_json /mnt/hwdata/cfy/RMBench/policy/Mem-0/llamafactory_data/cover_blocks_eval_subtask/cover_blocks_eval_subtask_high_level_finetune_data.json \
#     --image_base_dir /mnt/hwdata/cfy/RMBench/policy/Mem-0/llamafactory_data/cover_blocks_eval_subtask \
#     --base_url http://localhost:8123/v1 \
#     --model /mnt/hwdata/cfy/RMBench/policy/Mem-0/checkpoints/Qwen3-VL-8B-Instruct-cover_blocks_subtask \
#     --max_samples 150 \
#     --out planner_eval.jsonl



# python self/eval_planning_module.py \
#     --data_json /mnt/hwdata/cfy/RMBench/policy/Mem-0/llamafactory_data/cover_blocks_eval_uniform/cover_blocks_eval_uniform_high_level_finetune_data.json \
#     --image_base_dir /mnt/hwdata/cfy/RMBench/policy/Mem-0/llamafactory_data/cover_blocks_eval_uniform \
#     --base_url http://localhost:8123/v1 \
#     --model /mnt/hwdata/cfy/RMBench/policy/Mem-0/checkpoints/Qwen3-VL-8B-Instruct-cover_blocks_uniform \
#     --max_samples 10000 \
#     --out planner_eval.jsonl

python self/eval_planning_module.py \
    --data_json /mnt/hwdata/cfy/RMBench/policy/Mem-0/llamafactory_data/cover_blocks_uniform/cover_blocks_uniform_high_level_finetune_data.json \
    --image_base_dir /mnt/hwdata/cfy/RMBench/policy/Mem-0/llamafactory_data/cover_blocks_uniform \
    --base_url http://localhost:8123/v1 \
    --model /mnt/hwdata/cfy/RMBench/policy/Mem-0/checkpoints/Qwen3-VL-8B-Instruct-cover_blocks_uniform \
    --max_samples 10000 \
    --out planner_train_eval.jsonl