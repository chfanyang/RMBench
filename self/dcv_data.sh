python ./self/convert_cover_blocks_to_videollama3.py \
  --rmbench_root /mnt/hwdata/cfy/RMBench \
  --task_config demo_clean \
  --out_root ./datasets/rmbench_cover_blocks_vl3_debug \
  --max_episodes 2 \
  --fps 30 \
  --modes full subtask next \
  --overwrite