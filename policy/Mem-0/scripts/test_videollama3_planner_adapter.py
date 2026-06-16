"""Smoke test for the VideoLLaMA3 planner adapter.

By default this does not load model weights, so it can run without GPU or
VideoLLaMA3 checkpoints. Pass --base_model to exercise one real generation.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

MEM0_ROOT = Path(__file__).resolve().parents[1]
if str(MEM0_ROOT) not in sys.path:
    sys.path.insert(0, str(MEM0_ROOT))

from source.models.planning_module.videollama3_planner import VideoLLaMA3Planner


def build_config(args, frame_dir: str):
    return {
        "global_task": "Cover the blocks from right to left, then uncover them by color order.",
        "videollama3": {
            "base_model": args.base_model or "",
            "lora_path": args.lora_path or "",
            "device": args.device,
            "fps": 1,
            "max_frames": 8,
            "max_new_tokens": 64,
            "frame_stride": args.frame_stride,
            "frame_dir": frame_dir,
            "load_model": bool(args.base_model),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, default="", help="Optional real VideoLLaMA3 base model path.")
    parser.add_argument("--lora_path", type=str, default="", help="Optional LoRA adapter path for real generation.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--frame_dir", type=str, default="")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_dir = args.frame_dir or os.path.join(tmpdir, "vl3_frames")
        planner = VideoLLaMA3Planner(
            config=build_config(args, frame_dir),
            global_task="Cover blocks streaming smoke test.",
        )

        for i in range(4):
            rgb = np.zeros((64, 64, 3), dtype=np.uint8)
            rgb[..., 0] = i * 40
            rgb[..., 1] = 128
            rgb[..., 2] = 255 - i * 30
            planner.append_frame_array(rgb)

        saved = sorted(Path(frame_dir).glob("*.jpg"))
        assert saved, "append_frame_array did not save any frames"

        raw = 'prefix {"current_subgoal":"cover right block","current_status":"completed","next_subgoal":"cover middle block","should_switch":true,"task_status":"running"} suffix'
        parsed = planner.parse_planning_json(raw)
        assert parsed["next_subgoal"] == "cover middle block"
        assert planner._choose_subgoal(parsed, raw) == "cover middle block"

        invalid_json = 'assistant says next_subgoal: "uncover red block" after checking the scene'
        assert planner.parse_planning_json(invalid_json) == {}
        assert planner._choose_subgoal({}, invalid_json) == "uncover red block"

        planner.reset_episode()
        assert planner.initial_observation is None
        assert planner.key_information == []
        assert planner.finished_subtasks == []
        assert planner._last_json == {}
        assert planner._last_subgoal == ""

        if args.base_model:
            result = planner.generate_anwser()
            assert result.startswith("next_subtask: ")
            print(result)
        else:
            print("VideoLLaMA3Planner smoke test passed without loading model weights.")


if __name__ == "__main__":
    main()
