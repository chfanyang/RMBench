"""HTTP strict smoke test for a running VideoLLaMA3 planner server.

This script does not import VideoLLaMA3, peft, or transformers. It only posts a
real frame_dir and training-format prompt to /plan and prints the key response
fields.
"""

import argparse
import json
import sys


def build_training_prompt(global_task: str) -> str:
    return (
        f"Global task: {global_task}\n"
        "Based only on the video so far, output the robot planning state as compact JSON.\n"
        "Use exactly these keys: current_subgoal, current_status, next_subgoal, should_switch, task_status.\n"
        'current_status must be either "in_progress" or "completed".\n'
        'task_status must be either "running" or "completed".\n'
        "If the current subgoal is still in progress, next_subgoal should be the current subgoal and should_switch should be false.\n"
        "If the current subgoal has just been completed, next_subgoal should be the next subgoal and should_switch should be true.\n"
        'If the whole task has been completed, next_subgoal should be null, should_switch should be false, and task_status should be "completed".'
    )


def get_requests():
    try:
        import requests
    except Exception as exc:
        raise ImportError("test_videollama3_server_strict_smoke.py requires requests.") from exc
    return requests


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8009")
    parser.add_argument("--frame_dir", type=str, required=True)
    parser.add_argument("--global_task", type=str, required=True)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()

    payload = {
        "frame_dir": args.frame_dir,
        "global_task": args.global_task,
        "prompt": build_training_prompt(args.global_task),
        "finished_subtasks": [],
        "previous_subgoal": "",
        "fps": args.fps,
        "max_frames": args.max_frames,
        "max_new_tokens": args.max_new_tokens,
        "strict": args.strict,
    }

    requests = get_requests()
    response = requests.post(
        args.url.rstrip("/") + "/plan",
        json=payload,
        timeout=args.timeout,
    )
    response.raise_for_status()
    data = response.json()

    for key in [
        "ok",
        "error",
        "raw_output",
        "parsed_json",
        "regex_subgoal",
        "subgoal",
        "used_fallback",
        "instruction",
    ]:
        value = data.get(key)
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        print(f"{key}: {value}")

    if args.strict and not data.get("ok", False):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
