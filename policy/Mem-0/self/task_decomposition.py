#!/usr/bin/env python3
"""
Use LLaMAFactory's ChatModel to call Qwen3-VL-8B-Instruct for task decomposition.

Given an initial observation image and a task instruction, this script asks the
model to decompose the task into a sequence of language subgoals, outputting the
total number of steps and each step's description.

Usage:
    python self/task_decomposition.py \
        --image <path_to_image> \
        --instruction "<task instruction>"

    # Or with defaults (uses cover_blocks task):
    python self/task_decomposition.py --image <path_to_image>
"""

#  python self/task_decomposition.py --image self/cover_block.png

import argparse
import os
import sys

workspace = os.path.dirname(os.path.abspath(__file__))
Mem0_workspace = os.path.join(workspace, "..")
llamafactory_src = os.path.join(Mem0_workspace, "LlamaFactory", "src")
sys.path.insert(0, llamafactory_src)

from llamafactory.chat.chat_model import ChatModel


DEFAULT_INSTRUCTION = (
    "On the table, red, green, and blue blocks are arranged randomly along "
    "with three lids. From the current viewpoint, cover the blocks from right "
    "to left using the lids, and then uncover them again in the sequence red, "
    "green, and blue."
    #"There is a button and three colored cubes arranged in a random row on the table. Each time the cubes are rearranged, the arm presses the button until the arrangement is successful."
    #"Observe the two numbers on the table. Press the left button the number of times corresponding to the number on the left, and press the middle button the number of times corresponding to the number on the right. Then press the right button once to confirm."
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Decompose a robot manipulation task into subtasks using "
                    "LLaMAFactory + Qwen3-VL-8B-Instruct."
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to the initial observation image.",
    )
    parser.add_argument(
        "--instruction", type=str, default=DEFAULT_INSTRUCTION,
        help="Task instruction for the robot.",
    )
    parser.add_argument(
        "--model_path", type=str,
        default=os.path.join(Mem0_workspace, "checkpoints", "Qwen3-VL-8B-Instruct"),
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--infer_backend", type=str, default="huggingface",
        choices=["huggingface", "vllm"],
        help="Inference backend: huggingface (self-contained) or vllm."
    )
    parser.add_argument(
        "--max_tokens", type=int, default=512,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (0 = greedy).",
    )
    return parser.parse_args()


def build_system_prompt() -> str:
    return (
        "You are a robotic task planning assistant. "
        "Your job is to analyze a task instruction and an initial observation "
        "image, then decompose the task into a sequence of clear, actionable "
        "subtasks (language subgoals).\n\n"
        "Output format:\n"
        "Total steps: <number>\n"
        "Step 1: <subtask description>\n"
        "Step 2: <subtask description>\n"
        "...\n\n"
        "Requirements:\n"
        "- Each step must be a single, clearly defined action that the robot can "
        "execute.\n"
        "- Steps must be ordered logically to accomplish the overall task.\n"
        "- Use the visual information in the image to ground your plan in the "
        "actual scene.\n"
        "- Output ONLY the steps in the specified format, no extra commentary."
    )


def build_user_message(instruction: str) -> str:
    return (
        "<image>\n"
        "This is the initial observation image of a robot manipulation task.\n\n"
        f"Task instruction: {instruction}\n\n"
        "Please decompose this task into a sequence of subtasks (language "
        "subgoals). Output the total number of steps and each step's description."
    )


def main():
    args = parse_args()

    # Validate image
    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    print(f"{'=' * 60}")
    print(f"Task Decomposition via LLaMAFactory + Qwen3-VL-8B-Instruct")
    print(f"{'=' * 60}")
    print(f"Model:      {args.model_path}")
    print(f"Backend:    {args.infer_backend}")
    print(f"Image:      {args.image}")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Temp:       {args.temperature}")
    print(f"{'=' * 60}")

    # Initialize ChatModel via LLaMAFactory
    print("\n[1/2] Loading model via LLaMAFactory ChatModel ...", flush=True)
    chat_model = ChatModel({
        "model_name_or_path": args.model_path,
        "template": "qwen3_vl_nothink",
        "infer_backend": args.infer_backend,
        "trust_remote_code": True,
    })
    print("Model loaded successfully.", flush=True)

    # Build prompt
    system_prompt = build_system_prompt()
    user_message = build_user_message(args.instruction)

    # Run inference
    print("\n[2/2] Running inference ...", flush=True)
    responses = chat_model.chat(
        messages=[{"role": "user", "content": user_message}],
        system=system_prompt,
        images=[args.image],
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        do_sample=args.temperature > 0,
    )

    # Output
    response_text = responses[0].response_text
    print(f"\n{'=' * 60}")
    print("Model Output:")
    print(f"{'=' * 60}")
    print(response_text)
    print(f"{'=' * 60}")

    finish = responses[0].finish_reason
    prompt_len = responses[0].prompt_length
    resp_len = responses[0].response_length
    print(f"\n[Info] finish_reason={finish}, prompt_len={prompt_len}, "
          f"response_len={resp_len}")


if __name__ == "__main__":
    main()
