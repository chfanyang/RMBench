#!/usr/bin/env python3
"""
Offline evaluation script for LoRA-finetuned Qwen-VL planning module.

Evaluates whether the merged Qwen3-VL planning model can predict the correct
next_subtask from LLaMA-Factory SFT multi-modal data, using the same
prompt-construction pattern as MemoryMattersPlanner.prepare_qwen_input().

Usage:
    python policy/Mem-0/scripts/eval_planning_module.py \
        --data_json policy/Mem-0/llamafactory_data/<dataset>/<dataset>_high_level_finetune_data.json \
        --image_base_dir policy/Mem-0/llamafactory_data/<dataset> \
        --base_url http://localhost:8123/v1 \
        --model mem0-planner \
        --max_samples 200 \
        --out planner_eval.jsonl
"""

import argparse
import base64
import io
import json
import os
import re
import time
from difflib import SequenceMatcher

from openai import OpenAI
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
#  Image utilities (mirrors MemoryMattersPlanner._image_to_data_url)
# ---------------------------------------------------------------------------

def image_to_data_url(image_path: str) -> str:
    """Convert a local image file to a base64 data URL for the vLLM API."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ---------------------------------------------------------------------------
#  Prompt construction (reuses MemoryMattersPlanner.prepare_qwen_input logic)
# ---------------------------------------------------------------------------

def build_prompt_messages(sample: dict, image_base_dir: str) -> list:
    """
    Build OpenAI-compatible multimodal chat messages from a ShareGPT-format sample.

    The data uses ``<image>`` placeholders inside the user text (LLaMA-Factory
    convention).  This function splits the text on those placeholders and
    inserts real ``image_url`` content blocks — exactly the same interleaved
    text/image structure that ``MemoryMattersPlanner.prepare_qwen_input()``
    produces at inference time.

    Args:
        sample: dict with keys ``messages`` (list of {role, content}) and
                ``images`` (list of relative image paths).
        image_base_dir: base directory for resolving relative image paths.

    Returns:
        List of message dicts ready for ``client.chat.completions.create()``.
    """
    messages = []
    images = sample["images"]
    image_idx = 0

    for msg in sample["messages"]:
        role = msg["role"]
        content = msg["content"]

        if role == "system":
            # Multimodal content format — same as prepare_qwen_input
            messages.append({
                "role": "system",
                "content": [{"type": "text", "text": content}],
            })

        elif role == "user":
            # Split on <image> placeholders and interleave text / image_url blocks
            parts = content.split("<image>")
            user_content = []
            for i, part in enumerate(parts):
                if part:
                    user_content.append({"type": "text", "text": part})
                if i < len(parts) - 1:
                    if image_idx >= len(images):
                        raise ValueError(
                            f"More <image> placeholders in user message than images "
                            f"in sample (image_idx={image_idx}, total_images={len(images)})."
                        )
                    img_abs = os.path.join(image_base_dir, images[image_idx])
                    data_url = image_to_data_url(img_abs)
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": data_url},
                    })
                    image_idx += 1
            messages.append({"role": "user", "content": user_content})

        # assistant messages are skipped — we only build the *prompt*

    if image_idx != len(images):
        raise ValueError(
            f"Image count mismatch: {image_idx} <image> placeholders found in "
            f"user message but sample has {len(images)} images."
        )

    return messages


# ---------------------------------------------------------------------------
#  Target extraction & output parsing
# ---------------------------------------------------------------------------

def extract_target(sample: dict) -> str:
    """Extract the ground-truth assistant content from the sample."""
    for msg in sample["messages"]:
        if msg["role"] == "assistant":
            return msg["content"]
    raise ValueError("Sample has no assistant message.")


def parse_prediction(raw_answer: str):
    """
    Extract ``next_subtask: ...`` from the model output.

    Returns:
        (prediction_text, format_ok)
          - prediction_text: the content after ``next_subtask:`` (or the full
            raw_answer when the format is missing).
          - format_ok: True if the expected format was found.
    """
    m = re.search(r"next_subtask:\s*(.+)", raw_answer, re.IGNORECASE)
    if m:
        pred = m.group(1).strip()
        # Strip a single trailing period (system prompt asks for one)
        pred = pred.rstrip(".").strip()
        return pred, True
    return raw_answer, False


def normalize(text: str) -> str:
    """
    Normalize a string for comparison.

    Steps:
    1. Strip ``next_subtask:`` prefix (in case the raw target is passed).
    2. Strip leading/trailing whitespace, single/double quotes, periods.
    3. Lowercase.
    4. Collapse multiple whitespace characters.
    """
    text = re.sub(r"^next_subtask:\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().strip("\"'").strip(".")
    text = text.strip()
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text


def fuzzy_ratio(a: str, b: str) -> float:
    """SequenceMatcher-based fuzzy string similarity (0.0 – 1.0)."""
    return SequenceMatcher(None, a, b).ratio()


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate LoRA-finetuned Qwen-VL planning module on "
                    "next_subtask prediction."
    )
    parser.add_argument(
        "--data_json", type=str, required=True,
        help="Path to the *_high_level_finetune_data.json generated by "
             "llamafactory_data_preparation.py.",
    )
    parser.add_argument(
        "--image_base_dir", type=str, default=None,
        help="Base directory for image paths.  Defaults to the directory "
             "containing --data_json.",
    )
    parser.add_argument(
        "--base_url", type=str, default="http://localhost:8123/v1",
        help="vLLM OpenAI-compatible server base URL.",
    )
    parser.add_argument(
        "--model", type=str, default="mem0-planner",
        help="Served model name.",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Only evaluate the first N samples (useful for quick sanity checks).",
    )
    parser.add_argument(
        "--out", type=str, default="planner_eval.jsonl",
        help="Output JSONL file for per-sample predictions.",
    )
    args = parser.parse_args()

    # ---- resolve paths ----------------------------------------------------
    data_json = os.path.abspath(args.data_json)
    image_base_dir = os.path.abspath(
        args.image_base_dir if args.image_base_dir else os.path.dirname(data_json)
    )

    # ---- load data --------------------------------------------------------
    print(f"Loading data from: {data_json}")
    with open(data_json, "r") as f:
        samples = json.load(f)

    if args.max_samples is not None:
        samples = samples[: args.max_samples]

    n_total = len(samples)
    print(f"Evaluating {n_total} samples")
    print(f"Image base dir: {image_base_dir}")
    print(f"vLLM endpoint:  {args.base_url}")
    print(f"Model:          {args.model}")

    # ---- init client ------------------------------------------------------
    client = OpenAI(api_key="EMPTY", base_url=args.base_url, timeout=3600)

    # ---- eval loop --------------------------------------------------------
    results = []
    exact_match_total = 0
    format_ok_total = 0
    fuzzy_sum = 0.0
    time_sum = 0.0
    by_step = {}  # step -> {"total": N, "exact": M}

    for idx, sample in enumerate(tqdm(samples, desc="Evaluating")):
        try:
            # --- build prompt (same logic as MemoryMattersPlanner) ----------
            messages = build_prompt_messages(sample, image_base_dir)

            # --- ground truth -----------------------------------------------
            target_raw = extract_target(sample)
            target_norm = normalize(target_raw)

            # step = number of finished subtasks = len(images) - 1
            step = len(sample["images"]) - 1

            # --- inference --------------------------------------------------
            t0 = time.perf_counter()
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                max_tokens=128,
                temperature=0,
            )
            t1 = time.perf_counter()
            inference_time = round(t1 - t0, 4)
            raw_answer = response.choices[0].message.content

            # --- parse & compare -------------------------------------------
            pred_raw, format_ok = parse_prediction(raw_answer)
            pred_norm = normalize(pred_raw)

            exact = (pred_norm == target_norm)
            fuzz = fuzzy_ratio(pred_norm, target_norm)

            # --- record ----------------------------------------------------
            result = {
                "idx": idx,
                "step": step,
                "target": target_raw,
                "prediction": pred_raw,
                "raw_answer": raw_answer,
                "exact_match": exact,
                "format_ok": format_ok,
                "fuzzy_ratio": round(fuzz, 4),
                "inference_time_s": inference_time,
            }
            results.append(result)

            # --- aggregate -------------------------------------------------
            exact_match_total += int(exact)
            format_ok_total += int(format_ok)
            fuzzy_sum += fuzz
            time_sum += inference_time

            by_step.setdefault(step, {"total": 0, "exact": 0})
            by_step[step]["total"] += 1
            by_step[step]["exact"] += int(exact)

        except Exception as exc:
            tqdm.write(f"[ERROR] sample {idx}: {exc}")
            step = len(sample.get("images", [])) - 1
            results.append({
                "idx": idx,
                "step": step,
                "target": "",
                "prediction": "",
                "raw_answer": "",
                "exact_match": False,
                "format_ok": False,
                "fuzzy_ratio": 0.0,
                "inference_time_s": 0.0,
                "error": str(exc),
            })

    # ---- save results -----------------------------------------------------
    out_dir = os.path.dirname(os.path.abspath(args.out)) or "."
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nPer-sample results saved to: {args.out}")

    # ---- summary ----------------------------------------------------------
    n_evaluated = len(results)
    print(f"\n{'=' * 52}")
    print("Evaluation Summary")
    print(f"{'=' * 52}")
    if n_evaluated > 0:
        print(f"  Total samples:       {n_evaluated}")
        print(f"  Exact match:         {exact_match_total}/{n_evaluated} "
              f"({exact_match_total / n_evaluated * 100:.2f}%)")
        print(f"  Format rate:         {format_ok_total}/{n_evaluated} "
              f"({format_ok_total / n_evaluated * 100:.2f}%)")
        print(f"  Avg fuzzy ratio:     {fuzzy_sum / n_evaluated:.4f}")
        print(f"  Avg inference time:  {time_sum / n_evaluated:.4f}s "
              f"({time_sum:.2f}s total)")

        print(f"\n  Accuracy by finished subtask count (step):")
        print(f"  {'Step':<8} {'Total':<8} {'Correct':<8} {'Accuracy':<10}")
        print(f"  {'-' * 34}")
        for step in sorted(by_step.keys()):
            s = by_step[step]
            acc = s["exact"] / s["total"] * 100 if s["total"] > 0 else 0.0
            print(f"  {step:<8} {s['total']:<8} {s['exact']:<8} {acc:.2f}%")
    else:
        print("  No samples were successfully evaluated.")

    print(f"{'=' * 52}\n")


if __name__ == "__main__":
    main()
