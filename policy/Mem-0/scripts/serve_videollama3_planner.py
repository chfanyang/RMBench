"""Serve VideoLLaMA3 high-level planning over HTTP.

Run this script inside the VideoLLaMA3 conda environment. RMBench/Mem-0 only
talks to this service and does not import VideoLLaMA3 dependencies.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Keep the existing HuggingFace cache for model files, but place dynamic
# remote-code modules in /tmp so restricted worktrees do not block startup.
os.environ.setdefault("HF_MODULES_CACHE", "/tmp/videollama3_hf_modules")

try:
    import torch
except Exception:
    torch = None


def cprint(text, *args, **kwargs):
    print(text)


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        value = json.loads(match.group(0))
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def extract_subgoal_field(text: str) -> str:
    if not text:
        return ""
    for field in ("next_subgoal", "current_subgoal"):
        match = re.search(
            rf'["\']?{field}["\']?\s*:\s*["\']([^"\']+)["\']',
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
    return ""


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


def build_prompt(payload: Dict[str, Any]) -> str:
    for key in ("prompt", "question"):
        if payload.get(key):
            return str(payload[key])

    return build_training_prompt(str(payload.get("global_task", "")))


def validate_frame_dir(frame_dir: str) -> Optional[str]:
    path = Path(frame_dir)
    if not path.exists():
        return f"frame_dir does not exist: {frame_dir}"
    if not path.is_dir():
        return f"frame_dir is not a directory: {frame_dir}"
    frame_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    if not any(p.is_file() and p.suffix.lower() in frame_exts for p in path.iterdir()):
        return f"frame_dir is empty or contains no image frames: {frame_dir}"
    return None


def make_response(
    ok: bool,
    raw_output: str = "",
    parsed_json: Optional[Dict[str, Any]] = None,
    regex_subgoal: Optional[str] = None,
    subgoal: str = "",
    used_fallback: bool = False,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    clean_subgoal = (subgoal or "").strip().rstrip(".")
    if not clean_subgoal:
        clean_subgoal = "continue the task"
        used_fallback = True
    return {
        "ok": ok,
        "raw_output": raw_output or "",
        "parsed_json": parsed_json or {},
        "regex_subgoal": regex_subgoal,
        "subgoal": clean_subgoal,
        "used_fallback": used_fallback,
        "instruction": f"next_subtask: {clean_subgoal}.",
        "error": error,
    }


def request_fallback_subgoal(payload: Dict[str, Any]) -> str:
    previous = str(payload.get("previous_subgoal") or "").strip()
    if previous:
        return previous
    finished = payload.get("finished_subtasks") or []
    if finished:
        return str(finished[-1]).strip() or "continue the task"
    return "continue the task"


def processor_fallback_from_config(base_model: str) -> str:
    config_path = Path(base_model) / "config.json"
    if not config_path.is_file():
        return ""
    try:
        with config_path.open("r", encoding="utf-8") as f:
            config = json.load(f)
    except Exception:
        return ""
    return str(config.get("_name_or_path") or "")


class VideoLLaMA3PlannerService:
    def __init__(self, args):
        self.args = args
        self.model = None
        self.processor = None
        self.model_loaded = False
        self.last_subgoal = ""

        if args.no_load_model or args.dry_run:
            cprint("[VideoLLaMA3 server] dry run mode: model loading disabled")
        else:
            self.load_model()

    def load_model(self) -> None:
        if torch is None:
            raise ImportError("PyTorch is required to serve VideoLLaMA3.")
        repo_path = os.path.abspath(os.path.expanduser(self.args.repo_path))
        if repo_path and repo_path not in sys.path:
            sys.path.insert(0, repo_path)

        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
        except Exception as exc:
            raise ImportError(
                "Failed to import transformers. Run this server inside the VideoLLaMA3 conda env."
            ) from exc

        dtype = torch.float32 if str(self.args.device) == "cpu" else torch.float16
        kwargs = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
            "device_map": {"": self.args.device},
        }
        if self.args.attn_implementation:
            kwargs["attn_implementation"] = self.args.attn_implementation

        cprint("[VideoLLaMA3 server] loading planner model")
        cprint(f"  repo_path: {repo_path}")
        cprint(f"  base_model: {self.args.base_model}")
        cprint(f"  lora_path: {self.args.lora_path or 'none'}")
        cprint(f"  device: {self.args.device}")
        cprint(f"  dtype: {dtype}")
        cprint(f"  host/port: {self.args.host}:{self.args.port}")

        self.model = AutoModelForCausalLM.from_pretrained(self.args.base_model, **kwargs)
        if self.args.lora_path:
            if not os.path.exists(self.args.lora_path):
                raise FileNotFoundError(f"VideoLLaMA3 LoRA path does not exist: {self.args.lora_path}")
            try:
                from peft import PeftModel
            except Exception as exc:
                raise ImportError("peft is required because --lora_path is configured.") from exc
            cprint(f"[VideoLLaMA3 server] loading LoRA adapter: {self.args.lora_path}")
            self.model = PeftModel.from_pretrained(self.model, self.args.lora_path)
            try:
                self.model = self.model.merge_and_unload()
                cprint("[VideoLLaMA3 server] merged LoRA adapter for inference")
            except Exception as exc:
                cprint(f"[VideoLLaMA3 server] merge_and_unload failed; continuing with PeftModel: {exc}")

        self.model.eval()
        processor_path = self.args.base_model
        try:
            self.processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
        except OSError as exc:
            fallback = processor_fallback_from_config(self.args.base_model)
            if not fallback or fallback == processor_path:
                raise
            cprint(
                f"[VideoLLaMA3 server] processor load failed from {processor_path}; "
                f"falling back to {fallback}: {exc}"
            )
            self.processor = AutoProcessor.from_pretrained(fallback, trust_remote_code=True)
        self.model_loaded = True

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "model_loaded": self.model_loaded,
            "device": self.args.device,
        }

    def plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        frame_dir = str(payload.get("frame_dir") or "")
        global_task = str(payload.get("global_task") or "")
        if not frame_dir or not global_task:
            return make_response(False, used_fallback=True, error="frame_dir and global_task are required")

        frame_error = validate_frame_dir(frame_dir)
        if frame_error:
            return make_response(False, used_fallback=True, error=frame_error)

        if self.args.no_load_model or self.args.dry_run:
            return make_response(
                True,
                raw_output='{"next_subgoal":"continue the task"}',
                parsed_json={"next_subgoal": "continue the task"},
                subgoal="continue the task",
                used_fallback=False,
            )

        try:
            raw_output = self._generate(payload, frame_dir)
            parsed = parse_json_object(raw_output) or {}
            regex_subgoal = None if parsed else extract_subgoal_field(raw_output)
            subgoal = str(parsed.get("next_subgoal") or parsed.get("current_subgoal") or regex_subgoal or "").strip()
            used_fallback = False
            if not subgoal:
                fallback = request_fallback_subgoal(payload)
                if bool(payload.get("strict", False)):
                    return make_response(
                        False,
                        raw_output=raw_output,
                        parsed_json=parsed,
                        regex_subgoal=regex_subgoal,
                        subgoal=fallback,
                        used_fallback=True,
                        error="planner-output-invalid",
                    )
                subgoal = fallback
                used_fallback = True
            self.last_subgoal = subgoal.rstrip(".")
            return make_response(
                True,
                raw_output=raw_output,
                parsed_json=parsed,
                regex_subgoal=regex_subgoal,
                subgoal=subgoal,
                used_fallback=used_fallback,
            )
        except Exception as exc:
            return make_response(False, used_fallback=True, error=repr(exc))

    def _generate(self, payload: Dict[str, Any], frame_dir: str) -> str:
        fps = int(payload.get("fps") or self.args.fps)
        max_frames = int(payload.get("max_frames") or self.args.max_frames)
        max_new_tokens = int(payload.get("max_new_tokens") or self.args.max_new_tokens)
        prompt = build_prompt(payload)
        start_time = payload.get("start_time")
        end_time = payload.get("end_time")
        frame_indices = payload.get("frame_indices")

        from videollama3.mm_utils import load_video

        frames, timestamps = load_video(
            video_path=str(frame_dir),
            start_time=start_time,
            end_time=end_time,
            fps=fps,
            max_frames=max_frames,
            frame_indices=frame_indices,
        )
        video_content = {
            "type": "video",
            "video": frames,
            "num_frames": len(frames),
            "timestamps": timestamps,
        }

        conversation = [
            {"role": "system", "content": "You are a helpful robot planning assistant."},
            {
                "role": "user",
                "content": [
                    video_content,
                    {"type": "text", "text": prompt},
                ],
            },
        ]
        model_inputs = self.processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        for key, value in list(model_inputs.items()):
            if torch is not None and isinstance(value, torch.Tensor):
                if key == "pixel_values":
                    dtype = torch.float32 if str(self.args.device) == "cpu" else torch.float16
                    model_inputs[key] = value.to(device=self.args.device, dtype=dtype)
                else:
                    model_inputs[key] = value.to(device=self.args.device)

        input_len = model_inputs["input_ids"].shape[1] if "input_ids" in model_inputs else None
        with torch.inference_mode():
            output_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                use_cache=True,
            )
        gen_ids = output_ids[:, input_len:] if input_len is not None else output_ids
        text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        if not text:
            text = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        return text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8009)
    parser.add_argument("--repo_path", type=str, default="/mnt/hwdata/cfy/VideoLLaMA3")
    parser.add_argument("--base_model", type=str, default="")
    parser.add_argument("--lora_path", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:1")
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--max_frames", type=int, default=128)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")
    parser.add_argument("--dry_run", action="store_true", help="Serve schema-compatible fallback responses without loading model.")
    parser.add_argument("--no_load_model", action="store_true", help="Alias for --dry_run.")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel, Field
        import uvicorn
    except Exception as exc:
        raise ImportError(
            "FastAPI, pydantic, and uvicorn are required. Install them in the "
            "VideoLLaMA3 conda env before running serve_videollama3_planner.py."
        ) from exc

    class PlanRequest(BaseModel):
        frame_dir: str
        global_task: str
        initial_observation: str = ""
        finished_subtasks: List[str] = Field(default_factory=list)
        previous_subgoal: str = ""
        key_information: List[Any] = Field(default_factory=list)
        fps: Optional[int] = None
        max_frames: Optional[int] = None
        max_new_tokens: Optional[int] = None
        frame_indices: Optional[List[int]] = None
        start_time: Optional[float] = None
        end_time: Optional[float] = None
        strict: bool = False

    service = VideoLLaMA3PlannerService(args)
    app = FastAPI(title="VideoLLaMA3 Planner Server")

    @app.get("/health")
    def health():
        return service.health()

    @app.post("/plan")
    def plan(request: PlanRequest):
        if hasattr(request, "model_dump"):
            payload = request.model_dump()
        else:
            payload = request.dict()
        return service.plan(payload)

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
