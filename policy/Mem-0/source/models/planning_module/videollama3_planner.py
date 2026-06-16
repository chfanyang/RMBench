"""VideoLLaMA3 high-level planner adapter for Mem-0 deployment.

Phase 1 only replaces the high-level planner. The existing execution module,
subtask-end classifier, threshold, and switching timing remain owned by
MemoryMattersAgent.
"""

import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image
from termcolor import cprint


class VideoLLaMA3Planner:
    """Adapter with the same call surface used by MemoryMattersPlanner."""

    def __init__(
        self,
        config: Optional[Any] = None,
        device: Optional[torch.device] = None,
        global_task: Optional[str] = None,
        **kwargs,
    ):
        self.config = self._select_vl3_config(config)
        self.global_task = global_task or self._cfg_get(config, "global_task", "") or ""

        self.base_model = self._cfg_get(self.config, "base_model", "")
        self.lora_path = self._cfg_get(self.config, "lora_path", "")
        self.device = str(self._cfg_get(self.config, "device", device or ("cuda" if torch.cuda.is_available() else "cpu")))
        self.fps = int(self._cfg_get(self.config, "fps", 1))
        self.max_frames = int(self._cfg_get(self.config, "max_frames", 128))
        self.max_new_tokens = int(self._cfg_get(self.config, "max_new_tokens", 128))
        self.frame_stride = max(1, int(self._cfg_get(self.config, "frame_stride", 1)))
        self.frame_dir = Path(str(self._cfg_get(self.config, "frame_dir", "./_tmp_visual/vl3_stream_frames")))
        self.attn_implementation = self._cfg_get(self.config, "attn_implementation", None)
        self.load_model_on_init = bool(self._cfg_get(self.config, "load_model", True))

        self.model = None
        self.processor = None
        self.initial_observation = None
        self.key_information = []
        self.finished_subtasks = []
        self._seen_frames = 0
        self._saved_frames = 0
        self._last_json: Dict[str, Any] = {}
        self._last_subgoal = ""

        self.reset_stream()
        if self.load_model_on_init:
            self._load_model()

    @staticmethod
    def _cfg_get(cfg: Optional[Any], key: str, default: Any = None) -> Any:
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        try:
            return cfg.get(key, default)
        except Exception:
            return getattr(cfg, key, default)

    def _select_vl3_config(self, cfg: Optional[Any]) -> Any:
        nested = self._cfg_get(cfg, "videollama3", None)
        return nested if nested is not None else (cfg or {})

    def _load_model(self) -> None:
        if not self.base_model:
            raise ValueError("VideoLLaMA3Planner requires videollama3.base_model in config.")

        try:
            from transformers import AutoModelForCausalLM, AutoProcessor
        except Exception as exc:
            raise ImportError(
                "Failed to import transformers. Install the VideoLLaMA3 environment "
                "before using planner_type=videollama3."
            ) from exc

        dtype = torch.float32 if self.device == "cpu" else torch.float16
        kwargs = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
            "device_map": {"": self.device},
        }
        if self.attn_implementation:
            kwargs["attn_implementation"] = self.attn_implementation

        cprint(f"[VideoLLaMA3Planner] loading base model: {self.base_model}", "cyan")
        self.model = AutoModelForCausalLM.from_pretrained(self.base_model, **kwargs)

        if self.lora_path:
            if not os.path.exists(self.lora_path):
                raise FileNotFoundError(f"VideoLLaMA3 LoRA path does not exist: {self.lora_path}")
            try:
                from peft import PeftModel
            except Exception as exc:
                raise ImportError(
                    "peft is required because videollama3.lora_path is configured."
                ) from exc
            cprint(f"[VideoLLaMA3Planner] loading LoRA adapter: {self.lora_path}", "cyan")
            self.model = PeftModel.from_pretrained(self.model, self.lora_path)
            try:
                self.model = self.model.merge_and_unload()
                cprint("[VideoLLaMA3Planner] merged LoRA adapter for inference", "cyan")
            except Exception as exc:
                cprint(f"[VideoLLaMA3Planner] merge_and_unload failed; using PeftModel: {exc}", "yellow")

        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.base_model, trust_remote_code=True)

    def reset_stream(self, *args, **kwargs) -> None:
        """Clear accumulated streaming frames at episode start."""
        shutil.rmtree(self.frame_dir, ignore_errors=True)
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self._seen_frames = 0
        self._saved_frames = 0

    def append_frame_array(self, rgb: np.ndarray) -> None:
        """Append one RGB frame from the environment, honoring frame_stride."""
        if rgb is None:
            return
        frame_index = self._seen_frames
        self._seen_frames += 1
        if frame_index % self.frame_stride != 0:
            return

        arr = np.asarray(rgb)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            cprint(f"[VideoLLaMA3Planner] skip invalid RGB frame shape: {arr.shape}", "yellow")
            return
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

        self.frame_dir.mkdir(parents=True, exist_ok=True)
        path = self.frame_dir / f"{self._saved_frames:06d}.jpg"
        Image.fromarray(arr, mode="RGB").save(path, quality=95)
        self._saved_frames += 1

    def update_initial_observation(self, observation: Any):
        self.initial_observation = observation

    def update_image_or_video_input(self, image_or_video_inputs, subtasks=None):
        if subtasks is not None:
            for image_or_video_input, subtask in zip(image_or_video_inputs, subtasks):
                self.key_information.append(image_or_video_input)
                self.finished_subtasks.append(subtask)

    def prepare_qwen_input(self):
        prompt = self._build_prompt()
        return {
            "video_path": str(self.frame_dir),
            "prompt": prompt,
        }

    def _build_prompt(self) -> str:
        previous = self.finished_subtasks[-1] if self.finished_subtasks else ""
        return (
            "You are the high-level planner for a robot manipulation task. "
            "Given all video frames accumulated so far in the current episode, "
            "return only compact JSON with exactly these fields: "
            '{"current_subgoal":"...","current_status":"in_progress or completed",'
            '"next_subgoal":"...","should_switch":true,"task_status":"running or completed"}. '
            f"Global task: {self.global_task}. "
            f"Previous subgoal: {previous or 'none'}. "
            "Choose next_subgoal as the instruction the existing low-level executor should follow next."
        )

    @torch.inference_mode()
    def generate_anwser(self, inputs=None):
        """Generate planning JSON and return Mem-0 compatible next_subtask text."""
        if self._saved_frames == 0 and self.initial_observation is not None:
            self._append_initial_observation_as_frame()

        if self.model is None or self.processor is None:
            self._load_model()

        inputs = inputs or self.prepare_qwen_input()
        prompt = inputs["prompt"] if isinstance(inputs, dict) else self._build_prompt()
        video_path = inputs.get("video_path", str(self.frame_dir)) if isinstance(inputs, dict) else str(self.frame_dir)

        conversation = [
            {"role": "system", "content": "You are a helpful robot planning assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": {"video_path": video_path, "fps": self.fps, "max_frames": self.max_frames}},
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
            if isinstance(value, torch.Tensor):
                if key == "pixel_values":
                    model_inputs[key] = value.to(device=self.device, dtype=torch.float16 if self.device != "cpu" else torch.float32)
                else:
                    model_inputs[key] = value.to(device=self.device)

        input_len = model_inputs["input_ids"].shape[1] if "input_ids" in model_inputs else None
        output_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=self.max_new_tokens,
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

        plan = self.parse_planning_json(text)
        subgoal = self._choose_subgoal(plan, text)
        return f"next_subtask: {subgoal}."

    def _append_initial_observation_as_frame(self) -> None:
        obs = self.initial_observation
        try:
            if isinstance(obs, (str, os.PathLike)):
                img = Image.open(obs).convert("RGB")
            elif isinstance(obs, Image.Image):
                img = obs.convert("RGB")
            else:
                return
            self.append_frame_array(np.asarray(img))
        except Exception as exc:
            cprint(f"[VideoLLaMA3Planner] failed to append initial observation: {exc}", "yellow")

    def parse_planning_json(self, text: str) -> Dict[str, Any]:
        parsed = self._parse_json_object(text)
        if parsed is None:
            cprint(f"[VideoLLaMA3Planner] failed to parse JSON, raw output: {text}", "yellow")
            return dict(self._last_json)
        self._last_json = parsed
        return parsed

    def _choose_subgoal(self, plan: Dict[str, Any], raw_text: str) -> str:
        subgoal = str(plan.get("next_subgoal") or plan.get("current_subgoal") or "").strip()
        if not subgoal and raw_text:
            subgoal = raw_text.strip()
        if not subgoal:
            subgoal = self._last_subgoal or self.global_task or "continue the task"
        self._last_subgoal = subgoal
        return subgoal.rstrip(".")

    @staticmethod
    def _parse_json_object(text: str) -> Optional[Dict[str, Any]]:
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
        candidate = match.group(0)
        try:
            value = json.loads(candidate)
            return value if isinstance(value, dict) else None
        except Exception:
            return None
