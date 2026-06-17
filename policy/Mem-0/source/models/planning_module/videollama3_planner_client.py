"""HTTP client planner for a VideoLLaMA3 planner server.

This keeps VideoLLaMA3 model dependencies out of the RMBench/Mem-0 process.
It only stores streaming RGB frames and asks the external server for the next
high-level subgoal.
"""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

try:
    from termcolor import cprint
except Exception:
    def cprint(text, *args, **kwargs):
        print(text)

requests = None


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


class VideoLLaMA3PlannerClient:
    """Mem-0 compatible high-level planner backed by HTTP."""

    def __init__(self, config: Optional[Any] = None, global_task: Optional[str] = None, **kwargs):
        self.config = self._select_server_config(config)
        self.global_task = global_task or self._cfg_get(config, "global_task", "") or ""

        self.url = str(self._cfg_get(self.config, "url", "http://127.0.0.1:8009")).rstrip("/")
        self.timeout = float(self._cfg_get(self.config, "timeout", 3600))
        self.fps = int(self._cfg_get(self.config, "fps", self._cfg_get(self._select_local_config(config), "fps", 1)))
        self.max_frames = int(self._cfg_get(self.config, "max_frames", self._cfg_get(self._select_local_config(config), "max_frames", 128)))
        self.max_new_tokens = int(self._cfg_get(self.config, "max_new_tokens", self._cfg_get(self._select_local_config(config), "max_new_tokens", 128)))
        self.frame_stride = max(1, int(self._cfg_get(self.config, "frame_stride", self._cfg_get(self._select_local_config(config), "frame_stride", 1))))
        frame_dir = self._cfg_get(
            self.config,
            "frame_dir",
            self._cfg_get(self._select_local_config(config), "frame_dir", "./_tmp_visual/vl3_stream_frames"),
        )
        self.frame_dir = Path(str(frame_dir)).expanduser().resolve()
        self.strict = bool(self._cfg_get(self.config, "strict", False))

        self.initial_observation = None
        self.key_information = []
        self.finished_subtasks = []
        self._seen_frames = 0
        self._saved_frames = 0
        self._last_subgoal = ""

        self.last_raw_output = ""
        self.last_parsed_json: Dict[str, Any] = {}
        self.last_regex_subgoal = None
        self.last_used_fallback = False
        self.last_instruction = ""
        self.last_server_response: Dict[str, Any] = {}

        self.reset_stream()

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

    def _select_server_config(self, cfg: Optional[Any]) -> Any:
        nested = self._cfg_get(cfg, "videollama3_server", None)
        return nested if nested is not None else {}

    def _select_local_config(self, cfg: Optional[Any]) -> Any:
        nested = self._cfg_get(cfg, "videollama3", None)
        return nested if nested is not None else {}

    def _get_requests(self):
        global requests
        if requests is not None:
            return requests
        try:
            import requests as requests_mod
        except Exception as exc:
            raise ImportError(
                "VideoLLaMA3PlannerClient requires the requests package in the "
                "RMBench/Mem-0 environment."
            ) from exc
        requests = requests_mod
        return requests

    def reset_stream(self, *args, **kwargs) -> None:
        shutil.rmtree(self.frame_dir, ignore_errors=True)
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self._seen_frames = 0
        self._saved_frames = 0

    def reset_episode(self) -> None:
        self.reset_stream()
        self.initial_observation = None
        self.key_information = []
        self.finished_subtasks = []
        self._last_subgoal = ""
        self.last_raw_output = ""
        self.last_parsed_json = {}
        self.last_regex_subgoal = None
        self.last_used_fallback = False
        self.last_instruction = ""
        self.last_server_response = {}

    def append_frame_array(self, rgb: np.ndarray) -> None:
        if rgb is None:
            return
        frame_index = self._seen_frames
        self._seen_frames += 1
        if frame_index % self.frame_stride != 0:
            return

        arr = np.asarray(rgb)
        if arr.ndim != 3 or arr.shape[-1] != 3:
            cprint(f"[VideoLLaMA3PlannerClient] skip invalid RGB frame shape: {arr.shape}", "yellow")
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

    def prepare_qwen_input(self) -> Dict[str, Any]:
        return self._build_request_payload()

    def _build_request_payload(self) -> Dict[str, Any]:
        previous_subgoal = self.finished_subtasks[-1] if self.finished_subtasks else self._last_subgoal
        return {
            "frame_dir": str(self.frame_dir),
            "global_task": self.global_task,
            "prompt": build_training_prompt(self.global_task),
            "initial_observation": str(self.initial_observation or ""),
            "finished_subtasks": list(self.finished_subtasks),
            "previous_subgoal": previous_subgoal or "",
            "key_information": list(self.key_information),
            "fps": self.fps,
            "max_frames": self.max_frames,
            "max_new_tokens": self.max_new_tokens,
            "strict": self.strict,
        }

    def generate_anwser(self, inputs=None):
        payload = inputs if isinstance(inputs, dict) else self._build_request_payload()
        try:
            requests_mod = self._get_requests()
            response = requests_mod.post(f"{self.url}/plan", json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            fallback = self._fallback_instruction()
            cprint(f"[VideoLLaMA3PlannerClient] planner server request failed: {exc}; using {fallback}", "yellow")
            self.last_used_fallback = True
            self.last_instruction = fallback
            self.last_server_response = {"ok": False, "error": str(exc)}
            return fallback

        self.last_server_response = data
        self.last_raw_output = str(data.get("raw_output", ""))
        parsed = data.get("parsed_json", {})
        self.last_parsed_json = parsed if isinstance(parsed, dict) else {}
        self.last_regex_subgoal = data.get("regex_subgoal")
        self.last_used_fallback = bool(data.get("used_fallback", False))

        instruction = str(data.get("instruction") or "").strip()
        if not instruction:
            instruction = self._fallback_instruction()
            self.last_used_fallback = True
            cprint("[VideoLLaMA3PlannerClient] server returned empty instruction; using fallback", "yellow")

        self.last_instruction = instruction
        if not bool(data.get("ok", False)):
            raw_preview = self.last_raw_output[:240].replace("\n", "\\n")
            cprint(
                "[VideoLLaMA3PlannerClient] planner server returned ok=false; "
                f"error={data.get('error')}; "
                f"used_fallback={self.last_used_fallback}; "
                f"raw_output_prefix={raw_preview!r}; "
                f"instruction={instruction}",
                "yellow",
            )
        subgoal = str(data.get("subgoal") or "").strip()
        if subgoal:
            self._last_subgoal = subgoal.rstrip(".")
        elif instruction.startswith("next_subtask: "):
            self._last_subgoal = instruction.split("next_subtask: ", 1)[-1].rsplit(".", 1)[0].strip()
        return instruction

    def _fallback_instruction(self) -> str:
        subgoal = self._last_subgoal or (self.finished_subtasks[-1] if self.finished_subtasks else "") or "continue the task"
        return f"next_subtask: {subgoal.rstrip('.')}."
