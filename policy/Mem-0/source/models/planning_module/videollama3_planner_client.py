"""HTTP client planner for a VideoLLaMA3 planner server.

This keeps VideoLLaMA3 model dependencies out of the RMBench/Mem-0 process.
It only stores streaming RGB frames and asks the external server for the next
high-level subgoal.
"""

import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from PIL import Image

from source.models.planning_module.planner_state import (
    PlannerState,
    PlannerStateError,
    planner_state_from_dict,
)

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
        self.planner_frame_interval = max(1, int(self._cfg_get(self.config, "planner_frame_interval", 30)))
        frame_dir = self._cfg_get(
            self.config,
            "frame_dir",
            self._cfg_get(self._select_local_config(config), "frame_dir", "./_tmp_visual/vl3_stream_frames"),
        )
        self.frame_dir = Path(str(frame_dir)).expanduser().resolve()
        self.strict = self._cfg_bool(self._cfg_get(self.config, "strict", True), default=True)
        cprint(
            "[VideoLLaMA3PlannerClient] config; "
            f"url={self.url}; timeout={self.timeout}; "
            f"frame_stride={self.frame_stride}; "
            f"planner_frame_interval={self.planner_frame_interval}; "
            f"frame_dir={self.frame_dir}; strict={self.strict}",
            "cyan",
        )

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
        self.last_plan: Optional[PlannerState] = None

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

    @staticmethod
    def _cfg_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return default
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in ("true", "1", "yes", "y"):
                return True
            if normalized in ("false", "0", "no", "n"):
                return False
        return bool(value)

    def _select_server_config(self, cfg: Optional[Any]) -> Any:
        nested = self._cfg_get(cfg, "videollama3_server", None)
        merged = {}
        if nested is not None:
            try:
                merged.update(dict(nested))
            except Exception:
                return nested
        if isinstance(cfg, dict):
            prefix = "videollama3_server."
            for key, value in cfg.items():
                if isinstance(key, str) and key.startswith(prefix):
                    merged[key[len(prefix):]] = value
        return merged

    def _select_local_config(self, cfg: Optional[Any]) -> Any:
        nested = self._cfg_get(cfg, "videollama3", None)
        merged = {}
        if nested is not None:
            try:
                merged.update(dict(nested))
            except Exception:
                return nested
        if isinstance(cfg, dict):
            prefix = "videollama3."
            for key, value in cfg.items():
                if isinstance(key, str) and key.startswith(prefix):
                    merged[key[len(prefix):]] = value
        return merged

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
        self.last_plan = None

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
        frame_indices = self._build_planner_frame_indices()
        cprint(
            "[VideoLLaMA3PlannerClient] build payload; "
            f"saved_frames={self._saved_frames}; "
            f"planner_frame_interval={self.planner_frame_interval}; "
            f"frame_indices={frame_indices}",
            "cyan",
        )
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
            "frame_indices": frame_indices,
            "strict": self.strict,
        }

    def _build_planner_frame_indices(self) -> list[int]:
        """Use training-style sparse frames: 0, interval, ..., current frame."""
        if self._saved_frames <= 0:
            return []

        last_idx = self._saved_frames - 1
        indices = list(range(0, self._saved_frames, self.planner_frame_interval))
        if not indices or indices[-1] != last_idx:
            indices.append(last_idx)
        return indices

    def generate_anwser(self, inputs=None):
        plan = self.plan(inputs)
        instruction = plan.as_next_subtask_text(self._last_subgoal)
        self.last_instruction = instruction
        if plan.execution_instruction(self._last_subgoal):
            self._last_subgoal = plan.execution_instruction(self._last_subgoal).rstrip(".")
        return instruction

    def plan(self, inputs=None) -> PlannerState:
        payload = inputs if isinstance(inputs, dict) else self._build_request_payload()
        request_t0 = time.perf_counter()
        frame_indices = payload.get("frame_indices") or []
        cprint(
            "[VideoLLaMA3PlannerClient] POST /plan start; "
            f"frame_dir={payload.get('frame_dir')}; "
            f"saved_frames={self._saved_frames}; "
            f"frame_indices_count={len(frame_indices)}; "
            f"last_frame_index={frame_indices[-1] if frame_indices else None}; "
            f"strict={payload.get('strict')}",
            "cyan",
        )
        try:
            requests_mod = self._get_requests()
            response = requests_mod.post(f"{self.url}/plan", json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            cprint(
                "[VideoLLaMA3PlannerClient] POST /plan done; "
                f"elapsed={time.perf_counter() - request_t0:.2f}s; "
                f"ok={data.get('ok')}; error={data.get('error')}",
                "cyan",
            )
        except Exception as exc:
            if self.strict:
                raise RuntimeError(f"VideoLLaMA3 planner server request failed: {exc}") from exc
            fallback = self._fallback_state(error=str(exc))
            cprint(
                f"[VideoLLaMA3PlannerClient] planner server request failed: {exc}; "
                f"using {fallback.as_next_subtask_text(self._last_subgoal)}",
                "yellow",
            )
            self.last_used_fallback = True
            self.last_instruction = fallback.as_next_subtask_text(self._last_subgoal)
            self.last_server_response = {"ok": False, "error": str(exc)}
            self.last_plan = fallback
            return fallback

        self.last_server_response = data
        self.last_raw_output = str(data.get("raw_text", data.get("raw_output", "")))
        parsed = data.get("parsed_json", {})
        self.last_parsed_json = parsed if isinstance(parsed, dict) else {}
        self.last_regex_subgoal = data.get("regex_subgoal")
        self.last_used_fallback = bool(data.get("used_fallback", False))

        if not bool(data.get("ok", False)):
            raw_preview = self.last_raw_output[:240].replace("\n", "\\n")
            instruction = str(data.get("instruction") or self._fallback_instruction()).strip()
            cprint(
                "[VideoLLaMA3PlannerClient] planner server returned ok=false; "
                f"error={data.get('error')}; "
                f"used_fallback={self.last_used_fallback}; "
                f"raw_output_prefix={raw_preview!r}; "
                f"instruction={instruction}",
                "yellow",
            )
            if self.strict:
                raise RuntimeError(f"VideoLLaMA3 planner server returned ok=false: {data.get('error')}")

        try:
            plan = self._state_from_server_response(data)
        except PlannerStateError as exc:
            raw_preview = self.last_raw_output[:240].replace("\n", "\\n")
            cprint(
                "[VideoLLaMA3PlannerClient] invalid planner schema; "
                f"error={exc}; raw_output_prefix={raw_preview!r}",
                "red",
            )
            if self.strict:
                raise
            plan = self._fallback_state(error=str(exc))
            self.last_used_fallback = True

        self.last_plan = plan
        self.last_instruction = plan.as_next_subtask_text(self._last_subgoal)
        new_instruction = plan.execution_instruction(self._last_subgoal)
        if new_instruction:
            self._last_subgoal = new_instruction.rstrip(".")
        return plan

    def _state_from_server_response(self, data: Dict[str, Any]) -> PlannerState:
        state = {
            "current_subgoal": data.get("current_subgoal"),
            "current_status": data.get("current_status"),
            "next_subgoal": data.get("next_subgoal"),
            "should_switch": data.get("should_switch"),
            "task_status": data.get("task_status"),
            "instruction": data.get("instruction"),
        }
        return planner_state_from_dict(state, raw_text=self.last_raw_output)

    def _fallback_state(self, error: str = "") -> PlannerState:
        subgoal = self._last_subgoal or (self.finished_subtasks[-1] if self.finished_subtasks else "") or "continue the task"
        return PlannerState(
            current_subgoal=subgoal.rstrip("."),
            current_status="in_progress",
            next_subgoal=subgoal.rstrip("."),
            should_switch=False,
            task_status="running",
            instruction=f"next_subtask: {subgoal.rstrip('.')}.",
            raw_text=error,
        )

    def _fallback_instruction(self) -> str:
        subgoal = self._last_subgoal or (self.finished_subtasks[-1] if self.finished_subtasks else "") or "continue the task"
        return f"next_subtask: {subgoal.rstrip('.')}."
