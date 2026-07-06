"""MemoryMatters low-level agent wrapper for deployment."""

import sys, os
sys.path.append (os.path.abspath (os.path.join (os.path.dirname(__file__), '../..')))
import subprocess

from typing import Dict, Optional

import shutil
import numpy as np
import torch
from PIL import Image
from termcolor import cprint
from omegaconf import OmegaConf

from source.models.execution_module.memorymatters_executor import MemoryMattersExecutor
from source.models.planning_module.memorymatters_planner import MemoryMattersPlanner
from source.models.planning_module.videollama3_planner import VideoLLaMA3Planner
from source.models.planning_module.videollama3_planner_client import VideoLLaMA3PlannerClient
from source.models.planning_module.planner_state import PlannerState
from source.training.utils.trainer_tools import resize_images
import source.utils.pil_tools as pil_tools


class MemoryMattersAgent:
    """Lightweight wrapper around MemoryMattersLowLevelPolicy for inference."""

    def __init__(self, cfg: OmegaConf, ckpt_path: str, device: torch.device):
        self.config = cfg
        
        self.task_type = "Mn"
        task_name = self.config.get("task_name", "unknown_task")
        if task_name in ["swap_blocks", "swap_T", "observe_and_pickup", "put_back_block", "rearrange_blocks"]:
            self.task_type = "M1"
        cprint (f"task name: {task_name}", "red")
        cprint (f"task type: {self.task_type}", "red")
        
        self.device = device
        self.episode_id = 0
        self.is_init = 0
        
        self.action_horizon = self.config.get("action_horizon", 30)
        self.action_strip = self.action_horizon
        self.threshold = self.config.get("threshold", 2)
        self.planner_type = self._normalize_planner_type(self.config.get("planner_type", "memorymatters"))
        self.structured_planner = self.planner_type in ("videollama3_local", "videollama3_server")
        self.use_classifier_switch = self._cfg_bool(
            self.config.get("use_classifier_switch", not self.structured_planner),
            default=not self.structured_planner,
        )
        self.planner_query_interval = max(1, int(self.config.get("planner_query_interval", 30)))
        self.planner_switch_confirm_steps = max(1, int(self.config.get("planner_switch_confirm_steps", 3)))
        
        self._last_fused = None
        self._last_original = None
        self._last_state = None
        self._time_action_history = {}
        
        self.executor = MemoryMattersExecutor(self.config, device=device).to(device)
        self.executor.eval()
        self._load_ckpt(ckpt_path)
        
        self.iter = 0
        self.stage = 0
        self.end_signal_count = 0
        self.action_count = 0
       
        self.high_model = self._build_high_model()
        cprint (f"global_task = {self.config.get ('global_task', '')}", "red")  
        cprint(
            f"[deploy] planner_type={self.planner_type}; "
            f"use_classifier_switch={self.use_classifier_switch}; "
            f"planner_query_interval={self.planner_query_interval}; "
            f"planner_switch_confirm_steps={self.planner_switch_confirm_steps}",
            "cyan",
        )
        
        # reset tmp video folder
        shutil.rmtree ("./_tmp_visual/", ignore_errors = True)
        os.makedirs ("./_tmp_visual/", exist_ok = True)
        
        self.instruction = ""
        self.last_plan: Optional[PlannerState] = None
        self.pending_switch_plan: Optional[PlannerState] = None
        self.pending_switch_step: Optional[int] = None
        self.task_finished = False

    @staticmethod
    def _normalize_planner_type(planner_type):
        planner_type = str(planner_type or "memorymatters").lower()
        if planner_type == "videollama3":
            cprint("[deploy] planner_type=videollama3 is deprecated; using videollama3_local", "yellow")
            planner_type = "videollama3_local"
        return planner_type

    @staticmethod
    def _cfg_bool(value, default=False):
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

    def _build_high_model(self):
        if self.planner_type == "videollama3_local":
            cprint("[deploy] high-level planner: VideoLLaMA3Planner (local debug; loads model in RMBench process)", "cyan")
            return VideoLLaMA3Planner(
                config=self.config,
                global_task=self.config.get("global_task", ""),
                device=self.device,
            )

        if self.planner_type == "videollama3_server":
            cprint("[deploy] high-level planner: VideoLLaMA3PlannerClient (HTTP server)", "cyan")
            return VideoLLaMA3PlannerClient(
                config=self.config,
                global_task=self.config.get("global_task", ""),
            )

        if self.planner_type != "memorymatters":
            cprint(f"[deploy] unknown planner_type={self.planner_type}; falling back to memorymatters", "yellow")

        cprint("[deploy] high-level planner: MemoryMattersPlanner", "cyan")
        return MemoryMattersPlanner (
            config = OmegaConf.load (self.config.get ("planning_module_config_path", "")),
            global_task = self.config.get ("global_task", ""),
            vllm_url = self.config.get ("vllm_url", ""),
        )

    def append_planner_frame(self, rgb):
        if self.task_type == "Mn" and hasattr(self.high_model, "append_frame_array"):
            self.high_model.append_frame_array(rgb)

    def accumulate_actions_chunk(self, actions_model: np.ndarray) -> None:
        """
        Accumulate one predicted normalized action chunk at absolute time indices.

        Each eval produces predictions for [iter, iter + action_horizon). We append
        these per-time predictions into a history for later smoothing across overlapping evals.

        Args:
            actions_model: np.ndarray with shape (T, D) or (1, T, D) of normalized actions.
        """
        if actions_model is None:
            return

        # squeeze batch if present
        if actions_model.ndim == 3:
            actions_model = actions_model[0] if actions_model.shape[0] > 0 else actions_model.squeeze(0)
        if actions_model.ndim == 1:
            actions_model = actions_model.reshape(1, -1)

        T, _ = actions_model.shape
        horizon = min(self.action_horizon, T)
        for k in range(horizon):
            t = self.iter + k
            hist = self._time_action_history.get(t, [])
            hist.append(actions_model[k])
            self._time_action_history[t] = hist

    def get_smoothed_actions(self, start: int, length: int) -> np.ndarray:
        """
        Get smoothed normalized actions for absolute time range [start, start+length).

        Smoothing is the mean over all accumulated predictions for each time index.

        Args:
            start: absolute starting time index (typically current `self.iter`).
            length: number of steps to return (typically `self.action_strip`).

        Returns:
            np.ndarray of shape (length, D) with smoothed normalized actions.
        """
        steps = []
        for i in range(length):
            t = start + i
            hist = self._time_action_history.get(t)
            if not hist:
                cprint(f"[deploy] missing predictions for time {t}; falling back to zeros", "yellow")
                # Determine D from any available history entry
                if steps:
                    D = steps[0].shape[-1]
                else:
                    # attempt to fetch from a nearby index
                    any_hist = next((v for v in self._time_action_history.values() if v), None)
                    D = any_hist[0].shape[-1] if any_hist else 16
                steps.append(np.zeros((D,), dtype=np.float32))
            else:
                steps.append(np.mean(np.stack(hist, axis=0), axis=0))

        return np.stack(steps, axis=0)

    def discard_action_history_from(self, start: int) -> None:
        """Drop cached future action predictions so the next eval recomputes them."""
        stale_keys = [t for t in self._time_action_history.keys() if t >= start]
        for t in stale_keys:
            del self._time_action_history[t]
        
    def _set_video_ffmpeg (self):
        self.ffmpeg = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "rgb24",
                "-video_size", "320x240",
                "-framerate", "4",
                "-i",
                "-",
                "-pix_fmt",
                "yuv420p",
                "-vcodec",
                "libx264",
                "-crf",
                "23",
                f"./_tmp_visual/video_{self.stage}.mp4",
            ],
            stdin=subprocess.PIPE,
        )
    
    def _del_video_ffmpeg(self):
        if self.ffmpeg:
            self.ffmpeg.stdin.close()
            self.ffmpeg.wait()
            del self.ffmpeg
        
    def get_instruction (self):
        if self.structured_planner:
            plan = self.query_planner(update_instruction=False)
            selected = plan.execution_instruction(self.instruction)
            if selected:
                self.instruction = selected
            return

        qwen_inputs = self.high_model.prepare_qwen_input()
        answer = self.high_model.generate_anwser (qwen_inputs)
        subtask = answer.split("next_subtask: ")[-1].split(".")[0]
        self.instruction = subtask
        cprint (f"[deploy] high-level instruction: {self.instruction}", "cyan")

    def query_planner(self, update_instruction: bool = False, current_step: Optional[int] = None) -> PlannerState:
        if not self.structured_planner:
            raise RuntimeError("query_planner is only available for VideoLLaMA3 structured planners")
        if not hasattr(self.high_model, "plan"):
            raise RuntimeError("VideoLLaMA3 planner does not expose plan()")

        inputs = self.high_model.prepare_qwen_input()
        plan = self.high_model.plan(inputs)
        self.last_plan = plan

        previous_instruction = self.instruction
        if plan.task_status == "completed":
            self.task_finished = True
        elif update_instruction:
            selected = plan.execution_instruction(previous_instruction)
            if selected:
                self.instruction = selected

        cprint(
            "[deploy][VideoLLaMA3-only switch] "
            f"step={current_step if current_step is not None else self.iter}; "
            f"iter_base={self.iter}; current_instruction={previous_instruction!r}; "
            f"raw={plan.raw_text}; current_subgoal={plan.current_subgoal!r}; "
            f"current_status={plan.current_status}; next_subgoal={plan.next_subgoal!r}; "
            f"should_switch={plan.should_switch}; task_status={plan.task_status}; "
            f"selected_instruction={self.instruction!r}; update_instruction={update_instruction}; "
            f"finished_subtasks={getattr(self.high_model, 'finished_subtasks', [])}",
            "cyan",
        )
        return plan
    
    def init_high_with_image (self):
        self.high_model.update_initial_observation ("./_tmp_visual/init.png")
        self.get_instruction ()
        self._set_video_ffmpeg ()
        
    def update_high_observation (self):
        self.high_model.update_image_or_video_input([f"./_tmp_visual/image_{self.stage}.png"], [self.instruction])
        self.get_instruction ()
        self.stage += 1
        self._set_video_ffmpeg ()

    def apply_planner_switch(self, rgb=None) -> None:
        """Commit a VideoLLaMA3-decided subtask switch and reset low-level state."""
        if not self.structured_planner:
            return
        switch_plan = self.pending_switch_plan or self.last_plan
        completed = switch_plan.current_subgoal if switch_plan else self.instruction
        if switch_plan is not None:
            selected = switch_plan.execution_instruction(self.instruction)
            if selected:
                self.instruction = selected
        if rgb is not None:
            Image.fromarray(rgb).save(f"./_tmp_visual/image_{self.stage}.png")
        if hasattr(self.high_model, "update_image_or_video_input"):
            self.high_model.update_image_or_video_input([f"./_tmp_visual/image_{self.stage}.png"], [completed])
        self.pending_switch_plan = None
        self.pending_switch_step = None
        self.stage += 1
        self.end_signal_count = 0
        self.action_count = 0
        self.executor.memory_bank.reset()
        cprint(
            f"[deploy][VideoLLaMA3-only switch] switching to stage {self.stage}; "
            f"instruction={self.instruction!r}; completed={completed!r}",
            "green",
        )

    def _load_ckpt(self, ckpt_path: str) -> None:
        """Load checkpoint if provided; warn otherwise."""
        if not ckpt_path:
            cprint("[deploy] no checkpoint path provided, using randomly initialized weights", "red")
            return
        if not os.path.isfile(ckpt_path):
            cprint(f"[deploy] checkpoint not found: {ckpt_path}", "red")
            return
        try:
            payload = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        except TypeError:
            # PyTorch <2.6 fallback without weights_only arg
            payload = torch.load(ckpt_path, map_location=self.device)
        state_dict = payload.get("model_state_dict", payload)
        missing, unexpected = self.executor.load_state_dict(state_dict, strict=False)
        cprint(f"[deploy] checkpoint loaded: {ckpt_path}", "green")
        if missing:
            cprint(f"[deploy] missing keys: {missing}", "yellow")
        if unexpected:
            cprint(f"[deploy] unexpected keys: {unexpected}", "yellow")

    def reset(self) -> None:
        """Reset MemoryBank for a new episode."""
        if self.is_init == 1:
            self._del_video_ffmpeg()
        
        self.episode_id += 1
        cprint(f"[deploy] memory reset (episode {self.episode_id})", "cyan")
        self.executor.memory_bank.reset()
        
        self.is_init = 0
        
        self._last_fused = None
        self._last_original = None
        self._last_state = None
        
        self.iter = 0
        self.stage = 0
        self.end_signal_count = 0
        self.action_count = 0
        self._time_action_history = {}
        self.last_plan = None
        self.pending_switch_plan = None
        self.pending_switch_step = None
        self.task_finished = False
        
        if hasattr(self.high_model, "reset_episode"):
            self.high_model.reset_episode()
        elif hasattr(self.high_model, "reset_stream"):
            self.high_model.reset_stream()
        else:
            self.high_model = self._build_high_model()
        
        # reset tmp video folder
        shutil.rmtree ("./_tmp_visual/", ignore_errors = True)
        os.makedirs ("./_tmp_visual/", exist_ok = True)
        
        self.instruction = ""

    @torch.inference_mode()
    def update_obs(self, obs_payload: Dict[str, object], use_classifier: Optional[bool] = None):
        """
        Update MemoryBank without running action head; cache fused feature for next action.

        Args:
            obs_payload: dict with keys image (PIL) and optional state/instruction.
            instruction: override string; falls back to obs_payload["instruction"] or empty.
        """
        images = [[obs_payload["image"]]]
        images = resize_images(images, target_size=(224, 224))
        instruction = [obs_payload.get("instruction", "")]
        qwen_inputs = self.executor.qwen_model.build_qwenvl_inputs(
            images, instruction,
            system_prompt=None, add_summary_token=False, add_generation_prompt=False, max_length=128
        )

        qwenvl_outputs = self.executor.qwen_model(
            **qwen_inputs,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden_state = qwenvl_outputs.hidden_states[-1]    # (batch_size, sequence_length, hidden_size). final layer hidden state.
        image_feature, text_feature = self.executor.qwen_model.extract_features(qwen_inputs.input_ids, last_hidden_state) # (batch_size, 1, hidden_size), (batch_size, 1, hidden_size)
        
        # Step 2: Memory Fusion and Update
        if use_classifier is None:
            use_classifier = self.use_classifier_switch
        memory_fusion_output, anchor_output, sub_end_flag = self.executor.memory_bank.update_on_eval(
            image_feature,
            text_feature,
            self.executor.classifier,
            episode_id=self.episode_id,
            use_classifier=use_classifier,
        ) # anchor_output: (batch_size, 1, hidden_size), memory_fusion_output: (batch_size, 1, hidden_size), sub_end_flag: bool

        summary_features = torch.cat([memory_fusion_output, anchor_output, text_feature], dim=1) # (batch_size, 3, hidden_size)
        fused = summary_features
        self._last_original = fused  # (B, 1, H)
        self._last_fused = fused  # (B, 1, H)
        
        state = obs_payload.get("state")
        if state is not None:
            state_tensor = torch.from_numpy(np.array(state)).to(
                device=fused.device, dtype=fused.dtype
            )
            if state_tensor.dim() == 1:
                state_tensor = state_tensor.unsqueeze(0)
            if state_tensor.dim() == 2:  # (B, state_dim) -> (B, 1, state_dim)
                state_tensor = state_tensor.unsqueeze(1)
            self._last_state = state_tensor
        else:
            self._last_state = None

        if use_classifier:
            self.end_signal_count += sub_end_flag
        self.action_count += 1
        if use_classifier and self.end_signal_count != self.executor.memory_bank.end_signal_count[self.episode_id]:
            cprint(
                f"mismatch in end_signal_count: agent {self.end_signal_count} vs memory_bank {self.executor.memory_bank.end_signal_count[self.episode_id]}",
                "red"
            )
        return sub_end_flag

    @torch.inference_mode()
    def update_obs_without_classifier(self, obs_payload: Dict[str, object]):
        return self.update_obs(obs_payload, use_classifier=False)

    @torch.inference_mode()
    def get_action(self) -> dict:
        """
        Predict action chunk (model layout) using latest cached observation and MemoryBank.
        """
        if self._last_fused is None:
            cprint("[deploy] obs_cache is empty; call update_obs first", "red")
            return None 
        
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.executor.action_model.predict_action(self._last_fused, self._last_state)  # (B, chunk_len, action_dim)
        
        if pred_actions is None:
            cprint("[deploy] model produced no actions", "red")
            return None
        
        return {
            "normalized_actions": pred_actions.detach().cpu().numpy(),
        }
