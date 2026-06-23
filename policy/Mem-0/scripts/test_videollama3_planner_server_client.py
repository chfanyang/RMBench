"""Smoke tests for the VideoLLaMA3 planner server/client path.

This test does not load VideoLLaMA3. It monkeypatches the HTTP layer used by
VideoLLaMA3PlannerClient and checks the request/response contract.
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np

MEM0_ROOT = Path(__file__).resolve().parents[1]
if str(MEM0_ROOT) not in sys.path:
    sys.path.insert(0, str(MEM0_ROOT))

from source.models.planning_module import videollama3_planner_client as client_module
from source.models.planning_module.videollama3_planner_client import VideoLLaMA3PlannerClient


class FakeResponse:
    def __init__(self, payload, status_error=None):
        self.payload = payload
        self.status_error = status_error

    def raise_for_status(self):
        if self.status_error:
            raise self.status_error

    def json(self):
        return self.payload


class FakeRequests:
    def __init__(self):
        self.calls = []
        self.fail = False

    def post(self, url, json, timeout):
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        if self.fail:
            raise RuntimeError("mock server unavailable")
        return FakeResponse(
            {
                "ok": True,
                "raw_output": '{"current_subgoal":"cover the right block","current_status":"in_progress","next_subgoal":"cover the right block","should_switch":false,"task_status":"running"}',
                "raw_text": '{"current_subgoal":"cover the right block","current_status":"in_progress","next_subgoal":"cover the right block","should_switch":false,"task_status":"running"}',
                "parsed_json": {
                    "current_subgoal": "cover the right block",
                    "current_status": "in_progress",
                    "next_subgoal": "cover the right block",
                    "should_switch": False,
                    "task_status": "running",
                },
                "regex_subgoal": None,
                "subgoal": "cover the right block",
                "used_fallback": False,
                "current_subgoal": "cover the right block",
                "current_status": "in_progress",
                "next_subgoal": "cover the right block",
                "should_switch": False,
                "task_status": "running",
                "instruction": "next_subtask: cover the right block.",
                "error": None,
            }
        )


def build_config(frame_dir):
    return {
        "global_task": "Cover the blocks from right to left.",
        "videollama3_server": {
            "url": "http://127.0.0.1:8009",
            "timeout": 12,
            "fps": 1,
            "max_frames": 8,
            "max_new_tokens": 32,
            "frame_stride": 1,
            "planner_frame_interval": 30,
            "frame_dir": frame_dir,
            "strict": False,
        },
    }


def main():
    fake_requests = FakeRequests()
    client_module.requests = fake_requests

    with tempfile.TemporaryDirectory() as tmpdir:
        frame_dir = os.path.join(tmpdir, "vl3_client_frames")
        planner = VideoLLaMA3PlannerClient(
            config=build_config(frame_dir),
            global_task="Cover the blocks from right to left.",
        )
        planner.update_initial_observation("./_tmp_visual/init.png")

        rgb = np.zeros((32, 32, 3), dtype=np.uint8)
        rgb[..., 0] = 255
        planner.append_frame_array(rgb)

        saved = sorted(Path(frame_dir).glob("*.jpg"))
        assert len(saved) == 1, "client did not save one RGB frame"

        instruction = planner.generate_anwser()
        assert instruction == "next_subtask: cover the right block."
        assert fake_requests.calls, "client did not POST to server"
        call = fake_requests.calls[-1]
        assert call["url"] == "http://127.0.0.1:8009/plan"
        assert call["timeout"] == 12
        assert call["json"]["frame_dir"] == str(Path(frame_dir).resolve())
        assert call["json"]["global_task"] == "Cover the blocks from right to left."
        assert call["json"]["fps"] == 1
        assert call["json"]["max_frames"] == 8
        assert call["json"]["max_new_tokens"] == 32
        assert call["json"]["frame_indices"] == [0]
        assert planner.last_parsed_json["next_subgoal"] == "cover the right block"
        assert planner.last_plan.current_subgoal == "cover the right block"
        assert planner.last_plan.current_status == "in_progress"
        assert planner.last_used_fallback is False

        for _ in range(1, 193):
            planner.append_frame_array(rgb)
        instruction = planner.generate_anwser()
        assert instruction == "next_subtask: cover the right block."
        call = fake_requests.calls[-1]
        assert call["json"]["frame_indices"] == [0, 30, 60, 90, 120, 150, 180, 192]

        fake_requests.fail = True
        fallback = planner.generate_anwser()
        assert fallback == "next_subtask: cover the right block."
        assert planner.last_used_fallback is True

        planner.reset_episode()
        assert not list(Path(frame_dir).glob("*.jpg"))
        assert planner.initial_observation is None
        assert planner.key_information == []
        assert planner.finished_subtasks == []
        assert planner.last_raw_output == ""
        assert planner.last_parsed_json == {}
        assert planner.last_regex_subgoal is None
        assert planner.last_used_fallback is False
        assert planner.last_instruction == ""
        assert planner.last_server_response == {}
        assert planner.last_plan is None

    print("VideoLLaMA3PlannerClient server/client smoke test passed without loading model weights.")


if __name__ == "__main__":
    main()
