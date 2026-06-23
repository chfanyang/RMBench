"""Smoke tests for VideoLLaMA3-only planner state and classifier-free updates."""

import sys
from pathlib import Path

import torch

MEM0_ROOT = Path(__file__).resolve().parents[1]
if str(MEM0_ROOT) not in sys.path:
    sys.path.insert(0, str(MEM0_ROOT))

from source.models.execution_module.memory_bank.memory_bank import MemoryBank
from source.models.planning_module.planner_state import planner_state_from_dict


class RaisingClassifier:
    def predict(self, *args, **kwargs):
        raise AssertionError("classifier.predict() must not be called when use_classifier=False")


def test_planner_state_validation():
    plan = planner_state_from_dict(
        {
            "current_subgoal": "cover the left block",
            "current_status": "ongoing",
            "next_subgoal": "cover the left block",
            "should_switch": False,
            "task_status": "ongoing",
        },
        raw_text='{"current_status":"ongoing"}',
    )
    assert plan.current_status == "in_progress"
    assert plan.task_status == "running"
    assert plan.execution_instruction("") == "cover the left block"

    switch_plan = planner_state_from_dict(
        {
            "current_subgoal": "cover the left block",
            "current_status": "done",
            "next_subgoal": "cover the middle block",
            "should_switch": True,
            "task_status": "running",
        }
    )
    assert switch_plan.current_status == "completed"
    assert switch_plan.execution_instruction("cover the left block") == "cover the middle block"

    done_plan = planner_state_from_dict(
        {
            "current_subgoal": "open the right cover",
            "current_status": "completed",
            "next_subgoal": None,
            "should_switch": False,
            "task_status": "done",
        }
    )
    assert done_plan.task_status == "completed"


def test_memory_bank_without_classifier():
    bank = MemoryBank(hidden_dim=8, window_size=4, num_heads=2, dropout=0.0, device=torch.device("cpu"))
    bank.eval()
    image_feature = torch.randn(1, 1, 8)
    text_feature = torch.randn(1, 1, 8)
    fused, anchor, sub_end = bank.update_on_eval(
        image_feature,
        text_feature,
        RaisingClassifier(),
        episode_id=0,
        use_classifier=False,
    )
    assert fused.shape == (1, 1, 8)
    assert anchor.shape == (1, 1, 8)
    assert sub_end is False
    assert bank.end_signal_count[0] == 0


def main():
    test_planner_state_validation()
    test_memory_bank_without_classifier()
    print("VideoLLaMA3 planner state and classifier-free switch smoke test passed.")


if __name__ == "__main__":
    main()
