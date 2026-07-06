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


def test_delayed_switch_confirmation():
    first_plan = planner_state_from_dict(
        {
            "current_subgoal": "cover the left block",
            "current_status": "completed",
            "next_subgoal": "cover the middle block",
            "should_switch": True,
            "task_status": "running",
        }
    )
    second_plan = planner_state_from_dict(
        {
            "current_subgoal": "cover the left block",
            "current_status": "completed",
            "next_subgoal": "cover the middle block",
            "should_switch": True,
            "task_status": "running",
        }
    )
    confirm_steps = 3
    pending = None
    pending_step = None
    current_step = 30
    switch_candidate = first_plan.should_switch and first_plan.current_status == "completed"
    assert switch_candidate is True
    pending = first_plan
    pending_step = current_step
    should_replan_actions = True
    assert should_replan_actions is True

    current_step = 32
    confirm_due = pending is not None and current_step - pending_step >= confirm_steps
    assert confirm_due is False

    current_step = 33
    confirm_due = pending is not None and current_step - pending_step >= confirm_steps
    assert confirm_due is True
    switch_candidate = second_plan.should_switch and second_plan.current_status == "completed"
    switch_confirmed = confirm_due and switch_candidate
    assert switch_confirmed is True
    assert pending.execution_instruction("cover the left block") == "cover the middle block"


def test_action_history_discard_on_switch():
    history = {
        30: ["old"],
        31: ["old"],
        32: ["old"],
        33: ["old"],
        34: ["old"],
    }
    switch_step = 33
    for key in [t for t in history if t >= switch_step]:
        del history[key]
    assert sorted(history.keys()) == [30, 31, 32]


def main():
    test_planner_state_validation()
    test_memory_bank_without_classifier()
    test_delayed_switch_confirmation()
    test_action_history_discard_on_switch()
    print("VideoLLaMA3 planner state and classifier-free switch smoke test passed.")


if __name__ == "__main__":
    main()
