"""Structured state returned by high-level planners."""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional


class PlannerStateError(ValueError):
    """Raised when planner output cannot be validated as a structured state."""


@dataclass
class PlannerState:
    current_subgoal: str
    current_status: str
    next_subgoal: Optional[str]
    should_switch: bool
    task_status: str
    instruction: Optional[str] = None
    raw_text: str = ""

    def execution_instruction(self, previous_instruction: str = "") -> str:
        if self.task_status == "completed":
            return previous_instruction
        if self.should_switch:
            return (self.next_subgoal or self.current_subgoal or previous_instruction).strip()
        return (self.current_subgoal or previous_instruction).strip()

    def as_next_subtask_text(self, previous_instruction: str = "") -> str:
        instruction = self.execution_instruction(previous_instruction).strip().rstrip(".")
        if not instruction:
            instruction = "continue the task"
        return f"next_subtask: {instruction}."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_subgoal": self.current_subgoal,
            "current_status": self.current_status,
            "next_subgoal": self.next_subgoal,
            "should_switch": self.should_switch,
            "task_status": self.task_status,
            "instruction": self.instruction,
            "raw_text": self.raw_text,
        }


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


def normalize_current_status(value: Any) -> str:
    status = str(value or "").strip().lower()
    mapping = {
        "in_progress": "in_progress",
        "in progress": "in_progress",
        "ongoing": "in_progress",
        "running": "in_progress",
        "not_done": "in_progress",
        "not done": "in_progress",
        "completed": "completed",
        "complete": "completed",
        "done": "completed",
        "finished": "completed",
    }
    if status not in mapping:
        raise PlannerStateError(f"invalid current_status: {value!r}")
    return mapping[status]


def normalize_task_status(value: Any) -> str:
    status = str(value or "").strip().lower()
    mapping = {
        "running": "running",
        "in_progress": "running",
        "in progress": "running",
        "ongoing": "running",
        "completed": "completed",
        "complete": "completed",
        "done": "completed",
        "finished": "completed",
    }
    if status not in mapping:
        raise PlannerStateError(f"invalid task_status: {value!r}")
    return mapping[status]


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "yes", "1"):
            return True
        if normalized in ("false", "no", "0"):
            return False
    raise PlannerStateError(f"invalid should_switch: {value!r}")


def planner_state_from_dict(data: Dict[str, Any], raw_text: str = "") -> PlannerState:
    if not isinstance(data, dict):
        raise PlannerStateError("planner output is not a JSON object")

    required = ("current_subgoal", "current_status", "next_subgoal", "should_switch", "task_status")
    missing = [key for key in required if key not in data]
    if missing:
        raise PlannerStateError(f"planner output missing fields: {missing}")

    current_subgoal = str(data.get("current_subgoal") or "").strip().rstrip(".")
    current_status = normalize_current_status(data.get("current_status"))
    task_status = normalize_task_status(data.get("task_status"))
    should_switch = normalize_bool(data.get("should_switch"))

    raw_next = data.get("next_subgoal")
    next_subgoal = None if raw_next is None else str(raw_next).strip().rstrip(".")
    if next_subgoal == "":
        next_subgoal = None

    if not current_subgoal and task_status != "completed":
        raise PlannerStateError("current_subgoal is empty while task is still running")
    if should_switch and task_status != "completed" and not next_subgoal:
        raise PlannerStateError("should_switch=true requires non-empty next_subgoal")

    instruction = data.get("instruction")
    return PlannerState(
        current_subgoal=current_subgoal,
        current_status=current_status,
        next_subgoal=next_subgoal,
        should_switch=should_switch,
        task_status=task_status,
        instruction=str(instruction).strip() if instruction else None,
        raw_text=raw_text,
    )


def planner_state_from_text(text: str) -> PlannerState:
    parsed = parse_json_object(text)
    if parsed is None:
        raise PlannerStateError("planner output is not valid JSON")
    return planner_state_from_dict(parsed, raw_text=text)
