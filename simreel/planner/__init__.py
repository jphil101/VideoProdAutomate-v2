from __future__ import annotations

from ..config import PlannerSettings
from ..plan_schema import StoryPlan
from ..utils import log
from .heuristic import plan_heuristic


def make_plan(events_text: str, script_text: str, sentences: list[str], settings: PlannerSettings) -> tuple[StoryPlan, str]:
    choice = settings.planner
    if choice in ("auto", "claude"):
        from .llm import credentials_available, plan_with_claude

        if choice == "claude" or credentials_available():
            try:
                return plan_with_claude(events_text, sentences, settings.model, settings.effort), "claude"
            except Exception as exc:
                if choice == "claude":
                    raise
                log(f"   Claude planner unavailable ({exc}); falling back to the heuristic planner")
        else:
            log("   no Claude credentials found; using the offline heuristic planner")
    return plan_heuristic(events_text, script_text), "heuristic"
