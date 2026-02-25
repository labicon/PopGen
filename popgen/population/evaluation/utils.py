from typing import Any


def combine_feedback(instance_metrics: list[dict[str, Any]]) -> str:
    """Combine feedback from failing rollouts into a string for the next generation attempt."""

    def _build_feedback_str(i: int, reasoning: str, feedback: str) -> str:
        s = f"- Rollout #{i}:\n"
        s += f"  Failure Reason: {reasoning}\n"
        s += f"  Feedback: {feedback}\n"
        return s

    bullets = [_build_feedback_str(i, m["reasoning"], m["feedback"]) for i, m in enumerate(instance_metrics) if not m.get("evaluation")]
    if not bullets:
        return "No specific feedback provided."
    return "Feedback from failed rollouts:\n" + "\n".join(bullets)
