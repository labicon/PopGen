from typing import Any


def average_values(values: list[Any]) -> Any:
    """Average a homogeneous list of numbers or nested dicts of numbers."""
    if not values:
        return 0.0

    first = values[0]
    if isinstance(first, dict):
        out = {}
        for key in first.keys():
            key_values = [v[key] for v in values if isinstance(v, dict) and key in v]
            out[key] = average_values(key_values) if key_values else 0.0
        return out

    return sum(float(v) for v in values) / len(values)
