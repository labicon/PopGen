from pathlib import Path


def find_reward_checkpoints(save_dir: str | Path) -> list[Path]:
    """Return candidate reward-training checkpoint files under save_dir."""
    base = Path(save_dir)
    patterns = (
        "checkpoints/checkpoint_*.pt",
        "checkpoints/checkpoint*.pt",
        "**/checkpoints/checkpoint_*.pt",
        "**/checkpoints/checkpoint*.pt",
    )
    matches: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in base.glob(pattern):
            resolved = path.resolve()
            if resolved not in seen:
                seen.add(resolved)
                matches.append(path)
    return sorted(matches, key=lambda p: p.stat().st_mtime)
