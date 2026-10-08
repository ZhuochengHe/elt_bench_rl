"""Select ELT-Bench tasks for training rollouts."""

from __future__ import annotations

import random


def select_trainable_tasks(
    names: list[str], *, task_name: str | None, seed: int, max_tasks: int
) -> list[str]:
    if task_name is not None:
        if task_name not in names:
            raise ValueError(
                f"Task {task_name!r} is not trainable for this destination. "
                f"Available tasks: {', '.join(names)}"
            )
        return [task_name]

    selected = list(names)
    random.Random(seed).shuffle(selected)
    return selected[:max_tasks] if max_tasks else selected
