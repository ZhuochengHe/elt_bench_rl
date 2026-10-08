"""Tests for deterministic task selection in the training dataset."""

from __future__ import annotations

import pytest

from eltbench.task_selection import select_trainable_tasks


def test_explicit_task_selection_returns_only_requested_task():
    assert select_trainable_tasks(
        ["address", "books"], task_name="address", seed=0, max_tasks=0
    ) == ["address"]


def test_explicit_unavailable_task_fails_with_available_tasks():
    with pytest.raises(ValueError, match="Available tasks: address, books"):
        select_trainable_tasks(
            ["address", "books"], task_name="unknown", seed=0, max_tasks=0
        )


def test_random_task_selection_preserves_seeded_max_tasks_behavior():
    names = ["address", "books", "twilio"]
    result = select_trainable_tasks(
        names, task_name=None, seed=42, max_tasks=2
    )
    assert result == select_trainable_tasks(
        names, task_name=None, seed=42, max_tasks=2
    )
    assert len(result) == 2
    assert names == ["address", "books", "twilio"]
