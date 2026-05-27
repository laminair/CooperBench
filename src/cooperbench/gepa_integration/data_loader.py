"""Task DataLoader for GEPA integration.

Wraps CooperBench's ``discover_tasks()`` to produce a GEPA-compatible
``DataLoader`` that yields task dicts as ``DataInst`` objects.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from cooperbench.runner.tasks import DEFAULT_DATASET_DIR, discover_tasks


class CooperBenchDataLoader:
    """GEPA DataLoader backed by CooperBench's task discovery.

    Args:
        subset: Subset name (e.g. ``"9b_easy"``, ``"core"``).
        dataset_dir: Root of the dataset tree.
        split: ``"train"``, ``"val"``, or ``"all"``.
        val_ratio: Fraction of tasks to hold out for validation when split is
            ``"train"`` or ``"val"``.
        seed: Random seed for reproducible train/val splits.
    """

    def __init__(
        self,
        subset: str | None = None,
        dataset_dir: Path | str | None = None,
        split: str = "all",
        val_ratio: float = 0.2,
        seed: int = 42,
    ):
        self._subset = subset
        self._dataset_dir = Path(dataset_dir) if dataset_dir else DEFAULT_DATASET_DIR
        self._split = split
        self._val_ratio = val_ratio
        self._seed = seed

        all_tasks = discover_tasks(subset=subset, dataset_dir=self._dataset_dir)
        all_tasks = self._dedup_tasks(all_tasks)

        rng = random.Random(seed)
        indices = list(range(len(all_tasks)))
        rng.shuffle(indices)

        val_count = max(1, int(len(all_tasks) * val_ratio))
        val_indices = set(indices[:val_count])
        train_indices = set(indices[val_count:])

        if split == "train":
            self._tasks = [all_tasks[i] for i in sorted(train_indices)]
        elif split == "val":
            self._tasks = [all_tasks[i] for i in sorted(val_indices)]
        else:
            self._tasks = all_tasks

    def _dedup_tasks(self, tasks: list[dict]) -> list[dict]:
        """Deduplicate tasks by (repo, task_id, features) key."""
        seen = set()
        result = []
        for t in tasks:
            key = (t["repo"], t["task_id"], tuple(t["features"]))
            if key not in seen:
                seen.add(key)
                result.append(t)
        return result

    def all_ids(self) -> list[int]:
        return list(range(len(self._tasks)))

    def fetch(self, ids: list[int]) -> list[dict[str, Any]]:
        return [self._tasks[i] for i in ids]

    def __len__(self) -> int:
        return len(self._tasks)

    def __repr__(self) -> str:
        return f"CooperBenchDataLoader(subset={self._subset!r}, split={self._split!r}, tasks={len(self._tasks)})"
