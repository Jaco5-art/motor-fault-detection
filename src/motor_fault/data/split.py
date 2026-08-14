from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from motor_fault.config import read_yaml
from motor_fault.data.archive import KaggleRobotArchive, MOTORS


@dataclass(frozen=True)
class SplitDefinition:
    name: str
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]
    threshold_policy: str

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SplitDefinition":
        payload = read_yaml(path)
        return cls(
            name=str(payload["name"]),
            train=tuple(map(str, payload["train"])),
            validation=tuple(map(str, payload["validation"])),
            test=tuple(map(str, payload["test"])),
            threshold_policy=str(payload["threshold_policy"]),
        )

    @property
    def by_name(self) -> dict[str, tuple[str, ...]]:
        return {
            "train": self.train,
            "validation": self.validation,
            "test": self.test,
        }

    def validate(self, archive_run_ids: list[str] | tuple[str, ...]) -> None:
        groups = {name: set(ids) for name, ids in self.by_name.items()}
        names = list(groups)
        for i, left in enumerate(names):
            for right in names[i + 1 :]:
                overlap = groups[left] & groups[right]
                if overlap:
                    raise ValueError(f"Run leakage between {left} and {right}: {sorted(overlap)}")
        configured = set().union(*groups.values())
        available = set(map(str, archive_run_ids))
        if configured != available:
            raise ValueError(
                f"Split coverage mismatch. missing={sorted(available-configured)}, "
                f"unknown={sorted(configured-available)}"
            )

    def build_run_manifest(self, archive: KaggleRobotArchive) -> pd.DataFrame:
        self.validate(archive.run_ids("training"))
        records = []
        for split_name, run_ids in self.by_name.items():
            for run_id in run_ids:
                run = archive.load_run(run_id)
                record = {
                    "run_id": run_id,
                    "split": split_name,
                    "activity": run.activity,
                    "n_rows": run.n_rows,
                    "start_time": float(run.time[0]),
                    "end_time": float(run.time[-1]),
                    "declared_failure_motors": ",".join(map(str, run.declared_failures)),
                }
                for motor in MOTORS:
                    y = run.labels[:, motor - 1]
                    record[f"m{motor}_fault_points"] = int(y.sum())
                    record[f"m{motor}_fault_ratio"] = float(y.mean())
                records.append(record)
        return pd.DataFrame.from_records(records).sort_values("run_id").reset_index(drop=True)
