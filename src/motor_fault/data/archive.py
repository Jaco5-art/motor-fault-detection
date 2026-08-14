from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = ("time", "position", "temperature", "voltage", "label")
SIGNALS = ("position", "temperature", "voltage")
MOTORS = tuple(range(1, 7))
MEMBER_PATTERN = re.compile(
    r"(?:^|/)(training_data|testing_data)/(?:training_data/|testing_data/)?"
    r"(?P<run_id>\d{8}_\d+)/data_motor_(?P<motor>[1-6])\.csv$"
)


@dataclass(frozen=True)
class RunData:
    run_id: str
    time: np.ndarray
    features: np.ndarray
    labels: np.ndarray | None
    channel_names: tuple[str, ...]
    activity: str
    declared_failures: tuple[int, ...]

    @property
    def n_rows(self) -> int:
        return int(self.features.shape[0])


class KaggleRobotArchive:
    """Read the Kaggle archive directly without depending on extraction depth."""

    def __init__(self, archive_path: str | Path):
        self.path = Path(archive_path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self._members = self._index_members()
        self._conditions = self._read_conditions()

    @property
    def archive_sha256(self) -> str:
        digest = sha256()
        with self.path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(
            f"m{motor}_{signal}" for motor in MOTORS for signal in SIGNALS
        )

    def _index_members(self) -> dict[str, dict[str, dict[int, str]]]:
        indexed: dict[str, dict[str, dict[int, str]]] = {
            "training": {},
            "testing": {},
        }
        with ZipFile(self.path) as archive:
            for name in archive.namelist():
                match = MEMBER_PATTERN.search(name)
                if match is None:
                    continue
                split = "training" if match.group(1) == "training_data" else "testing"
                run_id = match.group("run_id")
                motor = int(match.group("motor"))
                indexed[split].setdefault(run_id, {})[motor] = name

        for split, runs in indexed.items():
            for run_id, motor_files in runs.items():
                missing = sorted(set(MOTORS) - set(motor_files))
                if missing:
                    raise ValueError(f"{split}/{run_id} is missing motors {missing}")
        if not indexed["training"]:
            raise ValueError("No labeled training runs were found in the archive")
        return indexed

    def _read_conditions(self) -> pd.DataFrame:
        with ZipFile(self.path) as archive:
            matches = [
                name
                for name in archive.namelist()
                if "/training_data/" in f"/{name}"
                and PurePosixPath(name).name == "Test conditions.xlsx"
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected one training conditions workbook, found {matches}")
            frame = pd.read_excel(BytesIO(archive.read(matches[0])))

        required = {"Test id", "Description"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Conditions workbook lacks {sorted(required - set(frame.columns))}")
        frame = frame.copy()
        def normalize_run_id(value: object) -> str:
            digits = re.sub(r"\D", "", str(value))
            if len(digits) != 14:
                raise ValueError(f"Unexpected Test id in conditions workbook: {value!r}")
            return f"{digits[:8]}_{digits[8:]}"

        frame["run_id"] = frame["Test id"].map(normalize_run_id)
        frame = frame.set_index("run_id", drop=False)
        return frame

    def run_ids(self, split: str = "training") -> list[str]:
        if split not in self._members:
            raise ValueError(f"Unknown archive split: {split}")
        return sorted(self._members[split])

    def member_name(self, run_id: str, motor: int, split: str = "training") -> str:
        try:
            return self._members[split][str(run_id)][int(motor)]
        except KeyError as exc:
            raise KeyError(f"Missing {split} run={run_id}, motor={motor}") from exc

    def load_motor_frame(
        self, run_id: str, motor: int, split: str = "training"
    ) -> pd.DataFrame:
        member = self.member_name(run_id, motor, split)
        with ZipFile(self.path) as archive:
            frame = pd.read_csv(archive.open(member))
        if tuple(frame.columns) != EXPECTED_COLUMNS:
            raise ValueError(
                f"Unexpected schema in {member}: {tuple(frame.columns)} != {EXPECTED_COLUMNS}"
            )
        numeric = frame.loc[:, list(EXPECTED_COLUMNS)].apply(pd.to_numeric, errors="coerce")
        if split == "training" and numeric.isna().any().any():
            bad = numeric.columns[numeric.isna().any()].tolist()
            raise ValueError(f"Missing/non-numeric labeled data in {member}: {bad}")
        if split == "training":
            values = set(numeric["label"].astype(int).unique())
            if not values.issubset({0, 1}):
                raise ValueError(f"Non-binary labels in {member}: {sorted(values)}")
        return numeric

    def activity(self, run_id: str) -> str:
        if str(run_id) not in self._conditions.index:
            return "unknown"
        return str(self._conditions.loc[str(run_id), "Description"])

    def declared_failures(self, run_id: str) -> tuple[int, ...]:
        if str(run_id) not in self._conditions.index:
            return ()
        row = self._conditions.loc[str(run_id)]
        failed = []
        for motor in MOTORS:
            column = f"Motor_{motor}_failure"
            if column in row.index and pd.notna(row[column]) and int(row[column]) == 1:
                failed.append(motor)
        return tuple(failed)

    def load_run(self, run_id: str, split: str = "training") -> RunData:
        frames = [self.load_motor_frame(run_id, motor, split) for motor in MOTORS]
        reference_time = frames[0]["time"].to_numpy(dtype=np.float64)
        if not pd.Series(reference_time).is_monotonic_increasing:
            raise ValueError(f"Time is not monotonic for {split}/{run_id}")
        if pd.Series(reference_time).duplicated().any():
            raise ValueError(f"Duplicate timestamps in {split}/{run_id}")

        feature_blocks = []
        label_blocks = []
        for motor, frame in zip(MOTORS, frames):
            motor_time = frame["time"].to_numpy(dtype=np.float64)
            if len(motor_time) != len(reference_time) or not np.allclose(
                motor_time, reference_time, rtol=0.0, atol=1e-9
            ):
                raise ValueError(f"Motor timestamps are not aligned in {split}/{run_id}")
            feature_blocks.append(frame.loc[:, list(SIGNALS)].to_numpy(dtype=np.float32))
            if split == "training":
                label_blocks.append(frame["label"].to_numpy(dtype=np.int8))

        features = np.concatenate(feature_blocks, axis=1)
        labels = np.column_stack(label_blocks) if label_blocks else None
        return RunData(
            run_id=str(run_id),
            time=reference_time,
            features=features,
            labels=labels,
            channel_names=self.channel_names,
            activity=self.activity(str(run_id)),
            declared_failures=self.declared_failures(str(run_id)),
        )
