from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from motor_fault.data.archive import KaggleRobotArchive, MOTORS


@dataclass(frozen=True)
class AuditResult:
    motor_table: pd.DataFrame
    run_table: pd.DataFrame
    summary: dict


def audit_archive(archive: KaggleRobotArchive) -> AuditResult:
    motor_records = []
    run_records = []
    for run_id in archive.run_ids("training"):
        run = archive.load_run(run_id)
        deltas = np.diff(run.time)
        run_record = {
            "run_id": run_id,
            "activity": run.activity,
            "n_rows": run.n_rows,
            "duration": float(run.time[-1] - run.time[0]),
            "sampling_delta_median": float(np.median(deltas)) if len(deltas) else np.nan,
            "sampling_delta_p95": float(np.percentile(deltas, 95)) if len(deltas) else np.nan,
            "declared_failure_motors": ",".join(map(str, run.declared_failures)),
        }
        for motor in MOTORS:
            frame = archive.load_motor_frame(run_id, motor)
            y = frame["label"].astype(np.int8)
            transitions = int(y.diff().fillna(0).ne(0).sum())
            record = {
                "run_id": run_id,
                "activity": run.activity,
                "motor": motor,
                "n_rows": len(frame),
                "fault_points": int(y.sum()),
                "fault_ratio": float(y.mean()),
                "label_transitions": transitions,
                "missing_values": int(frame.isna().sum().sum()),
                "time_monotonic": bool(frame["time"].is_monotonic_increasing),
                "duplicate_timestamps": int(frame["time"].duplicated().sum()),
            }
            motor_records.append(record)
            run_record[f"m{motor}_fault_points"] = record["fault_points"]
            run_record[f"m{motor}_fault_ratio"] = record["fault_ratio"]
        run_records.append(run_record)

    motor_table = pd.DataFrame.from_records(motor_records)
    run_table = pd.DataFrame.from_records(run_records)
    per_motor = {}
    for motor in MOTORS:
        subset = motor_table[motor_table["motor"] == motor]
        per_motor[f"m{motor}"] = {
            "fault_points": int(subset["fault_points"].sum()),
            "total_points": int(subset["n_rows"].sum()),
            "fault_ratio": float(subset["fault_points"].sum() / subset["n_rows"].sum()),
            "positive_runs": int((subset["fault_points"] > 0).sum()),
        }
    summary = {
        "archive_sha256": archive.archive_sha256,
        "schema": ["time", "position", "temperature", "voltage", "label"],
        "labeled_runs": len(run_table),
        "motors": len(MOTORS),
        "motor_csv_files": len(motor_table),
        "aligned_multivariate_rows": int(run_table["n_rows"].sum()),
        "input_channels": list(archive.channel_names),
        "missing_values": int(motor_table["missing_values"].sum()),
        "non_monotonic_files": int((~motor_table["time_monotonic"]).sum()),
        "duplicate_timestamps": int(motor_table["duplicate_timestamps"].sum()),
        "per_motor": per_motor,
    }
    return AuditResult(motor_table=motor_table, run_table=run_table, summary=summary)


def write_audit(result: AuditResult, output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    result.motor_table.to_csv(destination / "motor_audit.csv", index=False)
    result.run_table.to_csv(destination / "run_audit.csv", index=False)
    with (destination / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(result.summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
