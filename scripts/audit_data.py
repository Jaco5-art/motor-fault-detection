#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from motor_fault.data.archive import KaggleRobotArchive
from motor_fault.data.audit import audit_archive, write_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the labeled Kaggle archive")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/data_audit"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    archive = KaggleRobotArchive(args.archive)
    result = audit_archive(archive)
    write_audit(result, args.output)
    print(f"Audited {result.summary['labeled_runs']} labeled runs")
    print(f"Aligned rows: {result.summary['aligned_multivariate_rows']}")
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
