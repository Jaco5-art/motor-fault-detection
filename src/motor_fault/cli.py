from __future__ import annotations

import argparse
import json
from pathlib import Path

from motor_fault.inference import predict_archive_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen GMM fault detector on one robot run."
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--archive-split", choices=("training", "testing"), default="testing"
    )
    parser.add_argument("--window-length", type=int, default=200)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--bundle-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("predictions"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frame, summary = predict_archive_run(
        archive_path=args.archive,
        run_id=args.run_id,
        archive_split=args.archive_split,
        window_length=args.window_length,
        stride=args.stride,
        bundle_dir=args.bundle_dir,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / f"{args.run_id}_window_predictions.csv"
    json_path = args.output_dir / f"{args.run_id}_summary.json"
    frame.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {len(frame)} window predictions to {csv_path}")
    print(f"Saved run summary to {json_path}")
    for motor, metrics in summary["motor_summary"].items():
        print(
            f"{motor}: alarms={metrics['predicted_fault_windows']}, "
            f"max_score={metrics['maximum_anomaly_score']:.4f}, "
            f"threshold={metrics['threshold']:.4f}"
        )


if __name__ == "__main__":
    main()
