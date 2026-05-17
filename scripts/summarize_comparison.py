"""Summarize YOLOv8 model comparison runs into a CSV table."""

from __future__ import annotations

import csv
from pathlib import Path


RUNS = {
    "YOLOv8m": Path("runs/helmet_yolov8m_person-bike-helmet3/results.csv"),
    "YOLOv8n": Path("runs_compare/yolov8n_120ep/results.csv"),
    "YOLOv8s": Path("runs_compare/yolov8s_120ep/results.csv"),
}


def read_rows(path: Path):
    if not path.is_file():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def best_by_map50(rows):
    if not rows:
        return None
    return max(rows, key=lambda r: float(r["metrics/mAP50(B)"]))


def format_row(model_name: str, row):
    if row is None:
        return {
            "model": model_name,
            "epoch": "",
            "precision": "",
            "recall": "",
            "mAP50": "",
            "mAP50-95": "",
            "source": "missing",
        }
    return {
        "model": model_name,
        "epoch": row["epoch"],
        "precision": f'{float(row["metrics/precision(B)"]):.4f}',
        "recall": f'{float(row["metrics/recall(B)"]):.4f}',
        "mAP50": f'{float(row["metrics/mAP50(B)"]):.4f}',
        "mAP50-95": f'{float(row["metrics/mAP50-95(B)"]):.4f}',
        "source": "best_mAP50_validation",
    }


def main():
    out_path = Path("runs_compare/comparison_summary.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name, path in RUNS.items():
        rows.append(format_row(model_name, best_by_map50(read_rows(path))))

    fieldnames = ["model", "epoch", "precision", "recall", "mAP50", "mAP50-95", "source"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(out_path)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
