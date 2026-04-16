#!/usr/bin/env python3
"""Parse fine-tuning logs and export summary CSV."""

import argparse
import csv
import glob
import os
import re
from dataclasses import dataclass
from typing import List, Optional


ACC_RE = re.compile(r"Acc\. \(Correct/Total\):\s*([0-9]*\.?[0-9]+)")
MACRO_F1_RE = re.compile(r"Macro f1:\s*([0-9]*\.?[0-9]+)")
EPOCH_RE = re.compile(r"Epoch id:\s*(\d+)")
ROUND_RE = re.compile(r"r(\d+)")


@dataclass
class LogSummary:
    log_file: str
    round_id: Optional[int]
    best_dev_macro_f1: Optional[float]
    best_dev_acc: Optional[float]
    last_dev_macro_f1: Optional[float]
    last_dev_acc: Optional[float]
    test_macro_f1: Optional[float]
    test_acc: Optional[float]
    early_stopped: bool
    last_epoch_seen: Optional[int]


def _safe_float(v: Optional[float]) -> str:
    return "" if v is None else f"{v:.6f}"


def parse_single_log(path: str) -> LogSummary:
    dev_macro_f1: List[float] = []
    dev_acc: List[float] = []
    test_macro_f1: Optional[float] = None
    test_acc: Optional[float] = None

    in_test_phase = False
    early_stopped = False
    last_epoch_seen: Optional[int] = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()

            epoch_match = EPOCH_RE.search(line)
            if epoch_match:
                last_epoch_seen = int(epoch_match.group(1))

            if "early stopping" in line.lower():
                early_stopped = True

            if "Test set evaluation." in line:
                in_test_phase = True
                continue

            acc_match = ACC_RE.search(line)
            if acc_match:
                val = float(acc_match.group(1))
                if in_test_phase:
                    test_acc = val
                else:
                    dev_acc.append(val)
                continue

            f1_match = MACRO_F1_RE.search(line)
            if f1_match:
                val = float(f1_match.group(1))
                if in_test_phase:
                    test_macro_f1 = val
                else:
                    dev_macro_f1.append(val)
                continue

    best_dev_macro_f1 = max(dev_macro_f1) if dev_macro_f1 else None
    best_dev_acc = max(dev_acc) if dev_acc else None
    last_dev_macro_f1 = dev_macro_f1[-1] if dev_macro_f1 else None
    last_dev_acc = dev_acc[-1] if dev_acc else None

    filename = os.path.basename(path)
    round_match = ROUND_RE.search(filename)
    round_id = int(round_match.group(1)) if round_match else None

    return LogSummary(
        log_file=path,
        round_id=round_id,
        best_dev_macro_f1=best_dev_macro_f1,
        best_dev_acc=best_dev_acc,
        last_dev_macro_f1=last_dev_macro_f1,
        last_dev_acc=last_dev_acc,
        test_macro_f1=test_macro_f1,
        test_acc=test_acc,
        early_stopped=early_stopped,
        last_epoch_seen=last_epoch_seen,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-parse fine-tuning logs to CSV")
    parser.add_argument("--log_glob", type=str, default="logs/iscx_app_r*.log", help="Glob pattern for log files.")
    parser.add_argument("--output_csv", type=str, default="logs/iscx_app_record_auto.csv", help="Path to output CSV.")
    args = parser.parse_args()

    files = sorted(glob.glob(args.log_glob))
    if not files:
        raise FileNotFoundError(f"No log files matched: {args.log_glob}")

    summaries = [parse_single_log(p) for p in files]

    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as fw:
        writer = csv.writer(fw)
        writer.writerow([
            "round",
            "log_file",
            "best_dev_macro_f1",
            "best_dev_acc",
            "last_dev_macro_f1",
            "last_dev_acc",
            "test_macro_f1",
            "test_acc",
            "early_stopped",
            "last_epoch_seen",
        ])
        for s in sorted(summaries, key=lambda x: (x.round_id is None, x.round_id, x.log_file)):
            writer.writerow([
                "" if s.round_id is None else s.round_id,
                s.log_file,
                _safe_float(s.best_dev_macro_f1),
                _safe_float(s.best_dev_acc),
                _safe_float(s.last_dev_macro_f1),
                _safe_float(s.last_dev_acc),
                _safe_float(s.test_macro_f1),
                _safe_float(s.test_acc),
                int(s.early_stopped),
                "" if s.last_epoch_seen is None else s.last_epoch_seen,
            ])

    print(f"Parsed {len(summaries)} logs -> {args.output_csv}")


if __name__ == "__main__":
    main()
