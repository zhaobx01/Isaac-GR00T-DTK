"""CPU-only, streaming finite-value verification for a local checkpoint."""

import argparse
import json
from pathlib import Path

import torch

from gr00t.experiment.numerics import scan_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = scan_checkpoint(args.checkpoint)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "failures"}))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
