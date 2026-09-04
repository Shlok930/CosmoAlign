"""Standalone Phase 4 runner; never invokes the Phase 3 pipeline."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from src.phase4_geometry_experiment import run_viewpoint_geometry_experiment
from src.phase4_illumination_experiment import run_illumination_experiment
from src.phase4_reporting import run_closure_report
from src.phase4_scale_normalization import run_scale_normalization_experiment
from src.phase4_step7_validator_diagnostic import run_validator_diagnostic
from src.phase4_stress_experiment import run_categorized_stress_evaluation


ROOT = Path(__file__).resolve().parent
PHASE4_ROOT = ROOT / "outputs" / "phase4"


def _git(command: str) -> str:
    try:
        return subprocess.check_output(["git", *command.split()], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNAVAILABLE"


def _metadata(run_root: Path, selected: list[str]) -> None:
    manifest = ROOT / "data" / "stress_tests" / "manifest.json"
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else "UNAVAILABLE"
    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_branch": _git("branch --show-current"),
        "git_commit": _git("rev-parse HEAD"),
        "working_tree_dirty": bool(_git("status --porcelain")),
        "python_version": platform.python_version(),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "experiment_name": "Phase 4 closure",
        "experiments": selected,
        "dataset_manifest_sha256": manifest_hash,
        "random_seed": 42,
        "ransac_threshold_px": 5.0,
        "lowe_ratio_threshold": 0.75,
        "sift_parameters": {"nfeatures": 0, "contrastThreshold": 0.04, "edgeThreshold": 10.0, "sigma": 1.6},
    }
    (run_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run isolated Phase 4 experiments without invoking Phase 3.")
    parser.add_argument("--scale", action="store_true", help="Run scale normalization.")
    parser.add_argument("--illumination", action="store_true", help="Run standalone illumination experiments.")
    parser.add_argument("--viewpoint", action="store_true", help="Run viewpoint geometry comparison.")
    parser.add_argument("--combined", action="store_true", help="Run Combined stress evaluation.")
    parser.add_argument("--step7", action="store_true", help="Run Step 7 validator diagnostic.")
    parser.add_argument("--report", action="store_true", help="Build a closure report from available results.")
    parser.add_argument("--all", action="store_true", help="Run all Phase 4 closure components in a new timestamped directory.")
    args = parser.parse_args()
    selected = [name for name in ("scale", "illumination", "viewpoint", "combined", "step7", "report") if getattr(args, name)]
    if args.all or not selected:
        selected = ["scale", "illumination", "viewpoint", "combined", "step7", "report"]
    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_root = PHASE4_ROOT / "closure" / run_id
    run_root.mkdir(parents=True)
    _metadata(run_root, selected)
    paths = {}
    if "scale" in selected:
        run_scale_normalization_experiment(output_dir=str(run_root / "scale_normalization"))
        paths["scale_csv"] = str(run_root / "scale_normalization" / "scale_normalization_results.csv")
    if "illumination" in selected:
        run_illumination_experiment(output_dir=str(run_root / "illumination"))
        paths["illumination_csv"] = str(run_root / "illumination" / "illumination_results.csv")
    if "viewpoint" in selected:
        run_viewpoint_geometry_experiment(output_dir=str(run_root / "geometry"))
        paths["viewpoint_csv"] = str(run_root / "geometry" / "geometry_experiments.csv")
    if "combined" in selected:
        run_categorized_stress_evaluation(output_dir=str(run_root / "stress"))
        paths["stress_csv"] = str(run_root / "stress" / "stress_experiments.csv")
    if "step7" in selected:
        run_validator_diagnostic(output_dir=str(run_root / "step7_validator_diagnostic"))
        paths["step7_csv"] = str(run_root / "step7_validator_diagnostic" / "validator_diagnostic.csv")
    if "report" in selected:
        run_closure_report(output_dir=str(run_root / "report"), **paths)
    print(f"Phase 4 run complete: {run_root}")


if __name__ == "__main__":
    main()
