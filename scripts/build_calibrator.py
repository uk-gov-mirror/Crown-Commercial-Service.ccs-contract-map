#!/usr/bin/env python3
"""
Build confidence calibrator from evaluation results.

This script reads the latest evaluation CSV and creates a calibrated
confidence model that maps heuristic scores to well-calibrated probabilities.

Usage:
    python scripts/build_calibrator.py --results-dir data/results --output calibrator.pkl
"""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contractmap.calibration import create_calibrator_from_evaluation


def main():
    parser = argparse.ArgumentParser(
        description="Build confidence calibrator from evaluation results."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "data/results",
        help="Directory containing evaluation CSV files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "calibrator.pkl",
        help="Output path for calibrator pickle file",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["isotonic", "platt"],
        default="isotonic",
        help="Calibration method (default: isotonic)",
    )

    args = parser.parse_args()

    print(f"Building calibrator from: {args.results_dir}")
    print(f"Output: {args.output}")
    print(f"Method: {args.method}")

    try:
        calibrator = create_calibrator_from_evaluation(
            results_dir=args.results_dir,
            output_path=args.output,
        )
        
        print(f"\n✅ Calibrator built successfully!")
        print(f"   Fitted: {calibrator.is_fitted}")
        
        # Test calibration on some sample scores
        test_scores = [0, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
        print(f"\nCalibration test:")
        for score in test_scores:
            calibrated = calibrator.calibrate(score)
            print(f"   Score {score:3d} → Confidence {calibrated:.3f}")
            
    except Exception as e:
        print(f"❌ Failed to build calibrator: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()