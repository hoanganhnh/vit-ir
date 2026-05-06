"""
Master pipeline runner for SatLetter dataset collection
========================================================
Runs all dataset collection scripts in order:
  1. Download EMNIST Letters
  2. Generate rendered fonts
  3. Scrape NASA Landsat (automated + manual guide)
  4. Augment satellite images
  5. Generate satellite fonts (fonts on NASA textures)
  6. Verify all datasets

Usage:
  python scripts/run_pipeline.py              # Run steps 1, 3, 4, 5, 6
  python scripts/run_pipeline.py --all        # Run all including rendered fonts
  python scripts/run_pipeline.py --step 5     # Run specific step
"""

import os
import sys
import subprocess
import argparse


def run_step(step_num: int, script: str, args: list = None):
    """Run a pipeline step."""
    print(f"\n{'#' * 70}")
    print(f"# STEP {step_num}: {script}")
    print(f"{'#' * 70}\n")

    cmd = [sys.executable, script] + (args or [])
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)))
    if result.returncode != 0:
        print(f"\n❌ Step {step_num} failed with exit code {result.returncode}")
        return False
    print(f"\n✅ Step {step_num} completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description="SatLetter Dataset Pipeline")
    parser.add_argument("--step", type=int, help="Run specific step (1-6)")
    parser.add_argument("--all", action="store_true", help="Run all steps including augmentation")
    parser.add_argument("--emnist-max-train", type=int, default=2000, help="Max EMNIST train per class")
    parser.add_argument("--emnist-max-test", type=int, default=400, help="Max EMNIST test per class")
    parser.add_argument("--fonts-train", type=int, default=500, help="Rendered fonts train per class")
    parser.add_argument("--fonts-test", type=int, default=100, help="Rendered fonts test per class")
    parser.add_argument("--aug-multiplier", type=int, default=5, help="Satellite augmentation multiplier")
    args = parser.parse_args()

    steps = {
        1: ("scripts/01_download_emnist.py", [str(args.emnist_max_train), str(args.emnist_max_test)]),
        2: ("scripts/02_generate_rendered_fonts.py", [str(args.fonts_train), str(args.fonts_test)]),
        3: ("scripts/03_scrape_nasa_landsat.py", []),
        4: ("scripts/04_augment_satellite.py", [str(args.aug_multiplier)]),
        5: ("scripts/02_generate_rendered_fonts.py", [str(args.fonts_train), str(args.fonts_test), "--sat"]),
        6: ("scripts/05_verify_dataset.py", []),
    }

    if args.step:
        if args.step not in steps:
            print(f"Invalid step: {args.step}. Valid steps: 1-6")
            sys.exit(1)
        script, step_args = steps[args.step]
        run_step(args.step, script, step_args)
    else:
        # Default: 1→EMNIST, 3→NASA download, 4→augment NASA, 5→sat_fonts, 6→verify
        steps_to_run = [1, 2, 3, 4, 5, 6] if args.all else [1, 3, 4, 5, 6]
        for step_num in steps_to_run:
            script, step_args = steps[step_num]
            success = run_step(step_num, script, step_args)
            if not success:
                print(f"\n⚠️ Pipeline stopped at step {step_num}")
                sys.exit(1)

    print(f"\n{'#' * 70}")
    print("# PIPELINE COMPLETE")
    print(f"{'#' * 70}")


if __name__ == "__main__":
    main()
