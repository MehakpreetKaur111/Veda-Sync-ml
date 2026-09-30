"""Generate demo data and run the end-to-end risk modeling pipeline."""
import argparse
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def run(script: str, *arguments: str) -> None:
    command = [sys.executable, str(BASE_DIR / script), *arguments]
    print(f"\n{'=' * 60}\nRunning: {script}\n{'=' * 60}", flush=True)
    subprocess.run(command, cwd=BASE_DIR, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patients", type=int, default=60)
    parser.add_argument("--days", type=int, default=28)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run("00_generate_demo_data.py", "--patients", str(args.patients), "--days", str(args.days), "--seed", str(args.seed))
    run("01_data_loading.py")
    run("02_feature_engineering.py")
    run("03_train_model.py")
    run("04_evaluate.py")
    run("05_inference.py")


if __name__ == "__main__":
    main()